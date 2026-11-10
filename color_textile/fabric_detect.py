"""Lightweight classical-CV fabric detection on a carton/tray background.

Designed for stationary fabric samples on USB cameras (e.g. 2592x1944 YUYV):
1. Downscale to ~648 px wide and convert to CIE L*a*b*.
2. Model the background from the outer 5% frame border (or a captured reference frame).
3. Compute ΔE76 to the background, threshold with bounded Otsu + shadow refinement.
4. Clean with morphology, select the largest connected component, and flood-fill
   internal holes so centre text/logos never punch holes in the fabric mask.
5. Upscale mask and contour to full resolution and run mask quality checks.
"""

from dataclasses import dataclass, field
import time
import cv2
import numpy as np

from .decoders import yuyv_to_bgr

DEFAULT_DETECT_WIDTH = 648
BORDER_FRACTION = 0.05
MIN_CONTRAST_DE = 5.0
LOW_CONTRAST_WARN_DE = 8.0
MIN_SOLIDITY_WARN = 0.88
MIN_AREA_FRAC_WARN = 0.008
MIN_SHORT_SIDE_PX = 36


@dataclass
class BackgroundModel:
    """Reference background model captured from the border or an empty tray."""

    mode: str = "border"  # "border" | "reference"
    median_lab: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    spread_de: float = 1.0
    lab_small: np.ndarray = None  # (h, w, 3) float32 CIE L*a*b* when mode == "reference"


@dataclass
class FabricDetection:
    """Result of fabric detection on one frame."""

    found: bool
    mask: np.ndarray  # (H, W) uint8, 0 or 255 at full frame resolution
    contour: np.ndarray  # (N, 2) int32 full-res contour points, or empty (0, 2)
    box: np.ndarray  # (4, 2) int32 minAreaRect corners at full resolution
    center: tuple = (0.0, 0.0)
    long_side: float = 0.0
    short_side: float = 0.0
    angle: float = 0.0
    area_px: int = 0
    area_frac: float = 0.0
    solidity: float = 0.0
    contrast_de: float = 0.0
    threshold_de: float = 0.0
    bg_median_lab: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    bg_spread_de: float = 0.0
    touches_border: bool = False
    holes_filled_px: int = 0
    warnings: list = field(default_factory=list)
    elapsed_ms: float = 0.0


def frame_to_bgr(frame):
    """Accept either a packed YUYV (H, W, 2) array or a BGR (H, W, 3) image."""
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[2] == 2:
        return yuyv_to_bgr(arr)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    raise ValueError(f"Định dạng frame không hỗ trợ: {arr.shape}")


def bgr_to_lab_fast(bgr):
    """Fast BGR uint8 -> CIE L*a*b* float32 (L* 0..100, a* -128..127, b* -128..127)."""
    lab_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab_u8[..., 0] *= 100.0 / 255.0
    lab_u8[..., 1] -= 128.0
    lab_u8[..., 2] -= 128.0
    return lab_u8


def _downscale_bgr(bgr, max_width=DEFAULT_DETECT_WIDTH):
    h, w = bgr.shape[:2]
    if w <= max_width:
        return bgr, 1.0, 1.0
    small_w = max(32, int(max_width))
    small_h = max(32, int(round(h * small_w / w)))
    small = cv2.resize(bgr, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return small, w / float(small_w), h / float(small_h)


def build_background_model(frame, max_width=DEFAULT_DETECT_WIDTH):
    """Capture an explicit empty-tray background model from a YUYV or BGR frame."""
    bgr = frame_to_bgr(frame)
    small, _, _ = _downscale_bgr(bgr, max_width=max_width)
    lab = bgr_to_lab_fast(small)
    flat = lab.reshape(-1, 3)
    median = np.median(flat, axis=0).astype(np.float32)
    de = np.linalg.norm(flat - median, axis=-1)
    spread = max(0.5, float(np.median(de) * 1.4826))
    return BackgroundModel(mode="reference", median_lab=median, spread_de=spread, lab_small=lab)


def _find_tray_bounds(lab):
    """Detect the flat inner carton tray bounds when dark vertical box walls are visible."""
    h, w = lab.shape[:2]
    if h < 64 or w < 64:
        return 0, 0, w, h
    y_top_band = slice(int(h * 0.06), max(int(h * 0.06) + 2, int(h * 0.16)))
    y_bot_band = slice(min(h - 2, int(h * 0.84)), int(h * 0.94))
    prof_x = np.maximum(
        np.median(lab[y_top_band, :, 0], axis=0),
        np.median(lab[y_bot_band, :, 0], axis=0),
    )
    center_l = float(np.median(prof_x[int(w * 0.35):int(w * 0.65)]))
    thr_l = center_l - 8.0
    mid_x = w // 2
    left_bad = np.where(prof_x[:mid_x] < thr_l)[0]
    right_bad = np.where(prof_x[mid_x:] < thr_l)[0]
    pad_x = max(6, int(round(w * 0.04)))
    x0 = min(mid_x - 20, int(left_bad[-1]) + pad_x) if len(left_bad) else 0
    x1 = max(mid_x + 20, mid_x + int(right_bad[0]) - pad_x) if len(right_bad) else w

    tw = max(16, x1 - x0)
    x_left_band = slice(x0 + int(tw * 0.08), max(x0 + int(tw * 0.08) + 2, x0 + int(tw * 0.25)))
    x_right_band = slice(min(x1 - 2, x0 + int(tw * 0.75)), x0 + int(tw * 0.92))
    prof_y = np.maximum(
        np.median(lab[:, x_left_band, 0], axis=1),
        np.median(lab[:, x_right_band, 0], axis=1),
    )
    mid_y = h // 2
    top_bad = np.where(prof_y[:mid_y] < thr_l)[0]
    bot_bad = np.where(prof_y[mid_y:] < thr_l)[0]
    pad_y = max(6, int(round(h * 0.03)))
    y0 = min(mid_y - 20, int(top_bad[-1]) + pad_y) if len(top_bad) else 0
    y1 = max(mid_y + 20, mid_y + int(bot_bad[0]) - pad_y) if len(bot_bad) else h
    return x0, y0, x1, y1


def _border_background(lab, border_frac=BORDER_FRACTION):
    """Estimate background median Lab, spread, and 2D illumination-corrected ΔE map."""
    h, w = lab.shape[:2]
    x0, y0, x1, y1 = _find_tray_bounds(lab)
    tray = lab[y0:y1, x0:x1]
    th, tw = tray.shape[:2]
    bw = max(3, int(round(min(th, tw) * max(border_frac, 0.07))))
    mask = np.zeros((th, tw), dtype=bool)
    mask[:bw, :] = True
    mask[-bw:, :] = True
    mask[:, :bw] = True
    mask[:, -bw:] = True
    border_pixels = tray[mask]
    median = np.median(border_pixels, axis=0).astype(np.float32)

    tb = np.r_[0:bw, th - bw:th]
    lr = np.r_[0:bw, tw - bw:tw]
    raw_px = np.clip(np.median(tray[tb, :, 0], axis=0), median[0] - 8.0, median[0] + 8.0).astype(np.float32)
    raw_py = np.clip(np.median(tray[:, lr, 0], axis=1), median[0] - 8.0, median[0] + 8.0).astype(np.float32)
    k_px = max(3, (tw // 12) | 1)
    k_py = max(3, (th // 12) | 1)
    px = cv2.GaussianBlur(raw_px.reshape(1, -1), (k_px, 1), 0)
    py = cv2.GaussianBlur(raw_py.reshape(-1, 1), (1, k_py), 0)
    corner_l = float(np.clip(np.median(tray[np.ix_(tb, lr)][..., 0]), median[0] - 8.0, median[0] + 8.0))
    l_bg = py + px - corner_l

    dl = tray[..., 0] - l_bg
    da = tray[..., 1] - median[1]
    db = tray[..., 2] - median[2]
    tray_de = np.sqrt(dl * dl + da * da + db * db)

    b_de = tray_de[mask]
    spread = max(0.5, float(np.median(b_de) * 1.4826))
    b_p90 = float(np.percentile(b_de, 90))

    dist_de = np.zeros((h, w), dtype=np.float32)
    dist_de[y0:y1, x0:x1] = tray_de
    return median, spread, dist_de, b_p90, (x0, y0, x1, y1)


def _fill_internal_holes(binary_mask):
    """Fill all internal holes (such as printed text/logos) inside a binary 0/255 mask."""
    h, w = binary_mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = binary_mask
    flood = padded.copy()
    ff_mask = np.zeros((h + 4, w + 4), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood[1:-1, 1:-1])
    filled = cv2.bitwise_or(binary_mask, holes)
    holes_px = int(np.count_nonzero(filled) - np.count_nonzero(binary_mask))
    return filled, max(0, holes_px)


def mask_iou(mask_a, mask_b):
    """Intersection-over-Union of two binary masks."""
    if mask_a is None or mask_b is None or mask_a.shape != mask_b.shape:
        return 0.0
    a = mask_a > 0
    b = mask_b > 0
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 0.0
    inter = int(np.count_nonzero(a & b))
    return inter / float(union)


def detect_fabric(frame, background=None, max_width=DEFAULT_DETECT_WIDTH,
                  border_frac=BORDER_FRACTION, min_contrast_de=MIN_CONTRAST_DE, anchor=None):
    """Detect the main fabric piece on a carton/tray background.

    Parameters
    ----------
    frame : np.ndarray
        Packed YUYV (H, W, 2) or BGR (H, W, 3) image.
    background : BackgroundModel, optional
        Captured reference background; if None, uses the outer 5% frame border.
    max_width : int
        Working width for downscaled detection.
    border_frac : float
        Fraction of frame size used for border background estimation.
    min_contrast_de : float
        Minimum ΔE76 threshold floor between fabric and background.
    anchor : tuple[float, float], optional
        Normalised (x, y) in 0..1 to prefer the component containing/nearest to it.
    """
    t0 = time.perf_counter()
    bgr = frame_to_bgr(frame)
    full_h, full_w = bgr.shape[:2]
    small, sx, sy = _downscale_bgr(bgr, max_width=max_width)
    sh, sw = small.shape[:2]
    lab = bgr_to_lab_fast(small)

    if background is not None and background.mode == "reference" and background.lab_small is not None:
        ref_lab = background.lab_small
        if ref_lab.shape[:2] != (sh, sw):
            ref_lab = cv2.resize(ref_lab, (sw, sh), interpolation=cv2.INTER_LINEAR)
        bg_median = background.median_lab.copy()
        bg_spread = float(background.spread_de)
        x0, y0, x1, y1 = _find_tray_bounds(ref_lab)
        dist_de = np.zeros((sh, sw), dtype=np.float32)
        tray_diff = np.linalg.norm(lab[y0:y1, x0:x1] - ref_lab[y0:y1, x0:x1], axis=-1)
        if bg_spread > 4.0:
            tray_med = np.linalg.norm(lab[y0:y1, x0:x1] - bg_median, axis=-1)
            tray_diff = np.minimum(tray_diff, tray_med)
        dist_de[y0:y1, x0:x1] = tray_diff
        b_p90 = bg_spread * 1.5
    else:
        bg_median, bg_spread, dist_de, b_p90, (x0, y0, x1, y1) = _border_background(lab, border_frac=border_frac)

    tray_de = dist_de[y0:y1, x0:x1]
    floor_de = max(float(min_contrast_de), b_p90 + 4.5, bg_spread * 3.2 + 1.8)
    # Use 99.5th percentile inside the tray so small fabric pieces (~1-2% of frame) still scale Otsu properly
    p99 = float(np.percentile(tray_de, 99.5)) if tray_de.size else 16.0
    otsu_cap = max(16.0, min(36.0, p99 * 0.95))
    scaled = np.clip(tray_de * (255.0 / otsu_cap), 0, 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_de = float(otsu_val) * (otsu_cap / 255.0)

    cap_de = max(floor_de + 2.0, min(22.0, floor_de + 9.0))
    threshold_de = float(np.clip(otsu_de, floor_de, cap_de))

    raw_fg = dist_de >= threshold_de
    if np.count_nonzero(raw_fg) > 20:
        fg_med_de = float(np.median(dist_de[raw_fg]))
        # Trim faint drop shadows along the edge when the fabric body has strong contrast
        shadow_trim = min(13.0, 0.45 * fg_med_de)
        if shadow_trim > threshold_de:
            threshold_de = float(shadow_trim)
            raw_fg = dist_de >= threshold_de

    fg_u8 = (raw_fg.astype(np.uint8)) * 255
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, k_open)
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, k_close)

    # OpenCV external contour extraction: RETR_EXTERNAL naturally ignores internal text/logo holes
    contours_raw, _ = cv2.findContours(fg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_comp_px = max(25, int(sh * sw * 0.001))
    valid_contours = [c for c in contours_raw if cv2.contourArea(c) >= min_comp_px]

    if not valid_contours:
        elapsed = (time.perf_counter() - t0) * 1e3
        return FabricDetection(
            found=False,
            mask=np.zeros((full_h, full_w), dtype=np.uint8),
            contour=np.zeros((0, 2), dtype=np.int32),
            box=np.zeros((4, 2), dtype=np.int32),
            threshold_de=threshold_de,
            bg_median_lab=bg_median,
            bg_spread_de=bg_spread,
            warnings=["Không phát hiện được tấm vải. Kiểm tra độ tương phản với nền carton."],
            elapsed_ms=elapsed,
        )

    if anchor is not None:
        ax = float(np.clip(anchor[0] * (sw - 1), 0, sw - 1))
        ay = float(np.clip(anchor[1] * (sh - 1), 0, sh - 1))
        containing = [c for c in valid_contours if cv2.pointPolygonTest(c, (ax, ay), False) >= 0]
        if containing:
            chosen_contour = max(containing, key=cv2.contourArea)
        else:
            chosen_contour = max(valid_contours, key=lambda c: cv2.pointPolygonTest(c, (ax, ay), True))
    else:
        chosen_contour = max(valid_contours, key=cv2.contourArea)

    comp_mask = np.zeros((sh, sw), dtype=np.uint8)
    cv2.drawContours(comp_mask, [chosen_contour], -1, 255, thickness=cv2.FILLED)
    comp_mask = cv2.morphologyEx(comp_mask, cv2.MORPH_CLOSE, k_close)
    filled_small = cv2.morphologyEx(comp_mask, cv2.MORPH_OPEN, k_open)
    holes_small_px = int(np.count_nonzero((filled_small > 0) & (~raw_fg)))

    contours_small, _ = cv2.findContours(filled_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours_small:
        elapsed = (time.perf_counter() - t0) * 1e3
        return FabricDetection(
            found=False,
            mask=np.zeros((full_h, full_w), dtype=np.uint8),
            contour=np.zeros((0, 2), dtype=np.int32),
            box=np.zeros((4, 2), dtype=np.int32),
            threshold_de=threshold_de,
            bg_median_lab=bg_median,
            bg_spread_de=bg_spread,
            warnings=["Mặt nạ vải rỗng sau khi làm sạch."],
            elapsed_ms=elapsed,
        )

    main_contour_small = max(contours_small, key=cv2.contourArea)
    if (sw, sh) == (full_w, full_h):
        main_contour = main_contour_small
    else:
        pts = main_contour_small.astype(np.float32)
        pts[..., 0] *= sx
        pts[..., 1] *= sy
        main_contour = np.round(pts).astype(np.int32)
    clean_full = np.zeros((full_h, full_w), dtype=np.uint8)
    cv2.drawContours(clean_full, [main_contour], -1, 255, thickness=cv2.FILLED)

    rect = cv2.minAreaRect(main_contour)
    (cx, cy), (rw, rh), angle = rect
    long_side = float(max(rw, rh))
    short_side = float(min(rw, rh))
    box = np.int32(np.round(cv2.boxPoints(rect)))

    area_px = int(round(float(cv2.contourArea(main_contour))))
    if area_px <= 0:
        area_px = int(np.count_nonzero(clean_full))
    area_frac = area_px / float(full_h * full_w)
    hull = cv2.convexHull(main_contour)
    hull_area = max(1.0, float(cv2.contourArea(hull)))
    solidity = min(1.0, area_px / hull_area)
    contrast_de = float(np.median(dist_de[filled_small > 0])) if np.any(filled_small > 0) else 0.0
    holes_filled_px = int(round(holes_small_px * sx * sy))

    margin = max(2, int(round(min(full_h, full_w) * 0.004)))
    touches_border = bool(
        clean_full[:margin, :].any()
        or clean_full[-margin:, :].any()
        or clean_full[:, :margin].any()
        or clean_full[:, -margin:].any()
    )

    warnings = []
    if touches_border:
        warnings.append("Vải chạm mép khung hình — hãy đặt vải nằm trọn trong khung.")
    if area_frac < MIN_AREA_FRAC_WARN or short_side < MIN_SHORT_SIDE_PX:
        warnings.append(f"Vải quá nhỏ ({int(round(long_side))}×{int(round(short_side))} px).")
    if solidity < MIN_SOLIDITY_WARN:
        warnings.append(f"Độ đặc mặt nạ thấp ({solidity:.2f} < {MIN_SOLIDITY_WARN}) — kiểm tra bóng đổ hoặc nếp gấp.")
    if contrast_de < LOW_CONTRAST_WARN_DE:
        warnings.append(f"Độ tương phản vải–nền thấp (ΔE {contrast_de:.1f}) — nên dùng nút Chụp nền.")

    elapsed = (time.perf_counter() - t0) * 1e3
    return FabricDetection(
        found=True,
        mask=clean_full,
        contour=main_contour.reshape(-1, 2).astype(np.int32),
        box=box,
        center=(float(cx), float(cy)),
        long_side=long_side,
        short_side=short_side,
        angle=float(angle),
        area_px=area_px,
        area_frac=area_frac,
        solidity=float(solidity),
        contrast_de=contrast_de,
        threshold_de=threshold_de,
        bg_median_lab=bg_median,
        bg_spread_de=bg_spread,
        touches_border=touches_border,
        holes_filled_px=holes_filled_px,
        warnings=warnings,
        elapsed_ms=elapsed,
    )
