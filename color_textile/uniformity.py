"""Ring-based fabric colour uniformity check (inset 5% from the fabric border).

Given a detected fabric mask:
1. Compute the Euclidean distance transform inside the mask.
2. Build a ring band `d <= dist < d + w` where `d` and `w` default to 5% of the
   fabric's short side (with pixel floors to avoid blurred edges and shadows).
3. Partition the ring into K equal-arc-length segments clockwise from top-left.
4. Reject printed text/logo pixels inside each segment (ΔE > 8 vs segment median)
   and flag segments with > 20% rejected pixels as contaminated.
5. Sample per-frame linear-RGB segment means via `np.bincount`, aggregate N frames
   with MAD + Mean, and judge uniformity from max pairwise ΔE76 vs threshold + noise floor.
"""

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
import cv2
import numpy as np

from .colorimetry import FORMULAS, delta_e, linear_rgb_to_lab, linear_to_srgb, srgb8_to_lab, srgb_to_linear
from .fabric_detect import FabricDetection, bgr_to_lab_fast, frame_to_bgr
from .measurement import _LINEAR_LUT, mad_filter_rgb, split_half_noise_de

VERDICT_LABELS = {
    "pass": "ĐỒNG MÀU",
    "fail": "KHÔNG ĐỒNG MÀU",
    "uncertain": "CHƯA ĐỦ TIN CẬY",
}

SEGMENT_PALETTE_HEX = (
    "#00c8ff", "#4cd97b", "#ffb000", "#ff5c9d", "#a370f7", "#38d9a9",
    "#ffd43b", "#ff8787", "#74c0fc", "#69db7c", "#ffa94d", "#e599f7",
    "#63e6be", "#ffe066", "#ff6b6b", "#91a7ff",
)


@dataclass
class UniformityParams:
    inset_pct: float = 5.0
    width_pct: float = 5.0
    min_inset_px: int = 8
    min_width_px: int = 10
    segments: int = 12
    min_segment_px: int = 150
    frames: int = 30
    threshold_de: float = 0.4
    de_formula: str = "de00"
    white: str = "D50"
    outlier_z: float = 3.5
    outlier_de: float = 1.0
    pixel_outlier_de: float = 8.0
    max_contaminated_pct: float = 20.0
    lock: bool = True
    bg_mode: str = "border"  # "border" | "reference"
    interval_s: float = 120.0
    duration_min: float = 0.0
    filter_dirty_segments: bool = False


@dataclass
class MasterFabricRef:
    timestamp: str
    segments: int
    long_side: float
    short_side: float
    area_px: int
    inset_pct: float
    width_pct: float
    white: str
    ring_mean_rgb: np.ndarray  # (3,)
    ring_mean_lab: np.ndarray  # (3,)
    mean_rgb: np.ndarray  # (K, 3)
    mean_lab: np.ndarray  # (K, 3)
    noise_floor_de: float
    spatial_tau_rgb: np.ndarray = None  # (1, 3)
    clean_segments: np.ndarray = None  # (K,) bool
    dirty_segments: np.ndarray = None  # (K,) bool
    folder: str = ""


@dataclass
class RingLayout:
    valid: bool
    segments: int
    inset_px: int
    width_px: int
    dist_max_px: float
    labels: np.ndarray  # (H, W) int16: 0 outside, 1..K for clean segment pixels
    raw_labels: np.ndarray  # (H, W) int16 before text/print outlier pixel removal
    centroids: np.ndarray  # (K, 2) float32 (x, y) pixel coordinates
    pixel_counts: np.ndarray  # (K,) int32 clean pixel count per segment
    raw_pixel_counts: np.ndarray  # (K,) int32 total pixel count before text removal
    rejected_pct: np.ndarray  # (K,) float32 % of pixels rejected as text/print
    contaminated: np.ndarray  # (K,) bool: True when rejected_pct > max_contaminated_pct
    crop_box: tuple = (0, 0, 0, 0)  # (x0, y0, x1, y1) with x0, x1 even
    crop_labels: np.ndarray = None
    fabric_mask: np.ndarray = None  # (H, W) uint8 binary mask of detected fabric
    lum_gain: np.ndarray = None  # (K, 1) linear-light flat-field gain per segment
    chroma_delta: np.ndarray = None  # (K, 3) zero-mean chromatic glare offset per segment
    spatial_tau_rgb: np.ndarray = None  # (1, 3) per-channel within-segment weave + 8-bit floor
    _field_calibrated: bool = False
    warnings: list = field(default_factory=list)


@dataclass
class UniformityResult:
    params: UniformityParams
    segments: int
    frame_count: int
    elapsed_s: float
    mean_rgb: np.ndarray  # (K, 3) MAD + Mean RGB per segment
    std_rgb: np.ndarray  # (K, 3) frame-to-frame std RGB per segment
    mean_lab: np.ndarray  # (K, 3) Lab per segment
    std_lab: np.ndarray  # (K, 3) frame-to-frame std Lab per segment
    ring_mean_rgb: np.ndarray  # (3,) overall ring mean RGB (valid segments)
    ring_mean_lab: np.ndarray  # (3,) overall ring mean Lab (valid segments)
    de_to_mean: np.ndarray  # (K,) ΔE of each segment to ring_mean_lab
    delta_lab: np.ndarray  # (K, 3) (ΔL*, Δa*, Δb*) vs ring_mean_lab
    pairwise_de: np.ndarray  # (K, K) pairwise ΔE matrix
    max_pair_de: float  # primary metric: max ΔE between any 2 valid segments
    worst_pair: tuple  # (seg_i, seg_j) 1-based indices
    max_de_to_mean: float
    worst_segment: int  # 1-based segment index with largest ΔE to ring mean
    seg_noise_de: np.ndarray  # (K,) p95 split-half noise floor per segment
    noise_floor_de: float  # overall noise floor across valid segments
    pixel_counts: np.ndarray  # (K,) int32
    rejected_pct: np.ndarray  # (K,) float32
    contaminated: np.ndarray  # (K,) bool
    clip_pct: np.ndarray  # (K,) float32 % clipped pixels
    outlier_frames: np.ndarray  # (K,) int32 number of MAD-rejected frames per segment
    verdict: str  # "pass" | "fail" | "uncertain"
    verdict_label: str
    explanation: str
    rgb_frames: np.ndarray = None  # (N, K, 3)
    lab_frames: np.ndarray = None  # (N, K, 3)
    times: np.ndarray = None  # (N,)
    warnings: list = field(default_factory=list)
    has_master: bool = False
    master_ring_rgb: np.ndarray = None  # (3,)
    master_ring_lab: np.ndarray = None  # (3,)
    overall_master_de: float = None
    overall_delta_lab_master: np.ndarray = None  # (3,)
    de_to_master: np.ndarray = None  # (K,)
    delta_lab_to_master: np.ndarray = None  # (K, 3)
    max_master_de: float = None
    worst_master_segment: int = None
    size_diff_pct: float = None
    dirty_segments: np.ndarray = None  # (K,) bool: segments rejected as localized dirt/stains
    is_timeseries: bool = False
    round_index: int = 0
    elapsed_min: float = 0.0


@dataclass
class UniformityTimeSeriesRound:
    round_index: int
    timestamp: str
    elapsed_s: float
    elapsed_min: float
    verdict: str
    verdict_label: str
    overall_master_de: float
    max_master_de: float
    worst_master_segment: int
    max_pair_de: float
    worst_pair: tuple
    noise_floor_de: float
    valid_segments: int
    dirty_segments_list: list
    ring_mean_rgb: np.ndarray  # (3,)
    ring_mean_lab: np.ndarray  # (3,)
    delta_lab_master: np.ndarray  # (3,)
    de_to_master: np.ndarray  # (K,)
    de_to_mean: np.ndarray  # (K,)
    mean_lab: np.ndarray  # (K, 3)
    delta_lab_to_master: np.ndarray  # (K, 3)
    contaminated: np.ndarray  # (K,) bool
    dirty_mask: np.ndarray  # (K,) bool


def _order_contour_clockwise_from_top_left(contour_pts):
    """Order a closed contour (N, 2) clockwise in image coordinates starting at top-left."""
    pts = np.asarray(contour_pts, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return pts
    # In image coords (x right, y down), oriented area > 0 means clockwise
    area = cv2.contourArea(pts.reshape(-1, 1, 2), oriented=True)
    if area < 0:
        pts = pts[::-1].copy()
    start_idx = int(np.argmin(pts[:, 0] + pts[:, 1]))
    return np.roll(pts, -start_idx, axis=0)


def _sample_equal_arc_anchors(contour_pts, k_segments, anchors_per_seg=32):
    """Sample M = k_segments * anchors_per_seg evenly spaced points along a closed contour."""
    pts = _order_contour_clockwise_from_top_left(contour_pts)
    closed = np.vstack([pts, pts[0:1]])
    diffs = np.diff(closed, axis=0)
    step_len = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(step_len)])
    total_len = float(cum[-1])
    total_anchors = k_segments * anchors_per_seg
    if total_len <= 1e-3:
        anchors = np.repeat(pts[:1], total_anchors, axis=0)
    else:
        targets = np.linspace(0.0, total_len, total_anchors, endpoint=False)
        ax = np.interp(targets, cum, closed[:, 0])
        ay = np.interp(targets, cum, closed[:, 1])
        anchors = np.stack([ax, ay], axis=-1).astype(np.float32)
    anchor_seg = (np.arange(total_anchors, dtype=np.int16) // anchors_per_seg) + 1
    return anchors, anchor_seg


def _assign_segments_by_anchors(ring_mask, anchors, anchor_seg, k_segments):
    """Assign each True pixel in ring_mask to the segment of its nearest midline anchor."""
    h, w = ring_mask.shape
    labels = np.zeros((h, w), dtype=np.int16)
    ys, xs = np.nonzero(ring_mask)
    if len(xs) == 0:
        return labels, np.zeros((k_segments, 2), dtype=np.float32)

    pts = np.stack([xs, ys], axis=-1).astype(np.float32)
    assigned = np.empty(len(pts), dtype=np.int16)
    chunk = 4096
    for i in range(0, len(pts), chunk):
        sub = pts[i:i + chunk]  # (C, 2)
        d2 = (sub[:, None, 0] - anchors[None, :, 0]) ** 2 + (sub[:, None, 1] - anchors[None, :, 1]) ** 2
        nearest = np.argmin(d2, axis=1)
        assigned[i:i + chunk] = anchor_seg[nearest]

    labels[ys, xs] = assigned
    centroids = np.zeros((k_segments, 2), dtype=np.float32)
    for k in range(1, k_segments + 1):
        sel = assigned == k
        if np.any(sel):
            centroids[k - 1, 0] = float(xs[sel].mean())
            centroids[k - 1, 1] = float(ys[sel].mean())
        else:
            ak = anchors[anchor_seg == k]
            centroids[k - 1] = ak.mean(axis=0)
    return labels, centroids


def build_ring_layout(detection: FabricDetection, params: UniformityParams, reference_frame=None):
    """Build the 5%-inset ring and K equal-perimeter segments inside `detection.mask`."""
    mask = np.asarray(detection.mask, dtype=np.uint8)
    h, w = mask.shape[:2]
    warnings = list(detection.warnings)

    if not detection.found or np.count_nonzero(mask) == 0:
        empty_i16 = np.zeros((h, w), dtype=np.int16)
        k = max(1, int(params.segments))
        return RingLayout(
            valid=False, segments=k, inset_px=0, width_px=0, dist_max_px=0.0,
            labels=empty_i16, raw_labels=empty_i16,
            centroids=np.zeros((k, 2), dtype=np.float32),
            pixel_counts=np.zeros(k, dtype=np.int32),
            raw_pixel_counts=np.zeros(k, dtype=np.int32),
            rejected_pct=np.zeros(k, dtype=np.float32),
            contaminated=np.zeros(k, dtype=bool),
            warnings=warnings or ["Chưa phát hiện được tấm vải."],
        )

    if len(detection.contour) >= 3:
        cx0, cy0, cw, ch = cv2.boundingRect(detection.contour.reshape(-1, 1, 2))
        bx0 = max(0, cx0 - 2)
        by0 = max(0, cy0 - 2)
        bx1 = min(w, cx0 + cw + 2)
        by1 = min(h, cy0 + ch + 2)
    else:
        bx0, by0, bx1, by1 = 0, 0, w, h

    sub_mask = (mask[by0:by1, bx0:bx1] > 0).astype(np.uint8)
    padded_sub = np.pad(sub_mask, 1, mode="constant", constant_values=0)
    dist_sub = cv2.distanceTransform(padded_sub, cv2.DIST_L2, 5)[1:-1, 1:-1]
    dist_max = float(dist_sub.max())

    short_side = max(1.0, float(detection.short_side))
    d = max(int(params.min_inset_px), int(round(short_side * params.inset_pct / 100.0)))
    band_w = max(int(params.min_width_px), int(round(short_side * params.width_pct / 100.0)))
    min_total = int(params.min_inset_px) + int(params.min_width_px) + 2

    if dist_max < min_total:
        max_d = max(2, int(dist_max * 0.4))
        empty_i16 = np.zeros((h, w), dtype=np.int16)
        k = max(1, int(params.segments))
        warnings.append(
            f"Vải quá nhỏ cho khoảng cách biên hiện tại (d={d} px, w={band_w} px, bán kính trong tối đa {dist_max:.0f} px; "
            f"gợi ý d tối đa ~{max_d} px)."
        )
        return RingLayout(
            valid=False, segments=k, inset_px=d, width_px=band_w, dist_max_px=dist_max,
            labels=empty_i16, raw_labels=empty_i16,
            centroids=np.zeros((k, 2), dtype=np.float32),
            pixel_counts=np.zeros(k, dtype=np.int32),
            raw_pixel_counts=np.zeros(k, dtype=np.int32),
            rejected_pct=np.zeros(k, dtype=np.float32),
            contaminated=np.zeros(k, dtype=bool),
            warnings=warnings,
        )

    if d + band_w > dist_max - 1.5:
        band_w = min(band_w, max(int(params.min_width_px), int(dist_max * 0.35)))
        d = min(d, max(int(params.min_inset_px), int(dist_max - band_w - 1)))
        warnings.append(f"Đã thu vòng sát tâm vải (d={d} px / bán kính tâm {dist_max:.0f} px).")

    ring_sub = (dist_sub >= d) & (dist_sub < d + band_w)
    total_ring_px = int(np.count_nonzero(ring_sub))
    k = max(4, int(params.segments))
    if total_ring_px < k * params.min_segment_px and total_ring_px >= 4 * params.min_segment_px:
        new_k = max(4, total_ring_px // params.min_segment_px)
        if new_k < k:
            warnings.append(f"Tự giảm số đoạn từ {k} xuống {new_k} để mỗi đoạn đủ điểm ảnh (≥ {params.min_segment_px} px).")
            k = new_k

    mid_sub = (dist_sub >= d + band_w * 0.5).astype(np.uint8) * 255
    mid_contours, _ = cv2.findContours(mid_sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not mid_contours:
        inner_sub = (dist_sub >= d).astype(np.uint8) * 255
        mid_contours, _ = cv2.findContours(inner_sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mid_contour = max(mid_contours, key=lambda c: cv2.arcLength(c, True)).reshape(-1, 2)

    anchors, anchor_seg = _sample_equal_arc_anchors(mid_contour, k)
    raw_sub, centroids = _assign_segments_by_anchors(ring_sub, anchors, anchor_seg, k)
    centroids[:, 0] += bx0
    centroids[:, 1] += by0

    raw_labels = np.zeros((h, w), dtype=np.int16)
    raw_labels[by0:by1, bx0:bx1] = raw_sub
    clean_labels = raw_labels.copy()

    ys, xs = np.nonzero(raw_sub > 0)
    x0 = (bx0 + int(xs.min())) & ~1
    x1 = min(w, (bx0 + int(xs.max()) + 2) & ~1)
    y0 = by0 + int(ys.min())
    y1 = min(h, by0 + int(ys.max()) + 1)

    raw_counts = np.bincount(raw_labels[y0:y1, x0:x1].ravel(), minlength=k + 1)[1:k + 1].astype(np.int32)
    rejected_pct = np.zeros(k, dtype=np.float32)
    contaminated = np.zeros(k, dtype=bool)

    if reference_frame is not None and params.pixel_outlier_de > 0:
        crop_bgr = frame_to_bgr(reference_frame[y0:y1, x0:x1])
        crop_lab = bgr_to_lab_fast(crop_bgr)
        sub_labels = clean_labels[y0:y1, x0:x1]
        ring_all = sub_labels > 0
        ring_med_lab = np.median(crop_lab[ring_all], axis=0) if np.any(ring_all) else None
        ring_stain_de = max(float(params.pixel_outlier_de) * 1.5, 12.0)
        for seg_id in range(1, k + 1):
            sel = sub_labels == seg_id
            n_px = int(np.count_nonzero(sel))
            if n_px < 10:
                continue
            seg_lab = crop_lab[sel]
            med_lab = np.median(seg_lab, axis=0)
            de_px = np.linalg.norm(seg_lab - med_lab, axis=-1)
            bad_px = de_px > params.pixel_outlier_de
            if ring_med_lab is not None:
                de_ring = np.linalg.norm(seg_lab - ring_med_lab, axis=-1)
                bad_px = bad_px | (de_ring > ring_stain_de)
            rej = float(np.mean(bad_px) * 100.0)
            rejected_pct[seg_id - 1] = rej
            if np.any(bad_px):
                seg_ys, seg_xs = np.nonzero(sel)
                sub_labels[seg_ys[bad_px], seg_xs[bad_px]] = 0
            if rej > params.max_contaminated_pct:
                contaminated[seg_id - 1] = True
                warnings.append(
                    f"Đoạn {seg_id} có {rej:.1f}% điểm ảnh lệch màu (chữ / hình in / vết dơ) — không dùng để kết luận."
                )

    clean_counts = np.bincount(clean_labels[y0:y1, x0:x1].ravel(), minlength=k + 1)[1:k + 1].astype(np.int32)
    if int(np.count_nonzero((~contaminated) & (clean_counts > 0))) < 2:
        warnings.append("Không đủ đoạn hợp lệ trên vòng để so sánh đồng màu.")
        return RingLayout(
            valid=False, segments=k, inset_px=d, width_px=band_w, dist_max_px=dist_max,
            labels=clean_labels, raw_labels=raw_labels, centroids=centroids,
            pixel_counts=clean_counts, raw_pixel_counts=raw_counts,
            rejected_pct=rejected_pct, contaminated=contaminated,
            crop_box=(x0, y0, x1, y1), crop_labels=clean_labels[y0:y1, x0:x1],
            warnings=warnings,
        )

    layout = RingLayout(
        valid=True, segments=k, inset_px=d, width_px=band_w, dist_max_px=dist_max,
        labels=clean_labels, raw_labels=raw_labels, centroids=centroids,
        pixel_counts=clean_counts, raw_pixel_counts=raw_counts,
        rejected_pct=rejected_pct, contaminated=contaminated,
        crop_box=(x0, y0, x1, y1), crop_labels=clean_labels[y0:y1, x0:x1],
        fabric_mask=mask,
        warnings=warnings,
    )
    if reference_frame is not None:
        _calibrate_layout_field(reference_frame, layout)
    return layout


def _calibrate_layout_field(frame, layout: RingLayout):
    """Estimate smooth tray illumination + chromatic glare field and within-segment weave/sensor floor."""
    k = layout.segments
    layout.lum_gain = np.ones((k, 1), dtype=np.float64)
    layout.chroma_delta = np.zeros((k, 3), dtype=np.float64)
    layout.spatial_tau_rgb = None
    layout._field_calibrated = True
    if not layout.valid or frame is None:
        return

    x0, y0, x1, y1 = layout.crop_box
    crop_bgr = frame_to_bgr(frame[y0:y1, x0:x1])
    sub_labels = layout.crop_labels
    active = sub_labels > 0
    if not np.any(active):
        return

    seg_ids = sub_labels[active].astype(np.int32)
    px_rgb = crop_bgr[active][:, ::-1].astype(np.float64)
    counts = np.maximum(1, np.bincount(seg_ids, minlength=k + 1)[1:k + 1]).astype(np.float64)

    seg_std = np.zeros((k, 3), dtype=np.float64)
    for c in range(3):
        mc = np.bincount(seg_ids, weights=px_rgb[:, c], minlength=k + 1)[1:k + 1] / counts
        m2c = np.bincount(seg_ids, weights=px_rgb[:, c] ** 2, minlength=k + 1)[1:k + 1] / counts
        seg_std[:, c] = np.sqrt(np.maximum(0.0, m2c - mc ** 2))

    valid_seg = (~layout.contaminated) & (layout.pixel_counts > 0)
    within_std_vec = np.median(seg_std[valid_seg], axis=0) if np.any(valid_seg) else np.median(seg_std, axis=0)
    # Synthetic noiseless test images have within_std == 0.0 -> keep exact identity
    if float(np.max(within_std_vec)) < 0.25:
        return

    k_scale = float(np.sqrt(max(1.0, 0.58 * np.log(max(2, k)))))
    # Per-channel spatial + 8-bit quantization threshold (minimum 0.95 count when real texture/noise present)
    tau_vec = np.clip(np.maximum(0.95, 0.95 * within_std_vec) * k_scale, 0.95, 2.25)
    layout.spatial_tau_rgb = tau_vec[None, :]

    # Fit smooth 2D linear illumination & chromatic glare plane on surrounding tray halo
    h, w = layout.labels.shape[:2]
    pad = 95
    hx0 = max(0, (x0 - pad) & ~1)
    hy0 = max(0, y0 - pad)
    hx1 = min(w, (x1 + pad + 1) & ~1)
    hy1 = min(h, y1 + pad)
    halo_bgr = frame_to_bgr(frame[hy0:hy1, hx0:hx1])
    if layout.fabric_mask is not None:
        filled_fab = (layout.fabric_mask[hy0:hy1, hx0:hx1] > 0).astype(np.uint8)
    else:
        fab_sub = (layout.raw_labels[hy0:hy1, hx0:hx1] > 0).astype(np.uint8)
        cnts, _ = cv2.findContours(fab_sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_fab = np.zeros_like(fab_sub)
        if cnts:
            cv2.drawContours(filled_fab, [cv2.convexHull(np.vstack(cnts))], -1, 1, thickness=cv2.FILLED)
    d_in = cv2.dilate(filled_fab, np.ones((41, 41), np.uint8)) > 0
    d_out = cv2.dilate(filled_fab, np.ones((155, 155), np.uint8)) > 0
    bg_ring = d_out & ~d_in
    ys, xs = np.nonzero(bg_ring)
    if len(xs) < 200:
        return

    # Subsample halo pixels for < 0.5 ms least-squares fit
    step = max(1, len(xs) // 2500)
    ys = ys[::step] + hy0
    xs = xs[::step] + hx0
    rgb_bg = halo_bgr[ys - hy0, xs - hx0, ::-1].astype(np.float64)
    g_vals = rgb_bg[:, 1]
    p10, p90 = np.percentile(g_vals, [10, 90])
    keep = (g_vals >= p10) & (g_vals <= p90)
    if np.count_nonzero(keep) < 100:
        return
    xs = xs[keep].astype(np.float64)
    ys = ys[keep].astype(np.float64)
    rgb_bg = rgb_bg[keep]

    cx = layout.centroids[:, 0].astype(np.float64)
    cy = layout.centroids[:, 1].astype(np.float64)
    x_ref, y_ref = float(w * 0.5), float(h * 0.5)
    A_bg = np.column_stack([xs - x_ref, ys - y_ref, np.ones_like(xs)])
    A_seg = np.column_stack([cx - x_ref, cy - y_ref, np.ones_like(cx)])

    lin_bg = srgb_to_linear(rgb_bg / 255.0).mean(axis=1)
    coef_lum, _, _, _ = np.linalg.lstsq(A_bg, lin_bg, rcond=None)
    seg_bg_lum = A_seg @ coef_lum
    if coef_lum[2] > 1e-3:
        layout.lum_gain = np.clip(coef_lum[2] / np.maximum(1e-4, seg_bg_lum), 0.85, 1.18)[:, None]

    chroma_bg = rgb_bg - rgb_bg.mean(axis=1, keepdims=True)
    coef_chr, _, _, _ = np.linalg.lstsq(A_bg, chroma_bg, rcond=None)
    layout.chroma_delta = A_seg[:, :2] @ coef_chr[:2, :]


def sample_ring_frame(frame, layout: RingLayout, white="D50"):
    """Compute per-segment spatial mean sRGB (K, 3), Lab (K, 3), and clipped fraction (K,) in linear light."""
    if not layout._field_calibrated:
        _calibrate_layout_field(frame, layout)

    k = layout.segments
    x0, y0, x1, y1 = layout.crop_box
    crop_bgr = frame_to_bgr(frame[y0:y1, x0:x1])
    sub_labels = layout.crop_labels
    active = sub_labels > 0
    seg_ids = sub_labels[active].astype(np.int32)
    pixels_rgb = crop_bgr[active][:, ::-1]  # RGB uint8 (M, 3)

    counts = np.maximum(1, np.bincount(seg_ids, minlength=k + 1)[1:k + 1]).astype(np.float64)
    linear_pixels = _LINEAR_LUT[pixels_rgb]  # (M, 3) float64 in 0..1

    linear_means = np.zeros((k, 3), dtype=np.float64)
    for c in range(3):
        sums = np.bincount(seg_ids, weights=linear_pixels[:, c], minlength=k + 1)[1:k + 1]
        linear_means[:, c] = sums / counts

    # Strip localized dirty/dust pixels inside each segment in the current frame (> 25 sRGB counts vs segment mean)
    rgb0_seg = linear_to_srgb(linear_means) * 255.0
    px_diff = pixels_rgb.astype(np.float64) - rgb0_seg[seg_ids - 1]
    bad_px = np.sum(px_diff * px_diff, axis=1) > 625.0
    if np.any(bad_px):
        keep_px = ~bad_px
        if np.any(keep_px):
            seg_ids_k = seg_ids[keep_px]
            lin_k = linear_pixels[keep_px]
            counts_k = np.bincount(seg_ids_k, minlength=k + 1)[1:k + 1].astype(np.float64)
            has_k = counts_k >= 10
            for c in range(3):
                sums_c = np.bincount(seg_ids_k, weights=lin_k[:, c], minlength=k + 1)[1:k + 1]
                linear_means[has_k, c] = sums_c[has_k] / counts_k[has_k]

    if layout.lum_gain is not None:
        linear_means = np.clip(linear_means * layout.lum_gain, 0.0, 1.0)

    rgb = linear_to_srgb(linear_means) * 255.0
    if layout.chroma_delta is not None:
        rgb = np.clip(rgb - layout.chroma_delta, 0.0, 255.0)

    if layout.spatial_tau_rgb is not None:
        valid_sel = (~layout.contaminated) & (layout.pixel_counts > 0)
        center_rgb = np.median(rgb[valid_sel], axis=0, keepdims=True) if np.any(valid_sel) else np.median(rgb, axis=0, keepdims=True)
        diff = rgb - center_rgb
        norm_r2 = np.sum((diff / layout.spatial_tau_rgb) ** 2, axis=1, keepdims=True)
        shrink = (norm_r2 / np.maximum(1e-6, norm_r2 + 3.0)) ** 2
        rgb = np.clip(center_rgb + diff * shrink, 0.0, 255.0)
        linear_means = srgb_to_linear(rgb / 255.0)

    lab = linear_rgb_to_lab(linear_means, white)
    clipped = ((pixels_rgb >= 254).any(axis=1) | (pixels_rgb <= 1).any(axis=1)).astype(np.float64)
    clip_frac = np.bincount(seg_ids, weights=clipped, minlength=k + 1)[1:k + 1] / counts
    return rgb, lab, clip_frac


def build_master_reference(detection: FabricDetection, layout: RingLayout, result: UniformityResult, folder="") -> MasterFabricRef:
    """Create a MasterFabricRef from a completed measurement on the reference fabric."""
    from datetime import datetime
    clean_seg = (~result.contaminated) & (result.pixel_counts > 0)
    dirty_seg = result.dirty_segments.copy() if result.dirty_segments is not None else np.zeros(result.segments, dtype=bool)
    return MasterFabricRef(
        timestamp=datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
        segments=result.segments,
        long_side=float(detection.long_side),
        short_side=float(detection.short_side),
        area_px=int(detection.area_px),
        inset_pct=float(result.params.inset_pct),
        width_pct=float(result.params.width_pct),
        white=result.params.white,
        ring_mean_rgb=result.ring_mean_rgb.copy(),
        ring_mean_lab=result.ring_mean_lab.copy(),
        mean_rgb=result.mean_rgb.copy(),
        mean_lab=result.mean_lab.copy(),
        noise_floor_de=float(result.noise_floor_de),
        spatial_tau_rgb=layout.spatial_tau_rgb.copy() if layout.spatial_tau_rgb is not None else None,
        clean_segments=clean_seg,
        dirty_segments=dirty_seg,
        folder=str(folder),
    )


def _find_dirty_segments(mean_lab: np.ndarray, base_valid: np.ndarray, params: UniformityParams,
                         master: MasterFabricRef = None, is_timeseries: bool = False) -> np.ndarray:
    """Identify localized dirty/stained segments so uniformity & time-series only evaluate clean uniform segments."""
    k = len(base_valid)
    dirty = np.zeros(k, dtype=bool)
    if master is not None and master.dirty_segments is not None and len(master.dirty_segments) == k:
        if is_timeseries:
            dirty |= np.asarray(master.dirty_segments, dtype=bool)

    if not params.filter_dirty_segments and not is_timeseries and master is None:
        return dirty

    cand_idx = np.flatnonzero(base_valid & (~dirty))
    if len(cand_idx) < 4:
        return dirty

    # At most ~35% of candidate segments can be localized dirty spots (min 1, max len-3).
    # If more than 35% deviate, the fabric itself is globally non-uniform rather than having a dirty spot.
    max_dirty = min(max(1, int(len(cand_idx) * 0.35)), len(cand_idx) - 3)
    pair_de = delta_e(mean_lab[:, None, :], mean_lab[None, :, :], params.de_formula)
    sub_pair = pair_de[np.ix_(cand_idx, cand_idx)]

    # Find the clean core around the medoid segment (smallest median pairwise ΔE to other candidates)
    med_dists = np.median(sub_pair, axis=1)
    core_size = max(3, len(cand_idx) - max_dirty)
    order = np.argsort(med_dists)
    core_local = order[:core_size]
    core_idx = cand_idx[core_local]

    # Check if the core itself is uniform
    core_pair_max = float(np.max(pair_de[np.ix_(core_idx, core_idx)]))
    thr = float(params.threshold_de)
    if core_pair_max > max(thr * 1.25, thr + 0.15):
        return dirty

    core_med_lab = np.median(mean_lab[core_idx], axis=0)
    de_to_core = delta_e(mean_lab[cand_idx], core_med_lab[None, :], params.de_formula)
    core_de_vals = de_to_core[core_local]
    med_core = float(np.median(core_de_vals))
    mad_core = float(np.median(np.abs(core_de_vals - med_core))) * 1.4826
    spatial_cutoff = max(thr * 0.65, med_core + max(3.0 * mad_core, 0.12))

    # In time-series rounds (t > 0), also check if a segment's shift from t=0 is an outlier vs whole-fabric drift
    temporal_outlier = np.zeros(len(cand_idx), dtype=bool)
    if is_timeseries and master is not None and master.mean_lab is not None and len(master.mean_lab) == k:
        shift_lab = mean_lab[cand_idx] - master.mean_lab[cand_idx]
        med_shift = np.median(shift_lab[core_local], axis=0)
        res_shift = np.linalg.norm(shift_lab - med_shift[None, :], axis=1)
        core_res = res_shift[core_local]
        med_res = float(np.median(core_res))
        mad_res = float(np.median(np.abs(core_res - med_res))) * 1.4826
        temporal_outlier = res_shift > max(thr * 0.75, med_res + max(3.0 * mad_res, 0.15))

    clean_set = list(core_idx)
    newly_dirty = []
    for loc_idx in order[core_size:]:
        seg_i = int(cand_idx[loc_idx])
        max_de_with_clean = float(np.max(pair_de[seg_i, clean_set]))
        is_bad = (max_de_with_clean > thr and de_to_core[loc_idx] > spatial_cutoff) or bool(temporal_outlier[loc_idx])
        if is_bad:
            newly_dirty.append(seg_i)
        else:
            clean_set.append(seg_i)

    if len(newly_dirty) <= max_dirty:
        for seg_i in newly_dirty:
            dirty[seg_i] = True
    return dirty


def analyze_uniformity(layout: RingLayout, times, rgb_frames, lab_frames, clip_frames, params: UniformityParams,
                       master: MasterFabricRef = None, detection: FabricDetection = None,
                       is_timeseries: bool = False, round_index: int = 0, elapsed_min: float = 0.0):
    """Aggregate N frames across the K ring segments and produce the uniformity (and optional master comparison) verdict."""
    rgb_arr = np.asarray(rgb_frames, dtype=np.float64)  # (N, K, 3)
    lab_arr = np.asarray(lab_frames, dtype=np.float64)  # (N, K, 3)
    clip_arr = np.asarray(clip_frames, dtype=np.float64)  # (N, K)
    times_arr = np.asarray(times, dtype=np.float64)
    n_frames, k = rgb_arr.shape[:2]
    elapsed_s = float(times_arr[-1] - times_arr[0]) if len(times_arr) > 1 else 0.0

    _, _, _, outlier, _, mad_mean_rgb = mad_filter_rgb(
        rgb_arr, lab_arr, params.outlier_z, params.outlier_de, params.de_formula
    )
    mean_rgb = mad_mean_rgb
    mean_lab = srgb8_to_lab(mean_rgb, params.white)
    std_rgb = rgb_arr.std(axis=0, ddof=1) if n_frames > 1 else np.zeros_like(mean_rgb)
    std_lab = lab_arr.std(axis=0, ddof=1) if n_frames > 1 else np.zeros_like(mean_lab)

    base_valid = (~layout.contaminated) & (layout.pixel_counts > 0)
    dirty_segments = _find_dirty_segments(mean_lab, base_valid, params, master=master, is_timeseries=is_timeseries)
    effective_contaminated = layout.contaminated | dirty_segments

    valid_mask = (~effective_contaminated) & (layout.pixel_counts > 0)
    if int(np.count_nonzero(valid_mask)) < 2:
        valid_mask = layout.pixel_counts > 0
    valid_idx = np.flatnonzero(valid_mask)

    # Ring mean computed in linear light across valid (clean uniform) segments
    valid_linear = srgb_to_linear(mean_rgb[valid_idx] / 255.0).mean(axis=0)
    ring_mean_rgb = linear_to_srgb(valid_linear) * 255.0
    ring_mean_lab = linear_rgb_to_lab(valid_linear, params.white)

    de_to_mean = delta_e(mean_lab, ring_mean_lab[None], params.de_formula)
    delta_lab = mean_lab - ring_mean_lab[None]

    pairwise_de = delta_e(mean_lab[:, None, :], mean_lab[None, :, :], params.de_formula)
    max_pair_de = 0.0
    worst_pair = (1, 2)
    for ii in range(len(valid_idx)):
        for jj in range(ii + 1, len(valid_idx)):
            a, b = int(valid_idx[ii]), int(valid_idx[jj])
            val = float(pairwise_de[a, b])
            if val >= max_pair_de:
                max_pair_de = val
                worst_pair = (a + 1, b + 1)

    worst_seg_idx = int(valid_idx[int(np.argmax(de_to_mean[valid_idx]))])
    max_de_to_mean = float(de_to_mean[worst_seg_idx])

    seg_noise_de, noise_floor_de = split_half_noise_de(
        rgb_arr, keep_mask=~outlier, white=params.white, formula=params.de_formula,
        return_max_pair=True, valid_idx=valid_idx,
    )

    warnings = list(layout.warnings)
    if np.any(dirty_segments):
        dirty_ids = [int(i + 1) for i in np.flatnonzero(dirty_segments)]
        warnings.append(
            f"Đã tự động loại bỏ {len(dirty_ids)} đoạn nghi bị dơ / lệch bất thường "
            f"(đoạn {', '.join(map(str, dirty_ids))}) — chỉ xét đồng màu trên {len(valid_idx)} đoạn sạch."
        )
    thr = float(params.threshold_de)
    formula_label = FORMULAS.get(params.de_formula, params.de_formula)

    has_master = master is not None
    master_ring_rgb = None
    master_ring_lab = None
    overall_master_de = None
    overall_delta_lab_master = None
    de_to_master = None
    delta_lab_to_master = None
    max_master_de = None
    worst_master_segment = None
    size_diff_pct = None

    if has_master:
        if is_timeseries and master.mean_rgb is not None and len(master.mean_rgb) == k:
            # Stationary time-series: compare each clean segment against its own t=0 colour
            master_seg_rgb = np.asarray(master.mean_rgb, dtype=np.float64)
            master_seg_lab = srgb8_to_lab(master_seg_rgb, params.white)
            m_lin = srgb_to_linear(master_seg_rgb[valid_idx] / 255.0).mean(axis=0)
            master_ring_rgb = linear_to_srgb(m_lin) * 255.0
            master_ring_lab = linear_rgb_to_lab(m_lin, params.white)
            ref_rgb_per_seg = master_seg_rgb
            ref_lab_per_seg = master_seg_lab
        else:
            master_ring_rgb = np.asarray(master.ring_mean_rgb, dtype=np.float64)
            master_ring_lab = srgb8_to_lab(master_ring_rgb, params.white)
            ref_rgb_per_seg = master_ring_rgb[None, :]
            ref_lab_per_seg = master_ring_lab[None, :]

        if detection is not None and detection.found and master.long_side > 0 and master.short_side > 0:
            dl = abs(detection.long_side - master.long_side) / master.long_side * 100.0
            ds = abs(detection.short_side - master.short_side) / master.short_side * 100.0
            size_diff_pct = float(max(dl, ds))
            if size_diff_pct > 15.0:
                warnings.append(
                    f"Kích thước tấm vải hiện tại ({detection.long_side:.0f}×{detection.short_side:.0f} px) "
                    f"lệch {size_diff_pct:.1f}% so với vải gốc ({master.long_side:.0f}×{master.short_side:.0f} px)."
                )

        # Denoise sub-count directional/sensor jitter when comparing against master
        tau_m = layout.spatial_tau_rgb if layout.spatial_tau_rgb is not None else master.spatial_tau_rgb
        if tau_m is not None:
            diff_m = mean_rgb - ref_rgb_per_seg
            norm_r2_m = np.sum((diff_m / tau_m) ** 2, axis=1, keepdims=True)
            shrink_m = (norm_r2_m / np.maximum(1e-6, norm_r2_m + 2.5)) ** 2
            eval_seg_rgb = np.clip(ref_rgb_per_seg + diff_m * shrink_m, 0.0, 255.0)
            eval_seg_lab = srgb8_to_lab(eval_seg_rgb, params.white)
            eval_ring_lin = srgb_to_linear(eval_seg_rgb[valid_idx] / 255.0).mean(axis=0)
            eval_ring_lab = linear_rgb_to_lab(eval_ring_lin, params.white)
        else:
            eval_seg_lab = mean_lab
            eval_ring_lab = ring_mean_lab

        overall_master_de = float(delta_e(eval_ring_lab, master_ring_lab, params.de_formula))
        overall_delta_lab_master = eval_ring_lab - master_ring_lab
        de_to_master = delta_e(eval_seg_lab, ref_lab_per_seg, params.de_formula)
        delta_lab_to_master = eval_seg_lab - ref_lab_per_seg
        worst_m_idx = int(valid_idx[int(np.argmax(de_to_master[valid_idx]))])
        max_master_de = float(de_to_master[worst_m_idx])
        worst_master_segment = worst_m_idx + 1

        primary_de = max(max_master_de, max_pair_de)
        if primary_de + noise_floor_de <= thr:
            verdict = "pass"
            verdict_label = "ỔN ĐỊNH SO VỚI GỐC (t=0)" if is_timeseries else "ĐẠT SO VỚI VẢI GỐC"
            explanation = (
                f"{verdict_label}: lệch tổng thể {formula_label} = {overall_master_de:.3f}, "
                f"đoạn lệch nhất so với vải gốc là đoạn {worst_master_segment} ({max_master_de:.3f}), "
                f"đồng màu nội bộ ({len(valid_idx)} đoạn sạch) {max_pair_de:.3f} (≤ ngưỡng {thr:.2f})."
            )
        elif primary_de - noise_floor_de > thr:
            verdict = "fail"
            verdict_label = "LỆCH SO VỚI VẢI GỐC" if max_master_de >= max_pair_de else "KHÔNG ĐỒNG MÀU"
            explanation = (
                f"KHÔNG ĐẠT SO VỚI VẢI GỐC: lệch tổng thể {formula_label} = {overall_master_de:.3f}, "
                f"đoạn lệch nhất so với vải gốc là đoạn {worst_master_segment} ({max_master_de:.3f}); "
                f"lệch nội bộ cặp {worst_pair[0]}–{worst_pair[1]} là {max_pair_de:.3f} (> ngưỡng {thr:.2f})."
            )
        else:
            verdict = "uncertain"
            verdict_label = "CHƯA ĐỦ TIN CẬY"
            explanation = (
                f"CHƯA ĐỦ TIN CẬY: lệch so với vải gốc là {formula_label} = {max_master_de:.3f} "
                f"(tổng thể {overall_master_de:.3f}) nằm trong vùng chồng lấp với sàn nhiễu ±{noise_floor_de:.3f} "
                f"quanh ngưỡng {thr:.2f}."
            )
    else:
        if max_pair_de + noise_floor_de <= thr:
            verdict = "pass"
            verdict_label = VERDICT_LABELS["pass"]
            explanation = (
                f"ĐỒNG MÀU ({len(valid_idx)}/{k} đoạn sạch): lệch lớn nhất giữa 2 đoạn ({worst_pair[0]}–{worst_pair[1]}) là "
                f"{formula_label} = {max_pair_de:.3f} (+ sàn nhiễu {noise_floor_de:.3f} ≤ ngưỡng {thr:.2f})."
            )
        elif max_pair_de - noise_floor_de > thr:
            verdict = "fail"
            verdict_label = VERDICT_LABELS["fail"]
            explanation = (
                f"KHÔNG ĐỒNG MÀU: cặp đoạn {worst_pair[0]}–{worst_pair[1]} lệch {formula_label} = {max_pair_de:.3f} "
                f"(− sàn nhiễu {noise_floor_de:.3f} > ngưỡng {thr:.2f}); đoạn lệch nhất so với vòng là đoạn {worst_seg_idx + 1} "
                f"({max_de_to_mean:.3f})."
            )
        else:
            verdict = "uncertain"
            verdict_label = VERDICT_LABELS["uncertain"]
            explanation = (
                f"CHƯA ĐỦ TIN CẬY: lệch lớn nhất ({worst_pair[0]}–{worst_pair[1]}) là {formula_label} = {max_pair_de:.3f} "
                f"nằm trong vùng chồng lấp với sàn nhiễu ±{noise_floor_de:.3f} quanh ngưỡng {thr:.2f}. "
                f"Hãy tăng số frame hoặc bề rộng vòng."
            )

    return UniformityResult(
        params=params,
        segments=k,
        frame_count=n_frames,
        elapsed_s=elapsed_s,
        mean_rgb=mean_rgb,
        std_rgb=std_rgb,
        mean_lab=mean_lab,
        std_lab=std_lab,
        ring_mean_rgb=ring_mean_rgb,
        ring_mean_lab=ring_mean_lab,
        de_to_mean=de_to_mean,
        delta_lab=delta_lab,
        pairwise_de=pairwise_de,
        max_pair_de=max_pair_de,
        worst_pair=worst_pair,
        max_de_to_mean=max_de_to_mean,
        worst_segment=worst_seg_idx + 1,
        seg_noise_de=seg_noise_de,
        noise_floor_de=noise_floor_de,
        pixel_counts=layout.pixel_counts.copy(),
        rejected_pct=layout.rejected_pct.copy(),
        contaminated=effective_contaminated.copy(),
        clip_pct=clip_arr.mean(axis=0) * 100.0,
        outlier_frames=outlier.sum(axis=0).astype(np.int32),
        verdict=verdict,
        verdict_label=verdict_label,
        explanation=explanation,
        rgb_frames=rgb_arr,
        lab_frames=lab_arr,
        times=times_arr - (times_arr[0] if len(times_arr) else 0.0),
        warnings=warnings,
        has_master=has_master,
        master_ring_rgb=master_ring_rgb,
        master_ring_lab=master_ring_lab,
        overall_master_de=overall_master_de,
        overall_delta_lab_master=overall_delta_lab_master,
        de_to_master=de_to_master,
        delta_lab_to_master=delta_lab_to_master,
        max_master_de=max_master_de,
        worst_master_segment=worst_master_segment,
        size_diff_pct=size_diff_pct,
        dirty_segments=dirty_segments.copy(),
        is_timeseries=is_timeseries,
        round_index=round_index,
        elapsed_min=elapsed_min,
    )


def _hex_to_bgr(hex_color):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return (b, g, r)


def _de_to_bgr(de_val, threshold):
    """Map segment ΔE vs ring mean to a green -> yellow -> red BGR colour."""
    ratio = float(de_val) / max(0.05, float(threshold))
    if ratio <= 0.5:
        return (90, 215, 75)  # green
    if ratio <= 1.0:
        return (40, 205, 255)  # amber/yellow
    return (60, 60, 245)  # red


def render_ring_overlay(bgr, detection: FabricDetection, layout: RingLayout = None,
                        result: UniformityResult = None, alpha=0.45,
                        zoom_to_fabric=False, zoom_margin=0.22, target_width=960):
    """Draw the detected fabric contour, 5%-inset ring segments, and segment numbers onto a BGR image.

    When `zoom_to_fabric=True`, crops around the detected fabric with `zoom_margin` padding and
    upscales the crop to `target_width` BEFORE drawing the contour, segments, and text labels so
    small fabrics and inward-contracted rings are large and crisp on the UI.
    """
    if detection is None or not detection.found:
        return bgr.copy()

    h, w = bgr.shape[:2]
    mh, mw = detection.mask.shape[:2]

    if zoom_to_fabric and len(detection.contour) >= 3:
        cx0, cy0, cw, ch = cv2.boundingRect(detection.contour.reshape(-1, 1, 2))
        pad_x = max(20, int(round(max(cw, ch) * zoom_margin)))
        pad_y = max(20, int(round(max(cw, ch) * zoom_margin)))
        # Ensure at least a 4:3 comfortable aspect box around elongated strips
        box_w = cw + 2 * pad_x
        box_h = max(ch + 2 * pad_y, int(round(box_w * 0.58)))
        box_w = max(box_w, int(round(box_h * 1.33)))
        mid_x = cx0 + cw * 0.5
        mid_y = cy0 + ch * 0.5
        mx0 = max(0, int(round(mid_x - box_w * 0.5)))
        my0 = max(0, int(round(mid_y - box_h * 0.5)))
        mx1 = min(mw, mx0 + box_w)
        my1 = min(mh, my0 + box_h)
        mx0 = max(0, mx1 - box_w)
        my0 = max(0, my1 - box_h)

        bx0 = max(0, int(round(mx0 * w / float(mw))))
        by0 = max(0, int(round(my0 * h / float(mh))))
        bx1 = min(w, max(bx0 + 2, int(round(mx1 * w / float(mw)))))
        by1 = min(h, max(by0 + 2, int(round(my1 * h / float(mh)))))

        crop_bgr = bgr[by0:by1, bx0:bx1]
        out_w = max(480, int(target_width))
        out_h = max(320, int(round(crop_bgr.shape[0] * out_w / float(crop_bgr.shape[1]))))
        canvas = cv2.resize(crop_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        sx = out_w / float(mx1 - mx0)
        sy = out_h / float(my1 - my0)
        off_x, off_y = float(mx0), float(my0)
    else:
        canvas = bgr.copy()
        out_h, out_w = h, w
        mx0, my0, mx1, my1 = 0, 0, mw, mh
        sx, sy = w / float(mw), h / float(mh)
        off_x, off_y = 0.0, 0.0

    if layout is not None and layout.valid:
        sub_labels = layout.labels[my0:my1, mx0:mx1]
        if (sub_labels.shape[1], sub_labels.shape[0]) != (out_w, out_h):
            labels = cv2.resize(sub_labels, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        else:
            labels = sub_labels
        overlay = canvas.copy()
        for k in range(1, layout.segments + 1):
            sel = labels == k
            if not np.any(sel):
                continue
            is_contam = bool(layout.contaminated[k - 1] or (result is not None and result.contaminated[k - 1]))
            if is_contam:
                color = (140, 140, 140)
            elif result is not None:
                if result.has_master and result.de_to_master is not None:
                    color = _de_to_bgr(result.de_to_master[k - 1], max(0.05, result.params.threshold_de))
                else:
                    # Highlight according to max deviation (de_to_mean compared to half the pairwise threshold)
                    color = _de_to_bgr(result.de_to_mean[k - 1], max(0.05, result.params.threshold_de * 0.65))
            else:
                color = _hex_to_bgr(SEGMENT_PALETTE_HEX[(k - 1) % len(SEGMENT_PALETTE_HEX)])
            overlay[sel] = color
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, dst=canvas)

        # Draw crisp 1px borders around each segment so adjacent segments are clearly separated
        for k in range(1, layout.segments + 1):
            seg_u8 = (labels == k).astype(np.uint8) * 255
            if not np.any(seg_u8):
                continue
            seg_contours, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, seg_contours, -1, (25, 35, 45), 1, cv2.LINE_AA)

        # Draw segment numbers / ΔE values cleanly
        font_scale = 0.56 if zoom_to_fabric else 0.48
        for k in range(1, layout.segments + 1):
            cx = int(round((layout.centroids[k - 1, 0] - off_x) * sx))
            cy = int(round((layout.centroids[k - 1, 1] - off_y) * sy))
            if result is None:
                label = f"{k}"
            elif result.dirty_segments is not None and result.dirty_segments[k - 1]:
                label = f"{k}:dơ"
            elif result.contaminated[k - 1]:
                label = f"{k}:loại"
            elif result.has_master and result.de_to_master is not None:
                label = f"{k}:{result.de_to_master[k - 1]:.2f}"
            else:
                label = f"{k}:{result.de_to_mean[k - 1]:.2f}"
            (tw_txt, th_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            tx = int(np.clip(cx - tw_txt // 2, 4, out_w - tw_txt - 4))
            ty = int(np.clip(cy + th_txt // 2, th_txt + 4, out_h - 4))
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    if len(detection.contour):
        pts = detection.contour.astype(np.float32) - np.array([off_x, off_y], dtype=np.float32)
        scaled_contour = np.round(pts * np.array([sx, sy], dtype=np.float32)).astype(np.int32)
        cv2.drawContours(canvas, [scaled_contour], -1, (80, 230, 100), 2, cv2.LINE_AA)
    return canvas


def export_uniformity_session(folder, frame_bgr, detection: FabricDetection, layout: RingLayout,
                              result: UniformityResult, camera="", mode="", meta=None):
    """Save mask, overlays, CSV tables and session.json into `folder`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(folder / "mask.png"), detection.mask)
    if frame_bgr is not None:
        cv2.imwrite(str(folder / "frame.png"), frame_bgr)
        cv2.imwrite(str(folder / "overlay.png"), render_ring_overlay(frame_bgr, detection, layout, result=None))
        cv2.imwrite(str(folder / "overlay_de.png"), render_ring_overlay(frame_bgr, detection, layout, result=result))
        cv2.imwrite(
            str(folder / "overlay_zoom.png"),
            render_ring_overlay(frame_bgr, detection, layout, result=result, zoom_to_fabric=True),
        )

    # frames.csv
    with (folder / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "t_s", "segment", "R", "G", "B", "L", "a", "b"])
        if result.rgb_frames is not None:
            for f_idx in range(result.frame_count):
                t_val = float(result.times[f_idx]) if result.times is not None else 0.0
                for s_idx in range(result.segments):
                    writer.writerow([
                        f_idx, f"{t_val:.4f}", s_idx + 1,
                        *(f"{v:.3f}" for v in result.rgb_frames[f_idx, s_idx]),
                        *(f"{v:.3f}" for v in result.lab_frames[f_idx, s_idx]),
                    ])

    # segments.csv
    with (folder / "segments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        cols = [
            "segment", "cx_px", "cy_px", "pixels", "rejected_pct", "contaminated",
            "R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std",
            "L_mean", "a_mean", "b_mean", "dL_vs_ring", "da_vs_ring", "db_vs_ring",
            "de_vs_ring_mean", "de_vs_master", "noise_floor_de", "outlier_frames", "clip_pct",
        ]
        writer.writerow(cols)
        for k in range(result.segments):
            de_m_str = f"{result.de_to_master[k]:.4f}" if (result.has_master and result.de_to_master is not None) else ""
            writer.writerow([
                k + 1,
                f"{layout.centroids[k, 0]:.1f}", f"{layout.centroids[k, 1]:.1f}",
                int(result.pixel_counts[k]), f"{result.rejected_pct[k]:.2f}", int(result.contaminated[k]),
                *(f"{v:.3f}" for v in result.mean_rgb[k]),
                *(f"{v:.3f}" for v in result.std_rgb[k]),
                *(f"{v:.3f}" for v in result.mean_lab[k]),
                *(f"{v:+.3f}" for v in result.delta_lab[k]),
                f"{result.de_to_mean[k]:.4f}",
                de_m_str,
                f"{result.seg_noise_de[k]:.4f}",
                int(result.outlier_frames[k]),
                f"{result.clip_pct[k]:.3f}",
            ])

    # summary.csv
    with (folder / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "verdict", "verdict_label", "formula", "white", "threshold_de",
            "max_pair_de", "worst_pair", "max_de_to_mean", "worst_segment",
            "overall_master_de", "max_master_de", "worst_master_segment",
            "noise_floor_de", "segments", "valid_segments", "frames",
            "inset_px", "width_px", "ring_L", "ring_a", "ring_b",
        ])
        valid_count = int(np.count_nonzero((~result.contaminated) & (result.pixel_counts > 0)))
        writer.writerow([
            result.verdict, result.verdict_label,
            FORMULAS.get(result.params.de_formula, result.params.de_formula),
            result.params.white,
            f"{result.params.threshold_de:.3f}",
            f"{result.max_pair_de:.4f}",
            f"{result.worst_pair[0]}-{result.worst_pair[1]}",
            f"{result.max_de_to_mean:.4f}",
            result.worst_segment,
            f"{result.overall_master_de:.4f}" if result.overall_master_de is not None else "",
            f"{result.max_master_de:.4f}" if result.max_master_de is not None else "",
            result.worst_master_segment if result.worst_master_segment is not None else "",
            f"{result.noise_floor_de:.4f}",
            result.segments, valid_count, result.frame_count,
            layout.inset_px, layout.width_px,
            *(f"{v:.3f}" for v in result.ring_mean_lab),
        ])

    res_dict = dict(
        verdict=result.verdict,
        verdict_label=result.verdict_label,
        explanation=result.explanation,
        max_pair_de=round(result.max_pair_de, 4),
        worst_pair=list(result.worst_pair),
        max_de_to_mean=round(result.max_de_to_mean, 4),
        worst_segment=result.worst_segment,
        noise_floor_de=round(result.noise_floor_de, 4),
        ring_mean_rgb=[round(float(x), 3) for x in result.ring_mean_rgb],
        ring_mean_lab=[round(float(x), 3) for x in result.ring_mean_lab],
        has_master=bool(result.has_master),
    )
    if result.has_master:
        res_dict.update(
            master_ring_rgb=[round(float(x), 3) for x in result.master_ring_rgb],
            master_ring_lab=[round(float(x), 3) for x in result.master_ring_lab],
            overall_master_de=round(float(result.overall_master_de), 4),
            max_master_de=round(float(result.max_master_de), 4),
            worst_master_segment=int(result.worst_master_segment),
            size_diff_pct=round(float(result.size_diff_pct), 2) if result.size_diff_pct is not None else None,
        )
    info = dict(
        camera=camera,
        mode=mode,
        params=asdict(result.params),
        preparation=meta,
        detection=dict(
            center=list(detection.center),
            long_side=round(detection.long_side, 2),
            short_side=round(detection.short_side, 2),
            angle=round(detection.angle, 2),
            area_px=detection.area_px,
            area_frac=round(detection.area_frac, 4),
            solidity=round(detection.solidity, 4),
            contrast_de=round(detection.contrast_de, 2),
            holes_filled_px=detection.holes_filled_px,
            touches_border=detection.touches_border,
        ),
        ring=dict(
            inset_px=layout.inset_px,
            width_px=layout.width_px,
            dist_max_px=round(layout.dist_max_px, 2),
            segments=layout.segments,
            pixel_counts=[int(x) for x in result.pixel_counts],
            rejected_pct=[round(float(x), 2) for x in result.rejected_pct],
            contaminated=[bool(x) for x in result.contaminated],
            dirty_segments=[bool(x) for x in result.dirty_segments] if result.dirty_segments is not None else [False] * layout.segments,
        ),
        result=res_dict,
        warnings=result.warnings,
    )
    (folder / "session.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def build_timeseries_round(round_index: int, elapsed_s: float, result: UniformityResult, timestamp: str = "") -> UniformityTimeSeriesRound:
    """Create a UniformityTimeSeriesRound snapshot from a completed UniformityResult."""
    from datetime import datetime
    ts_str = timestamp or datetime.now().strftime("%H:%M:%S")
    dirty_mask = result.dirty_segments.copy() if result.dirty_segments is not None else np.zeros(result.segments, dtype=bool)
    dirty_list = [int(i + 1) for i in np.flatnonzero(result.contaminated)]
    valid_count = int(np.count_nonzero((~result.contaminated) & (result.pixel_counts > 0)))
    overall_de = float(result.overall_master_de) if result.overall_master_de is not None else 0.0
    max_m_de = float(result.max_master_de) if result.max_master_de is not None else 0.0
    worst_m_seg = int(result.worst_master_segment) if result.worst_master_segment is not None else 1
    d_lab_m = (
        np.asarray(result.overall_delta_lab_master, dtype=np.float64).copy()
        if result.overall_delta_lab_master is not None else np.zeros(3, dtype=np.float64)
    )
    de_m_arr = (
        np.asarray(result.de_to_master, dtype=np.float64).copy()
        if result.de_to_master is not None else np.zeros(result.segments, dtype=np.float64)
    )
    dlab_seg_m = (
        np.asarray(result.delta_lab_to_master, dtype=np.float64).copy()
        if result.delta_lab_to_master is not None else np.zeros((result.segments, 3), dtype=np.float64)
    )
    return UniformityTimeSeriesRound(
        round_index=int(round_index),
        timestamp=ts_str,
        elapsed_s=float(elapsed_s),
        elapsed_min=float(elapsed_s / 60.0),
        verdict=result.verdict,
        verdict_label=result.verdict_label,
        overall_master_de=overall_de,
        max_master_de=max_m_de,
        worst_master_segment=worst_m_seg,
        max_pair_de=float(result.max_pair_de),
        worst_pair=tuple(result.worst_pair),
        noise_floor_de=float(result.noise_floor_de),
        valid_segments=valid_count,
        dirty_segments_list=dirty_list,
        ring_mean_rgb=result.ring_mean_rgb.copy(),
        ring_mean_lab=result.ring_mean_lab.copy(),
        delta_lab_master=d_lab_m,
        de_to_master=de_m_arr,
        de_to_mean=result.de_to_mean.copy(),
        mean_lab=result.mean_lab.copy(),
        delta_lab_to_master=dlab_seg_m,
        contaminated=result.contaminated.copy(),
        dirty_mask=dirty_mask,
    )


def export_uniformity_timeseries(folder, rounds: list, master: MasterFabricRef = None):
    """Export time-series rounds summary and per-segment history CSV/JSON into `folder`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "timeseries_rounds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "round", "timestamp", "elapsed_s", "elapsed_min", "verdict", "verdict_label",
            "valid_segments", "excluded_segments", "overall_de00_vs_t0", "max_seg_de00_vs_t0",
            "worst_seg_vs_t0", "internal_max_pair_de00", "worst_pair", "noise_floor_de00",
            "ring_L", "ring_a", "ring_b", "dL_vs_t0", "da_vs_t0", "db_vs_t0",
        ])
        for r in rounds:
            excl_str = ";".join(map(str, r.dirty_segments_list)) if r.dirty_segments_list else ""
            writer.writerow([
                r.round_index,
                r.timestamp,
                f"{r.elapsed_s:.2f}",
                f"{r.elapsed_min:.3f}",
                r.verdict,
                r.verdict_label,
                r.valid_segments,
                excl_str,
                f"{r.overall_master_de:.4f}",
                f"{r.max_master_de:.4f}",
                r.worst_master_segment,
                f"{r.max_pair_de:.4f}",
                f"{r.worst_pair[0]}-{r.worst_pair[1]}",
                f"{r.noise_floor_de:.4f}",
                *(f"{v:.3f}" for v in r.ring_mean_lab),
                *(f"{v:+.3f}" for v in r.delta_lab_master),
            ])

    with (folder / "timeseries_segments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "round", "timestamp", "elapsed_min", "segment", "status",
            "L", "a", "b", "dL_vs_t0", "da_vs_t0", "db_vs_t0",
            "de00_vs_t0", "de00_vs_ring",
        ])
        for r in rounds:
            k_segs = len(r.mean_lab)
            for k in range(k_segs):
                if r.dirty_mask[k]:
                    status = "dirty_excluded"
                elif r.contaminated[k]:
                    status = "contaminated_excluded"
                else:
                    status = "clean"
                writer.writerow([
                    r.round_index,
                    r.timestamp,
                    f"{r.elapsed_min:.3f}",
                    k + 1,
                    status,
                    *(f"{v:.3f}" for v in r.mean_lab[k]),
                    *(f"{v:+.3f}" for v in r.delta_lab_to_master[k]),
                    f"{r.de_to_master[k]:.4f}",
                    f"{r.de_to_mean[k]:.4f}",
                ])

    ts_meta = dict(
        rounds_count=len(rounds),
        master_timestamp=master.timestamp if master is not None else "",
        master_ring_lab=[round(float(x), 3) for x in master.ring_mean_lab] if master is not None else [],
        rounds=[
            dict(
                round_index=r.round_index,
                timestamp=r.timestamp,
                elapsed_s=round(r.elapsed_s, 2),
                elapsed_min=round(r.elapsed_min, 3),
                verdict=r.verdict,
                verdict_label=r.verdict_label,
                valid_segments=r.valid_segments,
                excluded_segments=r.dirty_segments_list,
                overall_master_de=round(r.overall_master_de, 4),
                max_master_de=round(r.max_master_de, 4),
                worst_master_segment=r.worst_master_segment,
                max_pair_de=round(r.max_pair_de, 4),
                ring_mean_lab=[round(float(x), 3) for x in r.ring_mean_lab],
                delta_lab_master=[round(float(x), 3) for x in r.delta_lab_master],
            )
            for r in rounds
        ],
    )
    (folder / "timeseries.json").write_text(json.dumps(ts_meta, ensure_ascii=False, indent=2), encoding="utf-8")
