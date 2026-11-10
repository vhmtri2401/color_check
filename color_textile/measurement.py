"""Three-point colour stability measurement.

A session samples up to three ROIs on the fabric. Every `interval` seconds it records
a burst ("round") of N consecutive frames, so long-term drift (between rounds) and
frame-to-frame scatter (inside a round) can be separated. Garbage frames are rejected per
round with MAD on the ROI-mean RGB (robust z > threshold), then the valid frames are
averaged (MAD + Mean). Drift between rounds is monitored separately, because MAD inside
a short burst cannot see slow drift.
"""

import csv
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .colorimetry import FORMULAS, delta_e, linear_rgb_to_lab, linear_to_srgb, srgb8_to_lab, srgb_to_linear
from .decoders import yuyv_to_bgr

MAX_POINTS = 3
MAD_SCALE = 1.4826  # MAD -> sigma for normally distributed data
RGB_SIGMA_FLOOR = 0.25  # epsilon for robust sigma (RGB 0..255) when MAD is ~0
TRIM_FRACTION = 0.1
AGGREGATES = {"mean": "Mean", "median": "Median", "trimmed": "Trimmed mean 10%", "mad_mean": "MAD + Mean"}
MONITOR_BIN_S = 1.0  # continuous time-series resolution between rounds
_LINEAR_LUT = srgb_to_linear(np.arange(256) / 255)


@dataclass
class SessionParams:
    roi: int = 31
    frames: int = 30
    interval_s: float = 120.0
    duration_min: float = 30.0
    outlier_z: float = 3.5
    outlier_de: float = 1.0  # extra gate: also require ΔE to the round median above this; 0 = MAD only
    de_formula: str = "de00"  # "de00" (CIEDE2000) default
    white: str = "D50"  # Lab reference white: "D50" (Bradford-adapted, Nix default) or "D65"
    rois: tuple = ()  # optional per-point ROI sizes (ROI-size experiment); empty = `roi` for every point

    def point_rois(self, count):
        return list(self.rois)[:count] if len(self.rois) >= count else [self.roi] * count

    def labels(self, count):
        if self.rois:
            return [f"Điểm {i + 1} · ROI {size}" for i, size in enumerate(self.point_rois(count))]
        return [f"Điểm {i + 1}" for i in range(count)]
    lock: bool = True  # lock AE/AWB/exposure/gain after the automatic loops settle
    settle_window_s: float = 5.0
    settle_de: float = 0.3
    settle_timeout_s: float = 60.0

    @property
    def rounds_total(self):
        # Rounds at t = 0, interval, ..., duration (inclusive).
        return int(self.duration_min * 60 // self.interval_s) + 1


def roi_bounds(width, height, point, roi):
    """Pixel window (x0, y0, x1, y1) centred on a normalised point; x0 and width stay even for YUYV."""
    size = max(2, min(int(roi), width, height))
    size += size % 2
    x0 = int(round(point[0] * width - size / 2)) & ~1
    y0 = int(round(point[1] * height - size / 2))
    x0 = min(max(0, x0), (width - size) & ~1)
    y0 = min(max(0, y0), height - size)
    return x0, y0, x0 + size, y0 + size


def sample_points(yuyv, points, roi, white="D65"):
    """Mean colour of each ROI: sRGB 0..255 (P, 3), L*a*b* (P, 3), clipped pixel fraction (P,).

    `roi` is one size for every point, or a sequence with one size per point.

    The spatial mean is taken in linear light, as recommended for averaging colour.
    """
    height, width = yuyv.shape[:2]
    count = len(points)
    rgb, lab, clip = np.empty((count, 3)), np.empty((count, 3)), np.empty(count)
    sizes = list(roi) if isinstance(roi, (list, tuple)) else [roi] * count
    for i, point in enumerate(points):
        x0, y0, x1, y1 = roi_bounds(width, height, point, sizes[i])
        pixels = yuyv_to_bgr(np.ascontiguousarray(yuyv[y0:y1, x0:x1])).reshape(-1, 3)[:, ::-1]
        linear = _LINEAR_LUT[pixels].mean(axis=0)
        rgb[i] = linear_to_srgb(linear) * 255
        lab[i] = linear_rgb_to_lab(linear, white)
        clip[i] = np.mean((pixels >= 254).any(axis=1) | (pixels <= 1).any(axis=1))
    return rgb, lab, clip


def mad_filter_rgb(rgb, lab, outlier_z=3.5, outlier_de=1.0, formula="de76"):
    """Compute MAD robust-z on RGB (N, P, 3) and return (median_rgb, robust_z, de_median, outlier, outlier_channel, mad_mean)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    lab = np.asarray(lab, dtype=np.float64)
    median_rgb = np.median(rgb, axis=0)  # (P, 3)
    sigma = np.maximum(np.median(np.abs(rgb - median_rgb), axis=0) * MAD_SCALE, RGB_SIGMA_FLOOR)
    robust_z = np.abs(rgb - median_rgb) / sigma  # (N, P, 3)
    de_median = delta_e(lab, np.median(lab, axis=0)[None], formula)  # (N, P)
    outlier = (robust_z > outlier_z).any(axis=-1) & (de_median > outlier_de)
    outlier_channel = (robust_z > outlier_z) & outlier[..., None]  # (N, P, 3)
    keep = ~outlier
    mad_mean = np.array([
        rgb[keep[:, p], p].mean(axis=0) if keep[:, p].any() else median_rgb[p]
        for p in range(rgb.shape[1])
    ])
    return median_rgb, robust_z, de_median, outlier, outlier_channel, mad_mean


def split_half_noise_de(rgb, keep_mask=None, white="D50", formula="de76", splits=32, return_max_pair=False, valid_idx=None):
    """Estimate per-region p95 ΔE noise floor (and optional max-pair p95 noise floor) via split-halves."""
    rgb = np.asarray(rgb, dtype=np.float64)
    n_frames, n_regions = rgb.shape[:2]
    if n_frames < 4 or n_regions == 0:
        return (np.zeros(n_regions, dtype=np.float64), 0.0) if return_max_pair else np.zeros(n_regions, dtype=np.float64)
    rng = np.random.default_rng(0)
    half = n_frames // 2
    lab_frames = srgb8_to_lab(rgb, white)  # (N, K, 3)
    base_lab = lab_frames.mean(axis=0, keepdims=True)  # (1, K, 3)
    deltas = np.zeros((splits, n_regions, 3), dtype=np.float64)
    for s in range(splits):
        perm = rng.permutation(n_frames)
        idx_a, idx_b = perm[:half], perm[half:2 * half]
        for p in range(n_regions):
            if keep_mask is not None:
                ka = idx_a[keep_mask[idx_a, p]]
                kb = idx_b[keep_mask[idx_b, p]]
                if len(ka) >= 2 and len(kb) >= 2:
                    deltas[s, p] = (lab_frames[ka, p].mean(axis=0) - lab_frames[kb, p].mean(axis=0)) * 0.5
                    continue
            deltas[s, p] = (lab_frames[idx_a, p].mean(axis=0) - lab_frames[idx_b, p].mean(axis=0)) * 0.5

    # Per-region 95th percentile noise of the N-frame mean around its true value
    per_region_de = delta_e(base_lab + deltas, base_lab, formula)  # (splits, K)
    per_region = np.percentile(per_region_de, 95, axis=0)
    if not return_max_pair:
        return per_region

    v_idx = np.arange(n_regions) if valid_idx is None else np.asarray(valid_idx, dtype=int)
    if len(v_idx) < 2:
        return per_region, float(per_region.max()) if len(per_region) else 0.0
    ref = base_lab[:, v_idx, :].mean(axis=1, keepdims=True)  # (1, 1, 3)
    sim_lab = ref + deltas[:, v_idx, :]  # (splits, V, 3)
    pair_de = delta_e(sim_lab[:, :, None, :], sim_lab[:, None, :, :], formula)  # (splits, V, V)
    max_pairs = pair_de.reshape(splits, -1).max(axis=1)
    return per_region, float(np.percentile(max_pairs, 90))


class RoundResult:
    """One burst of frames plus its outlier analysis."""

    def __init__(self, index, start_s, times, rgb, lab, clip, decode_errors, params):
        self.index = index
        self.t_min = start_s / 60
        self.times = np.asarray(times) - times[0]
        self.rgb = np.asarray(rgb)  # (N, P, 3)
        self.lab = np.asarray(lab)
        self.clip = np.asarray(clip)  # (N, P)
        self.decode_errors = decode_errors
        self.formula = params.de_formula
        self.white = params.white
        # MAD outlier rejection on RGB: z = |c - median| / (1.4826 MAD + eps), per channel.
        median_rgb, self.robust_z, self.de_median, self.outlier, self.outlier_channel, mad_mean = mad_filter_rgb(
            self.rgb, self.lab, params.outlier_z, params.outlier_de, self.formula
        )
        count = len(self.rgb)
        trim = int(count * TRIM_FRACTION)
        self.aggregate_rgb = {
            "mean": self.rgb.mean(axis=0),
            "median": median_rgb,
            "trimmed": np.sort(self.rgb, axis=0)[trim:count - trim].mean(axis=0),
            "mad_mean": mad_mean,
        }
        self.aggregate_lab = {k: srgb8_to_lab(v, self.white) for k, v in self.aggregate_rgb.items()}
        # The reported round value is MAD + Mean.
        self.mean_rgb = self.aggregate_rgb["mad_mean"]
        self.mean_lab = self.aggregate_lab["mad_mean"]
        self.control_changes = {}  # filled by the worker: locked camera controls that moved
        self.std_rgb = self.rgb.std(axis=0, ddof=1) if len(self.rgb) > 1 else np.zeros_like(self.mean_rgb)
        self.std_lab = self.lab.std(axis=0, ddof=1) if len(self.lab) > 1 else np.zeros_like(self.mean_lab)

    @property
    def frame_count(self):
        return len(self.rgb)

    def garbage_frames(self):
        """Indices of frames flagged at any point."""
        return np.flatnonzero(self.outlier.any(axis=1))


class MeasurementSession:
    """Schedules rounds on the worker clock; fed one sample per decoded frame."""

    def __init__(self, points, params, start):
        self.points = [tuple(p) for p in points]
        self.params = params
        self.start = start
        self.next_due = start
        self.rounds = []
        self.current = None
        self.state = "running"  # running | done | stopped
        self.meta = None  # preparation record set by the worker (stabilisation, locked controls)
        # Continuous 1 s means over the whole session, for the drift time series.
        self.monitor_t, self.monitor_rgb, self.monitor_lab = [], [], []
        self._bin = None

    @property
    def active(self):
        return self.state == "running"

    def add(self, now, rgb, lab, clip):
        """Record a frame if a round is due; return the RoundResult when a round completes."""
        if not self.active:
            return None
        self._monitor(now, rgb, lab)
        if self.current is None:
            if now < self.next_due:
                return None
            self.current = dict(start=now, t=[], rgb=[], lab=[], clip=[], errors=0)
        cur = self.current
        cur["t"].append(now)
        cur["rgb"].append(rgb)
        cur["lab"].append(lab)
        cur["clip"].append(clip)
        if len(cur["t"]) < self.params.frames:
            return None
        result = RoundResult(len(self.rounds), cur["start"] - self.start, cur["t"], cur["rgb"], cur["lab"],
                             cur["clip"], cur["errors"], self.params)
        self.rounds.append(result)
        self.current = None
        self.next_due = self.start + len(self.rounds) * self.params.interval_s
        if len(self.rounds) >= self.params.rounds_total:
            self.state = "done"
        return result

    def _monitor(self, now, rgb, lab):
        if self._bin is None:
            self._bin = [now, [], []]
        self._bin[1].append(rgb)
        self._bin[2].append(lab)
        if now - self._bin[0] >= MONITOR_BIN_S:
            self.monitor_t.append(((self._bin[0] + now) / 2 - self.start) / 60)  # bin centre
            self.monitor_rgb.append(np.mean(self._bin[1], axis=0))
            self.monitor_lab.append(np.mean(self._bin[2], axis=0))
            self._bin = None

    def monitor(self):
        """(minutes (M,), rgb (M, P, 3), lab (M, P, 3)) of the continuous series."""
        return (np.array(self.monitor_t), np.array(self.monitor_rgb).reshape(-1, len(self.points), 3),
                np.array(self.monitor_lab).reshape(-1, len(self.points), 3))

    def note_decode_error(self):
        if self.current is not None:
            self.current["errors"] += 1

    def stop(self):
        if self.active:
            self.state = "stopped"
            self.current = None

    def progress(self, now):
        return dict(state=self.state, done=len(self.rounds), total=self.params.rounds_total,
                    collecting=len(self.current["t"]) if self.current else 0,
                    next_in=max(0.0, self.next_due - now) if self.active and self.current is None else 0.0,
                    elapsed=now - self.start)


class Stabilizer:
    """Wait until the 1 s ROI means stop moving.

    Stable when, over the last `window_s` seconds, every 1 s mean is within `threshold`
    ΔE of the window mean at every point. Gives up after `timeout_s`.
    """

    def __init__(self, window_s, threshold, timeout_s, start, formula="de00"):
        self.formula = formula
        self.window = max(2, int(round(window_s)))
        self.threshold = threshold
        self.timeout_s = timeout_s
        self.start = start
        self.means = deque(maxlen=self.window)
        self.spread = None
        self._bin = None

    def add(self, now, lab):
        """Return "stable", "timeout" or None."""
        if self._bin is None:
            self._bin = [now, []]
        self._bin[1].append(lab)
        if now - self._bin[0] >= MONITOR_BIN_S:
            self.means.append(np.mean(self._bin[1], axis=0))
            self._bin = None
            if len(self.means) == self.window:
                window = np.array(self.means)
                self.spread = float(delta_e(window, window.mean(axis=0)[None], self.formula).max())
                if self.spread <= self.threshold:
                    return "stable"
        if now - self.start >= self.timeout_s:
            return "timeout"
        return None

    def window_mean(self):
        return np.array(self.means).mean(axis=0) if self.means else None

    def progress(self, now):
        return dict(spread=self.spread, filled=len(self.means), window=self.window, elapsed=now - self.start,
                    timeout=self.timeout_s, threshold=self.threshold)


class LiveBuffer:
    """Short rolling window used while placing points."""

    def __init__(self, size=30):
        self.rgb = deque(maxlen=size)
        self.lab = deque(maxlen=size)

    def clear(self):
        self.rgb.clear()
        self.lab.clear()

    def add(self, rgb, lab):
        if self.rgb and len(self.rgb[-1]) != len(rgb):
            self.clear()
        self.rgb.append(rgb)
        self.lab.append(lab)

    def stats(self):
        if not self.rgb:
            return None
        rgb, lab = np.array(self.rgb), np.array(self.lab)
        std = rgb.std(axis=0, ddof=1) if len(rgb) > 1 else np.zeros(rgb.shape[1:])
        return dict(n=len(rgb), rgb=rgb.mean(axis=0), lab=lab.mean(axis=0), std_rgb=std)


def method_comparison(rounds):
    """Compare aggregation methods per point: spread between rounds, drift, and shift from plain Mean."""
    rows = []
    if not rounds:
        return rows
    formula = rounds[0].formula
    for p in range(rounds[0].rgb.shape[1]):
        for key, name in AGGREGATES.items():
            rgb = np.array([r.aggregate_rgb[key][p] for r in rounds])
            lab = np.array([r.aggregate_lab[key][p] for r in rounds])
            mean_lab = np.array([r.aggregate_lab["mean"][p] for r in rounds])
            rows.append(dict(point=p + 1, method=name,
                             between_std_rgb=rgb.std(axis=0, ddof=1) if len(rounds) > 1 else np.zeros(3),
                             cv_pct=(rgb.std(axis=0, ddof=1) / np.maximum(rgb.mean(axis=0), 1e-9) * 100
                                     if len(rounds) > 1 else np.zeros(3)),
                             spread_de=float(delta_e(lab, lab.mean(axis=0)[None], formula).mean()),
                             drift_de_max=float(delta_e(lab, lab[0][None], formula).max()),
                             shift_from_mean_de=float(delta_e(lab, mean_lab, formula).mean())))
    return rows


def session_summary(rounds):
    """Per-point statistics across all rounds, and colour difference between points."""
    if not rounds:
        return [], {}
    points = rounds[0].rgb.shape[1]
    formula = rounds[0].formula
    minutes = np.array([r.t_min for r in rounds])
    means_rgb = np.array([r.mean_rgb for r in rounds])  # (R, P, 3)
    means_lab = np.array([r.mean_lab for r in rounds])
    rows = []
    for p in range(points):
        drift = delta_e(means_lab[:, p], means_lab[0, p][None], formula)
        slope = (np.polyfit(minutes, means_rgb[:, p], 1)[0] if len(rounds) >= 3 else np.full(3, np.nan))
        frames = sum(r.frame_count for r in rounds)
        outliers = sum(int(r.outlier[:, p].sum()) for r in rounds)
        within = np.concatenate([r.de_median[:, p] for r in rounds])
        rows.append(dict(point=p + 1, mean_rgb=means_rgb[:, p].mean(axis=0), mean_lab=means_lab[:, p].mean(axis=0),
                         within_std_rgb=np.mean([r.std_rgb[p] for r in rounds], axis=0),
                         between_std_rgb=means_rgb[:, p].std(axis=0, ddof=1) if len(rounds) > 1 else np.zeros(3),
                         within_de_p95=float(np.percentile(within, 95)), within_de_max=float(within.max()),
                         drift_de_last=float(drift[-1]), drift_de_max=float(drift.max()),
                         slope_rgb_per_min=slope, outliers=outliers, frames=frames,
                         clip_pct=float(np.mean([r.clip[:, p].mean() for r in rounds]) * 100)))
    overall = means_lab.mean(axis=0)
    pairs = {f"{i + 1}-{j + 1}": float(delta_e(overall[i], overall[j], formula))
             for i in range(points) for j in range(i + 1, points)}
    return rows, pairs


# ----------------------------------------------------------------------------- export
def _append_rows(path, header, rows):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(header)
        writer.writerows(rows)


def write_session_info(folder, points, params, camera, mode, meta=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    info = dict(camera=camera, mode=mode, points=[list(p) for p in points], params=asdict(params),
                preparation=meta,
                formula=FORMULAS.get(params.de_formula, params.de_formula), lab_white=params.white,
                note="RGB from BT.601 YUYV treated as sRGB; Lab uses the white above (D50 via Bradford). "
                     "Values are relative unless the camera is calibrated.")
    (folder / "session.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def export_round(folder, result):
    """Append one round to frames.csv (every frame) and rounds.csv (per-round statistics)."""
    folder = Path(folder)
    frames = []
    for i in range(result.frame_count):
        for p in range(result.rgb.shape[1]):
            frames.append([result.index + 1, f"{result.t_min:.4f}", i, f"{result.times[i]:.4f}", p + 1,
                           *(f"{v:.3f}" for v in result.rgb[i, p]), *(f"{v:.3f}" for v in result.lab[i, p]),
                           f"{result.de_median[i, p]:.4f}", int(result.outlier[i, p]), f"{result.clip[i, p]:.4f}"])
    _append_rows(folder / "frames.csv", ["round", "round_min", "frame", "t_s", "point", "R", "G", "B", "L", "a", "b",
                                         "de_to_round_median", "outlier", "clip_fraction"], frames)
    rounds = []
    for p in range(result.rgb.shape[1]):
        rounds.append([result.index + 1, f"{result.t_min:.4f}", p + 1, result.frame_count,
                       *(f"{v:.3f}" for v in result.mean_rgb[p]), *(f"{v:.3f}" for v in result.std_rgb[p]),
                       *(f"{v:.3f}" for v in result.mean_lab[p]), *(f"{v:.3f}" for v in result.std_lab[p]),
                       f"{np.percentile(result.de_median[:, p], 95):.4f}", int(result.outlier[:, p].sum()),
                       result.decode_errors,
                       ";".join(f"{k}:{a}->{b}" for k, (a, b) in result.control_changes.items())])
    _append_rows(folder / "rounds.csv", ["round", "round_min", "point", "frames", "R_mean", "G_mean", "B_mean",
                                         "R_std", "G_std", "B_std", "L_mean", "a_mean", "b_mean", "L_std", "a_std",
                                         "b_std", "de_p95_in_round", "outlier_frames", "decode_errors",
                                         "camera_controls_changed"], rounds)


def export_monitor(folder, minutes, rgb, lab):
    """Rewrite monitor.csv with the continuous 1 s series."""
    with (Path(folder) / "monitor.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["minute", "point", "R", "G", "B", "L", "a", "b"])
        for i, minute in enumerate(minutes):
            for p in range(rgb.shape[1]):
                writer.writerow([f"{minute:.4f}", p + 1, *(f"{v:.3f}" for v in rgb[i, p]),
                                 *(f"{v:.3f}" for v in lab[i, p])])


def export_summary(folder, rounds):
    rows, pairs = session_summary(rounds)
    path = Path(folder) / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point", "R", "G", "B", "L", "a", "b", "within_std_R", "within_std_G", "within_std_B",
                         "between_std_R", "between_std_G", "between_std_B", "within_de_p95", "within_de_max",
                         "drift_de_last", "drift_de_max", "slope_R_per_min", "slope_G_per_min", "slope_B_per_min",
                         "outlier_frames", "frames", "clip_pct"])
        for r in rows:
            writer.writerow([r["point"], *(f"{v:.3f}" for v in (*r["mean_rgb"], *r["mean_lab"], *r["within_std_rgb"],
                                                                 *r["between_std_rgb"])),
                             f"{r['within_de_p95']:.4f}", f"{r['within_de_max']:.4f}", f"{r['drift_de_last']:.4f}",
                             f"{r['drift_de_max']:.4f}", *(f"{v:.5f}" for v in r["slope_rgb_per_min"]),
                             r["outliers"], r["frames"], f"{r['clip_pct']:.3f}"])
        writer.writerow([])
        writer.writerow(["pair", "de_between_points"])
        for pair, value in pairs.items():
            writer.writerow([pair, f"{value:.4f}"])
        writer.writerow([])
        writer.writerow(["point", "method", "between_rounds_std_R", "std_G", "std_B", "cv_R_pct", "cv_G_pct",
                         "cv_B_pct", "mean_de_to_session_mean", "max_drift_de_vs_round1", "mean_de_shift_from_mean"])
        for m in method_comparison(rounds):
            writer.writerow([m["point"], m["method"], *(f"{v:.4f}" for v in (*m["between_std_rgb"], *m["cv_pct"])),
                             f"{m['spread_de']:.4f}", f"{m['drift_de_max']:.4f}", f"{m['shift_from_mean_de']:.4f}"])
    return rows, pairs
