"""Single-owner camera worker; all driver I/O runs outside the GUI thread."""

import queue
import threading
import time
from collections import deque
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from .comparison import CompareReport
from .decoders import METHODS, jpeg_bytes, process_native_yuyv, yuyv_to_bgr
from .devices import discover_cameras, find_device
from . import camera_controls
from .colorimetry import delta_e
from .fabric_detect import build_background_model, detect_fabric, mask_iou
from .measurement import MAX_POINTS, LiveBuffer, MeasurementSession, SessionParams, Stabilizer, sample_points
from .uniformity import (
    UniformityParams,
    analyze_uniformity,
    build_master_reference,
    build_ring_layout,
    build_timeseries_round,
    sample_ring_frame,
)

DECODERS = {m[0]: m for m in METHODS}


class FrameDecodeError(RuntimeError):
    """A received buffer could not be decoded; a later frame may be valid."""


def fourcc_string(value):
    value = int(value)
    return "".join(chr((value >> (8 * i)) & 255) for i in range(4))


class CameraWorker(QThread):
    cameras_changed = pyqtSignal(object)
    state_changed = pyqtSignal(object)
    notice = pyqtSignal(str, str)
    busy_changed = pyqtSignal(bool, str)
    report_ready = pyqtSignal(object)
    settings_applied = pyqtSignal(object)
    measure_live = pyqtSignal(object)
    measure_round = pyqtSignal(object)
    measure_meta = pyqtSignal(object)  # preparation record (stabilisation + locked controls) at session start
    uniform_state = pyqtSignal(object)
    uniform_done = pyqtSignal(object)

    def __init__(self, settings, autostart=True, parent=None):
        super().__init__(parent)
        self.config = replace(settings)
        self.autostart = autostart
        self.commands = queue.Queue()
        self.shutdown_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.pending_frame = None
        self.cameras = []
        self.cap = None
        self.actual = None
        self.latest = None
        self.stats = deque(maxlen=90)
        self.last_preview = 0.0
        self.last_state = 0.0
        self.last_error = ""
        self.report = None
        self.invalid_frames = 0
        self.measure_points = []
        self.measure_params = SessionParams()
        self.live = LiveBuffer()
        self.session = None
        self.prep = None  # dict(phase, stabilizer, meta) while waiting for a stable image
        self.controls = {}  # last read camera controls (name -> value)
        self.last_measure_emit = 0.0
        self.uniform_active = False
        self.uniform_params = UniformityParams()
        self.uniform_bg = None
        self.uniform_master = None
        self.uniform_detection = None
        self.uniform_layout = None
        self.uniform_result = None
        self.uniform_job = None  # dict(times, rgb, lab, clip, meta, frame_bgr, shifted)
        self.uniform_ts = None  # dict(active, params, detection, layout, start_time, next_round_at, round_index, rounds, meta)
        self.last_uniform_detect = 0.0
        self.last_uniform_ts_emit = 0.0

    def submit(self, operation, payload=None):
        self.commands.put((operation, payload))

    def shutdown(self):
        self.shutdown_event.set()

    def take_frame(self):
        with self.frame_lock:
            image, self.pending_frame = self.pending_frame, None
        return image

    def _clear_frame(self):
        self.latest = None
        with self.frame_lock:
            self.pending_frame = None

    def _release(self):
        if self.prep is not None:
            self.prep = None
            self.notice.emit("Đã hủy bước chờ ổn định vì luồng camera bị dừng hoặc đổi cấu hình.", "warning")
        if self.session is not None and self.session.active:
            self.session.stop()
            self.notice.emit("Phiên đo màu đã dừng vì luồng camera bị dừng hoặc đổi cấu hình. "
                             "Các lần đo đã hoàn tất vẫn được lưu.", "warning")
            self._emit_measure()
        if self.uniform_job is not None or self.uniform_ts is not None:
            self.uniform_job = None
            self.uniform_ts = None
            self.notice.emit("Đã hủy phiên đo đồng màu vì luồng camera bị dừng hoặc đổi cấu hình.", "warning")
            self._emit_uniform()
        self.live.clear()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.actual = None
        self.stats.clear()
        self._clear_frame()

    def _device(self, settings=None):
        return find_device(self.cameras, (settings or self.config).camera_id)

    def _open(self, settings):
        settings.validate()
        device = self._device(settings)
        if device is None:
            raise RuntimeError("Camera đã mất kết nối. Bấm Tìm lại camera.")
        if settings.method not in DECODERS:
            raise ValueError("Decoder chưa được cài đặt.")
        matched = any(m.fourcc == settings.fourcc and m.width == settings.width and
                      m.height == settings.height and abs(m.fps - settings.fps) < 0.05
                      for m in device.modes)
        if not matched:
            raise ValueError("Mode này không nằm trong danh sách camera hỗ trợ. Hãy tìm lại camera.")
        raw = settings.fourcc == "YUYV" or DECODERS[settings.method][1]
        self._release()
        cap = cv2.VideoCapture(device.node, cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Không mở được {device.node}; kiểm tra kết nối hoặc ứng dụng đang dùng camera.")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings.fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
            cap.set(cv2.CAP_PROP_FPS, settings.fps)
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0 if raw else 1)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            actual = (fourcc_string(cap.get(cv2.CAP_PROP_FOURCC)),
                      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      cap.get(cv2.CAP_PROP_FPS))
            if actual[:3] != (settings.fourcc, settings.width, settings.height):
                raise RuntimeError(f"Camera trả về {actual[0]} {actual[1]}×{actual[2]}, khác mode đã chọn.")
            if abs(actual[3] - settings.fps) > 0.1:
                self.notice.emit(f"Camera chấp nhận {actual[3]:g} FPS; yêu cầu {settings.fps:g} FPS.", "warning")
            self.cap, self.actual = cap, actual
            self.last_error = ""
            self.last_preview = 0.0
            self.invalid_frames = 0
        except Exception:
            cap.release()
            raise

    def _apply(self, settings):
        old, was_running = self.config, self.cap is not None
        settings.validate()
        acquisition = ("camera_id", "fourcc", "width", "height", "fps", "method")
        must_open = not was_running or any(getattr(old, k) != getattr(settings, k) for k in acquisition)
        try:
            if must_open:
                self._open(settings)
        except Exception:
            if was_running and self.cap is None:
                try:
                    self._open(old)
                except Exception as rollback_error:
                    self.notice.emit(f"Không khôi phục được luồng trước: {rollback_error}", "error")
            raise
        self.config = replace(settings, camera_id=self._device(settings).identity)
        self.last_error = ""
        self.settings_applied.emit(replace(self.config))
        self.notice.emit("Đã áp dụng cấu hình camera.", "success")

    def _scan(self, initial=False):
        was_running = self.cap is not None
        devices, failures = discover_cameras()
        self.cameras = devices
        self.cameras_changed.emit(devices)
        if failures:
            self.notice.emit("; ".join(failures), "warning")
        selected = self._device()
        if selected is None and not self.config.camera_id and devices:
            from .devices import select_mode
            selected = devices[0]
            mode = select_mode(selected, self.config.fourcc, self.config.width, self.config.height, self.config.fps)
            self.config = replace(self.config, camera_id=selected.identity, fourcc=mode.fourcc,
                                  width=mode.width, height=mode.height, fps=mode.fps)
        if not selected:
            self._release()
            self.last_error = "Không thấy camera đã chọn. Chọn camera khác rồi bấm Bắt đầu."
            if not devices:
                self.last_error = "Chưa có camera MJPG/YUYV. Cắm camera rồi bấm Tìm lại camera."
            self.notice.emit(self.last_error, "warning")
        elif was_running or (initial and self.autostart):
            self._apply(replace(self.config, camera_id=selected.identity))
            # A rescan also reconnects a device whose /dev/video number changed.
            if was_running:
                self._open(self.config)
        if selected:
            self.notice.emit(f"Tìm thấy {len(devices)} camera có mode MJPG/YUYV.", "info")

    def _read(self):
        start = time.perf_counter()
        ok, frame = self.cap.read()
        read_end = time.perf_counter()
        if not ok or frame is None:
            raise RuntimeError("Không nhận được ảnh. Kiểm tra camera rồi bấm Tìm lại camera / Bắt đầu.")
        try:
            if self.config.fourcc == "YUYV":
                result = process_native_yuyv(frame, self.actual[1], self.actual[2])
            else:
                decoder = DECODERS[self.config.method]
                if decoder[1] and frame.ndim > 1 and frame.shape[0] != 1 and frame.shape[1] != 1:
                    raise RuntimeError("Backend không trả byte MJPEG thô như yêu cầu.")
                result = decoder[2](frame)
        except Exception as exc:
            raise FrameDecodeError(str(exc)) from exc
        self.invalid_frames = 0
        self.latest = result
        if self.measure_points:
            self._sample(read_end, result["yuyv"])
        if self.uniform_active or self.uniform_job is not None or self.uniform_ts is not None:
            self._sample_uniform(read_end, result["yuyv"])
        # Store timings only; retaining full frames here would consume gigabytes.
        self.stats.append((read_end, (read_end - start) * 1000, result["dec"], result["cvt"]))
        now = time.perf_counter()
        if now - self.last_preview >= 1 / self.config.preview_fps:
            self.last_preview = now
            image = self._preview(result)
            with self.frame_lock:
                self.pending_frame = image
        if now - self.last_state >= 0.5:
            self._emit_state()

    def _step(self):
        try:
            self._read()
        except FrameDecodeError as exc:
            if self.session is not None:
                self.session.note_decode_error()
            self.invalid_frames += 1
            if self.invalid_frames < 5:
                self.notice.emit(f"Bỏ qua frame lỗi ({self.invalid_frames}/5): {exc}", "warning")
                return
            self.last_error = f"Camera trả về 5 frame lỗi liên tiếp: {exc}"
            self._release()
            self.notice.emit(self.last_error, "error")
            self._emit_state()
        except Exception as exc:
            self.last_error = str(exc)
            self._release()
            self.notice.emit(str(exc), "error")
            self._emit_state()

    def _preview(self, result):
        packed = result["yuyv"]
        view = self.config.view
        if view == "luma":
            image = cv2.cvtColor(packed[..., 0], cv2.COLOR_GRAY2RGB)
        else:
            reconstructed = yuyv_to_bgr(packed)
            if view == "source" and result["bgr"] is not None:
                bgr = result["bgr"]
            elif view == "difference" and result["bgr"] is not None:
                bgr = cv2.convertScaleAbs(cv2.absdiff(reconstructed, result["bgr"]), alpha=8)
            else:
                bgr = reconstructed
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        if width > self.config.preview_width:
            image = cv2.resize(image, (self.config.preview_width, max(1, round(height * self.config.preview_width / width))),
                               interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(image)

    # ----- three-point colour measurement (runs on the worker thread) -----
    def _sample(self, now, yuyv):
        params = self.measure_params
        rgb, lab, clip = sample_points(yuyv, self.measure_points, params.point_rois(len(self.measure_points)), params.white)
        self.live.add(rgb, lab)
        if self.prep is not None:
            self._prepare(now, lab)
        elif self.session is not None:
            result = self.session.add(now, rgb, lab, clip)
            if result is not None:
                locked = (self.session.meta or {}).get("controls", {}).get("locked")
                if locked:
                    try:
                        result.control_changes = camera_controls.changed_controls(self._device().node, locked)
                    except Exception as exc:
                        result.control_changes = {"read_error": (str(exc), None)}
                    if result.control_changes:
                        self.notice.emit(f"Lần đo {result.index + 1}: thông số camera đã bị thay đổi: "
                                         f"{result.control_changes}", "warning")
                self.measure_round.emit(result)
                if self.session.state == "done":
                    self.notice.emit("Phiên đo màu đã hoàn tất.", "success")
                self._emit_measure(now)
        if now - self.last_measure_emit >= 0.25:
            self._emit_measure(now)

    def _prepare(self, now, lab):
        """Stabilise (auto loops), lock controls, match the auto image, stabilise again, then start."""
        prep = self.prep
        params = self.measure_params
        if prep["phase"] == "match":
            if prep["matcher"].add(now, lab):
                matcher = prep["matcher"]
                prep["meta"]["matched"] = matcher.result
                self._read_controls()
                if prep["meta"].get("controls"):
                    prep["meta"]["controls"]["locked"] = dict(self.controls)
                if matcher.exposure_failed:
                    prep["meta"]["warnings"].append("exposure_not_lockable")
                    self.notice.emit("Camera này không khóa được exposure: chế độ tay không đạt độ sáng của AE (AE dùng "
                                     "gain ẩn). Giữ AE tự động, chỉ khóa cân bằng trắng. Kết quả có thể trôi theo AE.",
                                     "warning")
                else:
                    self.notice.emit("Đã dò thông số khớp ảnh tự động: " + camera_controls.describe(self.controls), "info")
                prep["phase"] = "settle_locked"
                prep["stabilizer"] = Stabilizer(params.settle_window_s, params.settle_de, params.settle_timeout_s, now, params.de_formula)
            return
        outcome = prep["stabilizer"].add(now, lab)
        if outcome is None:
            return
        stab = prep["stabilizer"]
        prep["meta"]["stabilization"].append(dict(phase=prep["phase"], result=outcome, spread_de=stab.spread,
                                                  seconds=round(now - stab.start, 2)))
        if outcome == "timeout":
            self.notice.emit(f"Hết thời gian chờ ổn định ({prep['phase']}); dao động ΔE "
                             f"{stab.spread if stab.spread is not None else float('nan'):.2f}. Vẫn tiếp tục.", "warning")
        if prep["phase"] == "settle_auto":
            prep["auto_mean"] = stab.window_mean()
            parsed = self._lock(prep["meta"])
            if parsed is not None and prep["auto_mean"] is not None:
                matcher = camera_controls.AutoLockMatcher(self._device().node, prep["auto_mean"], parsed, now)
                if not matcher.done:
                    prep["phase"], prep["matcher"] = "match", matcher
                    return
            prep["phase"] = "settle_locked"
            prep["stabilizer"] = Stabilizer(params.settle_window_s, params.settle_de, params.settle_timeout_s, now, params.de_formula)
            return
        if prep.get("auto_mean") is not None and stab.window_mean() is not None:
            # Some UVC drivers report a stale exposure while AE is on, so locking can shift the image.
            shift = float(delta_e(stab.window_mean(), prep["auto_mean"], params.de_formula).max())
            prep["meta"]["lock_shift_de"] = shift
            if shift > 1.0:
                self.notice.emit(f"Ảnh thay đổi ΔE {shift:.2f} khi khóa thông số (driver có thể báo exposure cũ). "
                                 "Kiểm tra độ sáng; có thể chỉnh exposure bằng v4l2-ctl rồi khóa lại.", "warning")
        self.prep = None
        self.session = MeasurementSession(self.measure_points, params, now)
        self.session.meta = prep["meta"]
        self.measure_meta.emit(prep["meta"])
        self.notice.emit(f"Ảnh đã ổn định; bắt đầu {params.rounds_total} lần đo, mỗi lần {params.frames} frame.", "info")

    def _lock(self, meta):
        try:
            result, warnings = camera_controls.lock_controls(self._device().node)
        except Exception as exc:
            meta["warnings"].append(f"Không khóa được thông số: {exc}")
            self.notice.emit(f"Không khóa được thông số camera: {exc}", "warning")
            return
        parsed = result.pop("parsed_before")
        meta["controls"] = result
        meta["warnings"] += warnings
        self.controls = result["locked"]
        self.notice.emit("Đã khóa thông số camera: " + camera_controls.describe(self.controls)
                         + (f" · cảnh báo: {'; '.join(warnings)}" if warnings else ""), "warning" if warnings else "success")
        return parsed

    def _read_controls(self):
        try:
            self.controls = camera_controls.values(camera_controls.read_controls(self._device().node))
        except Exception:
            self.controls = {}

    def _emit_measure(self, now=None):
        now = time.perf_counter() if now is None else now
        self.last_measure_emit = now
        prep = None
        if self.prep is not None and self.prep["phase"] == "match":
            prep = dict(phase="match", description=self.prep["matcher"].description)
        elif self.prep is not None:
            prep = dict(phase=self.prep["phase"], **self.prep["stabilizer"].progress(now))
        self.measure_live.emit(dict(points=list(self.measure_points), params=replace(self.measure_params),
                                    live=self.live.stats(), prep=prep,
                                    controls=camera_controls.describe(self.controls) if self.controls else "",
                                    session=self.session.progress(now) if self.session is not None else None,
                                    monitor=self.session.monitor() if self.session is not None else None))

    def _measure(self, payload):
        action = payload["action"]
        running = (self.session is not None and self.session.active) or self.prep is not None
        if action == "points":
            if running:
                self.notice.emit("Đang có phiên đo; dừng phiên trước khi đổi điểm hoặc ROI.", "warning")
            else:
                self.measure_points = [tuple(p) for p in payload["points"]][:MAX_POINTS]
                self.measure_params = replace(self.measure_params, roi=payload["roi"],
                                              rois=tuple(payload.get("rois") or ()))
                self.live.clear()
        elif action == "start":
            if self.cap is None or not self.measure_points:
                self.notice.emit("Bật camera và đặt ít nhất một điểm trước khi đo.", "warning")
            elif not running:
                params = self.measure_params = replace(payload["params"], rois=self.measure_params.rois)
                self.session = None
                self._read_controls()
                meta = dict(stabilization=[], controls=None, warnings=[], controls_at_start=dict(self.controls))
                self.prep = dict(phase="settle_auto" if params.lock else "settle", meta=meta,
                                 stabilizer=Stabilizer(params.settle_window_s, params.settle_de,
                                                       params.settle_timeout_s, time.perf_counter(), params.de_formula))
                self.notice.emit("Đang chờ ảnh ổn định" + (" trước khi khóa thông số camera…" if params.lock else "…"),
                                 "info")
        elif action == "stop" and running:
            if self.prep is not None:
                self.prep = None
            else:
                self.session.stop()
            self.notice.emit("Đã dừng phiên đo màu.", "info")
        elif action in ("lock", "unlock") and self._device() is not None:
            if running:
                self.notice.emit("Không đổi thông số camera khi đang đo.", "warning")
            elif action == "lock":
                self._lock(dict(warnings=[]))
            else:
                try:
                    failures = camera_controls.unlock_controls(self._device().node)
                    self.notice.emit("Đã trả AE/AWB về tự động." if not failures else "; ".join(failures),
                                     "warning" if failures else "success")
                except Exception as exc:
                    self.notice.emit(f"Không trả được về tự động: {exc}", "error")
                self._read_controls()
        elif action == "controls" and self._device() is not None:
            self._read_controls()
        self._emit_measure()

    # ----- 5%-inset ring uniformity check (runs on the worker thread) -----
    def _detect_uniform_now(self, yuyv):
        bg = self.uniform_bg if self.uniform_params.bg_mode == "reference" else None
        self.uniform_detection = detect_fabric(yuyv, background=bg)
        self.uniform_layout = build_ring_layout(self.uniform_detection, self.uniform_params, reference_frame=yuyv)
        self.last_uniform_detect = time.perf_counter()

    def _sample_uniform(self, now, yuyv):
        if self.uniform_job is None and self.uniform_ts is not None and self.uniform_ts.get("active"):
            ts = self.uniform_ts
            if now >= ts["next_round_at"]:
                r_idx = ts["round_index"]
                params = ts["params"]
                bg = self.uniform_bg if params.bg_mode == "reference" else None
                check_det = detect_fabric(yuyv, background=bg)
                shifted = (not check_det.found) or (mask_iou(check_det.mask, ts["detection"].mask) < 0.95)
                self.uniform_job = dict(
                    params=params,
                    detection=ts["detection"],
                    layout=ts["layout"],
                    frame_bgr=yuyv_to_bgr(yuyv),
                    times=[],
                    rgb=[],
                    lab=[],
                    clip=[],
                    meta=ts["meta"],
                    shifted=shifted,
                    is_master=False,
                    is_timeseries=True,
                    round_index=r_idx,
                )
            else:
                if now - self.last_uniform_ts_emit >= 0.5:
                    self.last_uniform_ts_emit = now
                    rem_s = max(0.0, ts["next_round_at"] - now)
                    r_done = len(ts["rounds"])
                    last_de = ts["rounds"][-1].overall_master_de if r_done else 0.0
                    next_min = (ts["round_index"] * ts["params"].interval_s) / 60.0
                    self._emit_uniform(
                        progress=(
                            f"⏱ Đang theo dõi Time-Series (đã chụp {r_done} lần · ΔE00 vs gốc = {last_de:.3f}): "
                            f"còn {rem_s:.0f}s tới lần chụp #{ts['round_index']} (t = {next_min:.1f} phút)…"
                        )
                    )
                return

        if self.uniform_job is not None:
            job = self.uniform_job
            layout = job["layout"]
            params = job["params"]
            # Periodically verify the fabric has not shifted during measurement
            if len(job["times"]) > 0 and len(job["times"]) % 5 == 0:
                bg = self.uniform_bg if params.bg_mode == "reference" else None
                check_det = detect_fabric(yuyv, background=bg)
                if not check_det.found or mask_iou(check_det.mask, job["detection"].mask) < 0.98:
                    job["shifted"] = True
            rgb, lab, clip = sample_ring_frame(yuyv, layout, params.white)
            job["times"].append(now)
            job["rgb"].append(rgb)
            job["lab"].append(lab)
            job["clip"].append(clip)
            done_n = len(job["times"])
            is_master = bool(job.get("is_master"))
            is_timeseries = bool(job.get("is_timeseries"))
            round_index = int(job.get("round_index", 0))
            if is_timeseries:
                action_label = f"chụp Time-Series lần #{round_index} ({'vải gốc t=0' if round_index == 0 else f't={round_index * params.interval_s / 60.0:.1f}p'})"
            else:
                action_label = "chụp vải gốc" if is_master else ("so với vải gốc" if self.uniform_master else "đo đồng màu")
            if done_n < params.frames:
                self._emit_uniform(progress=f"Đang {action_label}: đã chụp {done_n}/{params.frames} frame…")
                return

            elapsed_s = 0.0
            if is_timeseries and self.uniform_ts is not None:
                if round_index == 0:
                    self.uniform_ts["start_time"] = float(job["times"][0])
                else:
                    elapsed_s = max(0.0, float(job["times"][0] - self.uniform_ts["start_time"]))

            master_ref = None if is_master else self.uniform_master
            result = analyze_uniformity(
                layout, job["times"], job["rgb"], job["lab"], job["clip"], params,
                master=master_ref, detection=job["detection"],
                is_timeseries=is_timeseries, round_index=round_index, elapsed_min=elapsed_s / 60.0,
            )
            if job["shifted"]:
                result.warnings.append("Vải bị xê dịch trong lúc đo (IoU < 0.98) — nên kiểm tra lại vị trí vải.")
            if is_master:
                self.uniform_master = build_master_reference(job["detection"], layout, result)
                if is_timeseries:
                    result.has_master = True
                    result.master_ring_rgb = self.uniform_master.ring_mean_rgb.copy()
                    result.master_ring_lab = self.uniform_master.ring_mean_lab.copy()
                    result.overall_master_de = 0.0
                    result.overall_delta_lab_master = np.zeros(3, dtype=np.float64)
                    result.de_to_master = np.zeros(result.segments, dtype=np.float64)
                    result.delta_lab_to_master = np.zeros((result.segments, 3), dtype=np.float64)
                    result.max_master_de = 0.0
                    result.worst_master_segment = 1
                    result.size_diff_pct = 0.0

            ts_rounds = None
            ts_active = False
            if is_timeseries and self.uniform_ts is not None:
                ts_round = build_timeseries_round(round_index, elapsed_s, result)
                self.uniform_ts["rounds"].append(ts_round)
                ts_rounds = list(self.uniform_ts["rounds"])
                next_idx = round_index + 1
                next_elapsed_s = next_idx * params.interval_s
                if params.duration_min > 0 and next_elapsed_s > params.duration_min * 60.0 + 1e-6:
                    self.uniform_ts["active"] = False
                else:
                    self.uniform_ts["round_index"] = next_idx
                    self.uniform_ts["next_round_at"] = self.uniform_ts["start_time"] + next_elapsed_s
                ts_active = bool(self.uniform_ts["active"])

            self.uniform_result = result
            self.uniform_job = None
            self.uniform_done.emit(dict(
                frame_bgr=job["frame_bgr"],
                detection=job["detection"],
                layout=layout,
                result=result,
                meta=job["meta"],
                is_master=is_master,
                master=self.uniform_master,
                is_timeseries=is_timeseries,
                round_index=round_index,
                ts_rounds=ts_rounds,
                ts_active=ts_active,
            ))
            if is_timeseries and self.uniform_ts is not None and not self.uniform_ts["active"]:
                self.uniform_ts = None
            self._emit_uniform()
            if is_timeseries:
                if ts_active:
                    self.notice.emit(
                        f"Hoàn tất Time-Series lần #{round_index} (t={elapsed_s / 60.0:.1f}p · ΔE00 vs gốc={result.overall_master_de:.3f}). "
                        f"Sẽ chụp lại sau {params.interval_s:.0f}s.",
                        "success" if result.verdict == "pass" else "warning",
                    )
                else:
                    self.notice.emit(
                        f"Đã hoàn tất chuỗi theo dõi Time-Series ({len(ts_rounds or [])} lần chụp).",
                        "success",
                    )
            elif is_master:
                l_val, a_val, b_val = self.uniform_master.ring_mean_lab
                self.notice.emit(
                    f"Đã lưu vải gốc chuẩn (L*={l_val:.2f}, a*={a_val:.2f}, b*={b_val:.2f}). Đặt các tấm vải khác vào rồi bấm 'Đo & so với vải gốc'.",
                    "success",
                )
            else:
                self.notice.emit(f"Hoàn tất kiểm tra: {result.verdict_label}", "success" if result.verdict == "pass" else "warning")
            return

        if self.uniform_active and now - self.last_uniform_detect >= 1.0:
            self._detect_uniform_now(yuyv)
            self._emit_uniform()

    def _emit_uniform(self, progress=None):
        ts_active = bool(self.uniform_ts is not None and self.uniform_ts.get("active"))
        self.uniform_state.emit(dict(
            active=self.uniform_active,
            running=(self.uniform_job is not None) or ts_active,
            ts_active=ts_active,
            ts_rounds=list(self.uniform_ts["rounds"]) if self.uniform_ts is not None else None,
            has_bg_ref=self.uniform_bg is not None,
            bg_mode=self.uniform_params.bg_mode,
            master=self.uniform_master,
            detection=self.uniform_detection,
            layout=self.uniform_layout,
            result=self.uniform_result,
            progress=progress,
        ))

    def _uniform(self, payload):
        action = payload["action"]
        measure_busy = (self.session is not None and self.session.active) or self.prep is not None
        uniform_busy = (self.uniform_job is not None) or (self.uniform_ts is not None and self.uniform_ts.get("active"))
        if action == "open":
            self.uniform_active = True
            if payload.get("params") is not None and not uniform_busy:
                self.uniform_params = replace(payload["params"])
            if self.latest is not None and not uniform_busy:
                self._detect_uniform_now(self.latest["yuyv"])
        elif action == "close":
            self.uniform_active = False
            self.uniform_job = None
            self.uniform_ts = None
        elif action == "params":
            if not uniform_busy and payload.get("params") is not None:
                self.uniform_params = replace(payload["params"])
                self.uniform_result = None
                if self.latest is not None and self.uniform_detection is not None:
                    self.uniform_layout = build_ring_layout(
                        self.uniform_detection, self.uniform_params, reference_frame=self.latest["yuyv"]
                    )
        elif action == "bg_capture":
            if self.latest is None:
                self.notice.emit("Hãy bật camera trước khi chụp nền trống.", "warning")
            elif not uniform_busy:
                self.uniform_bg = build_background_model(self.latest["yuyv"])
                self.uniform_params = replace(self.uniform_params, bg_mode="reference")
                self.uniform_result = None
                self._detect_uniform_now(self.latest["yuyv"])
                self.notice.emit("Đã chụp ảnh nền carton tham chiếu.", "success")
        elif action == "detect":
            if self.latest is None:
                self.notice.emit("Hãy bật camera trước khi phát hiện vải.", "warning")
            elif not uniform_busy:
                if payload.get("params") is not None:
                    self.uniform_params = replace(payload["params"])
                self.uniform_result = None
                self._detect_uniform_now(self.latest["yuyv"])
                if self.uniform_detection and self.uniform_detection.found:
                    self.notice.emit("Đã phát hiện tấm vải và tạo vòng chia đoạn.", "info")
                else:
                    self.notice.emit("Chưa phát hiện được vải trên khung hình.", "warning")
        elif action == "clear_master":
            if not uniform_busy:
                self.uniform_master = None
                self.uniform_result = None
                self.notice.emit("Đã xóa vải gốc chuẩn. Chuyển về chế độ đo đồng màu độc lập.", "info")
        elif action in ("start", "capture_master", "start_timeseries"):
            is_ts = action == "start_timeseries"
            is_master = (action == "capture_master") or is_ts
            if self.cap is None or self.latest is None:
                self.notice.emit("Hãy bật camera trước khi đo.", "warning")
            elif measure_busy:
                self.notice.emit("Đang có phiên đo màu 3 điểm; hãy dừng phiên đó trước.", "warning")
            elif not uniform_busy:
                if payload.get("params") is not None:
                    self.uniform_params = replace(payload["params"])
                if is_ts or is_master or self.uniform_master is not None:
                    self.uniform_params = replace(self.uniform_params, filter_dirty_segments=True)
                self.uniform_result = None
                self._detect_uniform_now(self.latest["yuyv"])
                if not self.uniform_detection.found or not self.uniform_layout.valid:
                    warn = "; ".join(self.uniform_layout.warnings) if self.uniform_layout else "Chưa phát hiện được vải."
                    self.notice.emit(f"Không thể bắt đầu đo: {warn}", "warning")
                else:
                    meta = dict(warnings=[], controls=None)
                    if self.uniform_params.lock and self._device() is not None:
                        self._lock(meta)
                    now_t = time.perf_counter()
                    if is_ts:
                        self.uniform_ts = dict(
                            active=True,
                            params=replace(self.uniform_params),
                            detection=self.uniform_detection,
                            layout=self.uniform_layout,
                            start_time=now_t,
                            next_round_at=now_t,
                            round_index=0,
                            rounds=[],
                            meta=meta,
                        )
                    self.uniform_job = dict(
                        params=replace(self.uniform_params),
                        detection=self.uniform_detection,
                        layout=self.uniform_layout,
                        frame_bgr=yuyv_to_bgr(self.latest["yuyv"]),
                        times=[],
                        rgb=[],
                        lab=[],
                        clip=[],
                        meta=meta,
                        shifted=False,
                        is_master=is_master,
                        is_timeseries=is_ts,
                        round_index=0,
                    )
                    if is_ts:
                        lbl = "Đang chụp vải gốc (Lần #0, t=0) cho chuỗi Time-Series"
                    elif is_master:
                        lbl = "Đang chụp vải gốc"
                    else:
                        lbl = "Đang đo & so với vải gốc" if self.uniform_master else "Đang đo đồng màu"
                    self._emit_uniform(progress=f"{lbl}: 0/{self.uniform_params.frames} frame…")
                    return
        elif action == "stop" and uniform_busy:
            was_ts = self.uniform_ts is not None
            self.uniform_job = None
            self.uniform_ts = None
            self.notice.emit("Đã dừng theo dõi Time-Series vải gốc." if was_ts else "Đã dừng đo đồng màu.", "info")
        self._emit_uniform()

    def _emit_state(self):
        self.last_state = time.perf_counter()
        timings = list(self.stats)
        fps = ((len(timings) - 1) / (timings[-1][0] - timings[0][0])) if len(timings) > 1 else 0
        averages = [sum(t[i] for t in timings) / len(timings) for i in (1, 2, 3)] if timings else [0, 0, 0]
        self.state_changed.emit(dict(running=self.cap is not None, actual=self.actual, config=replace(self.config),
                                     fps=fps, read=averages[0], decode=averages[1], convert=averages[2],
                                     has_frame=self.latest is not None, error=self.last_error))

    def _save_frame(self):
        if self.latest is None:
            raise RuntimeError("Chưa có ảnh để lưu.")
        folder = Path(self.config.outdir).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        height, width = self.latest["yuyv"].shape[:2]
        stem = folder / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{Path(self._device().node).name}_{width}x{height}"
        self.latest["yuyv"].tofile(str(stem) + ".yuyv")
        if not cv2.imwrite(str(stem) + ".png", yuyv_to_bgr(self.latest["yuyv"])):
            raise RuntimeError("Không lưu được ảnh PNG.")
        if self.latest["jpeg"] is not None:
            Path(str(stem) + ".jpg").write_bytes(self.latest["jpeg"])
        self.notice.emit(f"Đã lưu ảnh PNG và YUYV: {stem}", "success")

    def _grab_raw(self, device, mode):
        config = replace(self.config, camera_id=device.identity, fourcc=mode.fourcc,
                         width=mode.width, height=mode.height, fps=mode.fps, method="imdecode_cvt")
        self._open(config)
        end = time.perf_counter() + self.config.warmup
        count, frame = 0, None
        try:
            while count < 3 or time.perf_counter() < end:
                if self.shutdown_event.is_set():
                    raise RuntimeError("Đã hủy capture khi đóng ứng dụng.")
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Không chụp được frame {mode.fourcc}.")
                count += 1
            return frame.copy()
        finally:
            self._release()

    def _compare(self):
        device = self._device()
        if device is None or self.cap is None:
            raise RuntimeError("Hãy bật camera trước khi capture & compare.")
        modes = {fcc: [m for m in device.modes if m.fourcc == fcc and
                       (m.width, m.height) == (self.config.width, self.config.height)] for fcc in ("MJPG", "YUYV")}
        if not all(modes.values()):
            raise RuntimeError("Capture & compare cần độ phân giải có cả MJPG và YUYV.")
        self.report = None
        self._release()
        try:
            jpeg, _ = jpeg_bytes(self._grab_raw(device, max(modes["MJPG"], key=lambda m: m.fps)))
            native = self._grab_raw(device, max(modes["YUYV"], key=lambda m: m.fps))
            reference = process_native_yuyv(native, self.config.width, self.config.height)["yuyv"]
            if self.shutdown_event.is_set():
                return
            report = CompareReport(f"{device.name} {device.node}", reference, jpeg,
                                   self.config.jpeg_quality, str(Path(self.config.outdir).expanduser()))
            report.save()
            self.report = report
            self.report_ready.emit(report)
            self.notice.emit(f"Đã lưu bảng so sánh: {report.dir}", "success")
        finally:
            if not self.shutdown_event.is_set():
                self._open(self.config)

    def run(self):
        try:
            self.busy_changed.emit(True, "Đang tìm camera…")
            try:
                self._scan(initial=True)
            except Exception as exc:
                self.last_error = str(exc)
                self.notice.emit(str(exc), "error")
            finally:
                self.busy_changed.emit(False, "")
                self._emit_state()
            while not self.shutdown_event.is_set():
                try:
                    operation, payload = self.commands.get(timeout=0.05 if self.cap is None else 0)
                except queue.Empty:
                    operation = None
                if operation == "measure":
                    self._measure(payload)
                    operation = None
                elif operation == "uniform":
                    self._uniform(payload)
                    operation = None
                if operation:
                    self.busy_changed.emit(True, {"scan": "Đang tìm lại camera…", "apply": "Đang đổi cấu hình…",
                                                 "compare": "Đang chụp MJPEG và YUYV, vui lòng giữ yên camera…",
                                                 "save": "Đang lưu ảnh…", "stop": "Đang dừng camera…"}.get(operation, "Đang xử lý…"))
                    try:
                        if operation == "scan":
                            self._scan()
                        elif operation == "apply":
                            self._apply(payload)
                        elif operation == "stop":
                            self._release()
                            self.notice.emit("Đã dừng camera.", "info")
                        elif operation == "save":
                            self._save_frame()
                        elif operation == "compare":
                            self._compare()
                    except Exception as exc:
                        self.last_error = str(exc)
                        self.notice.emit(str(exc), "error")
                    finally:
                        self.busy_changed.emit(False, "")
                        self._emit_state()
                if self.cap is not None and not self.shutdown_event.is_set():
                    self._step()
        finally:
            self._release()
