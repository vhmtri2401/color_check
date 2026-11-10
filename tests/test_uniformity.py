"""Unit tests for the 5%-inset ring colour uniformity module (Milestone M2)."""

from pathlib import Path
import tempfile
import time
import unittest
import cv2
import numpy as np

from color_textile.decoders import bgr_to_yuyv
from color_textile.fabric_detect import detect_fabric
from color_textile.uniformity import (
    UniformityParams,
    analyze_uniformity,
    build_master_reference,
    build_ring_layout,
    export_uniformity_session,
    render_ring_overlay,
    sample_ring_frame,
)
from tests.test_fabric_detect import make_carton, place_rotated_rect


def simulate_frames(base_bgr, layout, params, n_frames=30, frame_noise_rgb=0.15, seed=11):
    """Simulate N camera frames from a base BGR image with per-frame sensor noise on segment means."""
    rng = np.random.default_rng(seed)
    rgb0, _, clip0 = sample_ring_frame(bgr_to_yuyv(base_bgr), layout, params.white)
    from color_textile.colorimetry import srgb8_to_lab
    times, rgb_list, lab_list, clip_list = [], [], [], []
    for i in range(n_frames):
        noisy_rgb = np.clip(rgb0 + rng.normal(0.0, frame_noise_rgb, rgb0.shape), 0, 255)
        times.append(i / 30.0)
        rgb_list.append(noisy_rgb)
        lab_list.append(srgb8_to_lab(noisy_rgb, params.white))
        clip_list.append(clip0)
    return analyze_uniformity(layout, times, rgb_list, lab_list, clip_list, params)


class UniformityTests(unittest.TestCase):
    def test_uniform_fabric_passes(self):
        canvas = make_carton(600, 800)
        place_rotated_rect(canvas, (400, 300), (420, 280), 0.0, (140, 158, 230))
        det = detect_fabric(canvas)
        params = UniformityParams(frames=30, threshold_de=0.4)
        layout = build_ring_layout(det, params, reference_frame=canvas)
        self.assertTrue(layout.valid)
        self.assertEqual(layout.segments, 12)
        self.assertTrue(np.all(layout.pixel_counts >= params.min_segment_px))

        res = simulate_frames(canvas, layout, params, n_frames=30, frame_noise_rgb=0.12)
        self.assertEqual(res.verdict, "pass")
        self.assertLess(res.max_pair_de + res.noise_floor_de, 0.4)

    def test_single_segment_shade_defect_fails_and_locates_segment(self):
        canvas = make_carton(600, 800)
        place_rotated_rect(canvas, (400, 300), (420, 280), 0.0, (140, 158, 230))
        det = detect_fabric(canvas)
        params = UniformityParams(frames=30, threshold_de=0.4)
        layout = build_ring_layout(det, params, reference_frame=canvas)
        self.assertTrue(layout.valid)

        # Inject a ~1.2 ΔE76 shade shift into segment 4
        target_seg = 4
        sel = layout.labels == target_seg
        canvas[sel] = np.clip(canvas[sel].astype(np.int32) + np.array([0, -4, 4]), 0, 255).astype(np.uint8)

        res = simulate_frames(canvas, layout, params, n_frames=30, frame_noise_rgb=0.12)
        self.assertEqual(res.verdict, "fail")
        self.assertGreater(res.max_pair_de - res.noise_floor_de, 0.4)
        self.assertEqual(res.worst_segment, target_seg)
        self.assertIn(target_seg, res.worst_pair)

    def test_borderline_defect_with_high_noise_returns_uncertain(self):
        canvas = make_carton(600, 800)
        place_rotated_rect(canvas, (400, 300), (420, 280), 0.0, (140, 158, 230))
        det = detect_fabric(canvas)
        params = UniformityParams(frames=10, threshold_de=0.4)
        layout = build_ring_layout(det, params, reference_frame=canvas)

        # Shift segment 2 by a borderline amount (~0.38 ΔE on channel B) and inject frame noise
        sel = layout.labels == 2
        canvas[sel] = np.clip(canvas[sel].astype(np.int32) + np.array([1, 0, 0]), 0, 255).astype(np.uint8)
        res = simulate_frames(canvas, layout, params, n_frames=10, frame_noise_rgb=0.35)
        self.assertEqual(res.verdict, "uncertain")

    def test_printed_text_in_segment_is_stripped_or_flagged(self):
        canvas = make_carton(600, 800)
        place_rotated_rect(canvas, (400, 300), (420, 280), 0.0, (140, 158, 230))
        det = detect_fabric(canvas)
        params = UniformityParams(frames=30, threshold_de=0.4)
        layout0 = build_ring_layout(det, params, reference_frame=canvas)

        # Paint 35% of segment 6 with bright white ink (heavy text intrusion)
        ys, xs = np.nonzero(layout0.raw_labels == 6)
        n_ink = int(len(xs) * 0.35)
        canvas[ys[:n_ink], xs[:n_ink]] = (250, 250, 250)

        # Paint 8% of segment 9 with bright white ink (minor text intrusion < 20%)
        ys9, xs9 = np.nonzero(layout0.raw_labels == 9)
        n_ink9 = int(len(xs9) * 0.08)
        canvas[ys9[:n_ink9], xs9[:n_ink9]] = (250, 250, 250)

        layout = build_ring_layout(det, params, reference_frame=canvas)
        self.assertTrue(layout.valid)
        self.assertTrue(bool(layout.contaminated[6 - 1]))
        self.assertFalse(bool(layout.contaminated[9 - 1]))
        self.assertGreater(layout.rejected_pct[6 - 1], 20.0)
        self.assertGreater(layout.rejected_pct[9 - 1], 5.0)

        # Because segment 6 is marked contaminated and segment 9's ink pixels were stripped,
        # the remaining uniform fabric still passes!
        res = simulate_frames(canvas, layout, params, n_frames=30, frame_noise_rgb=0.10)
        self.assertEqual(res.verdict, "pass")

    def test_too_small_fabric_for_inset_reports_invalid_layout(self):
        canvas = make_carton(600, 800)
        place_rotated_rect(canvas, (400, 300), (80, 28), 0.0, (140, 158, 230))
        det = detect_fabric(canvas)
        params = UniformityParams(min_inset_px=10, min_width_px=10)
        layout = build_ring_layout(det, params, reference_frame=canvas)
        self.assertFalse(layout.valid)
        self.assertTrue(any("quá nhỏ" in w for w in layout.warnings))

    def test_exports_and_sampling_speed(self):
        canvas = np.full((1944, 2592, 3), (105, 145, 180), dtype=np.uint8)
        cv2.rectangle(canvas, (800, 600), (1600, 1200), (140, 158, 230), thickness=cv2.FILLED)
        det = detect_fabric(canvas)
        params = UniformityParams(frames=10, threshold_de=0.4)
        layout = build_ring_layout(det, params, reference_frame=canvas)
        self.assertTrue(layout.valid)

        yuyv = bgr_to_yuyv(canvas)
        sample_ring_frame(yuyv, layout, params.white)  # warm up
        t0 = time.perf_counter()
        sample_ring_frame(yuyv, layout, params.white)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        self.assertLess(elapsed_ms, 15.0)

        res = simulate_frames(canvas, layout, params, n_frames=10, frame_noise_rgb=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            export_uniformity_session(tmp, canvas, det, layout, res, camera="test", mode="YUYV")
            files = {p.name for p in Path(tmp).iterdir()}
            self.assertTrue({"mask.png", "overlay.png", "overlay_de.png", "frames.csv", "segments.csv", "summary.csv", "session.json"} <= files)

    def test_uniformity_window_and_overlay(self):
        import sys
        from PyQt5.QtWidgets import QApplication
        from color_textile.uniformity_window import UniformityWindow

        _app = QApplication.instance() or QApplication(sys.argv)
        canvas = make_carton(600, 800)
        place_rotated_rect(canvas, (400, 300), (420, 280), 0.0, (140, 158, 230))
        det = detect_fabric(canvas)
        params = UniformityParams(frames=10, threshold_de=0.4)
        layout = build_ring_layout(det, params, reference_frame=canvas)
        res = simulate_frames(canvas, layout, params, n_frames=10, frame_noise_rgb=0.1)

        preview = cv2.resize(canvas, (400, 300), interpolation=cv2.INTER_AREA)
        rendered = render_ring_overlay(preview, det, layout, res)
        self.assertEqual(rendered.shape, preview.shape)

        window = UniformityWindow(params)
        window.update_state(dict(
            running=False,
            progress="Đang lấy mẫu frame 10/10…",
            detection=det,
            layout=layout,
            result=res,
            has_bg=False,
        ))
        with tempfile.TemporaryDirectory() as tmp:
            folder = window.show_result(
                tmp,
                "/dev/video2",
                "Test Camera",
                "YUYV 2592x1944 @ 2",
                dict(frame_bgr=canvas, detection=det, layout=layout, result=res),
            )
            self.assertEqual(window.seg_table.rowCount(), 12)
            self.assertEqual(window.pair_table.rowCount(), 12)
            self.assertTrue((folder / "chart_ring.png").exists())
            self.assertTrue((folder / "segments.csv").exists())
        window.close()

    def test_main_window_opens_uniform_and_buttons_work(self):
        import sys
        from unittest.mock import patch
        from PyQt5.QtWidgets import QApplication
        from color_textile.gui import MainWindow
        from color_textile.settings import AppSettings

        _app = QApplication.instance() or QApplication(sys.argv)
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = Path(tmp) / "settings.json"
            with patch("color_textile.capture.discover_cameras", return_value=([], [])):
                main_win = MainWindow(AppSettings(outdir=tmp), settings_path, autostart=False)
                main_win.uniform_button.click()
                self.assertIsNotNone(main_win.uniform_window)
                self.assertTrue(main_win.uniform_window.isVisible())
                main_win.uniform_window.detect_button.click()
                main_win.uniform_window.start_button.click()
                main_win.uniform_window.bg_button.click()
                main_win.uniform_window.inset_slider.setValue(40)  # 20%
                self.assertAlmostEqual(main_win.uniform_window.inset_pct.value(), 20.0)
                main_win.uniform_window.inset_slider.setValue(84)  # 42% (near center)
                self.assertAlmostEqual(main_win.uniform_window.inset_pct.value(), 42.0)
                main_win.uniform_window.width_slider.setValue(20)  # 10%
                self.assertAlmostEqual(main_win.uniform_window.width_pct.value(), 10.0)
                main_win.uniform_window.close()
                main_win.worker.shutdown()
                main_win.worker.wait(2000)
                main_win.close()

    def test_worker_and_gui_end_to_end_uniformity_flow(self):
        import sys
        from unittest.mock import patch
        from PyQt5.QtWidgets import QApplication
        from color_textile.devices import CameraDevice, CameraMode
        from color_textile.gui import MainWindow
        from color_textile.settings import AppSettings

        _app = QApplication.instance() or QApplication(sys.argv)
        canvas = make_carton(480, 640)
        place_rotated_rect(canvas, (320, 240), (300, 180), 0.0, (140, 158, 230))
        yuyv = bgr_to_yuyv(canvas)
        cam = CameraDevice("/dev/video2", "cam2", "Test Cam 2", (CameraMode("YUYV", 640, 480, 5.0),))

        with tempfile.TemporaryDirectory() as tmp:
            settings_path = Path(tmp) / "settings.json"
            cfg = AppSettings(camera_id="cam2", fourcc="YUYV", width=640, height=480, fps=5.0, outdir=tmp, uniform_frames=5, uniform_lock=False)
            with patch("color_textile.capture.discover_cameras", return_value=([cam], [])):
                main_win = MainWindow(cfg, settings_path, autostart=False)
                main_win.worker.shutdown()
                main_win.worker.wait(2000)

                main_win.devices = [cam]
                main_win.running = True
                main_win.worker.cameras = [cam]
                main_win.worker.cap = object()
                main_win.worker.latest = dict(yuyv=yuyv, bgr=canvas, dec=0.0, cvt=0.0, jpeg=None)

                main_win.uniform_button.click()
                # Process "open" command on worker
                op, payload = main_win.worker.commands.get_nowait()
                self.assertEqual(op, "uniform")
                self.assertEqual(payload["action"], "open")
                main_win.worker._uniform(payload)
                self.assertTrue(main_win.worker.uniform_detection.found)
                self.assertTrue(main_win.worker.uniform_layout.valid)

                # Click "Đo đồng màu"
                main_win.uniform_window.frames.setValue(4)
                while not main_win.worker.commands.empty():
                    _, p = main_win.worker.commands.get_nowait()
                    main_win.worker._uniform(p)

                main_win.uniform_window.start_button.click()
                op, payload = main_win.worker.commands.get_nowait()
                self.assertEqual(payload["action"], "start")
                main_win.worker._uniform(payload)
                self.assertIsNotNone(main_win.worker.uniform_job)

                done_payloads = []
                main_win.worker.uniform_done.connect(done_payloads.append)
                for i in range(4):
                    main_win.worker._sample_uniform(float(i), yuyv)
                self.assertIsNone(main_win.worker.uniform_job)
                self.assertEqual(len(done_payloads), 1)

                main_win._uniform_done(done_payloads[0])
                self.assertEqual(main_win.uniform_window.seg_table.rowCount(), 12)
                main_win.uniform_window.close()
                main_win.close()

    def test_master_fabric_capture_and_comparison(self):
        canvas_master = make_carton(600, 800)
        place_rotated_rect(canvas_master, (400, 300), (420, 280), 0.0, (140, 158, 230))
        det_m = detect_fabric(canvas_master)
        params = UniformityParams(frames=10, threshold_de=0.4)
        self.assertEqual(params.de_formula, "de00")
        layout_m = build_ring_layout(det_m, params, reference_frame=canvas_master)
        res_m = simulate_frames(canvas_master, layout_m, params, n_frames=10, frame_noise_rgb=0.1)
        master_ref = build_master_reference(det_m, layout_m, res_m)

        # 1. Same colour sample -> PASS vs Master
        rgb0, lab0, clip0 = sample_ring_frame(bgr_to_yuyv(canvas_master), layout_m, params.white)
        res_same = analyze_uniformity(
            layout_m, [0.0, 0.1], [rgb0, rgb0], [lab0, lab0], [clip0, clip0],
            params, master=master_ref, detection=det_m,
        )
        self.assertTrue(res_same.has_master)
        self.assertEqual(res_same.verdict, "pass")
        self.assertLess(res_same.overall_master_de, 0.1)

        # 2. Off-shade sample (whole fabric shifted by +6 BGR) -> FAIL vs Master even though internally uniform!
        canvas_off = make_carton(600, 800)
        place_rotated_rect(canvas_off, (400, 300), (420, 280), 0.0, (146, 152, 236))
        det_off = detect_fabric(canvas_off)
        layout_off = build_ring_layout(det_off, params, reference_frame=canvas_off)
        rgb1, lab1, clip1 = sample_ring_frame(bgr_to_yuyv(canvas_off), layout_off, params.white)
        res_off = analyze_uniformity(
            layout_off, [0.0, 0.1], [rgb1, rgb1], [lab1, lab1], [clip1, clip1],
            params, master=master_ref, detection=det_off,
        )
        self.assertTrue(res_off.has_master)
        self.assertEqual(res_off.verdict, "fail")
        self.assertEqual(res_off.verdict_label, "LỆCH SO VỚI VẢI GỐC")
        self.assertGreater(res_off.overall_master_de, 0.4)

        # 3. Same-colour sample with a dirty spot on segment 7 -> dirty segment on new fabric is NOT compared -> PASS!
        canvas_dirty_new = canvas_master.copy()
        canvas_dirty_new[layout_m.labels == 7] = (120, 138, 210)
        rgb2, lab2, clip2 = sample_ring_frame(bgr_to_yuyv(canvas_dirty_new), layout_m, params.white)
        res_dirty_new = analyze_uniformity(
            layout_m, [0.0, 0.1], [rgb2, rgb2], [lab2, lab2], [clip2, clip2],
            params, master=master_ref, detection=det_m,
        )
        self.assertTrue(bool(res_dirty_new.dirty_segments[7 - 1]))
        self.assertTrue(bool(res_dirty_new.contaminated[7 - 1]))
        self.assertEqual(res_dirty_new.verdict, "pass")
        self.assertLess(res_dirty_new.max_master_de, 0.1)

    def test_dirty_segment_is_excluded_and_clean_segments_pass(self):
        canvas = make_carton(600, 800)
        place_rotated_rect(canvas, (400, 300), (420, 280), 0.0, (140, 158, 230))
        det = detect_fabric(canvas)
        params = UniformityParams(frames=20, threshold_de=0.4, filter_dirty_segments=True)
        layout = build_ring_layout(det, params, reference_frame=canvas)

        # Inject localized dirt stains on segments 3 and 9 (~1.8 ΔE00) while 10 segments remain uniform
        for dirty_seg in (3, 9):
            sel = layout.labels == dirty_seg
            canvas[sel] = np.clip(canvas[sel].astype(np.int32) + np.array([-6, -5, 5]), 0, 255).astype(np.uint8)

        res = simulate_frames(canvas, layout, params, n_frames=20, frame_noise_rgb=0.10)
        self.assertEqual(res.verdict, "pass")
        self.assertTrue(bool(res.dirty_segments[3 - 1]))
        self.assertTrue(bool(res.dirty_segments[9 - 1]))
        self.assertTrue(bool(res.contaminated[3 - 1]))
        self.assertTrue(bool(res.contaminated[9 - 1]))
        self.assertEqual(int(np.count_nonzero(~res.contaminated)), 10)
        self.assertLess(res.max_pair_de + res.noise_floor_de, 0.4)

        # Conversely, if half the fabric (6 of 12 segments) has a different shade, it is NOT a localized dirt spot -> FAIL
        canvas_half = make_carton(600, 800)
        place_rotated_rect(canvas_half, (400, 300), (420, 280), 0.0, (140, 158, 230))
        for seg_id in range(1, 7):
            sel = layout.labels == seg_id
            canvas_half[sel] = np.clip(canvas_half[sel].astype(np.int32) + np.array([-6, -5, 5]), 0, 255).astype(np.uint8)
        res_half = simulate_frames(canvas_half, layout, params, n_frames=20, frame_noise_rgb=0.10)
        self.assertEqual(res_half.verdict, "fail")

    def test_timeseries_master_monitoring_rounds_and_export(self):
        import sys
        from unittest.mock import patch
        from PyQt5.QtWidgets import QApplication
        from color_textile.devices import CameraDevice, CameraMode
        from color_textile.gui import MainWindow
        from color_textile.settings import AppSettings

        _app = QApplication.instance() or QApplication(sys.argv)
        canvas_t0 = make_carton(480, 640)
        place_rotated_rect(canvas_t0, (320, 240), (300, 180), 0.0, (140, 158, 230))
        det0 = detect_fabric(canvas_t0)
        layout0 = build_ring_layout(det0, UniformityParams(), reference_frame=canvas_t0)
        # Stain segment 5 at t=0
        canvas_t0[layout0.labels == 5] = (128, 146, 218)
        yuyv_t0 = bgr_to_yuyv(canvas_t0)

        cam = CameraDevice("/dev/video2", "cam2", "Test Cam 2", (CameraMode("YUYV", 640, 480, 5.0),))
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = Path(tmp) / "settings.json"
            cfg = AppSettings(
                camera_id="cam2", fourcc="YUYV", width=640, height=480, fps=5.0,
                outdir=tmp, uniform_frames=4, uniform_lock=False,
                uniform_interval_s=120.0, uniform_duration_min=4.0, uniform_filter_dirty=True,
            )
            with patch("color_textile.capture.discover_cameras", return_value=([cam], [])):
                main_win = MainWindow(cfg, settings_path, autostart=False)
                main_win.worker.shutdown()
                main_win.worker.wait(2000)

                main_win.devices = [cam]
                main_win.running = True
                main_win.worker.cameras = [cam]
                main_win.worker.cap = object()
                main_win.worker.latest = dict(yuyv=yuyv_t0, bgr=canvas_t0, dec=0.0, cvt=0.0, jpeg=None)

                main_win.uniform_button.click()
                while not main_win.worker.commands.empty():
                    _, p = main_win.worker.commands.get_nowait()
                    main_win.worker._uniform(p)

                done_payloads = []
                main_win.worker.uniform_done.connect(done_payloads.append)

                # Start Time-Series (every 120s for 4 minutes -> rounds at t=0s, 120s, 240s)
                main_win.uniform_window.ts_button.click()
                _, payload = main_win.worker.commands.get_nowait()
                self.assertEqual(payload["action"], "start_timeseries")
                main_win.worker._uniform(payload)

                # Round 0 at t=0..3s
                for i in range(4):
                    main_win.worker._sample_uniform(float(i), yuyv_t0)
                self.assertEqual(len(done_payloads), 1)
                p0 = done_payloads[-1]
                self.assertTrue(p0["is_timeseries"])
                self.assertEqual(p0["round_index"], 0)
                self.assertTrue(p0["ts_active"])
                self.assertEqual(p0["result"].verdict, "pass")
                self.assertTrue(bool(p0["result"].dirty_segments[5 - 1]))
                self.assertAlmostEqual(p0["result"].overall_master_de, 0.0, places=4)
                main_win._uniform_done(p0)

                # At t=60s (< 120s), no new round should trigger
                main_win.worker._sample_uniform(60.0, yuyv_t0)
                self.assertEqual(len(done_payloads), 1)
                self.assertIsNone(main_win.worker.uniform_job)

                # Round 1 at t=120s (2 minutes): segment 10 gets a new dirt stain, rest of fabric unchanged
                canvas_t1 = canvas_t0.copy()
                canvas_t1[layout0.labels == 10] = (125, 142, 215)
                yuyv_t1 = bgr_to_yuyv(canvas_t1)
                for i in range(4):
                    main_win.worker._sample_uniform(120.0 + i, yuyv_t1)
                self.assertEqual(len(done_payloads), 2)
                p1 = done_payloads[-1]
                self.assertEqual(p1["round_index"], 1)
                self.assertTrue(p1["ts_active"])
                self.assertEqual(p1["result"].verdict, "pass")
                self.assertTrue(bool(p1["result"].dirty_segments[5 - 1]))
                self.assertTrue(bool(p1["result"].dirty_segments[10 - 1]))
                self.assertLess(p1["result"].overall_master_de, 0.1)
                main_win._uniform_done(p1)

                # Round 2 at t=240s (4 minutes): whole clean fabric drifts in colour (+7 BGR) -> FAIL
                canvas_t2 = make_carton(480, 640)
                place_rotated_rect(canvas_t2, (320, 240), (300, 180), 0.0, (147, 151, 237))
                yuyv_t2 = bgr_to_yuyv(canvas_t2)
                for i in range(4):
                    main_win.worker._sample_uniform(240.0 + i, yuyv_t2)
                self.assertEqual(len(done_payloads), 3)
                p2 = done_payloads[-1]
                self.assertEqual(p2["round_index"], 2)
                self.assertFalse(p2["ts_active"])  # duration_min=4.0 completed!
                self.assertEqual(p2["result"].verdict, "fail")
                self.assertGreater(p2["result"].overall_master_de, 0.4)
                main_win._uniform_done(p2)

                ts_folder = main_win.uniform_window.folder
                self.assertTrue((ts_folder / "timeseries_rounds.csv").exists())
                self.assertTrue((ts_folder / "timeseries_segments.csv").exists())
                self.assertTrue((ts_folder / "timeseries.json").exists())
                self.assertTrue((ts_folder / "chart_timeseries.png").exists())
                self.assertTrue((ts_folder / "round_000" / "summary.csv").exists())
                self.assertTrue((ts_folder / "round_001" / "summary.csv").exists())
                self.assertTrue((ts_folder / "round_002" / "summary.csv").exists())
                self.assertEqual(main_win.uniform_window.ts_table.rowCount(), 3)

                main_win.uniform_window.close()
                main_win.close()


if __name__ == "__main__":
    unittest.main()
