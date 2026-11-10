"""Regression tests for mode selection, settings, and capture failures."""

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import cv2
import numpy as np
from PyQt5.QtCore import QLibraryInfo
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)
from PyQt5.QtWidgets import QApplication

from color_textile.capture import CameraWorker, FrameDecodeError
from color_textile.comparison import compare_yuyv
from color_textile.controls import CameraControls
from color_textile.decoders import bgr_to_yuyv
from color_textile.devices import CameraDevice, CameraMode, parse_formats
from color_textile.settings import AppSettings, load_settings, save_settings
from color_textile.settings_dialog import SettingsDialog

FORMATS = """
[0]: 'MJPG' (Motion-JPEG)
    Size: Discrete 1920x1080
        Interval: Discrete 0.033s (30.000 fps)
        Interval: Discrete 0.033s (30.000 fps)
        Interval: Discrete 0.067s (15.000 fps)
    Size: Discrete 640x480
        Interval: Discrete 0.033s (30.000 fps)
[1]: 'YUYV' (YUYV 4:2:2)
    Size: Discrete 1920x1080
        Interval: Discrete 0.200s (5.000 fps)
    Size: Discrete 640x480
        Interval: Discrete 0.033s (30.000 fps)
"""
CAMERA = CameraDevice("/dev/video2", "/dev/v4l/by-id/test", "Test camera", parse_formats(FORMATS))


class DeviceTests(unittest.TestCase):
    def test_keep_all_rates_without_duplicates(self):
        modes = parse_formats(FORMATS)
        self.assertEqual(len(modes), 5)
        self.assertEqual({m.fps for m in modes if m.fourcc == "MJPG" and m.width == 1920}, {15.0, 30.0})
        self.assertEqual({m.fps for m in modes if m.fourcc == "YUYV" and m.width == 1920}, {5.0})

    def test_metadata_and_unhandled_formats_are_not_cameras(self):
        self.assertEqual(parse_formats("Type: Metadata Capture"), ())
        self.assertEqual(parse_formats(FORMATS.replace("'MJPG'", "'H264'").replace("'YUYV'", "'NV12'")), ())

    def test_continuous_interval_is_not_reported_as_discrete(self):
        output = "[0]: 'MJPG'\n Size: Discrete 640x480\n Interval: Stepwise 0.01s - 1s (1.000-100.000 fps)\n"
        self.assertEqual(parse_formats(output), ())


class SettingsTests(unittest.TestCase):
    def test_round_trip_and_unknown_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested/settings.json"
            config = AppSettings(camera_id=CAMERA.identity, width=1920, height=1080)
            save_settings(config, path)
            self.assertEqual(load_settings(path), (config, ""))
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_corrupt_or_invalid_settings_fall_back(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            for value in ('{"fps": 0}', '{"width": "bad"}', '[]', '{"fps": NaN}', 'broken'):
                path.write_text(value)
                settings, warning = load_settings(path)
                self.assertEqual(settings, AppSettings())
                self.assertTrue(warning)


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.config = AppSettings(camera_id=CAMERA.identity, width=1920, height=1080)
        self.form = CameraControls()
        self.form.set_devices([CAMERA], self.config)

    def test_requested_resolution_is_preserved(self):
        self.assertEqual(self.form.resolution.currentData(), (1920, 1080))
        self.assertEqual(self.form.read_config(self.config), self.config)

    def test_switch_format_recomputes_fps(self):
        self.form.fourcc.setCurrentIndex(self.form.fourcc.findData("YUYV"))
        self.assertEqual(self.form.fps.currentData(), 5)
        self.assertEqual(self.form.resolution.currentData(), (1920, 1080))
        self.assertFalse(self.form.method.isEnabled())
        self.form.resolution.setCurrentIndex(0)
        self.assertEqual(self.form.fps.currentData(), 30)

    def test_camera_change_uses_supported_fallback(self):
        second = CameraDevice("/dev/video0", "other", "Other", (CameraMode("MJPG", 640, 480, 25),))
        self.form.set_devices([CAMERA, second], self.config)
        self.form.camera.setCurrentIndex(1)
        self.assertEqual(self.form.resolution.currentData(), (640, 480))
        self.assertEqual(self.form.fps.currentData(), 25)

    def test_rescan_keeps_identity_after_renumber(self):
        moved = replace(CAMERA, node="/dev/video4")
        self.form.set_devices([moved], self.config)
        self.assertEqual(self.form.read_config(self.config).camera_id, CAMERA.identity)
        self.assertEqual(self.form.device().node, "/dev/video4")

    def test_no_camera_and_settings_cancel(self):
        self.form.set_devices([], self.config)
        self.assertIsNone(self.form.device())
        self.assertFalse(self.form.resolution.isEnabled())
        dialog = SettingsDialog(self.config, [])
        dialog.quality.setValue(50)
        dialog.reject()
        self.assertEqual(self.config.jpeg_quality, 90)
        self.assertIsNone(dialog.result_config)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.config = AppSettings(camera_id=CAMERA.identity, width=1920, height=1080)
        self.worker = CameraWorker(self.config, False)
        self.worker.cameras = [CAMERA]

    def test_unsupported_mode_does_not_release_running_camera(self):
        cap = object()
        self.worker.cap = cap
        with self.assertRaises(ValueError):
            self.worker._apply(replace(self.config, width=3264, height=2448))
        self.assertIs(self.worker.cap, cap)
        self.assertEqual(self.worker.config, self.config)
        self.worker.cap = None

    def test_failed_switch_restores_previous_config(self):
        self.worker.cap = object()
        requested = replace(self.config, fourcc="YUYV", fps=5)
        opened = []
        def fake_open(config):
            opened.append(config)
            self.worker.cap = None
            if config.fourcc == "YUYV":
                raise RuntimeError("driver failure")
            self.worker.cap = object()
        with patch.object(self.worker, "_open", side_effect=fake_open):
            with self.assertRaisesRegex(RuntimeError, "driver failure"):
                self.worker._apply(requested)
        self.assertEqual(opened, [requested, self.config])
        self.assertIsNotNone(self.worker.cap)
        self.assertEqual(self.worker.config, self.config)
        self.worker.cap = None

    def test_scan_removes_disconnected_camera(self):
        with patch("color_textile.capture.discover_cameras", return_value=([], [])):
            self.worker._scan()
        self.assertIsNone(self.worker.cap)
        self.assertIn("Chưa có camera", self.worker.last_error)

    def test_statistics_retain_no_image_arrays(self):
        class FakeCapture:
            def read(self):
                return True, np.zeros((8, 16, 3), np.uint8)
        self.worker.cap = FakeCapture()
        self.worker.actual = ("MJPG", 16, 8, 30)
        self.worker.config = replace(self.config, width=16, height=8, method="cv_auto")
        self.worker._read()
        self.assertEqual(len(self.worker.stats[0]), 4)
        self.assertTrue(all(isinstance(value, float) for value in self.worker.stats[0]))
        self.assertIsNotNone(self.worker.take_frame())
        self.assertIsNone(self.worker.take_frame())
        self.worker.cap = None

    def test_identical_reference_has_perfect_metrics(self):
        image = bgr_to_yuyv(np.full((32, 48, 3), 100, np.uint8))
        metrics, difference, _ = compare_yuyv(image, image)
        self.assertTrue(np.isinf(metrics["psnr_y"]))
        self.assertAlmostEqual(metrics["ssim_y"], 1)
        self.assertEqual(int(difference.max()), 0)

    def test_transient_decode_failure_keeps_camera_but_repeated_failure_stops(self):
        cap = MagicMock()
        self.worker.cap = cap
        with patch.object(self.worker, "_read", side_effect=FrameDecodeError("bad JPEG")):
            self.worker._step()
            self.assertIs(self.worker.cap, cap)
            cap.release.assert_not_called()
            for _ in range(4):
                self.worker._step()
        self.assertIsNone(self.worker.cap)
        cap.release.assert_called_once()
        self.assertIn("5 frame", self.worker.last_error)


if __name__ == "__main__":
    unittest.main()
