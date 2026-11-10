"""Tests for colour maths, ROI sampling, session scheduling and garbage-frame detection."""

import csv
import os
import re
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np
from PyQt5.QtCore import QLibraryInfo
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)
from PyQt5.QtWidgets import QApplication

from color_textile.camera_controls import AutoLockMatcher, lock_controls, parse_controls
from color_textile.colorimetry import delta_e, delta_e2000, srgb8_to_lab
from color_textile.decoders import bgr_to_yuyv
from color_textile.measurement import (MeasurementSession, Stabilizer, method_comparison, SessionParams, export_round, export_summary, roi_bounds,
                                       sample_points, session_summary)

# Sharma, Wu & Dalal (2005) CIEDE2000 test data, pairs 1, 7, 17, 25.
SHARMA = [((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425),
          ((50, 0, 0), (50, -1, 2), 2.3669),
          ((50, 2.5, 0), (73, 25, -18), 27.1492),
          ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644)]


def run_session(params, rgb_noise=0.3, spike=None, fps=30.0):
    """Feed a synthetic flat patch; `spike` = (round, frame, rgb offset) injects one garbage frame."""
    yuyv = bgr_to_yuyv(np.full((120, 160, 3), (40, 120, 200), np.uint8))
    points = [(0.3, 0.3), (0.7, 0.7)]
    rgb, _, clip = sample_points(yuyv, points, params.roi, params.white)
    session = MeasurementSession(points, params, 0.0)
    rng = np.random.default_rng(1)
    rounds, t = [], 0.0
    while session.active:
        noisy = rgb + rng.normal(0, rgb_noise, rgb.shape)
        current = session.current
        if spike and len(rounds) == spike[0] and current and len(current["t"]) == spike[1]:
            noisy = noisy + spike[2]
        result = session.add(t, noisy, srgb8_to_lab(noisy, params.white), clip)
        if result is not None:
            rounds.append(result)
        t += 1 / fps
    return session, rounds


class ColorimetryTests(unittest.TestCase):
    def test_ciede2000_reference_pairs(self):
        for lab1, lab2, expected in SHARMA:
            self.assertAlmostEqual(float(delta_e2000(lab1, lab2)), expected, places=4)

    def test_d50_lab_matches_icc_reference(self):
        np.testing.assert_allclose(srgb8_to_lab([255, 255, 255], "D50"), [100, 0, 0], atol=0.01)
        np.testing.assert_allclose(srgb8_to_lab([255, 0, 0], "D50"), [54.29, 80.80, 69.89], atol=0.05)

    def test_delta_e76_is_euclidean(self):
        self.assertAlmostEqual(float(delta_e([50, 0, 0], [53, 4, 0], "de76")), 5.0)
        self.assertAlmostEqual(float(delta_e(SHARMA[0][0], SHARMA[0][1], "de00")), 2.0425, places=4)

    def test_srgb_white_and_known_colour(self):
        np.testing.assert_allclose(srgb8_to_lab([255, 255, 255]), [100, 0, 0], atol=0.01)
        np.testing.assert_allclose(srgb8_to_lab([255, 0, 0]), [53.24, 80.09, 67.20], atol=0.02)


class SamplingTests(unittest.TestCase):
    def test_per_point_roi_sizes(self):
        image = np.full((200, 200, 3), (40, 120, 200), np.uint8)
        image[95:105, 95:105] = (200, 40, 40)  # small different patch in the centre
        rgb, _, _ = sample_points(bgr_to_yuyv(image), [(0.5, 0.5)] * 2, [10, 100])
        self.assertGreater(rgb[0, 2] - rgb[1, 2], 50)  # small ROI sees the patch, large ROI dilutes it
        params = SessionParams(rois=(31, 101))
        self.assertEqual(params.point_rois(2), [31, 101])
        self.assertEqual(params.labels(2), ["Điểm 1 · ROI 31", "Điểm 2 · ROI 101"])
        self.assertEqual(SessionParams(roi=50).point_rois(3), [50, 50, 50])

    def test_roi_is_even_aligned_and_clamped(self):
        for point in ((0, 0), (1, 1), (0.5, 0.5), (0.013, 0.98)):
            x0, y0, x1, y1 = roi_bounds(161 - 1, 90, point, 31)
            self.assertEqual(x0 % 2, 0)
            self.assertEqual((x1 - x0) % 2, 0)
            self.assertTrue(0 <= x0 < x1 <= 160 and 0 <= y0 < y1 <= 90)

    def test_flat_patch_values(self):
        yuyv = bgr_to_yuyv(np.full((60, 80, 3), (40, 120, 200), np.uint8))
        rgb, lab, clip = sample_points(yuyv, [(0.5, 0.5), (0, 0), (1, 1)], 16)
        np.testing.assert_allclose(rgb, [[200, 120, 40]] * 3, atol=1.5)
        np.testing.assert_allclose(lab[0], srgb8_to_lab(rgb[0]), atol=1e-9)
        self.assertTrue(np.all(clip == 0))


class SessionTests(unittest.TestCase):
    def test_rounds_follow_schedule(self):
        params = SessionParams(frames=10, interval_s=2, duration_min=0.1)  # rounds at 0, 2, 4, 6 s
        session, rounds = run_session(params)
        self.assertEqual(params.rounds_total, 4)
        self.assertEqual(session.state, "done")
        np.testing.assert_allclose([r.t_min * 60 for r in rounds], [0, 2, 4, 6], atol=0.05)
        self.assertTrue(all(r.frame_count == 10 for r in rounds))
        minutes, rgb, lab = session.monitor()
        self.assertGreater(len(minutes), 4)  # continuous 1 s series also between rounds
        self.assertEqual(rgb.shape[1:], (2, 3))

    def test_single_garbage_frame_is_flagged_and_excluded(self):
        params = SessionParams(frames=30, interval_s=1, duration_min=2 / 60)
        _, rounds = run_session(params, spike=(1, 12, np.array([15.0, 0, 0])))
        self.assertEqual([list(r.garbage_frames()) for r in rounds], [[], [12], []])
        clean = rounds[1].mean_lab[0]
        self.assertLess(float(delta_e2000(clean, rounds[0].mean_lab[0])), 0.2)

    def test_normal_noise_has_no_garbage(self):
        params = SessionParams(frames=60, interval_s=1, duration_min=3 / 60)
        _, rounds = run_session(params, rgb_noise=0.8)
        # MAD estimated from 60 frames is itself noisy; z > 3.5 should still flag only a small fraction.
        flagged = sum(len(r.garbage_frames()) for r in rounds)
        self.assertLess(flagged / sum(r.frame_count for r in rounds), 0.03)

    def test_stop_keeps_finished_rounds(self):
        params = SessionParams(frames=5, interval_s=1, duration_min=1)
        session = MeasurementSession([(0.5, 0.5)], params, 0.0)
        sample = (np.ones((1, 3)), np.ones((1, 3)), np.zeros(1))
        for i in range(5):
            session.add(i / 30, *sample)
        session.stop()
        self.assertEqual((session.state, len(session.rounds)), ("stopped", 1))
        self.assertIsNone(session.add(10, *sample))

    def test_exports_and_summary(self):
        params = SessionParams(frames=10, interval_s=1, duration_min=2 / 60)
        _, rounds = run_session(params)
        with tempfile.TemporaryDirectory() as folder:
            for r in rounds:
                export_round(folder, r)
            rows, pairs = export_summary(folder, rounds)
            with open(Path(folder) / "frames.csv", newline="") as handle:
                self.assertEqual(len(list(csv.reader(handle))), 1 + 3 * 10 * 2)
            with open(Path(folder) / "rounds.csv", newline="") as handle:
                self.assertEqual(len(list(csv.reader(handle))), 1 + 3 * 2)
            self.assertTrue((Path(folder) / "summary.csv").exists())
        self.assertEqual(len(rows), 2)
        self.assertLess(pairs["1-2"], 0.1)  # same patch at both points
        self.assertLess(session_summary(rounds)[0][0]["drift_de_max"], 0.3)  # ΔE76 of 10-frame means

    def test_mad_mean_removes_garbage_bias_but_mean_does_not(self):
        params = SessionParams(frames=30, interval_s=1, duration_min=1 / 60)
        _, rounds = run_session(params, spike=(1, 4, np.array([40.0, 40.0, 40.0])))
        shift = {m["method"]: m["shift_from_mean_de"] for m in method_comparison(rounds) if m["point"] == 1}
        reference = rounds[0].aggregate_lab["mad_mean"][0]
        biased = float(delta_e2000(rounds[1].aggregate_lab["mean"][0], reference))
        clean = float(delta_e2000(rounds[1].aggregate_lab["mad_mean"][0], reference))
        self.assertGreater(biased, 3 * clean)
        self.assertGreater(shift["MAD + Mean"], 0)
        self.assertEqual(shift["Mean"], 0)


class WindowTests(unittest.TestCase):
    def test_window_draws_rounds_and_time_series(self):
        from color_textile.measure_window import MeasurementWindow
        app = QApplication.instance() or QApplication([])
        params = SessionParams(frames=20, interval_s=1, duration_min=2 / 60)
        session, rounds = run_session(params, spike=(1, 5, np.array([15.0, 0, 0])))
        window = MeasurementWindow(params)
        with tempfile.TemporaryDirectory() as folder:
            window.begin_session(folder, session.points, params, "test", "MJPG")
            window.update_live(dict(points=session.points, params=params, live=None,
                                    session=session.progress(3.0), monitor=session.monitor()))
            for r in rounds:
                window.add_round(r)
            names = sorted(p.name for p in Path(folder).iterdir())
        self.assertIn("timeseries_rgb.png", names)
        self.assertIn("monitor.csv", names)
        self.assertEqual(window.round_table.rowCount(), 3 * 2)
        self.assertIn("1/", window.summary_table.item(0, 9).text())
        window.close()
        app.processEvents()


CONTROLS = """
User Controls

        white_balance_automatic 0x0098090c (bool)   : default=1 value=1
                           gain 0x00980913 (int)    : min=0 max=128 step=1 default=64 value=64
      white_balance_temperature 0x0098091a (int)    : min=2800 max=6500 step=10 default=4600 value=4600 flags=inactive

Camera Controls

                  auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3 (Aperture Priority Mode)
         exposure_time_absolute 0x009a0902 (int)    : min=50 max=10000 step=1 default=166 value=300 flags=inactive
"""


class CameraControlTests(unittest.TestCase):
    def test_parse_v4l2_controls(self):
        controls = parse_controls(CONTROLS)
        self.assertEqual(controls["auto_exposure"]["value"], 3)
        self.assertEqual(controls["exposure_time_absolute"]["value"], 300)
        self.assertTrue(controls["white_balance_temperature"]["inactive"])
        self.assertEqual(controls["gain"]["max"], 128)

    def test_lock_turns_off_auto_and_keeps_values(self):
        state = {k: v["value"] for k, v in parse_controls(CONTROLS).items()}
        def fake_run(cmd, **_kwargs):
            if "-c" in cmd:
                name, value = cmd[-1].split("=")
                state[name] = int(value)
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            text = CONTROLS
            for name, value in state.items():
                text = re.sub(rf"({name} .*?value=)-?\d+", rf"\g<1>{value}", text)
            return SimpleNamespace(returncode=0, stdout=text, stderr="")
        with patch("color_textile.camera_controls.subprocess.run", side_effect=fake_run):
            result, warnings = lock_controls("/dev/video9")
        self.assertEqual(warnings, [])
        self.assertEqual(result["locked"]["auto_exposure"], 1)
        self.assertEqual(result["locked"]["white_balance_automatic"], 0)
        self.assertEqual(result["locked"]["exposure_time_absolute"], 300)
        self.assertEqual(result["before"]["auto_exposure"], 3)


class StabilizerTests(unittest.TestCase):
    def test_waits_for_settled_colour(self):
        stab = Stabilizer(3, 0.3, 60, 0.0)
        outcome, t = None, 0.0
        while outcome is None:
            settling = 5 * np.exp(-t / 2)  # automatic exposure converging
            outcome = stab.add(t, np.array([[50 + settling, 0, 0]]))
            t += 1 / 30
        self.assertEqual(outcome, "stable")
        self.assertGreater(t, 4)
        self.assertLessEqual(stab.spread, 0.3)

    def test_times_out_when_never_stable(self):
        stab = Stabilizer(3, 0.3, 5, 0.0)
        outcome, t = None, 0.0
        while outcome is None:
            outcome = stab.add(t, np.array([[50 + 3 * np.sin(t), 0, 0]]))
            t += 1 / 30
        self.assertEqual(outcome, "timeout")
        self.assertAlmostEqual(t, 5, delta=0.1)


class AutoLockMatcherTests(unittest.TestCase):
    def test_finds_exposure_and_white_balance_of_auto_image(self):
        before = parse_controls(CONTROLS)
        camera = dict(exposure_time_absolute=300, white_balance_temperature=4600)

        def image():
            light = 20 * np.log(camera["exposure_time_absolute"])  # L* rises with log exposure
            return np.array([[light, 2.0, (camera["white_balance_temperature"] - 5200) / 100]])

        def fake_set(_node, values):
            camera.update(values)
            return []

        target = np.array([[20 * np.log(1234), 2.0, (3900 - 5200) / 100]])
        with patch("color_textile.camera_controls.set_controls", side_effect=fake_set):
            matcher = AutoLockMatcher("/dev/video9", target, before, 0.0)
            t = 0.0
            while not matcher.add(t, image()):
                t += 1 / 30
        self.assertLess(abs(camera["exposure_time_absolute"] - 1234) / 1234, 0.03)
        self.assertLessEqual(abs(camera["white_balance_temperature"] - 3900), 20)
        self.assertLess(matcher.result["white_balance_temperature"]["residual_de00"], 0.5)
        self.assertFalse(matcher.exposure_failed)

    def test_unreachable_exposure_keeps_auto_exposure(self):
        before = parse_controls(CONTROLS)
        camera = dict(exposure_time_absolute=300, white_balance_temperature=4600, auto_exposure=1)

        def image():
            light = 90 if camera["auto_exposure"] == 3 else min(12, camera["exposure_time_absolute"] / 500)
            return np.array([[light, 2.0, (camera["white_balance_temperature"] - 5200) / 100]])

        def fake_set(_node, values):
            camera.update(values)
            return []

        with patch("color_textile.camera_controls.set_controls", side_effect=fake_set):
            matcher = AutoLockMatcher("/dev/video9", np.array([[90, 2.0, -5.0]]), before, 0.0)
            t = 0.0
            while not matcher.add(t, image()):
                t += 1 / 30
        self.assertTrue(matcher.exposure_failed)
        self.assertEqual((camera["auto_exposure"], camera["exposure_time_absolute"]), (3, 300))
        self.assertLessEqual(abs(camera["white_balance_temperature"] - 4700), 20)


if __name__ == "__main__":
    unittest.main()
