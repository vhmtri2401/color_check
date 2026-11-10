"""Unit tests for lightweight classical-CV fabric detection (Milestone M1)."""

import unittest
import cv2
import numpy as np

from color_textile.decoders import bgr_to_yuyv
from color_textile.fabric_detect import (
    build_background_model,
    detect_fabric,
    mask_iou,
)


def make_carton(h=600, w=800, seed=7):
    """Synthetic brown carton background with gentle shading and grain."""
    rng = np.random.default_rng(seed)
    # Typical brown carton in BGR: around B=105, G=145, R=180
    base = np.full((h, w, 3), (105, 145, 180), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shading = 4.0 * np.cos((xx - w / 2) / w * np.pi) * np.cos((yy - h / 2) / h * np.pi)
    noise = rng.normal(0.0, 1.8, (h, w, 3)).astype(np.float32)
    return np.clip(base + shading[..., None] + noise, 0, 255).astype(np.uint8)


def place_rotated_rect(canvas, center, size, angle_deg, bgr_color, shadow_drop=0):
    """Draw a rotated fabric patch (and optional edge drop shadow) onto canvas; return ground-truth mask."""
    h, w = canvas.shape[:2]
    rect = (center, size, angle_deg)
    box = np.int32(np.round(cv2.boxPoints(rect)))
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(gt_mask, [box], -1, 255, thickness=cv2.FILLED)

    if shadow_drop > 0:
        shadow_rect = ((center[0] + 6, center[1] + 6), (size[0] + 10, size[1] + 10), angle_deg)
        sbox = np.int32(np.round(cv2.boxPoints(shadow_rect)))
        smask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(smask, [sbox], -1, 255, thickness=cv2.FILLED)
        shadow_only = (smask > 0) & (gt_mask == 0)
        canvas[shadow_only] = np.clip(canvas[shadow_only].astype(np.int32) - shadow_drop, 0, 255).astype(np.uint8)

    rng = np.random.default_rng(42)
    tex = rng.normal(0.0, 1.2, canvas.shape).astype(np.float32)
    patch = np.clip(np.array(bgr_color, dtype=np.float32) + tex, 0, 255).astype(np.uint8)
    canvas[gt_mask > 0] = patch[gt_mask > 0]
    return gt_mask


class FabricDetectTests(unittest.TestCase):
    def test_pink_label_flat_strip_with_centre_text(self):
        """450x60 flat pink label with bright white text in the centre."""
        canvas = make_carton(600, 800)
        gt = place_rotated_rect(canvas, (400, 300), (450, 72), 0.0, (140, 158, 230))
        # Draw bright white text "ciele" in the centre of the fabric
        cv2.putText(canvas, "ciele", (315, 312), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (250, 250, 250), 4)

        det = detect_fabric(bgr_to_yuyv(canvas))
        self.assertTrue(det.found)
        self.assertGreaterEqual(mask_iou(det.mask, gt), 0.95)
        # Ensure the centre text pixels are inside the filled mask
        self.assertEqual(int(det.mask[300, 400]), 255)
        self.assertFalse(det.touches_border)

    def test_dark_navy_fabric_rotated_with_shadow_and_text(self):
        """Dark navy fabric (RGB ~ 23, 32, 43) rotated 20 deg with soft edge shadow and logo."""
        canvas = make_carton(600, 800)
        gt = place_rotated_rect(canvas, (400, 300), (360, 240), 20.0, (43, 32, 23), shadow_drop=16)
        # Draw both white text and carton-colored print in the centre to test hole filling
        cv2.putText(canvas, "TEXTILE", (310, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (240, 240, 240), 3)
        cv2.putText(canvas, "CARTON", (320, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (105, 145, 180), 4)

        det = detect_fabric(canvas)
        self.assertTrue(det.found)
        self.assertGreaterEqual(mask_iou(det.mask, gt), 0.95)
        self.assertGreater(det.holes_filled_px, 0)
        self.assertEqual(det.warnings, [])

    def test_reference_background_model(self):
        """Explicit captured background model handles non-uniform tray shading."""
        tray = make_carton(600, 800, seed=99)
        bg_model = build_background_model(tray)
        sample = tray.copy()
        gt = place_rotated_rect(sample, (400, 300), (300, 200), -12.0, (90, 175, 160))

        det = detect_fabric(sample, background=bg_model)
        self.assertTrue(det.found)
        self.assertGreaterEqual(mask_iou(det.mask, gt), 0.96)

    def test_border_touch_and_small_fabric_warnings(self):
        canvas = make_carton(600, 800)
        place_rotated_rect(canvas, (100, 300), (220, 180), 0.0, (40, 50, 190))
        det_touch = detect_fabric(canvas)
        self.assertTrue(det_touch.found)
        self.assertTrue(det_touch.touches_border)
        self.assertTrue(any("chạm mép" in w for w in det_touch.warnings))

        canvas2 = make_carton(600, 800)
        place_rotated_rect(canvas2, (400, 300), (50, 24), 0.0, (40, 50, 190))
        det_small = detect_fabric(canvas2)
        self.assertTrue(det_small.found)
        self.assertTrue(any("quá nhỏ" in w for w in det_small.warnings))

    def test_empty_carton_returns_not_found(self):
        canvas = make_carton(600, 800)
        det = detect_fabric(canvas)
        self.assertFalse(det.found)
        self.assertTrue(det.warnings)

    def test_carton_box_with_dark_walls_and_radial_vignette(self):
        """Simulate the 3D carton box camera view with dark left/right walls and LED radial dome."""
        h, w = 600, 800
        canvas = make_carton(h, w)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dome = 16.0 * np.exp(-((xx - w * 0.5) ** 2 + (yy - h * 0.5) ** 2) / (2.0 * (w * 0.28) ** 2))
        canvas = np.clip(canvas.astype(np.float32) + dome[..., None], 0, 255).astype(np.uint8)
        # Darken left (<16%) and right (>82%) vertical box walls
        canvas[:, :int(w * 0.16)] = np.clip(canvas[:, :int(w * 0.16)].astype(np.int32) - 42, 0, 255).astype(np.uint8)
        canvas[:, int(w * 0.82):] = np.clip(canvas[:, int(w * 0.82):].astype(np.int32) - 42, 0, 255).astype(np.uint8)

        # Empty box should return found=False
        det_empty = detect_fabric(canvas)
        self.assertFalse(det_empty.found)

        # Place a small dark navy fabric in the middle of the tray floor
        place_rotated_rect(canvas, (410, 300), (160, 54), 0.0, (42, 26, 22))
        det = detect_fabric(canvas)
        self.assertTrue(det.found)
        self.assertAlmostEqual(det.long_side, 160, delta=12)
        self.assertAlmostEqual(det.short_side, 54, delta=10)
        self.assertGreater(det.solidity, 0.90)

    def test_full_res_speed_under_30ms(self):
        """Detection on a 2592x1944 image should run well under 30 ms."""
        canvas = np.full((1944, 2592, 3), (105, 145, 180), dtype=np.uint8)
        cv2.rectangle(canvas, (800, 600), (1700, 1300), (43, 32, 23), thickness=cv2.FILLED)
        # Warm-up call
        detect_fabric(canvas)
        det = detect_fabric(canvas)
        self.assertTrue(det.found)
        self.assertLess(det.elapsed_ms, 30.0)


if __name__ == "__main__":
    unittest.main()
