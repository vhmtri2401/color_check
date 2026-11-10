"""sRGB, CIE L*a*b* (D65 or Bradford-adapted D50), CIE76 and CIEDE2000 helpers.

Camera output is treated as sRGB after the BT.601 YUYV -> BGR conversion. Without
a colour chart calibration the values are relative (good for stability and
point-to-point comparison), not absolute colorimetry.
"""

import numpy as np

# sRGB (D65) linear RGB -> XYZ, IEC 61966-2-1
_RGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
_WHITE = {"D65": np.array([0.95047, 1.00000, 1.08883]), "D50": np.array([0.96422, 1.00000, 0.82521])}
# Bradford chromatic adaptation D65 -> D50 (Lindbloom), as used for ICC/D50 Lab (e.g. Nix default)
_BRADFORD_D65_TO_D50 = np.array([[1.0478112, 0.0228866, -0.0501270],
                                 [0.0295424, 0.9904844, -0.0170491],
                                 [-0.0092345, 0.0150436, 0.7521316]])
FORMULAS = {"de76": "ΔE76", "de00": "ΔE00"}


def srgb_to_linear(values):
    """Gamma-encoded sRGB in 0..1 -> linear light."""
    v = np.asarray(values, dtype=np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(values):
    """Linear light -> gamma-encoded sRGB in 0..1."""
    v = np.clip(np.asarray(values, dtype=np.float64), 0, 1)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)


def linear_rgb_to_lab(rgb, white="D65"):
    """Linear RGB (..., 3) in 0..1 -> L*a*b* (..., 3) for a D65 or D50 (Bradford-adapted) white."""
    xyz = np.asarray(rgb, dtype=np.float64) @ _RGB_TO_XYZ.T
    if white == "D50":
        xyz = xyz @ _BRADFORD_D65_TO_D50.T
    xyz = xyz / _WHITE[white]
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def srgb8_to_lab(rgb, white="D65"):
    """8-bit sRGB (..., 3) -> L*a*b*."""
    return linear_rgb_to_lab(srgb_to_linear(np.asarray(rgb, dtype=np.float64) / 255), white)


def delta_e76(lab1, lab2):
    """CIE76: Euclidean distance in L*a*b* (what Nix and many simple meters report as ΔE)."""
    return np.linalg.norm(np.asarray(lab1, dtype=np.float64) - np.asarray(lab2, dtype=np.float64), axis=-1)


def delta_e(lab1, lab2, formula="de00"):
    return delta_e76(lab1, lab2) if formula == "de76" else delta_e2000(lab1, lab2)


def delta_e2000(lab1, lab2, kl=1.0, kc=1.0, kh=1.0):
    """CIEDE2000 colour difference (Sharma et al. 2005), broadcast over leading axes."""
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    l1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    l2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    c_bar = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(c_bar ** 7 / (c_bar ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    h1p = np.where(c1p == 0, 0.0, h1p)
    h2p = np.where(c2p == 0, 0.0, h2p)

    dlp = l2 - l1
    dcp = c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(c1p * c2p == 0, 0.0, dh)
    dhp = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2))

    lbp = (l1 + l2) / 2
    cbp = (c1p + c2p) / 2
    hsum = h1p + h2p
    hbp = np.where(np.abs(h1p - h2p) <= 180, hsum / 2, np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2))
    hbp = np.where(c1p * c2p == 0, hsum, hbp)
    t = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    d_theta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    rc = 2 * np.sqrt(cbp ** 7 / (cbp ** 7 + 25.0 ** 7))
    sl = 1 + 0.015 * (lbp - 50) ** 2 / np.sqrt(20 + (lbp - 50) ** 2)
    sc = 1 + 0.045 * cbp
    sh = 1 + 0.015 * cbp * t
    rt = -np.sin(np.radians(2 * d_theta)) * rc
    return np.sqrt((dlp / (kl * sl)) ** 2 + (dcp / (kc * sc)) ** 2 + (dhp / (kh * sh)) ** 2
                   + rt * (dcp / (kc * sc)) * (dhp / (kh * sh)))
