"""MJPEG decoders and packed YUYV conversion, preserved from the original lab.

Source: ~/mjpeg-yuyv-lab/mjpeg_yuyv_lab.py, Claude session d3fe2976.
Each decoder returns bgr, yuyv, dec (ms), cvt (ms), and jpeg (raw bytes).
"""

import time
import cv2
import numpy as np

try:
    from turbojpeg import TurboJPEG
    _TJ = TurboJPEG()
except Exception:
    _TJ = None

SUPPORTED_FOURCC = ("MJPG", "YUYV")
HAS_CV_YUY2 = hasattr(cv2, "COLOR_BGR2YUV_YUY2")


_LUT_Y = np.clip(np.round(np.arange(256) * 219 / 255 + 16), 0, 255).astype(np.uint8)
_LUT_C = np.clip(np.round((np.arange(256) - 128) * 224 / 255 + 128), 0, 255).astype(np.uint8)


def bgr_to_yuyv_numpy(bgr):
    """Convert BGR to limited-range BT.601 YUYV with NumPy."""
    h, w = bgr.shape[:2]
    f = bgr.astype(np.int32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16

    p = (f[:, 0::2] + f[:, 1::2] + 1) >> 1
    pb, pg, pr = p[..., 0], p[..., 1], p[..., 2]
    u = ((-38 * pr - 74 * pg + 112 * pb + 128) >> 8) + 128
    v = ((112 * pr - 94 * pg - 18 * pb + 128) >> 8) + 128
    out = np.empty((h, w, 2), np.uint8)
    out[..., 0] = y
    out[:, 0::2, 1] = u
    out[:, 1::2, 1] = v
    return out


def bgr_to_yuyv(bgr):
    if HAS_CV_YUY2:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_YUY2)
    return bgr_to_yuyv_numpy(bgr)


def pack_yuyv(y, u, v):
    """Pack three chroma planes into a YUYV array of shape (h, w, 2)."""
    h, w = y.shape
    if u.shape != (h, w // 2):
        u = cv2.resize(u, (w // 2, h), interpolation=cv2.INTER_NEAREST)
        v = cv2.resize(v, (w // 2, h), interpolation=cv2.INTER_NEAREST)
    out = np.empty((h, w, 2), np.uint8)
    out[..., 0] = y
    out[:, 0::2, 1] = u
    out[:, 1::2, 1] = v
    return out


def yuyv_to_bgr(yuyv):
    return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)


def jpeg_bytes(frame):
    """Remove trailing bytes after JPEG EOI in an OpenCV raw capture buffer."""
    buf = frame.reshape(-1)
    raw = buf.tobytes()
    end = raw.rfind(b"\xff\xd9")
    if end > 0:
        return raw[: end + 2], buf[: end + 2]
    return raw, buf


def m_cv_auto(frame):
    t0 = time.perf_counter()
    yuyv = bgr_to_yuyv(frame)
    t1 = time.perf_counter()
    return dict(bgr=frame, yuyv=yuyv, dec=0.0, cvt=(t1 - t0) * 1e3, jpeg=None)


def m_imdecode_cvt(frame):
    raw, buf = jpeg_bytes(frame)
    t0 = time.perf_counter()
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    t1 = time.perf_counter()
    if bgr is None:
        raise RuntimeError("imdecode thất bại (frame JPEG hỏng?)")
    yuyv = bgr_to_yuyv(bgr)
    t2 = time.perf_counter()
    return dict(bgr=bgr, yuyv=yuyv, dec=(t1 - t0) * 1e3, cvt=(t2 - t1) * 1e3, jpeg=raw)


def m_imdecode_numpy(frame):
    raw, buf = jpeg_bytes(frame)
    t0 = time.perf_counter()
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    t1 = time.perf_counter()
    if bgr is None:
        raise RuntimeError("imdecode thất bại (frame JPEG hỏng?)")
    yuyv = bgr_to_yuyv_numpy(bgr)
    t2 = time.perf_counter()
    return dict(bgr=bgr, yuyv=yuyv, dec=(t1 - t0) * 1e3, cvt=(t2 - t1) * 1e3, jpeg=raw)


def m_turbojpeg_yuv(frame):
    """Decode directly to YCbCr planes, map to limited range, and pack YUYV."""
    raw, _ = jpeg_bytes(frame)
    t0 = time.perf_counter()
    planes = _TJ.decode_to_yuv_planes(raw)
    t1 = time.perf_counter()
    y = planes[0]
    if len(planes) >= 3:
        u, v = planes[1], planes[2]
    else:
        u = v = np.full((y.shape[0], y.shape[1] // 2), 128, np.uint8)
    yuyv = pack_yuyv(cv2.LUT(y, _LUT_Y), cv2.LUT(u, _LUT_C), cv2.LUT(v, _LUT_C))
    t2 = time.perf_counter()
    return dict(bgr=None, yuyv=yuyv, dec=(t1 - t0) * 1e3, cvt=(t2 - t1) * 1e3, jpeg=raw)


METHODS = [
    ("cv_auto", False, m_cv_auto, "OpenCV tu decode trong read() + cvtColor BGR2YUV_YUY2"),
    ("imdecode_cvt", True, m_imdecode_cvt, "raw MJPEG -> cv2.imdecode -> cvtColor BGR2YUV_YUY2"),
    ("imdecode_numpy", True, m_imdecode_numpy, "raw MJPEG -> cv2.imdecode -> BGR->YUYV numpy (tu viet)"),
]
if _TJ is not None:
    METHODS.append(("turbojpeg_yuv", True, m_turbojpeg_yuv, "raw MJPEG -> TurboJPEG decode_to_yuv -> pack YUYV"))


def process_native_yuyv(frame, w, h):
    """Validate and reshape an uncompressed native camera YUYV frame."""
    if frame.ndim == 3 and frame.shape[2] == 2:
        yuyv = frame
    elif frame.size == w * h * 2:
        yuyv = frame.reshape(h, w, 2)
    else:
        raise RuntimeError(f"frame YUYV sai kích thước: {frame.shape}")
    return dict(bgr=None, yuyv=yuyv, dec=0.0, cvt=0.0, jpeg=None)


