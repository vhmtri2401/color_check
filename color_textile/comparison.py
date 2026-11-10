"""Capture comparison metrics and exports, retained from the original lab.

Source: ~/mjpeg-yuyv-lab/mjpeg_yuyv_lab.py, Claude session d3fe2976.
Camera frames are sequential captures; roundtrip compares the same reference.
"""

import os
import time
from uuid import uuid4
import cv2
import numpy as np
from .decoders import METHODS, bgr_to_yuyv, yuyv_to_bgr


def draw_hud(image, lines):
    for i, line in enumerate(lines):
        cv2.putText(image, line, (8, 24 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)


SUBSAMP_NAMES = {(2, 1): "4:2:2", (2, 2): "4:2:0", (1, 1): "4:4:4", (1, 2): "4:4:0", (4, 1): "4:1:1"}
_SAMP_FLAGS = {"4:2:2": "IMWRITE_JPEG_SAMPLING_FACTOR_422", "4:2:0": "IMWRITE_JPEG_SAMPLING_FACTOR_420",
               "4:4:4": "IMWRITE_JPEG_SAMPLING_FACTOR_444", "4:4:0": "IMWRITE_JPEG_SAMPLING_FACTOR_440",
               "4:1:1": "IMWRITE_JPEG_SAMPLING_FACTOR_411"}


def jpeg_subsampling(buf):
    """Read the JPEG SOF marker to identify chroma subsampling."""
    i, n = 2, len(buf)
    while i + 14 < n:
        if buf[i] != 0xFF:
            i += 1
            continue
        mk = buf[i + 1]
        if mk in (0xC0, 0xC1, 0xC2):
            if buf[i + 9] == 1:
                return "gray"
            fy, fc = buf[i + 11], buf[i + 14]
            key = ((fy >> 4) // max(fc >> 4, 1), (fy & 15) // max(fc & 15, 1))
            return SUBSAMP_NAMES.get(key, "?")
        if mk == 0xFF:
            i += 1
        elif mk == 0x01 or 0xD0 <= mk <= 0xD9:
            i += 2
        else:
            i += 2 + ((buf[i + 2] << 8) | buf[i + 3])
    return "?"


def encode_jpeg(bgr, quality, subsamp):
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    flag = getattr(cv2, _SAMP_FLAGS.get(subsamp, ""), None)
    if flag is not None and hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR"):
        params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, flag]
    ok, buf = cv2.imencode(".jpg", bgr, params)
    if not ok:
        raise RuntimeError("imencode thất bại")
    return buf.tobytes()


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim_map(a, b):
    """Compute SSIM using an 11x11 Gaussian window with sigma 1.5."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu1, mu2 = blur(a), blur(b)
    s1 = blur(a * a) - mu1 * mu1
    s2 = blur(b * b) - mu2 * mu2
    s12 = blur(a * b) - mu1 * mu2
    return ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))


def split_yuyv(yuyv):
    return (np.ascontiguousarray(yuyv[..., 0]),
            np.ascontiguousarray(yuyv[:, 0::2, 1]),
            np.ascontiguousarray(yuyv[:, 1::2, 1]))


def compare_yuyv(ref, test):
    """Compare packed YUYV arrays and return metrics, luma error, and SSIM map."""
    h = min(ref.shape[0], test.shape[0])
    w = min(ref.shape[1], test.shape[1]) & ~1
    ref = np.ascontiguousarray(ref[:h, :w])
    test = np.ascontiguousarray(test[:h, :w])
    (ry, ru, rv), (ty, tu, tv) = split_yuyv(ref), split_yuyv(test)
    smap = ssim_map(ry, ty)
    dy = cv2.absdiff(ry, ty)
    m = dict(psnr_y=psnr(ry, ty), psnr_u=psnr(ru, tu), psnr_v=psnr(rv, tv),
             psnr_bgr=psnr(yuyv_to_bgr(ref), yuyv_to_bgr(test)),
             ssim_y=float(smap.mean()), mae_y=float(dy.mean()), max_y=int(dy.max()),
             bias_y=float(ty.mean()) - float(ry.mean()),
             bias_u=float(tu.mean()) - float(ru.mean()),
             bias_v=float(tv.mean()) - float(rv.mean()))
    return m, dy, smap


CMP_COLS = [
    ("source", 96, lambda r: r["src"]),
    ("method", 136, lambda r: r["method"]),
    ("PSNR Y", 62, lambda r: f"{r['m']['psnr_y']:.2f}"),
    ("PSNR U", 62, lambda r: f"{r['m']['psnr_u']:.2f}"),
    ("PSNR V", 62, lambda r: f"{r['m']['psnr_v']:.2f}"),
    ("PSNR BGR", 88, lambda r: f"{r['m']['psnr_bgr']:.2f}"),
    ("SSIM Y", 64, lambda r: f"{r['m']['ssim_y']:.4f}"),
    ("MAE Y", 54, lambda r: f"{r['m']['mae_y']:.2f}"),
    ("max dY", 66, lambda r: f"{r['m']['max_y']}"),
    ("dY", 56, lambda r: f"{r['m']['bias_y']:+.2f}"),
    ("dU", 56, lambda r: f"{r['m']['bias_u']:+.2f}"),
    ("dV", 56, lambda r: f"{r['m']['bias_v']:+.2f}"),
    ("dec ms", 62, lambda r: f"{r['dec']:.2f}"),
    ("cvt ms", 62, lambda r: f"{r['cvt']:.2f}"),
    ("JPEG KB", 66, lambda r: f"{r['jpeg_kb']:.0f}" if r["jpeg_kb"] else "-"),
]


class CompareReport:
    """Compare decoded MJPEG samples with a native camera YUYV reference."""

    def __init__(self, cam_label, ref, cam_jpeg, quality, outdir):
        self.cam_label = cam_label
        self.ref = ref
        self.cam_jpeg = cam_jpeg
        self.cam_sub = jpeg_subsampling(cam_jpeg)
        self.quality = quality
        h, w = ref.shape[:2]
        self.dir = os.path.join(outdir, f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}_"
                                        f"{os.path.basename(cam_label.split()[-1])}_{w}x{h}")
        self.sel = 0
        self.rows = []
        self.build()

    def _row(self, src, method, res):
        m, dy, smap = compare_yuyv(self.ref, res["yuyv"])
        return dict(src=src, method=method, yuyv=res["yuyv"], m=m, dy=dy, smap=smap,
                    dec=res["dec"], cvt=res["cvt"],
                    jpeg_kb=len(res["jpeg"]) / 1024 if res["jpeg"] is not None else 0.0)

    def build(self):
        ref_bgr = yuyv_to_bgr(self.ref)
        self.rt_jpeg = encode_jpeg(ref_bgr, self.quality, self.cam_sub)
        rows = []
        for src, jpeg in (("camera", self.cam_jpeg), ("roundtrip", self.rt_jpeg)):
            for name, raw, fn, _ in METHODS:
                if not raw:
                    continue
                try:
                    rows.append(self._row(src, name, fn(np.frombuffer(jpeg, np.uint8))))
                except Exception as e:
                    print(f"[compare] {src}/{name} loi: {e}")
        t0 = time.perf_counter()
        base = bgr_to_yuyv(ref_bgr)
        rows.append(self._row("no-jpeg", "convert_only",
                              dict(yuyv=base, dec=0.0, cvt=(time.perf_counter() - t0) * 1e3, jpeg=None)))
        self.rows = rows
        self.sel = min(self.sel, len(rows) - 1)
        self.print_table()

    def header(self):
        h, w = self.ref.shape[:2]
        raw_kb = self.ref.nbytes / 1024
        cam_kb, rt_kb = len(self.cam_jpeg) / 1024, len(self.rt_jpeg) / 1024
        return [
            f"CAPTURE {self.cam_label} {w}x{h} | chuan = YUYV goc camera ({raw_kb:.0f} KB)",
            f"camera MJPEG {self.cam_sub} {cam_kb:.0f} KB (nen {raw_kb / cam_kb:.1f}x) | "
            f"roundtrip: YUYV goc -> JPEG q{self.quality} {self.cam_sub} {rt_kb:.0f} KB (nen {raw_kb / rt_kb:.1f}x)",
            "PSNR dB: cao = giong hon | SSIM: 1 = giong het | dY/dU/dV: lech trung binh "
            "(camera: con do 2 che do chup lech nhau ve thoi gian/phoi sang)",
        ]

    def print_table(self):
        print()
        for line in self.header():
            print(line)
        print("  ".join(f"{c[0]:>{max(len(c[0]), 8) if i > 1 else 14}}" for i, c in enumerate(CMP_COLS)))
        for r in self.rows:
            print("  ".join(f"{c[2](r):>{max(len(c[0]), 8) if i > 1 else 14}}" for i, c in enumerate(CMP_COLS)))

    def render(self, width=1280):
        r = self.rows[self.sel]
        ref_bgr = yuyv_to_bgr(self.ref)
        test_bgr = yuyv_to_bgr(np.ascontiguousarray(r["yuyv"]))
        h, w = self.ref.shape[:2]
        tw = width // 2
        th = int(h * tw / w)

        def tile(img, title):
            t = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
            draw_hud(t, [title])
            return t

        heat = cv2.applyColorMap(cv2.convertScaleAbs(r["dy"], alpha=8), cv2.COLORMAP_JET)
        sm = cv2.applyColorMap(np.clip((1 - r["smap"]) * 255 * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        grid = np.vstack([
            np.hstack([tile(ref_bgr, "CHUAN: YUYV goc camera"),
                       tile(test_bgr, f"{r['src']} / {r['method']}")]),
            np.hstack([tile(heat, f"|dY| x8   (MAE {r['m']['mae_y']:.2f}, max {r['m']['max_y']})"),
                       tile(sm, f"(1 - SSIM Y) x4   (SSIM {r['m']['ssim_y']:.4f})")]),
        ])

        fs, lh = 0.45, 20
        head = self.header()
        foot = [f"[n] chon dong  [,][.] chat luong JPEG roundtrip -/+5  [x] dong  | da luu: {self.dir}"]
        n_lines = len(head) + 1 + len(self.rows) + len(foot)
        panel = np.zeros((lh * n_lines + 14, grid.shape[1], 3), np.uint8)
        y = lh
        for t in head:
            cv2.putText(panel, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (200, 200, 200), 1, cv2.LINE_AA)
            y += lh
        table = [(None, [c[0] for c in CMP_COLS])] + [(i, [c[2](row) for c in CMP_COLS]) for i, row in enumerate(self.rows)]
        for i, cells in table:
            color = (0, 220, 255) if i == self.sel else ((255, 255, 255) if i is not None else (150, 200, 150))
            x = 8
            if i == self.sel:
                cv2.putText(panel, ">", (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)
            x += 14
            for (_, cw, _), cell in zip(CMP_COLS, cells):
                cv2.putText(panel, cell, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)
                x += cw
            y += lh
        for t in foot:
            cv2.putText(panel, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (200, 200, 200), 1, cv2.LINE_AA)
            y += lh
        return np.vstack([grid, panel])

    def save(self, width=1280):
        os.makedirs(self.dir, exist_ok=True)
        h, w = self.ref.shape[:2]
        self.ref.tofile(os.path.join(self.dir, "ref_native.yuyv"))
        with open(os.path.join(self.dir, "camera.jpg"), "wb") as f:
            f.write(self.cam_jpeg)
        with open(os.path.join(self.dir, f"roundtrip_q{self.quality}.jpg"), "wb") as f:
            f.write(self.rt_jpeg)
        for r in self.rows:
            np.ascontiguousarray(r["yuyv"]).tofile(os.path.join(self.dir, f"{r['src']}_{r['method']}.yuyv"))
        if not cv2.imwrite(os.path.join(self.dir, "compare.png"), self.render(width)):
            raise OSError("Không lưu được compare.png")
        csv_path = os.path.join(self.dir, "metrics.csv")
        new = not os.path.exists(csv_path)
        keys = ["psnr_y", "psnr_u", "psnr_v", "psnr_bgr", "ssim_y", "mae_y", "max_y", "bias_y", "bias_u", "bias_v"]
        with open(csv_path, "a") as f:
            if new:
                f.write("source,method,roundtrip_quality,subsampling,width,height," + ",".join(keys)
                        + ",dec_ms,cvt_ms,jpeg_kb\n")
            for r in self.rows:
                f.write(f"{r['src']},{r['method']},{self.quality},{self.cam_sub},{w},{h},"
                        + ",".join(f"{r['m'][k]:.4f}" for k in keys)
                        + f",{r['dec']:.3f},{r['cvt']:.3f},{r['jpeg_kb']:.1f}\n")
        print(f"[compare] da luu {self.dir}/  (ref_native.yuyv, camera.jpg, *.yuyv, compare.png, metrics.csv)")
        print(f"          xem: ffplay -f rawvideo -pixel_format yuyv422 -video_size {w}x{h} {self.dir}/ref_native.yuyv")


