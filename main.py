#!/usr/bin/env python3
"""Launch the Color Textile camera lab."""

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

from color_textile.settings import DEFAULT_SETTINGS_PATH, load_settings


def main():
    parser = argparse.ArgumentParser(description="Camera Lab · chọn camera, resolution, mode và capture/compare.")
    parser.add_argument("--list", action="store_true", help="Liệt kê camera và mode, không mở giao diện")
    parser.add_argument("--cam", help="Đường dẫn /dev/videoN, /dev/v4l/by-id/... hoặc số thứ tự camera")
    parser.add_argument("--size", help="Độ phân giải WxH")
    parser.add_argument("--fourcc", choices=("MJPG", "YUYV"))
    parser.add_argument("--fps", type=float)
    parser.add_argument("--method", help="Tên decoder, ví dụ imdecode_cvt")
    parser.add_argument("--out", help="Thư mục lưu ảnh và báo cáo")
    parser.add_argument("--no-start", action="store_true", help="Mở giao diện, chưa bật luồng camera")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS_PATH, help="Tệp cài đặt riêng")
    args = parser.parse_args()
    config, warning = load_settings(args.settings)
    if args.cam is not None or args.list:
        from color_textile.devices import discover_cameras, find_device
        cameras, failures = discover_cameras()
        if args.list:
            for i, camera in enumerate(cameras):
                print(f"[{i}] {camera.label}\n    ID: {camera.identity}")
                for mode in camera.modes:
                    print(f"    {mode.fourcc}  {mode.width}x{mode.height} @ {mode.fps:g} FPS")
            for message in failures:
                print(message, file=sys.stderr)
            return 0 if cameras else 1
        camera = cameras[int(args.cam)] if args.cam.isdigit() and int(args.cam) < len(cameras) else find_device(cameras, args.cam)
        if camera is None:
            parser.error("Không tìm thấy camera đã chọn; xem --list.")
        config = replace(config, camera_id=camera.identity)
    if args.size:
        try:
            width, height = map(int, args.size.lower().split("x"))
            config = replace(config, width=width, height=height)
        except ValueError:
            parser.error("--size có dạng WxH, ví dụ 1920x1080.")
    for name, value in (("fourcc", args.fourcc), ("fps", args.fps), ("method", args.method), ("outdir", args.out)):
        if value is not None:
            config = replace(config, **{name: value})
    try:
        config.validate()
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))

    # OpenCV wheels may inject a different Qt plugin directory; use PyQt's plugins.
    import cv2
    from PyQt5.QtCore import QLibraryInfo
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    fonts = os.environ.get("QT_QPA_FONTDIR", "")
    if "/cv2/" in fonts:
        os.environ.pop("QT_QPA_FONTDIR", None)
    from color_textile.capture import DECODERS
    if config.method not in DECODERS:
        if args.method:
            parser.error("Decoder không khả dụng. Có: " + ", ".join(DECODERS))
        warning = f"Decoder {config.method} chưa được cài; dùng imdecode_cvt."
        config = replace(config, method="imdecode_cvt")
    from color_textile.gui import run_gui
    return run_gui(config, args.settings, not args.no_start, warning)


if __name__ == "__main__":
    sys.exit(main())
