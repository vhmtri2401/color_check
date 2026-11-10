"""Opt-in end-to-end checks against the connected USB cameras.

Run with: QT_QPA_PLATFORM=offscreen python3 -m tests.smoke_camera
"""

import os
import tempfile
import time
from pathlib import Path

import cv2
from PyQt5.QtCore import QLibraryInfo
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)
from PyQt5.QtWidgets import QApplication

from color_textile.controls import choose
from color_textile.gui import MainWindow, STYLE
from color_textile.settings import AppSettings, load_settings
from color_textile.settings_dialog import SettingsDialog


def main():
    app = QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    output = Path("/tmp/color-textile-qa")
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="color-textile-smoke-") as folder:
        config = AppSettings(outdir=str(Path(folder) / "captures"), warmup=0)
        window = MainWindow(config, Path(folder) / "settings.json", autostart=False)
        window.show()
        errors = []
        window.worker.notice.connect(lambda text, level: errors.append(text) if level == "error" else None)

        def wait_until(condition, timeout=25):
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                app.processEvents()
                if errors:
                    raise AssertionError("; ".join(errors))
                if condition():
                    return
                time.sleep(0.01)
            raise AssertionError("Timeout: " + window.message.text())

        def switch(camera, fourcc, size, method="imdecode_cvt"):
            choose(window.controls.camera, camera.identity)
            choose(window.controls.fourcc, fourcc)
            choose(window.controls.resolution, size)
            choose(window.controls.method, method)
            assert window.controls.resolution.currentData() == size
            expected = window.controls.fps.currentData()
            window.apply_button.click()
            wait_until(lambda: not window.busy and window.running and window.has_frame and
                       window.worker.actual[:3] == (fourcc, *size) and window.config.method == method)
            assert abs(window.worker.actual[3] - expected) < 0.1
            print("PASS switch", camera.node, fourcc, size, expected, method, flush=True)

        try:
            wait_until(lambda: not window.busy and bool(window.devices))
            first = window.devices[0]
            switch(first, "MJPG", (1280, 720))
            window.grab().save(str(output / "main.png"))
            window.rescan_button.click()
            wait_until(lambda: not window.busy and window.running and window.has_frame)
            assert window.config.camera_id == first.identity
            print("PASS rescan retains camera", flush=True)
            if len(window.devices) > 1:
                second = window.devices[1]
                switch(second, "MJPG", (1920, 1080))
                switch(second, "YUYV", (1920, 1080))
                assert not window.controls.method.isEnabled()
                window.grab().save(str(output / "native-yuyv.png"))
                switch(second, "MJPG", (640, 480), "cv_auto")
                switch(second, "MJPG", (640, 480), "imdecode_numpy")
                switch(second, "MJPG", (640, 480))
            window.save_button.click()
            wait_until(lambda: not window.busy and bool(list(Path(config.outdir).glob("*.png"))))
            print("PASS snapshot PNG/YUYV/JPEG", flush=True)
            settings = SettingsDialog(window.config, window.devices, window)
            settings.show()
            app.processEvents()
            settings.grab().save(str(output / "settings-camera.png"))
            settings.tabs.setCurrentIndex(1)
            app.processEvents()
            settings.grab().save(str(output / "settings-capture.png"))
            settings.quality.setValue(85)
            settings._accept()
            assert settings.result_config.jpeg_quality == 85
            window._submit("apply", settings.result_config)
            wait_until(lambda: not window.busy and window.config.jpeg_quality == 85)
            restored, warning = load_settings(Path(folder) / "settings.json")
            assert not warning and restored == window.config
            window.capture_button.click()
            wait_until(lambda: not window.busy and window.compare_dialog is not None, timeout=40)
            report = window.compare_dialog.report
            assert Path(report.dir, "metrics.csv").is_file()
            assert Path(report.dir, "ref_native.yuyv").stat().st_size == window.config.width * window.config.height * 2
            assert {r["src"] for r in report.rows} == {"camera", "roundtrip", "no-jpeg"}
            window.compare_dialog.grab().save(str(output / "compare.png"))
            print("PASS compare, metrics, native reference, stream restored", flush=True)
            window.compare_dialog.close()
            assert window.compare_dialog is None
            wait_until(lambda: window.has_frame)
            window.stop_button.click()
            wait_until(lambda: not window.busy and not window.running)
            assert window.worker.cap is None
            print("PASS stop releases camera", flush=True)
            print("Screenshots:", output, flush=True)
        finally:
            window.close()
            end = time.monotonic() + 20
            while window.worker.isRunning() and time.monotonic() < end:
                app.processEvents()
                time.sleep(0.01)
            app.processEvents()
            assert not window.worker.isRunning(), "Worker did not exit cleanly"


if __name__ == "__main__":
    main()
