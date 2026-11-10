"""Native desktop interface for camera acquisition and comparison."""

import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (QApplication, QDialog, QFrame, QGridLayout, QHBoxLayout, QHeaderView,
                            QLabel, QMainWindow, QPushButton, QProgressBar, QScrollArea, QShortcut,
                            QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .capture import CameraWorker
from .controls import CameraControls
from .decoders import yuyv_to_bgr
from .devices import find_device
from .measure_window import POINT_COLORS, MeasurementWindow, session_folder
from .measurement import MAX_POINTS, SessionParams, roi_bounds
from .settings import save_settings
from .settings_dialog import SettingsDialog
from .uniformity import UniformityParams, build_ring_layout, render_ring_overlay
from .uniformity_window import UniformityWindow

STYLE = """
QWidget { background: #111923; color: #e5edf6; font-family: 'Noto Sans', 'DejaVu Sans'; font-size: 13px; }
QMainWindow, QDialog { background: #111923; }
QLabel#title { font-size: 24px; font-weight: 700; }
QLabel { background: transparent; }
QLabel#sectionTitle { font-size: 18px; font-weight: 600; }
QLabel#muted { color: #9cacc0; }
QLabel#badge { color: #83c6ff; background: #1c3045; padding: 8px 14px; border-radius: 12px; }
QFrame#panel { background: #192430; border: 1px solid #2c3b4c; border-radius: 12px; }
QPushButton { background: #253446; border: 1px solid #3d5066; border-radius: 7px; padding: 9px 14px; min-height: 20px; }
QPushButton:hover { background: #30465f; border-color: #77baff; }
QPushButton:pressed { background: #1d3046; }
QPushButton#primary { background: #2477d4; border-color: #3d91ee; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #3288e6; }
QPushButton#capture { background: #9a6420; border-color: #c58a39; color: white; font-weight: 600; }
QPushButton:disabled { background: #1b2633; color: #627286; border-color: #2c3949; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background: #101b28; border: 1px solid #3b4c60; border-radius: 6px; padding: 7px; min-height: 22px; }
QComboBox:focus, QLineEdit:focus { border-color: #68b4ff; }
QComboBox:disabled { color: #6e7c8e; border-color: #2b3847; }
QComboBox::drop-down { width: 25px; border-left: 1px solid #3b4c60; }
QComboBox::down-arrow { image: url(@ASSET@/chevron-down.svg); width: 14px; height: 14px; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url(@ASSET@/chevron-up.svg); width: 12px; height: 12px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url(@ASSET@/chevron-down.svg); width: 12px; height: 12px; }
QSpinBox::up-button, QDoubleSpinBox::up-button { width: 22px; subcontrol-origin: border; subcontrol-position: top right; }
QSpinBox::down-button, QDoubleSpinBox::down-button { width: 22px; subcontrol-origin: border; subcontrol-position: bottom right; }
QComboBox QAbstractItemView { background: #1c2b3d; selection-background-color: #285e96; min-width: 280px; }
QScrollArea { border: 0; background: transparent; }
QScrollBar:vertical { width: 10px; background: #111923; }
QScrollBar::handle:vertical { background: #3d5066; border-radius: 4px; min-height: 30px; }
QTabWidget::pane { border: 1px solid #33465b; border-radius: 7px; padding: 12px; }
QTabBar::tab { background: #1c2c3e; padding: 12px 18px; }
QTabBar::tab:selected { background: #245b90; }
QTableWidget { background: #132030; alternate-background-color: #1c2a3b; gridline-color: #304257; selection-background-color: #285e96; }
QHeaderView::section { background: #22364d; border: 0; padding: 8px; }
QProgressBar { border: 0; max-height: 4px; background: #172534; }
QProgressBar::chunk { background: #419cfa; }
""".replace("@ASSET@", (Path(__file__).resolve().parent / "assets").as_posix())


class ImageView(QLabel):
    clicked = pyqtSignal(float, float, int)  # normalised x, y in the image, Qt mouse button

    def __init__(self, placeholder="", parent=None):
        super().__init__(parent)
        self.placeholder = placeholder
        self.source_pixmap = None
        self.setText(placeholder)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(240, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #090f16; color: #7d91aa; border: 1px solid #29384a; border-radius: 10px;")

    def show_rgb(self, image):
        height, width = image.shape[:2]
        qimage = QImage(image.data, width, height, image.strides[0], QImage.Format_RGB888).copy()
        self.source_pixmap = QPixmap.fromImage(qimage)
        self._fit()

    def clear_image(self, message=None):
        self.source_pixmap = None
        self.clear()
        self.setText(message if message is not None else self.placeholder)

    def _fit(self):
        if self.source_pixmap is not None:
            self.setPixmap(self.source_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def mousePressEvent(self, event):
        shown = self.pixmap()
        if shown is not None and not shown.isNull():
            # The pixmap is centred and aspect-fitted inside the label.
            x = (event.x() - (self.width() - shown.width()) / 2) / shown.width()
            y = (event.y() - (self.height() - shown.height()) / 2) / shown.height()
            if 0 <= x <= 1 and 0 <= y <= 1:
                self.clicked.emit(x, y, int(event.button()))
        super().mousePressEvent(event)


class ComparisonDialog(QDialog):
    def __init__(self, report, parent=None):
        super().__init__(parent)
        self.report = report
        self.setWindowTitle("Capture & compare · MJPEG / YUYV")
        self.resize(1200, 850)
        layout = QVBoxLayout(self)
        heading = QLabel(f"So sánh ảnh · {report.ref.shape[1]} × {report.ref.shape[0]} · JPEG q{report.quality}")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        note = QLabel("camera: hai lần chụp khác thời điểm / phơi sáng  •  roundtrip: cùng một ảnh, dùng để đánh giá codec\n"
                      "PSNR càng cao càng giống; SSIM gần 1 càng giống. Thời gian decode/convert chỉ đo trên một frame.")
        note.setWordWrap(True)
        note.setObjectName("muted")
        layout.addWidget(note)
        splitter = QSplitter(Qt.Vertical)
        images = QWidget()
        grid = QGridLayout(images)
        self.views = []
        for i, title in enumerate(("Chuẩn · YUYV gốc", "Ảnh đã decode", "Sai khác độ sáng |dY| ×8", "Bản đồ (1 − SSIM) ×4")):
            cell = QWidget()
            box = QVBoxLayout(cell)
            box.setContentsMargins(0, 0, 0, 0)
            box.addWidget(QLabel(title))
            view = ImageView()
            box.addWidget(view, 1)
            self.views.append(view)
            grid.addWidget(cell, i // 2, i % 2)
        splitter.addWidget(images)
        self.table = QTableWidget(len(report.rows), 15)
        self.table.setMinimumHeight(225)
        self.table.setHorizontalHeaderLabels(["Nguồn", "Decoder", "PSNR Y", "PSNR U", "PSNR V", "PSNR BGR",
                                             "SSIM Y", "MAE Y", "max dY", "dY", "dU", "dV", "Decode ms", "Convert ms", "JPEG KB"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(True)
        for i, row in enumerate(report.rows):
            m = row["m"]
            values = [row["src"], row["method"], *[f"{m[k]:.2f}" for k in ("psnr_y", "psnr_u", "psnr_v", "psnr_bgr")],
                      f"{m['ssim_y']:.4f}", f"{m['mae_y']:.2f}", str(m["max_y"]),
                      *[f"{m[k]:+.2f}" for k in ("bias_y", "bias_u", "bias_v")],
                      f"{row['dec']:.2f}", f"{row['cvt']:.2f}", f"{row['jpeg_kb']:.0f}"]
            for j, value in enumerate(values):
                self.table.setItem(i, j, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()
        self.table.currentCellChanged.connect(lambda row, *_: self._select(row))
        splitter.addWidget(self.table)
        splitter.setSizes([560, 210])
        layout.addWidget(splitter, 1)
        footer = QHBoxLayout()
        path = QLabel(f"Đã lưu: {report.dir}")
        path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        path.setWordWrap(True)
        path.setObjectName("muted")
        footer.addWidget(path, 1)
        open_folder = QPushButton("Mở kết quả")
        open_folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(report.dir)))
        footer.addWidget(open_folder)
        close = QPushButton("Đóng")
        close.clicked.connect(self.close)
        footer.addWidget(close)
        layout.addLayout(footer)
        self.table.selectRow(0)

    def _select(self, index):
        if not 0 <= index < len(self.report.rows):
            return
        row = self.report.rows[index]
        images = [yuyv_to_bgr(self.report.ref), yuyv_to_bgr(np.ascontiguousarray(row["yuyv"])),
                  cv2.applyColorMap(cv2.convertScaleAbs(row["dy"], alpha=8), cv2.COLORMAP_JET),
                  cv2.applyColorMap(np.clip((1 - row["smap"]) * 1020, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)]
        for view, image in zip(self.views, images):
            if image.shape[1] > 640:
                image = cv2.resize(image, (640, round(image.shape[0] * 640 / image.shape[1])), interpolation=cv2.INTER_AREA)
            view.show_rgb(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


class MainWindow(QMainWindow):
    def __init__(self, config, settings_path, autostart=True, warning="", parent=None):
        super().__init__(parent)
        self.config = replace(config)
        self.settings_path = settings_path
        self.devices = []
        self.running = False
        self.busy = True
        self.has_frame = False
        self.closing = False
        self.compare_dialog = None
        self.measure_window = None
        self.measure_points = []  # points sent to the worker (ROI experiment repeats point 1)
        self.measure_rois = []
        self.measure_anchor = []  # points the user placed
        self.measure_running = False
        self.uniform_window = None
        self.uniform_state_data = None
        self.uniform_running = False
        self.last_raw_rgb = None
        self.frame_width = 0
        self.setWindowTitle("Color Textile · Camera Lab")
        self.resize(1380, 900)
        self.setMinimumSize(1060, 720)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 16)
        layout.setSpacing(14)
        header = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel("COLOR TEXTILE  /  Camera Lab")
        title.setObjectName("title")
        titles.addWidget(title)
        subtitle = QLabel("Xem camera · chuyển mode · thử nghiệm MJPEG → YUYV")
        subtitle.setObjectName("muted")
        titles.addWidget(subtitle)
        header.addLayout(titles, 1)
        self.badge = QLabel("ĐANG TÌM CAMERA")
        self.badge.setObjectName("badge")
        header.addWidget(self.badge)
        self.settings_button = QPushButton("Cài đặt…")
        self.settings_button.setObjectName("settingsButton")
        self.settings_button.clicked.connect(self._settings)
        header.addWidget(self.settings_button)
        layout.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(18)
        sidebar = QFrame(objectName="panel")
        sidebar.setFixedWidth(335)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(16, 18, 16, 16)
        heading = QLabel("Điều khiển camera")
        heading.setObjectName("sectionTitle")
        side.addWidget(heading)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        wrapper = QWidget()
        inner = QVBoxLayout(wrapper)
        inner.setContentsMargins(0, 8, 4, 8)
        self.controls = CameraControls()
        self.controls.changed.connect(self._pending)
        inner.addWidget(self.controls)
        inner.addStretch()
        scroll.setWidget(wrapper)
        side.addWidget(scroll, 1)
        self.pending = QLabel("Đang đọc danh sách thiết bị…")
        self.pending.setObjectName("muted")
        self.pending.setWordWrap(True)
        side.addWidget(self.pending)
        self.apply_button = QPushButton("Bắt đầu camera")
        self.apply_button.setObjectName("primary")
        self.apply_button.clicked.connect(self._apply)
        side.addWidget(self.apply_button)
        row = QHBoxLayout()
        self.rescan_button = QPushButton("Tìm lại camera")
        self.rescan_button.setObjectName("rescanButton")
        self.rescan_button.clicked.connect(lambda: self._submit("scan"))
        self.stop_button = QPushButton("Dừng")
        self.stop_button.clicked.connect(lambda: self._submit("stop"))
        row.addWidget(self.rescan_button, 1)
        row.addWidget(self.stop_button)
        side.addLayout(row)
        body.addWidget(sidebar)

        preview_area = QVBoxLayout()
        self.actual = QLabel("Chưa có luồng camera")
        self.actual.setObjectName("muted")
        self.actual.setWordWrap(True)
        preview_area.addWidget(self.actual)
        self.preview = ImageView("Đang tìm camera…\n\nBạn có thể cắm camera USB rồi bấm Tìm lại camera.")
        self.preview.clicked.connect(self._preview_clicked)
        preview_area.addWidget(self.preview, 1)
        stats = QHBoxLayout()
        self.stats_labels = {}
        for key, label in (("fps", "FPS THỰC"), ("read", "ĐỌC / ms"), ("decode", "DECODE / ms"), ("convert", "CHUYỂN YUYV / ms")):
            panel = QFrame(objectName="panel")
            box = QVBoxLayout(panel)
            small = QLabel(label)
            small.setObjectName("muted")
            box.addWidget(small)
            number = QLabel("—")
            number.setObjectName("sectionTitle")
            box.addWidget(number)
            self.stats_labels[key] = number
            stats.addWidget(panel, 1)
        preview_area.addLayout(stats)
        self.timing_note = QLabel("FPS thực đo từ luồng camera; FPS xem trước có thể được giới hạn trong Cài đặt.")
        self.timing_note.setWordWrap(True)
        self.timing_note.setObjectName("muted")
        preview_area.addWidget(self.timing_note)
        actions = QHBoxLayout()
        self.capture_button = QPushButton("Capture & compare")
        self.capture_button.setObjectName("capture")
        self.capture_button.clicked.connect(lambda: self._submit("compare"))
        self.save_button = QPushButton("Lưu ảnh hiện tại")
        self.save_button.clicked.connect(lambda: self._submit("save"))
        self.measure_button = QPushButton("Đo màu 3 điểm…")
        self.measure_button.setObjectName("primary")
        self.measure_button.clicked.connect(self._open_measure)
        self.uniform_button = QPushButton("Kiểm tra đồng màu…")
        self.uniform_button.setObjectName("primary")
        self.uniform_button.clicked.connect(self._open_uniform)
        self.results_button = QPushButton("Mở thư mục kết quả")
        self.results_button.clicked.connect(self._open_output)
        actions.addWidget(self.capture_button)
        actions.addWidget(self.save_button)
        actions.addWidget(self.measure_button)
        actions.addWidget(self.uniform_button)
        actions.addStretch()
        actions.addWidget(self.results_button)
        preview_area.addLayout(actions)
        body.addLayout(preview_area, 1)
        layout.addLayout(body, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)
        self.message = QLabel("Đang khởi tạo…")
        self.message.setWordWrap(True)
        self.message.setMinimumHeight(28)
        self.message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.message)

        self.worker = CameraWorker(config, autostart, self)
        self.worker.cameras_changed.connect(self._devices_changed)
        self.worker.state_changed.connect(self._state)
        self.worker.settings_applied.connect(self._applied)
        self.worker.notice.connect(self._notice)
        self.worker.busy_changed.connect(self._busy)
        self.worker.report_ready.connect(self._report)
        self.worker.finished.connect(self._finished)
        self.worker.measure_live.connect(self._measure_live)
        self.worker.measure_round.connect(self._measure_round)
        self.worker.measure_meta.connect(lambda meta: self.measure_window and self.measure_window.set_meta(meta))
        self.worker.uniform_state.connect(self._uniform_state)
        self.worker.uniform_done.connect(self._uniform_done)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(16)
        for key, slot in (("Ctrl+R", lambda: self._submit("scan")), ("Ctrl+S", lambda: self._submit("save")),
                          ("Ctrl+,", self._settings)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(slot)
        self._enable_controls()
        if warning:
            self._notice(warning, "warning")
        self.worker.start()

    def _devices_changed(self, devices):
        draft = self.controls.read_config(self.config)
        self.devices = devices
        self.controls.set_devices(devices, draft)
        self._pending()

    def _pending(self):
        if self.controls.device():
            changed = self.controls.read_config(self.config) != self.config
            self.pending.setText("Có thay đổi chưa áp dụng." if changed else "Chọn cấu hình rồi bấm Áp dụng." if not self.running else "Cấu hình đang được sử dụng.")
        else:
            self.pending.setText("Chưa có camera. Cắm thiết bị và tìm lại.")
        self._enable_controls()

    def _enable_controls(self):
        available = self.controls.device() is not None
        self.apply_button.setEnabled(not self.busy and available)
        self.apply_button.setText("Áp dụng cấu hình" if self.running else "Bắt đầu camera")
        self.stop_button.setEnabled(not self.busy and self.running)
        self.rescan_button.setEnabled(not self.busy)
        self.settings_button.setEnabled(not self.busy)
        self.controls.setEnabled(not self.busy)
        self.save_button.setEnabled(not self.busy and self.running and self.has_frame)
        device = find_device(self.devices, self.config.camera_id)
        formats = {m.fourcc for m in device.modes if (m.width, m.height) == (self.config.width, self.config.height)} if device else set()
        can_compare = {"MJPG", "YUYV"} <= formats
        self.capture_button.setEnabled(not self.busy and self.running and self.has_frame and can_compare
                                       and not self.measure_running and not self.uniform_running)
        self.capture_button.setToolTip("Không dùng được khi đang có phiên đo màu."
                                       if (self.measure_running or self.uniform_running) else
                                       "Chụp lần lượt MJPEG và YUYV; giữ yên camera." if can_compare else
                                       "Cần độ phân giải hỗ trợ cả MJPG và YUYV.")

    def _submit(self, operation, payload=None):
        if self.busy or self.closing:
            return
        if operation == "save" and not self.save_button.isEnabled():
            return
        if operation == "compare" and not self.capture_button.isEnabled():
            return
        self._busy(True, "Đang xử lý…")
        self.worker.submit(operation, payload)

    def _apply(self):
        self._submit("apply", self.controls.read_config(self.config))

    def _busy(self, busy, message):
        self.busy = busy
        self.progress.setVisible(busy)
        if busy:
            self.badge.setText("ĐANG XỬ LÝ")
            self._notice(message, "info")
        else:
            self.badge.setText("ĐANG XEM" if self.running else "ĐÃ DỪNG")
        self._enable_controls()

    def _applied(self, config):
        self.config = replace(config)
        self.controls.set_config(config)
        self._persist()
        self._pending()

    def _persist(self):
        try:
            save_settings(self.config, self.settings_path)
        except OSError as exc:
            self._notice(f"Cấu hình đã áp dụng nhưng chưa lưu được: {exc}", "warning")

    def _state(self, state):
        self.running = state["running"]
        self.has_frame = state["has_frame"]
        for key, label in self.stats_labels.items():
            label.setText(f"{state[key]:.1f}" if self.running and self.has_frame else "—")
        actual = state["actual"]
        self.frame_width = actual[1] if actual else 0
        if actual:
            device = find_device(self.devices, state["config"].camera_id)
            self.actual.setText(f"{device.label if device else 'Camera'}  •  Thực nhận: {actual[0]} {actual[1]} × {actual[2]} @ {actual[3]:g} FPS")
        else:
            self.actual.setText("Chưa có luồng camera")
            self.preview.clear_image(state["error"] or "Camera đã dừng.\nChọn cấu hình rồi bấm Bắt đầu camera.")
        self.timing_note.setText("OpenCV tự decode: thời gian giải nén nằm trong cột ĐỌC."
                                 if state["config"].fourcc == "MJPG" and state["config"].method == "cv_auto"
                                 else "FPS thực đo từ luồng camera; giới hạn xem trước được đặt riêng trong Cài đặt.")
        if not self.busy:
            self.badge.setText("ĐANG XEM" if self.running else "CHƯA CÓ CAMERA" if not self.devices else "ĐÃ DỪNG")
        self._pending()

    def _notice(self, text, level):
        colors = {"error": "#ff9d9d", "warning": "#f0c47b", "success": "#90dab2", "info": "#a8bdd4"}
        self.message.setStyleSheet(f"color: {colors.get(level, '#a8bdd4')};")
        self.message.setText(text)

    def _refresh(self):
        frame = self.worker.take_frame()
        if frame is not None and not self.closing:
            self.last_raw_rgb = frame
            self._render_current_frame()

    def _render_current_frame(self):
        if self.last_raw_rgb is None or self.closing:
            return
        frame = self.last_raw_rgb.copy()
        uniform_visible = self.uniform_window is not None and self.uniform_window.isVisible()
        if uniform_visible and self.uniform_state_data and self.uniform_state_data.get("detection") is not None:
            raw_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            det = self.uniform_state_data["detection"]
            layout = self.uniform_state_data.get("layout")
            res = self.uniform_state_data.get("result")
            full_bgr = render_ring_overlay(raw_bgr, det, layout, res, zoom_to_fabric=False)
            frame = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2RGB)
            if self.uniform_window.zoom_enabled():
                zoom_bgr = render_ring_overlay(
                    raw_bgr,
                    det,
                    layout,
                    res,
                    zoom_to_fabric=True,
                    zoom_margin=self.uniform_window.zoom_margin(),
                )
                self.uniform_window.show_preview_rgb(cv2.cvtColor(zoom_bgr, cv2.COLOR_BGR2RGB))
            else:
                self.uniform_window.show_preview_rgb(frame)
        elif uniform_visible:
            self.uniform_window.show_preview_rgb(frame)
        if self.measure_points:
            self._draw_points(frame)
        self.preview.show_rgb(frame)

    # ----- three-point colour measurement -----
    def _measure_params(self):
        c = self.config
        return SessionParams(roi=c.measure_roi, frames=c.measure_frames, interval_s=c.measure_interval_s,
                             duration_min=c.measure_duration_min, outlier_z=c.measure_outlier_z,
                             outlier_de=c.measure_outlier_de, lock=c.measure_lock, settle_window_s=c.measure_settle_s,
                             settle_de=c.measure_settle_de, settle_timeout_s=c.measure_settle_timeout_s,
                             de_formula=c.measure_de_formula, white=c.measure_white)

    def _open_measure(self):
        if self.measure_window is None:
            window = MeasurementWindow(self._measure_params(), self)
            window.params_changed.connect(self._measure_params_changed)
            window.roi_changed.connect(lambda _roi: self._send_points(self.measure_anchor))
            window.start_requested.connect(self._measure_start)
            window.stop_requested.connect(lambda: self.worker.submit("measure", dict(action="stop")))
            window.clear_points.connect(lambda: self._send_points([]))
            window.roi_test.toggled.connect(lambda on: self._notice(
                "Thử ROI: bấm lên ảnh để đặt tâm; các ROI có cùng tâm, kích thước khác nhau." if on else "", "info"))
            window.controls_requested.connect(lambda action: self.worker.submit("measure", dict(action=action)))
            self.measure_window = window
            self.worker.submit("measure", dict(action="controls"))
        self.measure_window.show()
        self.measure_window.raise_()
        self._notice("Bấm lên ảnh xem trước để đặt tối đa 3 điểm đo trên tấm vải.", "info")

    def _measure_params_changed(self, params):
        self.config = replace(self.config, measure_roi=params.roi, measure_frames=params.frames,
                              measure_interval_s=params.interval_s, measure_duration_min=params.duration_min,
                              measure_outlier_z=params.outlier_z, measure_outlier_de=params.outlier_de,
                              measure_lock=params.lock, measure_settle_s=params.settle_window_s,
                              measure_settle_de=params.settle_de, measure_settle_timeout_s=params.settle_timeout_s,
                              measure_de_formula=params.de_formula, measure_white=params.white)
        self._persist()

    def _preview_clicked(self, x, y, button):
        if self.measure_window is None or not self.measure_window.isVisible() or self.measure_running:
            return
        points = list(self.measure_anchor)
        if self.measure_window.roi_test_sizes():
            self._send_points([(x, y)] if button != int(Qt.RightButton) else [])
            return
        nearest = min(range(len(points)), key=lambda i: (points[i][0] - x) ** 2 + (points[i][1] - y) ** 2,
                      default=None)
        if button == int(Qt.RightButton):
            if nearest is not None:
                points.pop(nearest)
        elif len(points) < MAX_POINTS:
            points.append((x, y))
        else:
            points[nearest] = (x, y)
        self._send_points(points)

    def _send_points(self, points):
        self.measure_anchor = list(points)
        roi = self.measure_window.roi.value() if self.measure_window else self.config.measure_roi
        sizes = self.measure_window.roi_test_sizes() if self.measure_window else []
        if sizes and points:
            points, rois = [points[0]] * len(sizes), sizes
        else:
            rois = [roi] * len(points)
        self.measure_points, self.measure_rois = list(points), list(rois)
        self.worker.submit("measure", dict(action="points", points=self.measure_points, roi=roi,
                                           rois=rois if sizes and points else None))

    def _measure_start(self, params):
        device = find_device(self.devices, self.config.camera_id)
        if device is None or not self.running:
            self._notice("Hãy bật camera trước khi đo màu.", "warning")
            return
        folder = session_folder(self.config.outdir, device.node)
        mode = f"{self.config.fourcc} {self.config.width}x{self.config.height} @ {self.config.fps:g}"
        try:
            self.measure_window.begin_session(folder, self.measure_points, params, device.label, mode)
        except OSError as exc:
            self._notice(f"Không tạo được thư mục phiên đo: {exc}", "error")
            return
        self.worker.submit("measure", dict(action="start", params=params))

    def _measure_live(self, state):
        self.measure_points = state["points"]
        self.measure_rois = state["params"].point_rois(len(self.measure_points))
        session = state["session"]
        self.measure_running = bool(state.get("prep") or (session and session["state"] == "running"))
        if self.measure_window is not None:
            self.measure_window.update_live(state)
        self._enable_controls()

    def _measure_round(self, result):
        if self.measure_window is not None:
            try:
                self.measure_window.add_round(result)
            except OSError as exc:
                self._notice(f"Không ghi được kết quả lần đo: {exc}", "error")

    def _draw_points(self, frame):
        """Draw ROI boxes on the preview frame (already resized RGB, owned by the GUI)."""
        height, width = frame.shape[:2]
        scale = width / self.frame_width if self.frame_width else 1.0
        rois = self.measure_rois or [self.config.measure_roi] * len(self.measure_points)
        for i, point in enumerate(self.measure_points):
            x0, y0, x1, y1 = roi_bounds(width, height, point, max(4, round(rois[i] * scale)))
            color = tuple(int(POINT_COLORS[i][k:k + 2], 16) for k in (1, 3, 5))
            cv2.rectangle(frame, (x0 - 1, y0 - 1), (x1, y1), (0, 0, 0), 3)
            cv2.rectangle(frame, (x0 - 1, y0 - 1), (x1, y1), color, 1)
            cv2.putText(frame, str(i + 1), (x1 + 4, max(14, y0 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(frame, str(i + 1), (x1 + 4, max(14, y0 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1,
                        cv2.LINE_AA)

    # ----- 1-ring fabric uniformity check -----
    def _uniform_params(self):
        c = self.config
        return UniformityParams(
            inset_pct=c.uniform_inset_pct,
            width_pct=c.uniform_width_pct,
            segments=c.uniform_segments,
            frames=c.uniform_frames,
            threshold_de=c.uniform_threshold_de,
            outlier_z=c.measure_outlier_z,
            outlier_de=c.measure_outlier_de,
            lock=c.uniform_lock,
            de_formula=c.measure_de_formula,
            white=c.measure_white,
            interval_s=c.uniform_interval_s,
            duration_min=c.uniform_duration_min,
            filter_dirty_segments=c.uniform_filter_dirty,
        )

    def _open_uniform(self):
        if self.uniform_window is None:
            window = UniformityWindow(self._uniform_params(), self)
            window.params_changed.connect(self._uniform_params_changed)
            window.action_requested.connect(self._uniform_action)
            window.visibility_changed.connect(self._uniform_visibility_changed)
            window.view_changed.connect(self._render_current_frame)
            self.uniform_window = window
        self.uniform_window.show()
        self.uniform_window.raise_()
        if not self.running:
            self._notice("Hãy bật camera rồi bấm 'Phát hiện vải ngay' hoặc 'Đo đồng màu'.", "info")

    def _uniform_visibility_changed(self, visible):
        if visible:
            params = self.uniform_window.params() if self.uniform_window else self._uniform_params()
            self.worker.submit("uniform", dict(action="open", params=params))
        else:
            self.worker.submit("uniform", dict(action="close"))

    def _uniform_params_changed(self, params):
        self.config = replace(
            self.config,
            uniform_inset_pct=params.inset_pct,
            uniform_width_pct=params.width_pct,
            uniform_segments=params.segments,
            uniform_frames=params.frames,
            uniform_threshold_de=params.threshold_de,
            uniform_lock=params.lock,
            uniform_interval_s=params.interval_s,
            uniform_duration_min=params.duration_min,
            uniform_filter_dirty=params.filter_dirty_segments,
        )
        self._persist()
        if (not self.uniform_running and self.uniform_state_data
                and self.uniform_state_data.get("detection") is not None
                and self.uniform_state_data["detection"].found):
            det = self.uniform_state_data["detection"]
            self.uniform_state_data["layout"] = build_ring_layout(det, params)
            self.uniform_state_data["result"] = None
            if self.uniform_window is not None:
                self.uniform_window.update_state(self.uniform_state_data)
            self._render_current_frame()

    def _uniform_action(self, action, params=None):
        if action in ("start", "capture_master", "start_timeseries", "detect", "bg_capture") and not self.running:
            self._notice("Hãy bật camera trước khi kiểm tra đồng màu.", "warning")
            return
        payload = dict(action=action)
        if params is not None:
            payload["params"] = params
        self.worker.submit("uniform", payload)

    def _uniform_state(self, state):
        self.uniform_state_data = state
        self.uniform_running = bool(state.get("running"))
        if self.uniform_window is not None:
            self.uniform_window.update_state(state)
        self._render_current_frame()
        self._enable_controls()

    def _uniform_done(self, payload):
        if self.uniform_window is None:
            return
        device = find_device(self.devices, self.config.camera_id)
        node = device.node if device else "/dev/video0"
        camera_label = device.label if device else "Camera"
        mode_str = f"{self.config.fourcc} {self.config.width}x{self.config.height} @ {self.config.fps:g}"
        try:
            folder = self.uniform_window.show_result(self.config.outdir, node, camera_label, mode_str, payload)
            res = payload["result"]
            self._notice(
                f"Đã lưu kết quả đồng màu ({res.verdict.upper()} · max ΔE={res.max_pair_de:.3f}) tại {folder}",
                "success" if res.verdict == "pass" else "warning",
            )
        except OSError as exc:
            self._notice(f"Không ghi được kết quả kiểm tra đồng màu: {exc}", "error")

    def _settings(self):
        if self.busy or self.closing:
            return
        dialog = SettingsDialog(self.controls.read_config(self.config), self.devices, self)
        if dialog.exec_() == QDialog.Accepted:
            config = dialog.result_config
            self.controls.set_config(config)
            if self.devices:
                self._submit("apply", config)
            else:
                self.config = config
                self._persist()
                self._notice("Đã lưu cài đặt. Cắm camera và tìm lại để bắt đầu.", "success")

    def _open_output(self):
        folder = Path(self.config.outdir).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))

    def _report(self, report):
        if self.closing:
            return
        if self.compare_dialog is not None:
            self.compare_dialog.close()
        self.compare_dialog = ComparisonDialog(report, self)
        self.compare_dialog.setAttribute(Qt.WA_DeleteOnClose)
        self.compare_dialog.finished.connect(self._report_closed)
        self.compare_dialog.show()

    def _report_closed(self, _result):
        self.compare_dialog = None

    def _finished(self):
        if self.closing:
            self.close()

    def closeEvent(self, event):
        if self.worker.isRunning():
            event.ignore()
            self.closing = True
            self.timer.stop()
            self.worker.shutdown()
            self.setEnabled(False)
            self.message.setText("Đang giải phóng camera…")
        else:
            if self.compare_dialog is not None:
                self.compare_dialog.close()
            if self.measure_window is not None:
                self.measure_window.close()
            if self.uniform_window is not None:
                self.uniform_window.close()
            event.accept()


def run_gui(config, settings_path, autostart=True, warning=""):
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Color Textile Camera Lab")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow(config, settings_path, autostart, warning)
    window.show()
    return app.exec_()
