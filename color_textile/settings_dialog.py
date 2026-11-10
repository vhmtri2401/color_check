"""Separate settings window with extension points for additional pages."""

from dataclasses import replace

from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                            QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox,
                            QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from .controls import CameraControls


class SettingsDialog(QDialog):
    def __init__(self, config, devices, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cài đặt · Color Textile")
        self.resize(820, 700)
        self.base = replace(config)
        self.result_config = None
        layout = QVBoxLayout(self)
        heading = QLabel("Cài đặt ứng dụng")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        camera_page = QWidget()
        camera_layout = QVBoxLayout(camera_page)
        self.camera_controls = CameraControls()
        self.camera_controls.set_devices(devices, config)
        camera_layout.addWidget(self.camera_controls)
        camera_layout.addStretch()
        self.add_page("Camera", camera_page)

        capture_page = QWidget()
        form = QFormLayout(capture_page)
        form.setSpacing(18)
        self.quality = QSpinBox()
        self.quality.setRange(5, 100)
        self.quality.setValue(config.jpeg_quality)
        self.warmup = QDoubleSpinBox()
        self.warmup.setRange(0, 30)
        self.warmup.setSingleStep(0.5)
        self.warmup.setSuffix(" giây")
        self.warmup.setValue(config.warmup)
        self.outdir = QLineEdit(config.outdir)
        browse = QPushButton("Chọn thư mục…")
        browse.clicked.connect(self._browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self.outdir, 1)
        path_row.addWidget(browse)
        form.addRow("Chất lượng JPEG roundtrip", self.quality)
        form.addRow("Chờ camera ổn định mỗi mode", self.warmup)
        form.addRow("Thư mục lưu ảnh và báo cáo", path_row)
        note = QLabel("Capture & compare chụp lần lượt MJPEG và YUYV. Giữ yên camera và vật mẫu.\n\n"
                      "Dòng camera chịu ảnh hưởng của chuyển động / phơi sáng. Dùng dòng roundtrip để so sánh codec trên cùng một ảnh.\n\n"
                      "Lưu ảnh giữ độ phân giải gốc; kích thước xem trước không làm thay đổi dữ liệu lưu.")
        note.setWordWrap(True)
        note.setObjectName("muted")
        form.addRow(note)
        self.add_page("Chụp & lưu", capture_page)

        preview_page = QWidget()
        form = QFormLayout(preview_page)
        form.setSpacing(18)
        self.preview_fps = QSpinBox()
        self.preview_fps.setRange(1, 60)
        self.preview_fps.setValue(config.preview_fps)
        self.preview_width = QSpinBox()
        self.preview_width.setRange(320, 3840)
        self.preview_width.setSingleStep(160)
        self.preview_width.setValue(config.preview_width)
        self.preview_width.setSuffix(" px")
        form.addRow("Giới hạn FPS xem trước", self.preview_fps)
        form.addRow("Chiều rộng tối đa xem trước", self.preview_width)
        note = QLabel("Các giới hạn này chỉ áp dụng cho phần hiển thị. FPS camera được chọn riêng trong mục Camera.")
        note.setWordWrap(True)
        note.setObjectName("muted")
        form.addRow(note)
        self.add_page("Hiển thị", preview_page)

        modes_page = QWidget()
        modes_layout = QVBoxLayout(modes_page)
        table = QTableWidget()
        rows = [(device, mode) for device in devices for mode in device.modes]
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Camera", "Định dạng", "Độ phân giải", "FPS"])
        table.setRowCount(len(rows))
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setAlternatingRowColors(True)
        for i, (device, mode) in enumerate(rows):
            for j, value in enumerate((device.label, mode.fourcc, f"{mode.width} × {mode.height}", f"{mode.fps:g}")):
                table.setItem(i, j, QTableWidgetItem(value))
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        modes_layout.addWidget(QLabel("Các mode MJPG/YUYV được camera công bố tại lần quét gần nhất."))
        modes_layout.addWidget(table)
        self.add_page("Mode hỗ trợ", modes_page)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("Lưu và áp dụng")
        buttons.button(QDialogButtonBox.Save).setObjectName("primary")
        buttons.button(QDialogButtonBox.Cancel).setText("Hủy")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def add_page(self, title, widget):
        """Register a settings page without changing navigation or capture code."""
        self.tabs.addTab(widget, title)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Thư mục lưu ảnh", self.outdir.text())
        if folder:
            self.outdir.setText(folder)

    def _accept(self):
        try:
            config = self.camera_controls.read_config(self.base)
            self.result_config = replace(config, jpeg_quality=self.quality.value(), warmup=self.warmup.value(),
                                         outdir=self.outdir.text().strip(), preview_fps=self.preview_fps.value(),
                                         preview_width=self.preview_width.value()).validate()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Kiểm tra cài đặt", str(exc))
            return
        self.accept()
