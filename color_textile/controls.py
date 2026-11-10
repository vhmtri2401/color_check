"""Reusable camera selection form shared by the main view and settings."""

from dataclasses import replace

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QComboBox, QFormLayout, QLabel, QWidget

from .capture import DECODERS
from .devices import find_device, select_mode

VIEW_LABELS = {"reconstructed": "Ảnh từ YUYV", "source": "Ảnh BGR nguồn",
               "luma": "Kênh sáng Y", "difference": "Sai khác BGR ×8"}
METHOD_LABELS = {"cv_auto": "OpenCV tự decode", "imdecode_cvt": "OpenCV imdecode + cvtColor",
                 "imdecode_numpy": "OpenCV + chuyển YUYV bằng NumPy",
                 "turbojpeg_yuv": "TurboJPEG → YUV → YUYV"}


def choose(combo, value):
    # PyQt stores tuple data as Python objects; QVariant lookup may use identity.
    index = next((i for i in range(combo.count()) if combo.itemData(i) == value), -1)
    combo.setCurrentIndex(index if index >= 0 else (0 if combo.count() else -1))


class CameraControls(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.devices = []
        self.updating = False
        self.camera = QComboBox(objectName="cameraCombo")
        self.fourcc = QComboBox(objectName="formatCombo")
        self.resolution = QComboBox(objectName="resolutionCombo")
        self.fps = QComboBox(objectName="fpsCombo")
        self.method = QComboBox(objectName="decoderCombo")
        self.view = QComboBox(objectName="viewCombo")
        for name in DECODERS:
            self.method.addItem(METHOD_LABELS.get(name, name), name)
            self.method.setItemData(self.method.count() - 1, DECODERS[name][3], 3)
        for name, label in VIEW_LABELS.items():
            self.view.addItem(label, name)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setObjectName("muted")
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        form.setSpacing(10)
        for title, widget in (("Camera", self.camera), ("Định dạng truyền", self.fourcc),
                              ("Độ phân giải", self.resolution), ("FPS camera", self.fps),
                              ("Thuật toán decode MJPEG", self.method), ("Chế độ xem", self.view)):
            form.addRow(title, widget)
        form.addRow(self.hint)
        self.camera.currentIndexChanged.connect(self._camera_changed)
        self.fourcc.currentIndexChanged.connect(self._format_changed)
        self.resolution.currentIndexChanged.connect(self._resolution_changed)
        for combo in (self.fps, self.method, self.view):
            combo.currentIndexChanged.connect(self._changed)

    def device(self):
        return find_device(self.devices, self.camera.currentData())

    def set_devices(self, devices, config):
        self.updating = True
        self.devices = list(devices)
        self.camera.clear()
        for device in devices:
            self.camera.addItem(device.label, device.identity)
            self.camera.setItemData(self.camera.count() - 1, device.identity, 3)
        self.camera.setEnabled(bool(devices))
        if not devices:
            self.camera.addItem("Chưa tìm thấy camera", "")
        self.updating = False
        self.set_config(config)

    def set_config(self, config):
        self.updating = True
        device = find_device(self.devices, config.camera_id)
        choose(self.camera, device.identity if device else config.camera_id)
        self._populate_formats(config.fourcc, (config.width, config.height), config.fps)
        choose(self.method, config.method)
        choose(self.view, config.view)
        self.updating = False
        self._update_hint()

    def _populate_formats(self, fourcc, size, fps):
        device = self.device()
        self.fourcc.clear()
        if device:
            for fmt in dict.fromkeys(m.fourcc for m in device.modes):
                self.fourcc.addItem("MJPEG · nén qua USB" if fmt == "MJPG" else "YUYV · ảnh thô 4:2:2", fmt)
            preferred = select_mode(device, fourcc, *size, fps)
            choose(self.fourcc, preferred.fourcc)
            size, fps = (preferred.width, preferred.height), preferred.fps
        self._populate_sizes(size, fps)

    def _populate_sizes(self, size, fps):
        self.resolution.clear()
        device = self.device()
        if device:
            sizes = sorted({(m.width, m.height) for m in device.modes if m.fourcc == self.fourcc.currentData()},
                           key=lambda s: (s[0] * s[1], s))
            for width, height in sizes:
                self.resolution.addItem(f"{width} × {height}", (width, height))
            if size not in sizes and sizes:
                size = min(sizes, key=lambda s: abs(s[0] * s[1] - size[0] * size[1]))
            choose(self.resolution, size)
        self._populate_fps(fps)

    def _populate_fps(self, fps):
        self.fps.clear()
        device = self.device()
        if device:
            rates = sorted({m.fps for m in device.modes if m.fourcc == self.fourcc.currentData() and
                            (m.width, m.height) == self.resolution.currentData()}, reverse=True)
            for rate in rates:
                self.fps.addItem(f"{rate:g} FPS", rate)
            if rates:
                choose(self.fps, min(rates, key=lambda r: abs(r - fps)))

    def _camera_changed(self):
        if self.updating:
            return
        self.updating = True
        self._populate_formats(self.fourcc.currentData() or "MJPG", self.resolution.currentData() or (1280, 720),
                               self.fps.currentData() or 30)
        self.updating = False
        self._changed()

    def _format_changed(self):
        if self.updating:
            return
        self.updating = True
        self._populate_sizes(self.resolution.currentData() or (1280, 720), self.fps.currentData() or 30)
        self.updating = False
        self._changed()

    def _resolution_changed(self):
        if self.updating:
            return
        self.updating = True
        self._populate_fps(self.fps.currentData() or 30)
        self.updating = False
        self._changed()

    def _update_hint(self):
        native = self.fourcc.currentData() == "YUYV"
        self.method.setEnabled(bool(self.device()) and not native)
        self.view.setEnabled(bool(self.device()))
        for combo in (self.fourcc, self.resolution, self.fps):
            combo.setEnabled(bool(self.device()))
        if not self.device():
            message = "Cắm camera rồi bấm Tìm lại camera."
        elif native:
            message = "YUYV là ảnh thô, không cần decoder. Ảnh nguồn / sai khác dùng ảnh YUYV khi không có BGR nguồn."
        else:
            message = "Danh sách mode và FPS đọc trực tiếp từ camera. Chọn xong, bấm Áp dụng để đổi luồng."
        self.hint.setText(message)

    def _changed(self):
        if not self.updating:
            self._update_hint()
            self.changed.emit()

    def read_config(self, base):
        device = self.device()
        if not device or self.resolution.currentData() is None or self.fps.currentData() is None:
            return replace(base)
        width, height = self.resolution.currentData()
        return replace(base, camera_id=device.identity, fourcc=self.fourcc.currentData(), width=width,
                       height=height, fps=self.fps.currentData(), method=self.method.currentData(), view=self.view.currentData())
