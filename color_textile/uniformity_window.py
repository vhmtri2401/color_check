"""Window for the 5%-inset ring fabric colour uniformity check."""

from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Qt5Agg")
import numpy as np
from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QColor, QDesktopServices, QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QSlider, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)

from .colorimetry import FORMULAS
from .measure_window import BG, FG, GRID, LAB_COLORS, OUTLIER_COLOR, ChartCanvas, _style
from .uniformity import (
    SEGMENT_PALETTE_HEX,
    UniformityParams,
    export_uniformity_session,
    export_uniformity_timeseries,
    render_ring_overlay,
)

VERDICT_COLORS = {
    "pass": "#4cd97b",
    "fail": "#ff5c5c",
    "uncertain": "#f2c94c",
}


def uniformity_folder(outdir, node, prefix="uniform"):
    return Path(outdir).expanduser() / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{Path(node).name}"


class UniformityWindow(QWidget):
    """Controls, live detection status, perimeter charts and segment tables for uniformity checks."""

    params_changed = pyqtSignal(object)  # UniformityParams
    action_requested = pyqtSignal(str, object)  # action, payload
    visibility_changed = pyqtSignal(bool)
    view_changed = pyqtSignal()

    def __init__(self, params: UniformityParams, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Kiểm tra đồng màu · 1 vòng cách biên 5%")
        self.resize(1380, 920)
        self.folder = None
        self.ts_folder = None
        self.running = False
        self.ts_active = False
        self.ts_rounds = []
        self.has_bg_ref = False
        self.master_ref = None
        self.last_result = None
        self._last_result_payload = None

        layout = QVBoxLayout(self)
        heading = QLabel("Kiểm tra đồng màu tấm vải · 1 vòng cách biên")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        hint = QLabel(
            "Đặt tấm vải nằm trọn trên nền carton. Phần mềm tự tách vải (lấp lỗ chữ ở giữa), lấy 1 vòng cách biên vào "
            "5% kích thước vật thể, chia thành các đoạn đều nhau theo chu vi (bắt đầu từ góc trên-trái, thuận chiều kim đồng hồ) "
            "và so sánh ΔE giữa các đoạn trên vòng."
        )
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        layout.addWidget(hint)

        top = QHBoxLayout()
        form = QGridLayout()
        self.inset_pct = self._spin(QDoubleSpinBox(), 1.0, 45.0, params.inset_pct, " %", step=0.5)
        self.inset_slider = QSlider(Qt.Horizontal)
        self.inset_slider.setRange(2, 90)  # 1.0% .. 45.0% in 0.5% steps
        self.inset_slider.setValue(int(round(params.inset_pct * 2)))
        self.inset_slider.setToolTip("Kéo thanh trượt để thu vòng vào giữa tấm vải theo tỉ lệ cách đều viền ngoài")
        self.inset_ratio_label = QLabel(f"~{min(100.0, params.inset_pct * 2.0):.0f}% vào tâm")
        self.inset_ratio_label.setObjectName("muted")
        self.inset_ratio_label.setMinimumWidth(115)

        inset_box = QWidget()
        inset_row = QHBoxLayout(inset_box)
        inset_row.setContentsMargins(0, 0, 0, 0)
        inset_row.setSpacing(8)
        inset_row.addWidget(self.inset_slider, 1)
        inset_row.addWidget(self.inset_pct)
        inset_row.addWidget(self.inset_ratio_label)

        self.width_pct = self._spin(QDoubleSpinBox(), 1.0, 40.0, params.width_pct, " %", step=0.5)
        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setRange(2, 60)  # 1.0% .. 30.0% in 0.5% steps
        self.width_slider.setValue(int(round(min(30.0, params.width_pct) * 2)))
        self.width_slider.setToolTip("Kéo thanh trượt để chỉnh bề rộng dải vòng đo")

        width_box = QWidget()
        width_row = QHBoxLayout(width_box)
        width_row.setContentsMargins(0, 0, 0, 0)
        width_row.setSpacing(8)
        width_row.addWidget(self.width_slider, 1)
        width_row.addWidget(self.width_pct)

        self.segments = self._spin(QSpinBox(), 4, 36, params.segments, " đoạn")
        self.frames = self._spin(QSpinBox(), 3, 3000, params.frames, " frame")
        self.threshold_de = self._spin(QDoubleSpinBox(), 0.05, 20.0, params.threshold_de, "", step=0.05)
        self.formula = QComboBox()
        self.formula.addItem("ΔE00 (CIEDE2000)", "de00")
        self.formula.setCurrentIndex(0)
        self.white = QComboBox()
        self.white.addItem("D50 (Bradford, mặc định Nix)", "D50")
        self.white.addItem("D65", "D65")
        self.white.setCurrentIndex(max(0, self.white.findData(params.white)))
        self.bg_mode = QComboBox()
        self.bg_mode.addItem("Màu nền từ viền khung hình (5%)", "border")
        self.bg_mode.addItem("Ảnh nền carton đã chụp", "reference")
        self.bg_mode.setCurrentIndex(max(0, self.bg_mode.findData(params.bg_mode)))
        self.interval_s = self._spin(QDoubleSpinBox(), 1.0, 3600.0, params.interval_s, " giây (~2 phút)", step=10.0)
        self.interval_s.setToolTip("Chu kỳ tự động chụp lại trong chế độ Time-Series (mặc định 120 giây = 2 phút)")
        self.duration_min = self._spin(QDoubleSpinBox(), 0.0, 1440.0, params.duration_min, " phút (0 = liên tục)", step=2.0)
        self.duration_min.setToolTip("Tổng thời gian theo dõi Time-Series (đặt 0 để chạy liên tục đến khi bấm Dừng)")
        self.lock = QCheckBox("Khóa thông số camera trước khi đo (AE/AWB/exposure/gain)")
        self.lock.setChecked(params.lock)
        self.dirty_filter = QCheckBox("Chỉ xét đồng màu trên các điểm đồng màu — tự động loại bỏ điểm/đoạn bị dơ")
        self.dirty_filter.setChecked(bool(params.filter_dirty_segments))

        form.addWidget(QLabel("Thu vòng từ biên vào tâm (d)"), 0, 0)
        form.addWidget(inset_box, 0, 1, 1, 3)
        form.addWidget(QLabel("Bề rộng dải vòng (w)"), 1, 0)
        form.addWidget(width_box, 1, 1, 1, 3)

        fields = (
            ("Số đoạn trên vòng (K)", self.segments),
            ("Số frame đo (N)", self.frames),
            ("Ngưỡng đồng màu ΔE00 ≤", self.threshold_de),
            ("Chế độ tách nền", self.bg_mode),
            ("Công thức ΔE", self.formula),
            ("Chuẩn trắng Lab", self.white),
            ("Chu kỳ chụp lại (Time-Series)", self.interval_s),
            ("Thời lượng theo dõi", self.duration_min),
        )
        for i, (label, widget) in enumerate(fields):
            row_idx = 2 + (i // 2)
            form.addWidget(QLabel(label), row_idx, (i % 2) * 2)
            form.addWidget(widget, row_idx, (i % 2) * 2 + 1)
        form.addWidget(self.lock, 6, 0, 1, 2)
        form.addWidget(self.dirty_filter, 6, 2, 1, 2)

        self.inset_slider.valueChanged.connect(self._inset_slider_moved)
        self.width_slider.valueChanged.connect(self._width_slider_moved)
        for w in (self.inset_pct, self.width_pct, self.segments, self.frames, self.threshold_de,
                  self.interval_s, self.duration_min):
            w.valueChanged.connect(self._params_edited)
        for cb in (self.formula, self.white, self.bg_mode):
            cb.currentIndexChanged.connect(self._params_edited)
        self.lock.toggled.connect(self._params_edited)
        self.dirty_filter.toggled.connect(self._params_edited)
        top.addLayout(form, 1)

        buttons = QVBoxLayout()
        self.ts_button = QPushButton("Chụp gốc & Theo dõi 2 phút/lần")
        self.ts_button.setObjectName("primary")
        self.ts_button.setToolTip(
            "Chụp vải gốc ở t=0 (tự loại bỏ các điểm bị dơ), giữ cố định các điểm đồng màu và chụp lại định kỳ "
            "để xem thay đổi theo Time-Series"
        )
        self.ts_button.clicked.connect(lambda: self.action_requested.emit("start_timeseries", self.params()))
        self.master_button = QPushButton("Chụp vải gốc (chuẩn)")
        self.master_button.setToolTip("Đặt tấm vải chuẩn lên khay rồi bấm để lưu màu vải gốc làm mốc so sánh")
        self.master_button.clicked.connect(lambda: self.action_requested.emit("capture_master", self.params()))
        self.start_button = QPushButton("Đo đồng màu")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(lambda: self.action_requested.emit("start", self.params()))
        self.clear_master_button = QPushButton("Xóa vải gốc")
        self.clear_master_button.clicked.connect(lambda: self.action_requested.emit("clear_master", None))
        self.detect_button = QPushButton("Phát hiện vải ngay")
        self.detect_button.clicked.connect(lambda: self.action_requested.emit("detect", self.params()))
        self.bg_button = QPushButton("Chụp nền trống")
        self.bg_button.clicked.connect(lambda: self.action_requested.emit("bg_capture", self.params()))
        self.stop_button = QPushButton("Dừng")
        self.stop_button.clicked.connect(lambda: self.action_requested.emit("stop", None))
        self.open_button = QPushButton("Mở thư mục kết quả")
        self.open_button.clicked.connect(self._open_folder)
        for btn in (
            self.ts_button, self.master_button, self.start_button, self.clear_master_button,
            self.detect_button, self.bg_button, self.stop_button, self.open_button,
        ):
            buttons.addWidget(btn)
        top.addLayout(buttons)
        layout.addLayout(top)

        self.detect_info = QLabel("Trạng thái phát hiện: đang chờ ảnh camera…")
        self.detect_info.setObjectName("muted")
        self.detect_info.setWordWrap(True)
        layout.addWidget(self.detect_info)

        self.master_info = QLabel(
            "Vải gốc chuẩn: chưa chụp — bấm 'Chụp vải gốc (chuẩn)' để lưu mẫu vải gốc, sau đó đặt các tấm vải khác vào để so sánh."
        )
        self.master_info.setWordWrap(True)
        self.master_info.setStyleSheet(
            "font-size: 13px; padding: 6px 12px; background: #141e28; color: #9ab0c7; "
            "border: 1px solid #263545; border-radius: 6px;"
        )
        layout.addWidget(self.master_info)

        self.verdict_banner = QLabel("Bấm 'Đo đồng màu' sau khi vòng chia đoạn hiển thị đúng trên tấm vải.")
        self.verdict_banner.setWordWrap(True)
        self.verdict_banner.setStyleSheet(
            "font-size: 16px; font-weight: 700; padding: 10px 14px; background: #192430; "
            "border: 1px solid #2c3b4c; border-radius: 8px;"
        )
        layout.addWidget(self.verdict_banner)

        splitter = QSplitter(Qt.Horizontal)
        preview_box = QWidget()
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_header = QHBoxLayout()
        preview_title = QLabel("Ảnh phát hiện contour & vòng 12 đoạn")
        preview_title.setObjectName("muted")
        self.zoom_check = QCheckBox("Phóng to vùng vải")
        self.zoom_check.setChecked(True)
        self.zoom_check.setToolTip("Tự động cắt sát và phóng to tấm vải lên toàn khung nhìn")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 100)
        self.zoom_slider.setValue(80)
        self.zoom_slider.setFixedWidth(130)
        self.zoom_slider.setToolTip("Kéo hoặc lăn chuột trên ảnh để chỉnh độ phóng to vùng vải")
        self.zoom_label = QLabel("80%")
        self.zoom_label.setObjectName("muted")
        self.zoom_label.setMinimumWidth(58)
        self.zoom_check.toggled.connect(self._zoom_changed)
        self.zoom_slider.valueChanged.connect(self._zoom_changed)
        preview_header.addWidget(preview_title)
        preview_header.addStretch(1)
        preview_header.addWidget(self.zoom_check)
        preview_header.addWidget(self.zoom_slider)
        preview_header.addWidget(self.zoom_label)
        preview_layout.addLayout(preview_header)

        self.preview_view = QLabel("Đang chờ ảnh từ camera…")
        self.preview_view.setAlignment(Qt.AlignCenter)
        self.preview_view.setMinimumSize(520, 360)
        self.preview_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_view.setStyleSheet(
            "background: #090f16; color: #7d91aa; border: 1px solid #29384a; border-radius: 10px;"
        )
        self._preview_pixmap = None
        preview_layout.addWidget(self.preview_view, 1)
        splitter.addWidget(preview_box)

        self.tabs = QTabWidget()
        self.ring_chart = ChartCanvas()
        self.tabs.addTab(self.ring_chart, "Biểu đồ quanh vòng")

        self.ts_tab = QWidget()
        ts_layout = QVBoxLayout(self.ts_tab)
        ts_layout.setContentsMargins(4, 4, 4, 4)
        self.ts_chart = ChartCanvas()
        self.ts_table = QTableWidget(0, 11)
        self.ts_table.setHorizontalHeaderLabels([
            "Lần chụp", "Thời điểm", "Phút (t)", "Đoạn sạch", "Đoạn dơ (loại)",
            "L* / a* / b*", "ΔL* / Δa* / Δb* vs t=0", "ΔE00 TB vs t=0",
            "Max ΔE00 đoạn vs t=0", "ΔE00 nội bộ", "Kết luận",
        ])
        self.ts_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.ts_table.setAlternatingRowColors(True)
        self.ts_table.setMaximumHeight(220)
        ts_layout.addWidget(self.ts_chart, 1)
        ts_layout.addWidget(self.ts_table)
        self.tabs.addTab(self.ts_tab, "Time-Series (theo thời gian)")

        self.seg_table = QTableWidget(0, 15)
        self.seg_table.setHorizontalHeaderLabels([
            "Đoạn", "Điểm ảnh", "Loại chữ %", "R", "G", "B", "σR / σG / σB",
            "L*", "a*", "b*", "ΔL* / Δa* / Δb*", "ΔE00 vs vòng", "ΔE00 vs vải gốc", "Sàn nhiễu ΔE00", "Ghi chú",
        ])
        self.seg_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.seg_table.setAlternatingRowColors(True)
        self.tabs.addTab(self.seg_table, "Bảng từng đoạn")

        self.pair_table = QTableWidget(0, 0)
        self.pair_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tabs.addTab(self.pair_table, "Ma trận ΔE giữa các đoạn")

        splitter.addWidget(self.tabs)
        splitter.setSizes([660, 720])
        layout.addWidget(splitter, 1)

        self.folder_label = QLabel("")
        self.folder_label.setObjectName("muted")
        self.folder_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.folder_label)

        self.ring_chart.empty("Chưa có dữ liệu đo đồng màu. Bấm 'Đo đồng màu' để chụp và phân tích các đoạn trên vòng.")
        self.ts_chart.empty(
            "Chưa có dữ liệu Time-Series. Bấm 'Chụp gốc & Theo dõi 2 phút/lần' để chụp vải gốc ở t=0\n"
            "và tự động chụp lại sau mỗi chu kỳ (chỉ xét trên các đoạn đồng màu, loại bỏ các điểm bị dơ)."
        )
        self._update_ts_button_label()
        self._update_buttons()

    @staticmethod
    def _spin(widget, low, high, value, suffix, step=None):
        widget.setRange(low, high)
        if step is not None:
            widget.setSingleStep(step)
        widget.setValue(value)
        widget.setSuffix(suffix)
        return widget

    def params(self) -> UniformityParams:
        return UniformityParams(
            inset_pct=float(self.inset_pct.value()),
            width_pct=float(self.width_pct.value()),
            segments=int(self.segments.value()),
            frames=int(self.frames.value()),
            threshold_de=float(self.threshold_de.value()),
            de_formula=self.formula.currentData(),
            white=self.white.currentData(),
            lock=self.lock.isChecked(),
            bg_mode=self.bg_mode.currentData(),
            interval_s=float(self.interval_s.value()),
            duration_min=float(self.duration_min.value()),
            filter_dirty_segments=self.dirty_filter.isChecked(),
        )

    def _update_ts_button_label(self):
        iv = float(self.interval_s.value())
        if iv >= 60.0 and abs(iv % 60.0) < 0.5:
            mins = int(round(iv / 60.0))
            self.interval_s.setSuffix(f" giây ({mins} phút)")
            self.ts_button.setText(f"Chụp gốc & Theo dõi {mins} phút/lần")
        else:
            self.interval_s.setSuffix(f" giây ({iv / 60.0:.1f}p)")
            self.ts_button.setText(f"Chụp gốc & Theo dõi {iv:g}s/lần")

    def _inset_slider_moved(self, val):
        pct = val / 2.0
        self.inset_pct.blockSignals(True)
        self.inset_pct.setValue(pct)
        self.inset_pct.blockSignals(False)
        self._params_edited()

    def _width_slider_moved(self, val):
        pct = val / 2.0
        self.width_pct.blockSignals(True)
        self.width_pct.setValue(pct)
        self.width_pct.blockSignals(False)
        self._params_edited()

    def _params_edited(self):
        self.inset_slider.blockSignals(True)
        self.inset_slider.setValue(int(round(self.inset_pct.value() * 2)))
        self.inset_slider.blockSignals(False)
        self.width_slider.blockSignals(True)
        self.width_slider.setValue(int(round(min(30.0, self.width_pct.value()) * 2)))
        self.width_slider.blockSignals(False)
        self.inset_ratio_label.setText(f"~{min(100.0, self.inset_pct.value() * 2.0):.0f}% vào tâm")
        self._update_ts_button_label()
        p = self.params()
        self.params_changed.emit(p)
        self.action_requested.emit("params", p)

    def _update_buttons(self):
        for w in (self.inset_pct, self.inset_slider, self.width_pct, self.width_slider,
                  self.segments, self.frames, self.threshold_de, self.formula, self.white,
                  self.bg_mode, self.interval_s, self.duration_min, self.lock, self.dirty_filter,
                  self.ts_button, self.master_button, self.start_button,
                  self.detect_button, self.bg_button):
            w.setEnabled(not self.running)
        self.clear_master_button.setEnabled((not self.running) and (self.master_ref is not None))
        self.start_button.setText("Đo & so với vải gốc" if self.master_ref is not None else "Đo đồng màu")
        self.stop_button.setEnabled(self.running)

    def _update_master_banner(self):
        m = self.master_ref
        if m is None:
            self.master_info.setStyleSheet(
                "font-size: 13px; padding: 6px 12px; background: #141e28; color: #9ab0c7; "
                "border: 1px solid #263545; border-radius: 6px;"
            )
            self.master_info.setText(
                "Vải gốc chuẩn: chưa chụp — bấm 'Chụp vải gốc (chuẩn)' hoặc 'Chụp gốc & Theo dõi 2 phút/lần' để lưu mẫu vải gốc."
            )
        else:
            l_v, a_v, b_v = m.ring_mean_lab
            r_v, g_v, bl_v = m.ring_mean_rgb
            clean_n = int(np.count_nonzero(m.clean_segments)) if m.clean_segments is not None else m.segments
            dirty_ids = (
                [int(i + 1) for i in np.flatnonzero(m.dirty_segments)]
                if m.dirty_segments is not None and np.any(m.dirty_segments) else []
            )
            dirty_txt = f" (đã loại đoạn dơ: {', '.join(map(str, dirty_ids))})" if dirty_ids else ""
            mode_txt = (
                f"Đang theo dõi Time-Series định kỳ ({len(self.ts_rounds)} lần chụp)"
                if self.ts_active else "Đặt tấm vải cần kiểm tra lên khay và bấm 'Đo & so với vải gốc'"
            )
            self.master_info.setStyleSheet(
                "font-size: 13px; font-weight: 600; padding: 6px 12px; background: #132a23; color: #69db7c; "
                "border: 1px solid #2b6cb0; border-radius: 6px;"
            )
            self.master_info.setText(
                f"📌 Đang dùng VẢI GỐC CHUẨN (chụp lúc {m.timestamp}) · "
                f"Lab = ({l_v:.2f}, {a_v:+.2f}, {b_v:+.2f}) · RGB = ({r_v:.1f}, {g_v:.1f}, {bl_v:.1f}) · "
                f"{clean_n}/{m.segments} đoạn đồng màu sạch{dirty_txt} — {mode_txt}."
            )

    def update_state(self, state):
        """Receive live detection & measurement progress from the worker."""
        self.running = bool(state.get("running"))
        self.ts_active = bool(state.get("ts_active"))
        self.has_bg_ref = bool(state.get("has_bg_ref"))
        if state.get("ts_rounds") is not None:
            self.ts_rounds = list(state["ts_rounds"])
        if "master" in state:
            self.master_ref = state.get("master")
            self._update_master_banner()
        if self.has_bg_ref and self.bg_mode.currentData() != "reference" and state.get("bg_mode") == "reference":
            self.bg_mode.blockSignals(True)
            self.bg_mode.setCurrentIndex(self.bg_mode.findData("reference"))
            self.bg_mode.blockSignals(False)
        self._update_buttons()

        det = state.get("detection")
        layout = state.get("layout")
        progress = state.get("progress")

        if det is None or not det.found:
            msg = "Chưa phát hiện được vải."
            if det is not None and det.warnings:
                msg += " " + "; ".join(det.warnings)
            self.detect_info.setText(msg)
        else:
            info = (
                f"Đã phát hiện vải: {det.long_side:.0f}×{det.short_side:.0f} px (góc {det.angle:.1f}°, "
                f"độ đặc {det.solidity:.2f}, tương phản nền ΔE {det.contrast_de:.1f}, "
                f"thời gian {det.elapsed_ms:.1f} ms)"
            )
            if layout is not None and layout.valid:
                center_ratio = min(100.0, layout.inset_px / max(1.0, layout.dist_max_px) * 100.0)
                self.inset_ratio_label.setText(f"d={layout.inset_px} px ({center_ratio:.0f}% vào tâm)")
                info += (
                    f"  ·  Vòng: cách biên d={layout.inset_px} px ({center_ratio:.0f}% vào tâm), "
                    f"rộng w={layout.width_px} px, {layout.segments} đoạn (mỗi đoạn ~{int(layout.pixel_counts.mean())} px)"
                )
                if layout.warnings:
                    info += "  ·  Lưu ý: " + "; ".join(layout.warnings)
            elif layout is not None and layout.warnings:
                info += "  ·  " + "; ".join(layout.warnings)
            self.detect_info.setText(info)

        if progress:
            self.verdict_banner.setStyleSheet(
                "font-size: 15px; font-weight: 600; color: #83c6ff; padding: 10px 14px; "
                "background: #1c3045; border: 1px solid #3d5066; border-radius: 8px;"
            )
            self.verdict_banner.setText(str(progress))

    def show_result(self, outdir, node, camera_label, mode_str, payload):
        """Render completed uniformity result and export session files."""
        is_ts = bool(payload.get("is_timeseries"))
        round_idx = int(payload.get("round_index", 0))
        self.ts_active = bool(payload.get("ts_active"))
        self.running = self.ts_active
        is_master = bool(payload.get("is_master"))
        if payload.get("ts_rounds") is not None:
            self.ts_rounds = list(payload["ts_rounds"])
        if "master" in payload:
            self.master_ref = payload.get("master")
            self._update_master_banner()
        self._update_buttons()
        self._last_result_payload = payload
        frame_bgr = payload["frame_bgr"]
        det = payload["detection"]
        layout = payload["layout"]
        res = payload["result"]
        meta = payload.get("meta")
        self.last_result = res

        color = VERDICT_COLORS.get(res.verdict, "#c9d6e3")
        icon = {"pass": "✅", "fail": "❌", "uncertain": "⚠️"}.get(res.verdict, "")
        warn_suffix = ("  ·  " + "; ".join(res.warnings)) if res.warnings else ""
        self.verdict_banner.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {color}; padding: 10px 14px; "
            f"background: #192430; border: 2px solid {color}; border-radius: 8px;"
        )
        if is_ts:
            prefix_txt = f"[TIME-SERIES LẦN #{round_idx} · t={res.elapsed_min:.1f}p] "
        elif is_master:
            prefix_txt = "[ĐÃ LƯU VẢI GỐC CHUẨN] "
        else:
            prefix_txt = ""
        self.verdict_banner.setText(f"{icon} {prefix_txt}{res.explanation}{warn_suffix}")

        self._draw_chart(res)
        self._fill_tables(res)
        self._render_result_preview(payload)

        if is_ts:
            if round_idx == 0 or self.ts_folder is None:
                self.ts_folder = uniformity_folder(outdir, node, prefix="uniform_ts")
            folder = self.ts_folder
            self.folder = folder
            round_dir = folder / f"round_{round_idx:03d}"
            export_uniformity_session(round_dir, frame_bgr, det, layout, res, camera=camera_label, mode=mode_str, meta=meta)
            export_uniformity_session(folder, frame_bgr, det, layout, res, camera=camera_label, mode=mode_str, meta=meta)
            self._draw_ts_chart(res)
            self._fill_ts_table(res)
            export_uniformity_timeseries(folder, self.ts_rounds, master=self.master_ref)
            self.ring_chart.figure.savefig(folder / "chart_ring.png", dpi=110, facecolor=BG)
            self.ts_chart.figure.savefig(folder / "chart_timeseries.png", dpi=110, facecolor=BG)
            if round_idx >= 1:
                self.tabs.setCurrentWidget(self.ts_tab)
            self.folder_label.setText(
                f"Đã lưu Time-Series (lần #{round_idx}, tổng {len(self.ts_rounds)} lần chụp): {folder}"
            )
            return folder

        prefix = "master" if is_master else ("compare" if res.has_master else "uniform")
        folder = uniformity_folder(outdir, node, prefix=prefix)
        self.folder = folder
        export_uniformity_session(folder, frame_bgr, det, layout, res, camera=camera_label, mode=mode_str, meta=meta)
        self.ring_chart.figure.savefig(folder / "chart_ring.png", dpi=110, facecolor=BG)
        self.folder_label.setText(f"Đã lưu kết quả kiểm tra đồng màu: {folder}")
        return folder

    def zoom_enabled(self) -> bool:
        return self.zoom_check.isChecked()

    def zoom_margin(self) -> float:
        return max(0.05, (105 - self.zoom_slider.value()) / 100.0)

    def _zoom_changed(self):
        on = self.zoom_check.isChecked()
        self.zoom_slider.setEnabled(on)
        self.zoom_label.setText(f"{self.zoom_slider.value()}%" if on else "Toàn khung")
        if self._last_result_payload is not None and not self.running:
            self._render_result_preview(self._last_result_payload)
        self.view_changed.emit()

    def _render_result_preview(self, payload):
        over_bgr = render_ring_overlay(
            payload["frame_bgr"],
            payload["detection"],
            payload["layout"],
            payload["result"],
            zoom_to_fabric=self.zoom_enabled(),
            zoom_margin=self.zoom_margin(),
        )
        if over_bgr.shape[1] > 960:
            over_bgr = cv2.resize(
                over_bgr,
                (960, max(1, round(over_bgr.shape[0] * 960 / over_bgr.shape[1]))),
                interpolation=cv2.INTER_AREA,
            )
        self.show_preview_rgb(cv2.cvtColor(over_bgr, cv2.COLOR_BGR2RGB))

    def wheelEvent(self, event):
        if self.preview_view.underMouse() and self.zoom_check.isChecked():
            step = 5 if event.angleDelta().y() > 0 else -5
            self.zoom_slider.setValue(max(10, min(100, self.zoom_slider.value() + step)))
            event.accept()
            return
        super().wheelEvent(event)

    def _draw_chart(self, res):
        fig = self.ring_chart.figure
        fig.clear()
        ax_de, ax_lab = fig.subplots(2, 1, sharex=True)
        for ax in (ax_de, ax_lab):
            _style(ax)

        k = res.segments
        xs = np.arange(1, k + 1)
        formula_label = FORMULAS.get(res.params.de_formula, res.params.de_formula)
        thr = res.params.threshold_de
        use_master = bool(res.has_master and res.de_to_master is not None)
        raw_primary = res.de_to_master if use_master else res.de_to_mean
        primary_vals = np.where(res.contaminated, 0.0, raw_primary)
        plot_yerr = np.where(res.contaminated, 0.0, res.seg_noise_de)

        bar_colors = []
        for i in range(k):
            if res.contaminated[i]:
                bar_colors.append("#778899")
            elif primary_vals[i] > (thr if use_master else thr * 0.65):
                bar_colors.append(OUTLIER_COLOR)
            else:
                bar_colors.append(SEGMENT_PALETTE_HEX[i % len(SEGMENT_PALETTE_HEX)])

        bar_label = f"{formula_label} đoạn sạch so với Vải gốc" if use_master else f"{formula_label} đoạn sạch so với TB vòng"
        ax_de.bar(xs, primary_vals, color=bar_colors, alpha=0.85, width=0.6, label=bar_label)
        if use_master:
            clean_de_mean = np.where(res.contaminated, np.nan, res.de_to_mean)
            ax_de.plot(xs, clean_de_mean, marker="D", markersize=4, linestyle="--", linewidth=1.1,
                       color="#a5d8ff", label=f"{formula_label} nội bộ so với TB vòng")
            ax_de.axhline(res.overall_master_de, color="#69db7c", linestyle="-.", linewidth=1.2,
                          label=f"Lệch tổng thể so với Vải gốc: {res.overall_master_de:.3f}")
        ax_de.errorbar(xs, primary_vals, yerr=plot_yerr, fmt="none", ecolor="#e5edf6", elinewidth=1.2, capsize=3)
        ax_de.axhline(thr, color=OUTLIER_COLOR, linestyle="--", linewidth=1.1, label=f"Ngưỡng {thr:.2f}")
        ax_de.axhline(res.max_pair_de, color="#ffb000", linestyle=":", linewidth=1.2,
                      label=f"Max giữa 2 đoạn ({res.worst_pair[0]}–{res.worst_pair[1]}): {res.max_pair_de:.3f}")
        if res.noise_floor_de > 0:
            ax_de.axhspan(0, res.noise_floor_de, color="#83c6ff", alpha=0.14, label=f"Sàn nhiễu: {res.noise_floor_de:.3f}")
        ax_de.set_ylabel(formula_label, fontsize=10)
        ax_de.set_title(
            f"Độ lệch màu từng đoạn quanh chu vi ({res.frame_count} frame) · Kết luận: {res.verdict_label}",
            fontsize=10,
        )
        leg1 = ax_de.legend(fontsize=8, facecolor=BG, edgecolor=GRID, loc="upper right")
        for text in leg1.get_texts():
            text.set_color(FG)

        d_lab = res.delta_lab_to_master if (use_master and res.delta_lab_to_master is not None) else res.delta_lab
        for c, (label, col) in enumerate(zip(("ΔL*", "Δa*", "Δb*"), LAB_COLORS)):
            ax_lab.plot(xs, d_lab[:, c], marker="o", markersize=4, linewidth=1.3, color=col, label=label)
        ax_lab.axhline(0.0, color="#8899aa", linestyle="--", linewidth=0.8)
        ax_lab.set_xticks(xs)
        ax_lab.set_xticklabels([f"Đoạn {i}" for i in xs])
        ax_lab.set_ylabel("Độ lệch so với Vải gốc" if use_master else "Độ lệch so với TB vòng", fontsize=10)
        ax_lab.set_xlabel("Vị trí đoạn quanh chu vi (từ góc trên-trái, theo chiều kim đồng hồ)", fontsize=9)
        leg2 = ax_lab.legend(fontsize=8, facecolor=BG, edgecolor=GRID, loc="upper right")
        for text in leg2.get_texts():
            text.set_color(FG)

        self.ring_chart.draw_idle()

    def _draw_ts_chart(self, res):
        if not self.ts_rounds:
            return
        fig = self.ts_chart.figure
        fig.clear()
        ax_de, ax_lab = fig.subplots(2, 1, sharex=True)
        for ax in (ax_de, ax_lab):
            _style(ax)

        ts_mins = np.array([r.elapsed_min for r in self.ts_rounds], dtype=np.float64)
        overall_de = np.array([r.overall_master_de for r in self.ts_rounds], dtype=np.float64)
        max_m_de = np.array([r.max_master_de for r in self.ts_rounds], dtype=np.float64)
        pair_de = np.array([r.max_pair_de for r in self.ts_rounds], dtype=np.float64)
        d_labs = np.array([r.delta_lab_master for r in self.ts_rounds], dtype=np.float64)
        thr = float(res.params.threshold_de)
        formula_label = FORMULAS.get(res.params.de_formula, res.params.de_formula)

        ax_de.plot(
            ts_mins, overall_de, marker="o", markersize=6, linewidth=1.8,
            color="#4cd97b", label=f"{formula_label} TB đoạn sạch so với Vải gốc (t=0)",
        )
        ax_de.plot(
            ts_mins, max_m_de, marker="s", markersize=5, linewidth=1.4, linestyle="--",
            color="#ffb000", label=f"Max {formula_label} đoạn sạch so với Vải gốc (t=0)",
        )
        ax_de.plot(
            ts_mins, pair_de, marker="^", markersize=4, linewidth=1.2, linestyle=":",
            color="#74c0fc", label=f"Max {formula_label} nội bộ giữa các đoạn sạch",
        )
        ax_de.axhline(thr, color=OUTLIER_COLOR, linestyle="--", linewidth=1.2, label=f"Ngưỡng {thr:.2f}")
        ax_de.set_ylabel(formula_label, fontsize=10)
        ax_de.set_title(
            f"Biến thiên màu theo thời gian (Time-Series · {len(self.ts_rounds)} lần chụp · chỉ xét các đoạn sạch)",
            fontsize=10,
        )
        leg1 = ax_de.legend(fontsize=8, facecolor=BG, edgecolor=GRID, loc="upper left")
        for text in leg1.get_texts():
            text.set_color(FG)

        for c, (label, col) in enumerate(zip(("ΔL* vs t=0", "Δa* vs t=0", "Δb* vs t=0"), LAB_COLORS)):
            ax_lab.plot(ts_mins, d_labs[:, c], marker="o", markersize=5, linewidth=1.4, color=col, label=label)
        ax_lab.axhline(0.0, color="#8899aa", linestyle="--", linewidth=0.9)
        ax_lab.set_ylabel("Độ lệch Lab vs t=0", fontsize=10)
        ax_lab.set_xlabel("Thời gian từ lúc chụp vải gốc (phút)", fontsize=9)
        leg2 = ax_lab.legend(fontsize=8, facecolor=BG, edgecolor=GRID, loc="upper left")
        for text in leg2.get_texts():
            text.set_color(FG)

        self.ts_chart.draw_idle()

    def _fill_ts_table(self, res):
        n_rounds = len(self.ts_rounds)
        self.ts_table.setRowCount(n_rounds)
        k_total = res.segments
        for idx, r in enumerate(self.ts_rounds):
            dirty_str = ", ".join(f"Đ{d}" for d in r.dirty_segments_list) if r.dirty_segments_list else "Không"
            row = [
                f"Lần #{r.round_index}" + (" (Gốc)" if r.round_index == 0 else ""),
                r.timestamp,
                f"{r.elapsed_min:.2f} p",
                f"{r.valid_segments}/{k_total}",
                dirty_str,
                " / ".join(f"{v:.2f}" for v in r.ring_mean_lab),
                " / ".join(f"{v:+.2f}" for v in r.delta_lab_master),
                f"{r.overall_master_de:.3f}",
                f"{r.max_master_de:.3f} (Đ{r.worst_master_segment})",
                f"{r.max_pair_de:.3f} ({r.worst_pair[0]}–{r.worst_pair[1]})",
                r.verdict_label,
            ]
            v_col = VERDICT_COLORS.get(r.verdict, "#c9d6e3")
            for j, val in enumerate(row):
                item = QTableWidgetItem(val)
                if j in (7, 8, 10):
                    item.setForeground(QColor(v_col))
                self.ts_table.setItem(idx, j, item)
        self.ts_table.resizeColumnsToContents()
        self.ts_table.scrollToBottom()

    def _fill_tables(self, res):
        k = res.segments
        use_master = bool(res.has_master and res.de_to_master is not None)
        d_lab = res.delta_lab_to_master if (use_master and res.delta_lab_to_master is not None) else res.delta_lab
        self.seg_table.setRowCount(k)
        for i in range(k):
            if res.dirty_segments is not None and res.dirty_segments[i]:
                note = "Loại (điểm bị dơ)"
            elif res.contaminated[i]:
                note = "Loại (chữ/in hoặc dơ)"
            elif use_master and (i + 1) == res.worst_master_segment and res.verdict == "fail":
                note = "Lệch vải gốc lớn nhất"
            elif (i + 1) == res.worst_segment and not use_master:
                note = "Lệch lớn nhất"
            elif (i + 1) in res.worst_pair and res.max_pair_de > res.params.threshold_de:
                note = f"Cặp lệch {res.worst_pair[0]}–{res.worst_pair[1]}"
            else:
                note = "Đạt chuẩn (đồng màu)" if res.verdict == "pass" else "Hợp lệ"
            is_excl = bool(res.contaminated[i])
            row = [
                f"Đoạn {i + 1}",
                str(int(res.pixel_counts[i])),
                f"{res.rejected_pct[i]:.1f}%",
                *(f"{v:.2f}" for v in res.mean_rgb[i]),
                " / ".join(f"{v:.2f}" for v in res.std_rgb[i]),
                *(f"{v:.2f}" for v in res.mean_lab[i]),
                "— (đã loại)" if is_excl else " / ".join(f"{v:+.2f}" for v in d_lab[i]),
                "— (đã loại)" if is_excl else f"{res.de_to_mean[i]:.3f}",
                "— (đã loại)" if is_excl else (f"{res.de_to_master[i]:.3f}" if use_master else "—"),
                f"{res.seg_noise_de[i]:.3f}",
                note,
            ]
            for j, val in enumerate(row):
                item = QTableWidgetItem(val)
                if j == 0:
                    item.setForeground(QColor(SEGMENT_PALETTE_HEX[i % len(SEGMENT_PALETTE_HEX)]))
                elif res.contaminated[i]:
                    item.setForeground(QColor("#8899aa"))
                elif j in (11, 12, 14) and res.verdict == "fail":
                    if (use_master and res.de_to_master[i] > res.params.threshold_de) or ((i + 1) in res.worst_pair):
                        item.setForeground(QColor(OUTLIER_COLOR))
                self.seg_table.setItem(i, j, item)
        self.seg_table.resizeColumnsToContents()

        self.pair_table.setRowCount(k)
        self.pair_table.setColumnCount(k)
        headers = [f"Đ{i + 1}" for i in range(k)]
        self.pair_table.setHorizontalHeaderLabels(headers)
        self.pair_table.setVerticalHeaderLabels(headers)
        thr = res.params.threshold_de
        for i in range(k):
            for j in range(k):
                val = float(res.pairwise_de[i, j])
                item = QTableWidgetItem("—" if i == j else f"{val:.3f}")
                if i != j:
                    if res.contaminated[i] or res.contaminated[j]:
                        item.setForeground(QColor("#6e7c8e"))
                    elif val > thr:
                        item.setForeground(QColor(OUTLIER_COLOR))
                self.pair_table.setItem(i, j, item)
        self.pair_table.resizeColumnsToContents()

    def show_preview_rgb(self, rgb_image):
        if rgb_image is None:
            return
        rgb = np.ascontiguousarray(rgb_image)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        self._preview_pixmap = QPixmap.fromImage(qimg)
        self._fit_preview()

    def _fit_preview(self):
        if self._preview_pixmap is not None:
            self.preview_view.setPixmap(
                self._preview_pixmap.scaled(self.preview_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_preview()

    def _open_folder(self):
        if self.folder is not None and self.folder.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.folder)))

    def showEvent(self, event):
        super().showEvent(event)
        self.visibility_changed.emit(True)

    def closeEvent(self, event):
        self.visibility_changed.emit(False)
        super().closeEvent(event)
