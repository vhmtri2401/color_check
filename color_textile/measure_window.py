"""Window for the three-point colour stability session: parameters, live values, charts, tables."""

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg")
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QPushButton,
                            QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from .colorimetry import FORMULAS, delta_e
from .measurement import (SessionParams, export_monitor, export_round, export_summary, method_comparison,
                          session_summary, write_session_info)

POINT_COLORS = ("#ffb000", "#00c8ff", "#ff50c8")
RGB_COLORS = ("#ff5c5c", "#4cd97b", "#5c9dff")
LAB_COLORS = ("#d8dee9", "#ff6fa8", "#f2c94c")
OUTLIER_COLOR = "#ff3b3b"
BG, FG, GRID = "#111923", "#c9d6e3", "#2c3b4c"


def _style(ax):
    ax.set_facecolor("#0c141d")
    ax.tick_params(colors=FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.title.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)


class ChartCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.figure = Figure(facecolor=BG, layout="constrained")
        super().__init__(self.figure)

    def empty(self, text):
        self.figure.clear()
        self.figure.text(0.5, 0.5, text, ha="center", va="center", color=FG, fontsize=12)
        self.draw_idle()


class MeasurementWindow(QWidget):
    """Owns the session folder and exports; the worker only produces samples and rounds."""

    params_changed = pyqtSignal(object)  # SessionParams, so the main window can persist them
    start_requested = pyqtSignal(object)
    stop_requested = pyqtSignal()
    clear_points = pyqtSignal()
    roi_changed = pyqtSignal(int)
    controls_requested = pyqtSignal(str)  # "lock" | "unlock" | "controls"

    def __init__(self, params, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Đo màu 3 điểm · độ ổn định RGB / LAB")
        self.resize(1400, 950)
        self.rounds = []
        self.points = []
        self.labels = []
        self.folder = None
        self.running = False
        self.monitor = None
        self.monitor_dirty = False
        layout = QVBoxLayout(self)
        heading = QLabel("Đo màu 3 điểm trên tấm vải")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        hint = QLabel("Bấm chuột trái lên ảnh xem trước ở cửa sổ chính để đặt tối đa 3 điểm; khi đã đủ 3 điểm, "
                      "bấm gần một điểm để dời nó. Chuột phải để xóa điểm gần nhất. Nên khóa phơi sáng tự động (AE) "
                      "và cân bằng trắng tự động (AWB) của camera, và để đèn ổn định trước khi đo.")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        layout.addWidget(hint)

        top = QHBoxLayout()
        form = QGridLayout()
        self.roi = self._spin(QSpinBox(), 4, 400, params.roi, " px")
        self.frames = self._spin(QSpinBox(), 3, 3000, params.frames, " frame")
        self.interval = self._spin(QDoubleSpinBox(), 1, 3600, params.interval_s, " s")
        self.duration = self._spin(QDoubleSpinBox(), 0, 1440, params.duration_min, " phút")
        self.outlier_z = self._spin(QDoubleSpinBox(), 1, 50, params.outlier_z, "")
        self.outlier_de = self._spin(QDoubleSpinBox(), 0, 50, params.outlier_de, "")
        self.outlier_de.setSingleStep(0.1)
        self.settle_window = self._spin(QDoubleSpinBox(), 2, 120, params.settle_window_s, " s")
        self.settle_de = self._spin(QDoubleSpinBox(), 0.01, 20, params.settle_de, "")
        self.settle_de.setSingleStep(0.05)
        self.settle_timeout = self._spin(QDoubleSpinBox(), 5, 3600, params.settle_timeout_s, " s")
        self.formula = QComboBox()
        self.formula.addItem("ΔE00 (CIEDE2000)", "de00")
        self.formula.setCurrentIndex(max(0, self.formula.findData(params.de_formula)))
        self.white = QComboBox()
        self.white.addItem("D50 (Bradford, mặc định Nix)", "D50")
        self.white.addItem("D65", "D65")
        self.white.setCurrentIndex(max(0, self.white.findData(params.white)))
        self.roi_test = QCheckBox("Thử nhiều kích thước ROI tại cùng một điểm (chỉ dùng điểm 1):")
        self.roi_sizes = QLineEdit("31, 101, 201")
        self.roi_sizes.setPlaceholderText("ví dụ 31, 101, 201 (tối đa 3)")
        self.lock = QCheckBox("Khóa thông số camera trước khi đo (tắt AE/AWB, giữ exposure, gain, nhiệt độ màu)")
        self.lock.setChecked(params.lock)
        for i, (label, widget) in enumerate((("Kích thước ROI", self.roi), ("Số frame mỗi lần đo", self.frames),
                                             ("Chu kỳ đo", self.interval), ("Thời lượng phiên", self.duration),
                                             ("Frame rác: MAD z (RGB) >", self.outlier_z),
                                             ("và ΔE tới trung vị > (0 = chỉ MAD)", self.outlier_de),
                                             ("Chờ ổn định: cửa sổ", self.settle_window),
                                             ("ổn định khi dao động ΔE ≤", self.settle_de),
                                             ("Chờ ổn định tối đa", self.settle_timeout))):
            form.addWidget(QLabel(label), i // 2, (i % 2) * 2)
            form.addWidget(widget, i // 2, (i % 2) * 2 + 1)
        self.roi.valueChanged.connect(lambda value: self.roi_changed.emit(value))
        for widget in self._param_widgets():
            widget.valueChanged.connect(self._params_edited)
        self.lock.toggled.connect(self._params_edited)
        form.addWidget(QLabel("Công thức ΔE"), 5, 0)
        form.addWidget(self.formula, 5, 1)
        form.addWidget(QLabel("Chuẩn trắng Lab"), 5, 2)
        form.addWidget(self.white, 5, 3)
        self.formula.currentIndexChanged.connect(self._params_edited)
        self.white.currentIndexChanged.connect(self._params_edited)
        form.addWidget(self.lock, 6, 0, 1, 4)
        form.addWidget(self.roi_test, 7, 0, 1, 2)
        form.addWidget(self.roi_sizes, 7, 2, 1, 2)
        self.roi_test.toggled.connect(lambda _on: self.roi_changed.emit(self.roi.value()))
        self.roi_sizes.editingFinished.connect(lambda: self.roi_changed.emit(self.roi.value()))
        top.addLayout(form, 1)
        buttons = QVBoxLayout()
        self.start_button = QPushButton("Bắt đầu phiên đo")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(lambda: self.start_requested.emit(self.params()))
        self.stop_button = QPushButton("Dừng phiên")
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.clear_button = QPushButton("Xóa điểm")
        self.clear_button.clicked.connect(self.clear_points.emit)
        self.open_button = QPushButton("Mở thư mục phiên đo")
        self.open_button.clicked.connect(self._open_folder)
        self.lock_button = QPushButton("Khóa thông số ngay")
        self.lock_button.clicked.connect(lambda: self.controls_requested.emit("lock"))
        self.unlock_button = QPushButton("Trả AE/AWB về tự động")
        self.unlock_button.clicked.connect(lambda: self.controls_requested.emit("unlock"))
        for button in (self.start_button, self.stop_button, self.clear_button, self.lock_button, self.unlock_button,
                       self.open_button):
            buttons.addWidget(button)
        top.addLayout(buttons)
        layout.addLayout(top)
        self.plan = QLabel()
        self.plan.setObjectName("muted")
        layout.addWidget(self.plan)
        self.controls_label = QLabel("Thông số camera: —")
        self.controls_label.setObjectName("muted")
        self.controls_label.setWordWrap(True)
        layout.addWidget(self.controls_label)
        self.status = QLabel("Chưa đặt điểm.")
        self.status.setStyleSheet("font-size: 15px; font-weight: 600;")
        layout.addWidget(self.status)

        self.live_table = QTableWidget(0, 10)
        self.live_table.setHorizontalHeaderLabels(["Điểm", "Vị trí", "R", "G", "B", "L*", "a*", "b*",
                                                   "σR σG σB (30 frame gần nhất)", "Frame"])
        self.live_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.live_table.setMaximumHeight(135)
        self.live_table.verticalHeader().setVisible(False)
        layout.addWidget(self.live_table)

        self.relative = QCheckBox("Time series: vẽ độ lệch so với đầu phiên (dễ thấy trôi màu khi các điểm có màu khác nhau)")
        self.relative.setChecked(True)
        self.relative.toggled.connect(self._redraw_timeseries)
        layout.addWidget(self.relative)
        self.tabs = QTabWidget()
        self.rgb_series, self.lab_series = ChartCanvas(), ChartCanvas()
        self.rgb_chart, self.lab_chart, self.de_chart = ChartCanvas(), ChartCanvas(), ChartCanvas()
        self.tabs.addTab(self.rgb_series, "Time series RGB")
        self.tabs.addTab(self.lab_series, "Time series LAB")
        self.tabs.addTab(self.rgb_chart, "Phân phối frame RGB")
        self.tabs.addTab(self.lab_chart, "Phân phối frame LAB")
        self.tabs.addTab(self.de_chart, "ΔE · trôi màu & frame rác")
        self.round_table = QTableWidget(0, 14)
        self.round_table.setHorizontalHeaderLabels(["Lần", "Phút", "Điểm", "R", "G", "B", "σR/σG/σB", "L*", "a*", "b*",
                                                    "ΔE so với lần 1", "ΔE p95 trong lần", "Frame rác",
                                                    "Thông số camera"])
        self.round_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.round_table.setAlternatingRowColors(True)
        self.tabs.addTab(self.round_table, "Bảng các lần đo")
        self.summary_table = QTableWidget(0, 11)
        self.summary_table.setHorizontalHeaderLabels(["Điểm", "R̄ Ḡ B̄", "L* a* b*", "σ trong lần (RGB)",
                                                      "σ giữa các lần (RGB)", "Dốc RGB / phút", "ΔE p95 trong lần",
                                                      "Trôi ΔE cuối", "Trôi ΔE max", "Frame rác", "Clip %"])
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        summary_page = QWidget()
        summary_layout = QVBoxLayout(summary_page)
        summary_layout.addWidget(self.summary_table)
        self.pairs = QLabel()
        self.pairs.setWordWrap(True)
        summary_layout.addWidget(self.pairs)
        note = QLabel("σ trong lần: dao động giữa các frame liên tiếp (nhiễu, nhấp nháy đèn). σ giữa các lần và "
                      "Trôi ΔE: thay đổi trung bình theo thời gian (nhiệt, AWB/AE, đèn trôi). Trung bình N frame "
                      "giảm nhiễu ngẫu nhiên theo σ/√N nhưng không loại bỏ trôi màu.\n"
                      "Frame rác (MAD): z = |c − trung vị| / (1.4826·MAD + ε) trên R, G, B của từng lần đo; frame có kênh "
                      "vượt ngưỡng z (và ΔE tới trung vị vượt ngưỡng tối thiểu, nếu đặt) bị loại. Giá trị mỗi lần đo = "
                      "Mean các frame hợp lệ (MAD + Mean). MAD chỉ bắt frame bất thường trong một lần chụp; trôi màu dài "
                      "hạn được theo dõi riêng bằng time series và ΔE so với lần 1.\n"
                      "RGB từ YUYV BT.601 được coi là sRGB; Lab tính theo chuẩn trắng đã chọn (D50 dùng Bradford như Nix). Chưa hiệu chuẩn bằng bảng màu chuẩn, "
                      "nên dùng để so sánh tương đối.")
        note.setWordWrap(True)
        note.setObjectName("muted")
        summary_layout.addWidget(note)
        self.tabs.addTab(summary_page, "Tổng kết")
        self.method_table = QTableWidget(0, 7)
        self.method_table.setHorizontalHeaderLabels(["Điểm", "Phương pháp", "σ giữa các lần (R/G/B)", "CV % (R/G/B)",
                                                     "ΔE TB tới TB phiên", "Trôi ΔE max", "ΔE lệch so với Mean"])
        self.method_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.method_table.setAlternatingRowColors(True)
        self.tabs.addTab(self.method_table, "So sánh Mean / Median / Trimmed / MAD")
        layout.addWidget(self.tabs, 1)
        self.folder_label = QLabel("")
        self.folder_label.setObjectName("muted")
        self.folder_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.folder_label)
        self._params_edited()
        self._redraw()
        self._redraw_timeseries()
        # Time series refresh between rounds; the 1 s monitor series arrives continuously.
        self.series_timer = QTimer(self)
        self.series_timer.timeout.connect(self._tick)
        self.series_timer.start(3000)

    @staticmethod
    def _spin(widget, low, high, value, suffix):
        widget.setRange(low, high)
        widget.setValue(value)
        widget.setSuffix(suffix)
        return widget

    def _param_widgets(self):
        return (self.roi, self.frames, self.interval, self.duration, self.outlier_z, self.outlier_de,
                self.settle_window, self.settle_de, self.settle_timeout)

    def roi_test_sizes(self):
        """Sizes for the ROI experiment, or [] when it is off or the text is invalid."""
        if not self.roi_test.isChecked():
            return []
        try:
            sizes = [int(v) for v in self.roi_sizes.text().replace(";", ",").split(",") if v.strip()]
        except ValueError:
            return []
        return [min(400, max(4, v)) for v in sizes][:3]

    def _label(self, index):
        return self.labels[index] if index < len(self.labels) else f"Điểm {index + 1}"

    def params(self):
        return SessionParams(roi=self.roi.value(), frames=self.frames.value(), interval_s=self.interval.value(),
                             duration_min=self.duration.value(), outlier_z=self.outlier_z.value(),
                             outlier_de=self.outlier_de.value(), lock=self.lock.isChecked(),
                             settle_window_s=self.settle_window.value(), settle_de=self.settle_de.value(),
                             settle_timeout_s=self.settle_timeout.value(), de_formula=self.formula.currentData(),
                             white=self.white.currentData(), rois=tuple(self.roi_test_sizes()))

    def _params_edited(self):
        p = self.params()
        self.plan.setText(f"{FORMULAS[p.de_formula]} · Lab {p.white}/2° · Kế hoạch: chờ ảnh ổn định{' → khóa thông số → chờ ổn định lại' if p.lock else ''}, rồi "
                          f"{p.rounds_total} lần đo (phút 0 đến phút {p.duration_min:g}, mỗi {p.interval_s:g} s), "
                          f"mỗi lần {p.frames} frame liên tiếp.")
        self.params_changed.emit(p)

    # ----- session lifecycle (called by the main window) -----
    def begin_session(self, folder, points, params, camera, mode):
        self.rounds = []
        self.monitor = None
        self.points = list(points)
        self.folder = Path(folder)
        self.session_info = (list(points), params, camera, mode)
        write_session_info(self.folder, points, params, camera, mode)
        self.folder_label.setText(f"Kết quả phiên đo: {self.folder}")
        self._redraw()

    def set_meta(self, meta):
        """Record stabilisation and locked controls once the session actually starts."""
        if self.folder is not None:
            write_session_info(self.folder, *self.session_info, meta=meta)

    def add_round(self, result):
        if self.folder is None:
            return
        self.rounds.append(result)
        export_round(self.folder, result)
        export_summary(self.folder, self.rounds)
        if self.monitor is not None and len(self.monitor[0]):
            export_monitor(self.folder, *self.monitor)
        self._redraw()
        self._redraw_timeseries()
        for name, chart in (("timeseries_rgb.png", self.rgb_series), ("timeseries_lab.png", self.lab_series),
                            ("distribution_rgb.png", self.rgb_chart), ("distribution_lab.png", self.lab_chart),
                            ("chart_de00.png", self.de_chart)):
            chart.figure.savefig(self.folder / name, dpi=110, facecolor=BG)

    def update_live(self, state):
        self.points = state["points"]
        self.labels = state["params"].labels(len(self.points))
        session = state["session"]
        if state.get("monitor") is not None and self.folder is not None:
            if self.monitor is None or len(state["monitor"][0]) != len(self.monitor[0]):
                self.monitor_dirty = True
            self.monitor = state["monitor"]
        prep = state.get("prep")
        self.running = bool(prep or (session and session["state"] == "running"))
        for widget in (*self._param_widgets(), self.lock, self.formula, self.white, self.lock_button, self.unlock_button):
            widget.setEnabled(not self.running)
        self.start_button.setEnabled(not self.running and bool(self.points))
        self.stop_button.setEnabled(self.running)
        self.clear_button.setEnabled(not self.running and bool(self.points))
        if state.get("controls"):
            self.controls_label.setText("Thông số camera: " + state["controls"])
        if prep is not None and prep["phase"] == "match":
            self.status.setText("Đã khóa · đang dò thông số để ảnh khớp với lúc tự động · " + prep["description"])
        elif prep is not None:
            phase = {"settle_auto": "Chờ ổn định (camera đang tự động) trước khi khóa",
                     "settle_locked": "Đã khóa thông số · chờ ổn định lại",
                     "settle": "Chờ ảnh ổn định"}[prep["phase"]]
            spread = "đang gom dữ liệu" if prep["spread"] is None else f"dao động ΔE {prep['spread']:.2f}"
            self.status.setText(f"{phase} · {spread} / ngưỡng {prep['threshold']:.2f} · cửa sổ "
                                f"{prep['filled']}/{prep['window']} s · {_clock(prep['elapsed'])} / {_clock(prep['timeout'])}")
        elif session is None:
            self.status.setText(f"Đã đặt {len(self.points)}/3 điểm. Bấm Bắt đầu phiên đo khi sẵn sàng."
                                if self.points else "Chưa đặt điểm.")
        else:
            garbage = sum(len(r.garbage_frames()) for r in self.rounds)
            elapsed = _clock(session["elapsed"])
            if session["state"] == "running":
                step = (f"đang chụp {session['collecting']}/{state['params'].frames} frame" if session["collecting"]
                        else f"lần tiếp theo sau {_clock(session['next_in'])}")
                text = f"Đang đo · {elapsed} · đã xong {session['done']}/{session['total']} lần · {step}"
            else:
                text = ("Hoàn tất" if session["state"] == "done" else "Đã dừng") + \
                       f" · {session['done']}/{session['total']} lần đo"
            self.status.setText(f"{text} · frame rác: {garbage}")
        live = state["live"]
        self.live_table.setRowCount(len(self.points))
        for i, point in enumerate(self.points):
            values = [self._label(i), f"{point[0] * 100:.1f}%, {point[1] * 100:.1f}%"]
            if live is not None and i < len(live["rgb"]):
                values += [f"{v:.1f}" for v in live["rgb"][i]] + [f"{v:.2f}" for v in live["lab"][i]]
                values += [" / ".join(f"{v:.2f}" for v in live["std_rgb"][i]), str(live["n"])]
            for j, value in enumerate(values):
                item = QTableWidgetItem(value)
                if j == 0:
                    item.setForeground(_qcolor(POINT_COLORS[i]))
                self.live_table.setItem(i, j, item)
        self.live_table.resizeColumnsToContents()

    def _open_folder(self):
        if self.folder is not None and self.folder.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.folder)))

    def _tick(self):
        if self.monitor_dirty:
            self.monitor_dirty = False
            self._redraw_timeseries()
            if self.folder is not None and not self.running and self.monitor is not None:
                try:
                    export_monitor(self.folder, *self.monitor)
                except OSError:
                    pass  # the folder may have been removed; round exports report their own errors

    def _redraw_timeseries(self):
        have_monitor = self.monitor is not None and len(self.monitor[0]) > 0
        if not have_monitor and not self.rounds:
            for chart in (self.rgb_series, self.lab_series):
                chart.empty("Time series bắt đầu khi phiên đo chạy (giá trị trung bình mỗi giây).")
            return
        self._timeseries(self.rgb_series, 1, "rgb", "mean_rgb", "std_rgb", ("R", "G", "B"), RGB_COLORS)
        self._timeseries(self.lab_series, 2, "lab", "mean_lab", "std_lab", ("L*", "a*", "b*"), LAB_COLORS)

    def _timeseries(self, chart, monitor_index, key, mean_key, std_key, labels, colors):
        """One panel per channel: 1 s line per point, round means ± σ, garbage frames as red crosses."""
        fig = chart.figure
        fig.clear()
        axes = fig.subplots(3, 1, sharex=True)
        minutes = self.monitor[0] if self.monitor is not None else np.empty(0)
        series = self.monitor[monitor_index] if self.monitor is not None else np.empty((0, 0, 3))
        count = series.shape[1] if len(series) else self.rounds[0].rgb.shape[1]
        relative = self.relative.isChecked()
        for c, ax in enumerate(axes):
            _style(ax)
            for p in range(min(count, len(POINT_COLORS))):
                if not relative:
                    base = 0.0
                elif len(series):
                    base = series[0, p, c]
                else:
                    base = getattr(self.rounds[0], mean_key)[p, c]
                if len(series):
                    ax.plot(minutes, series[:, p, c] - base, color=POINT_COLORS[p], linewidth=1.1,
                            label=self._label(p))
                if self.rounds:
                    ax.errorbar([r.t_min for r in self.rounds], [getattr(r, mean_key)[p, c] - base for r in self.rounds],
                                yerr=[getattr(r, std_key)[p, c] for r in self.rounds], fmt="o", markersize=4,
                                color=POINT_COLORS[p], ecolor=POINT_COLORS[p], elinewidth=1, capsize=3,
                                label=None if len(series) else self._label(p))
                    for r in self.rounds:
                        bad = r.outlier_channel[:, p, c] if key == "rgb" else r.outlier[:, p]
                        if bad.any():
                            ax.scatter(r.t_min + r.times[bad] / 60, getattr(r, key)[bad, p, c] - base, s=40,
                                       marker="x", color=OUTLIER_COLOR, linewidths=1.5, zorder=5)
            if relative:
                ax.axhline(0, color="#8899aa", linewidth=0.8, linestyle="--")
            ax.set_ylabel(f"Δ{labels[c]}" if relative else labels[c], color=colors[c], fontsize=10)
        axes[0].set_title("Đường: trung bình mỗi giây · chấm: trung bình mỗi lần đo ± σ giữa các frame · × frame rác",
                          fontsize=9)
        axes[-1].set_xlabel("Thời gian từ đầu phiên (phút)", fontsize=9)
        legend = axes[0].legend(fontsize=8, facecolor=BG, edgecolor=GRID, loc="upper left")
        for text in legend.get_texts():
            text.set_color(FG)
        chart.draw_idle()

    # ----- charts and tables -----
    def _redraw(self):
        if not self.rounds:
            for chart in (self.rgb_chart, self.lab_chart, self.de_chart):
                chart.empty("Biểu đồ được cập nhật sau mỗi lần đo.")
        else:
            self._distribution(self.rgb_chart, "rgb", "mean_rgb", ("R", "G", "B"), RGB_COLORS)
            self._distribution(self.lab_chart, "lab", "mean_lab", ("L*", "a*", "b*"), LAB_COLORS)
            self._delta_e()
        self._tables()

    def _distribution(self, chart, key, mean_key, labels, colors):
        """Box plot of every frame in each round, frames as dots, garbage frames as red crosses."""
        fig = chart.figure
        fig.clear()
        count = self.rounds[0].rgb.shape[1]
        axes = fig.subplots(count, 3, squeeze=False, sharex=True)
        minutes = np.array([r.t_min for r in self.rounds])
        width = max(0.25, (np.diff(minutes).min() if len(minutes) > 1 else 1) * 0.45)
        rng = np.random.default_rng(0)
        for p in range(count):
            for c in range(3):
                ax = axes[p][c]
                _style(ax)
                data = [getattr(r, key)[:, p, c] for r in self.rounds]
                box = ax.boxplot(data, positions=minutes, widths=width, showfliers=False, manage_ticks=False,
                                 patch_artist=True, medianprops=dict(color="white", linewidth=1.2),
                                 whiskerprops=dict(color=colors[c]), capprops=dict(color=colors[c]))
                for patch in box["boxes"]:
                    patch.set(facecolor=colors[c] + "55", edgecolor=colors[c])
                for r, values in zip(self.rounds, data):
                    bad = r.outlier_channel[:, p, c] if key == "rgb" else r.outlier[:, p]
                    x = r.t_min + rng.uniform(-width / 2, width / 2, len(values))
                    ax.scatter(x[~bad], values[~bad], s=5, color=colors[c], alpha=0.45, linewidths=0)
                    if bad.any():
                        ax.scatter(x[bad], values[bad], s=34, marker="x", color=OUTLIER_COLOR, linewidths=1.4)
                ax.plot(minutes, [getattr(r, mean_key)[p, c] for r in self.rounds], color=POINT_COLORS[p],
                        linewidth=1.2, marker="o", markersize=3)
                ax.set_title(f"{self._label(p)} · {labels[c]}", fontsize=9, color=POINT_COLORS[p])
                if p == count - 1:
                    ax.set_xlabel("Thời gian (phút)", fontsize=8)
        chart.draw_idle()

    def _delta_e(self):
        fig = self.de_chart.figure
        fig.clear()
        count = self.rounds[0].rgb.shape[1]
        drift_ax, frame_ax = fig.subplots(1, 2)
        for ax in (drift_ax, frame_ax):
            _style(ax)
        minutes = np.array([r.t_min for r in self.rounds])
        means = np.array([r.mean_lab for r in self.rounds])
        monitor = self.monitor if self.monitor is not None and len(self.monitor[0]) else None
        formula, name = self.rounds[0].formula, FORMULAS[self.rounds[0].formula]
        for p in range(count):
            if monitor is not None:
                drift_ax.plot(monitor[0], delta_e(monitor[2][:, p], means[0, p][None], formula), color=POINT_COLORS[p],
                              linewidth=0.9, alpha=0.7)
            drift = delta_e(means[:, p], means[0, p][None], formula)
            drift_ax.plot(minutes, drift, marker="o", markersize=4, linestyle="none", color=POINT_COLORS[p],
                          label=self._label(p))
            for r in self.rounds:
                values, bad = r.de_median[:, p], r.outlier[:, p]
                x = r.t_min + (p - 1) * 0.12 + np.linspace(-0.05, 0.05, len(values))
                frame_ax.scatter(x[~bad], values[~bad], s=6, color=POINT_COLORS[p], alpha=0.5, linewidths=0,
                                 label=self._label(p) if r is self.rounds[0] else None)
                if bad.any():
                    frame_ax.scatter(x[bad], values[bad], s=36, marker="x", color=OUTLIER_COLOR, linewidths=1.4)
        drift_ax.axhline(1.0, color="#8899aa", linestyle="--", linewidth=0.8)
        drift_ax.set_title(f"Trôi màu: {name} trung bình mỗi lần so với lần 1", fontsize=10)
        drift_ax.set_xlabel("Thời gian (phút)", fontsize=8)
        frame_ax.axhline(self.outlier_de.value(), color=OUTLIER_COLOR, linestyle="--", linewidth=0.8)
        frame_ax.set_title(f"Phân phối frame: {name} từng frame so với trung vị lần đo (× = frame rác)", fontsize=10)
        frame_ax.set_xlabel("Thời gian (phút)", fontsize=8)
        for ax in (drift_ax, frame_ax):
            legend = ax.legend(fontsize=8, facecolor=BG, edgecolor=GRID)
            for text in legend.get_texts():
                text.set_color(FG)
        self.de_chart.draw_idle()

    def _tables(self):
        rows = []
        for r in self.rounds:
            for p in range(r.rgb.shape[1]):
                drift = float(delta_e(r.mean_lab[p], self.rounds[0].mean_lab[p], r.formula))
                rows.append([str(r.index + 1), f"{r.t_min:.2f}", self._label(p), *(f"{v:.2f}" for v in r.mean_rgb[p]),
                             " / ".join(f"{v:.2f}" for v in r.std_rgb[p]), *(f"{v:.2f}" for v in r.mean_lab[p]),
                             f"{drift:.3f}", f"{np.percentile(r.de_median[:, p], 95):.3f}",
                             f"{int(r.outlier[:, p].sum())}/{r.frame_count}"
                             + (f" · lỗi decode {r.decode_errors}" if r.decode_errors else ""),
                             "ĐÃ ĐỔI: " + ", ".join(r.control_changes) if r.control_changes else "giữ nguyên"])
        _fill(self.round_table, rows, bad_column=12)
        summary, pairs = session_summary(self.rounds)
        rows = [[self._label(s["point"] - 1), " ".join(f"{v:.1f}" for v in s["mean_rgb"]), " ".join(f"{v:.2f}" for v in s["mean_lab"]),
                 " / ".join(f"{v:.2f}" for v in s["within_std_rgb"]), " / ".join(f"{v:.2f}" for v in s["between_std_rgb"]),
                 " / ".join("—" if np.isnan(v) else f"{v:+.3f}" for v in s["slope_rgb_per_min"]),
                 f"{s['within_de_p95']:.3f}", f"{s['drift_de_last']:.3f}", f"{s['drift_de_max']:.3f}",
                 f"{s['outliers']}/{s['frames']}", f"{s['clip_pct']:.2f}"] for s in summary]
        _fill(self.summary_table, rows, bad_column=9)
        rows = [[self._label(m["point"] - 1), m["method"], " / ".join(f"{v:.3f}" for v in m["between_std_rgb"]),
                 " / ".join(f"{v:.3f}" for v in m["cv_pct"]), f"{m['spread_de']:.3f}", f"{m['drift_de_max']:.3f}",
                 f"{m['shift_from_mean_de']:.3f}"] for m in method_comparison(self.rounds)]
        _fill(self.method_table, rows, bad_column=-1)
        self.pairs.setText(f"{FORMULAS[self.rounds[0].formula] if self.rounds else 'ΔE'} giữa các điểm (trung bình cả phiên): " +
                           ("  ·  ".join(f"điểm {k}: {v:.3f}" for k, v in pairs.items()) if pairs else "—"))


def _fill(table, rows, bad_column):
    table.setRowCount(len(rows))
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            item = QTableWidgetItem(value)
            if j == bad_column and not value.startswith("0/"):
                item.setForeground(_qcolor(OUTLIER_COLOR))
            table.setItem(i, j, item)
    table.resizeColumnsToContents()
    if rows:
        table.scrollToBottom()


def _qcolor(hex_color):
    from PyQt5.QtGui import QColor
    return QColor(hex_color)


def _clock(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def session_folder(outdir, node):
    return Path(outdir).expanduser() / f"measure_{datetime.now():%Y%m%d_%H%M%S}_{Path(node).name}"
