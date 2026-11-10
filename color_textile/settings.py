"""Validated application preferences, independent of the interface."""

import json
import math
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "color_textile/settings.json"


@dataclass
class AppSettings:
    camera_id: str = ""
    fourcc: str = "MJPG"
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    method: str = "imdecode_cvt"
    view: str = "reconstructed"
    jpeg_quality: int = 90
    warmup: float = 1.5
    outdir: str = str(ROOT / "captures")
    preview_fps: int = 30
    preview_width: int = 1280
    measure_roi: int = 31
    measure_frames: int = 30
    measure_interval_s: float = 120.0
    measure_duration_min: float = 30.0
    measure_outlier_z: float = 3.5
    measure_outlier_de: float = 1.0
    measure_lock: bool = True
    measure_settle_s: float = 5.0
    measure_settle_de: float = 0.3
    measure_settle_timeout_s: float = 60.0
    measure_de_formula: str = "de00"
    measure_white: str = "D50"
    uniform_inset_pct: float = 5.0
    uniform_width_pct: float = 5.0
    uniform_segments: int = 12
    uniform_frames: int = 30
    uniform_threshold_de: float = 0.4
    uniform_lock: bool = True
    uniform_interval_s: float = 120.0
    uniform_duration_min: float = 0.0
    uniform_filter_dirty: bool = True

    def validate(self):
        if self.fourcc not in ("MJPG", "YUYV"):
            raise ValueError("Định dạng phải là MJPG hoặc YUYV.")
        if type(self.width) is not int or type(self.height) is not int or self.width <= 0 or self.height <= 0 or self.width % 2:
            raise ValueError("Kích thước ảnh không hợp lệ; chiều rộng YUYV phải chẵn.")
        if not math.isfinite(self.fps) or not 0 < self.fps <= 1000:
            raise ValueError("FPS không hợp lệ.")
        if type(self.jpeg_quality) is not int or not 5 <= self.jpeg_quality <= 100:
            raise ValueError("Chất lượng JPEG phải nằm trong khoảng 5–100.")
        if not math.isfinite(self.warmup) or not 0 <= self.warmup <= 30:
            raise ValueError("Thời gian ổn định phải nằm trong khoảng 0–30 giây.")
        if type(self.preview_fps) is not int or not 1 <= self.preview_fps <= 60:
            raise ValueError("FPS xem trước phải nằm trong khoảng 1–60.")
        if type(self.preview_width) is not int or not 320 <= self.preview_width <= 3840:
            raise ValueError("Chiều rộng xem trước phải nằm trong khoảng 320–3840.")
        if self.view not in ("reconstructed", "source", "luma", "difference"):
            raise ValueError("Chế độ xem không hợp lệ.")
        if not isinstance(self.camera_id, str) or not isinstance(self.method, str):
            raise ValueError("Lựa chọn camera/decoder không hợp lệ.")
        if type(self.measure_roi) is not int or not 4 <= self.measure_roi <= 400:
            raise ValueError("Kích thước ROI đo màu phải nằm trong khoảng 4–400 px.")
        if type(self.measure_frames) is not int or not 3 <= self.measure_frames <= 3000:
            raise ValueError("Số frame mỗi lần đo phải nằm trong khoảng 3–3000.")
        for value, low, high, name in ((self.measure_interval_s, 1, 3600, "Chu kỳ đo"),
                                       (self.measure_duration_min, 0, 1440, "Thời lượng phiên đo"),
                                       (self.measure_outlier_z, 1, 50, "Ngưỡng z"),
                                       (self.measure_outlier_de, 0, 50, "Ngưỡng ΔE00"),
                                       (self.measure_settle_s, 2, 120, "Cửa sổ ổn định"),
                                       (self.measure_settle_de, 0.01, 20, "Ngưỡng ổn định ΔE00"),
                                       (self.measure_settle_timeout_s, 5, 3600, "Thời gian chờ ổn định"),
                                       (self.uniform_inset_pct, 1, 45, "% khoảng cách biên"),
                                       (self.uniform_width_pct, 1, 40, "% bề rộng vòng"),
                                       (self.uniform_threshold_de, 0.05, 20, "Ngưỡng đồng màu ΔE"),
                                       (self.uniform_interval_s, 1, 3600, "Chu kỳ chụp lại đồng màu"),
                                       (self.uniform_duration_min, 0, 1440, "Thời lượng theo dõi đồng màu")):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} phải nằm trong khoảng {low}–{high}.")
        if type(self.uniform_segments) is not int or not 4 <= self.uniform_segments <= 36:
            raise ValueError("Số đoạn trên vòng phải nằm trong khoảng 4–36.")
        if type(self.uniform_frames) is not int or not 3 <= self.uniform_frames <= 3000:
            raise ValueError("Số frame đo đồng màu phải nằm trong khoảng 3–3000.")
        if self.measure_de_formula not in ("de76", "de00") or self.measure_white not in ("D50", "D65"):
            raise ValueError("Công thức ΔE hoặc chuẩn trắng Lab không hợp lệ.")
        if (not isinstance(self.measure_lock, bool)
                or not isinstance(self.uniform_lock, bool)
                or not isinstance(self.uniform_filter_dirty, bool)):
            raise ValueError("Tùy chọn khóa thông số camera hoặc lọc điểm dơ không hợp lệ.")
        if not isinstance(self.outdir, str) or not self.outdir.strip():
            raise ValueError("Hãy chọn thư mục lưu ảnh.")
        return self


def load_settings(path=DEFAULT_SETTINGS_PATH):
    path = Path(path)
    if not path.exists():
        return AppSettings(), ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        allowed = {f.name for f in fields(AppSettings)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        if filtered.get("measure_de_formula") == "de76":
            filtered["measure_de_formula"] = "de00"
        result = AppSettings(**filtered)
        return result.validate(), ""
    except (ValueError, TypeError, AttributeError, OSError) as exc:
        return AppSettings(), f"Không đọc được cài đặt; dùng mặc định. {exc}"


def save_settings(settings, path=DEFAULT_SETTINGS_PATH):
    settings.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
