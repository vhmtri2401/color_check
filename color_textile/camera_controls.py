"""Read, lock and restore UVC image controls through v4l2-ctl.

Controls can be changed while another handle streams from the device, so the
camera worker keeps its OpenCV capture open while locking.
"""

import re
import subprocess

# Controls that drift on their own when left automatic, and the value meaning "manual".
AUTO_EXPOSURE_MANUAL = 1
AUTO_EXPOSURE_AUTO = 3  # aperture priority, the usual UVC default
MAX_EXPOSURE_RESIDUAL_L = 2.0  # L* error above which a manual exposure cannot replace AE
LOCKED_KEYS = ("auto_exposure", "exposure_time_absolute", "exposure_dynamic_framerate", "gain",
               "white_balance_automatic", "white_balance_temperature", "backlight_compensation")

_LINE = re.compile(r"^\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)$")


def parse_controls(text):
    """Parse `v4l2-ctl -l` output into {name: {"type", "value", "min", "max", "default", "inactive"}}."""
    controls = {}
    for line in text.splitlines():
        match = _LINE.match(line)
        if not match:
            continue
        name, kind, rest = match.groups()
        fields = dict(re.findall(r"(\w+)=(-?\d+)", rest))
        if "value" not in fields:
            continue
        controls[name] = dict(type=kind, value=int(fields["value"]),
                              min=int(fields["min"]) if "min" in fields else None,
                              max=int(fields["max"]) if "max" in fields else None,
                              default=int(fields["default"]) if "default" in fields else None,
                              inactive="inactive" in rest)
    return controls


def read_controls(node):
    result = subprocess.run(["v4l2-ctl", "-d", node, "-l"], capture_output=True, text=True, timeout=5)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"v4l2-ctl không đọc được thông số {node}")
    return parse_controls(result.stdout)


def set_controls(node, values):
    """Apply controls one by one (the order matters: auto modes before manual values)."""
    failures = []
    for name, value in values.items():
        result = subprocess.run(["v4l2-ctl", "-d", node, "-c", f"{name}={int(value)}"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            failures.append(f"{name}: {result.stderr.strip() or 'lỗi'}")
    return failures


def values(controls, keys=LOCKED_KEYS):
    return {k: controls[k]["value"] for k in keys if k in controls}


def lock_controls(node):
    """Freeze exposure, gain and white balance at the values the automatic loops settled on.

    Returns (locked values, warnings). Values read while a control is automatic are what the
    driver reports; most UVC cameras report the live auto value, a few report the last manual one.
    """
    before = read_controls(node)
    plan = {}
    if "auto_exposure" in before:
        plan["auto_exposure"] = AUTO_EXPOSURE_MANUAL
    if "exposure_dynamic_framerate" in before:
        plan["exposure_dynamic_framerate"] = 0
    if "exposure_time_absolute" in before:
        plan["exposure_time_absolute"] = before["exposure_time_absolute"]["value"]
    if "white_balance_automatic" in before:
        plan["white_balance_automatic"] = 0
    if "white_balance_temperature" in before:
        plan["white_balance_temperature"] = before["white_balance_temperature"]["value"]
    if "gain" in before:
        plan["gain"] = before["gain"]["value"]
    if "backlight_compensation" in before:
        plan["backlight_compensation"] = 0
    warnings = set_controls(node, plan)
    after = read_controls(node)
    locked = values(after)
    for name, wanted in plan.items():
        if name in locked and locked[name] != wanted:
            warnings.append(f"{name} = {locked[name]} (yêu cầu {wanted})")
    missing = [k for k in ("auto_exposure", "white_balance_automatic") if k not in before]
    if missing:
        warnings.append("Camera không có điều khiển: " + ", ".join(missing))
    return dict(before=values(before), locked=locked, parsed_before=before), warnings


def unlock_controls(node):
    current = read_controls(node)
    plan = {}
    if "auto_exposure" in current:
        plan["auto_exposure"] = AUTO_EXPOSURE_AUTO
    if "white_balance_automatic" in current:
        plan["white_balance_automatic"] = 1
    return set_controls(node, plan)


def changed_controls(node, locked):
    """Locked controls whose value differs now: {name: (locked, now)}."""
    now = values(read_controls(node))
    return {k: (v, now.get(k)) for k, v in locked.items() if now.get(k) != v}


def describe(controls_values):
    names = dict(auto_exposure="AE", exposure_time_absolute="exposure", gain="gain",
                 white_balance_automatic="AWB", white_balance_temperature="WB K", exposure_dynamic_framerate="dyn FPS")
    mode = {1: "tay", 3: "tự động", 0: "tự động", 2: "ưu tiên shutter"}
    parts = []
    for key, label in names.items():
        if key in controls_values:
            v = controls_values[key]
            if key == "auto_exposure":
                parts.append(f"{label}: {mode.get(v, v)}")
            elif key == "white_balance_automatic":
                parts.append(f"{label}: {'bật' if v else 'tắt'}")
            else:
                parts.append(f"{label}: {v}")
    return " · ".join(parts) if parts else "không đọc được thông số"


class AutoLockMatcher:
    """Search manual exposure / white-balance values that reproduce the auto-settled image.

    Drivers that report a stale value while AE/AWB is on (flag "inactive") would otherwise
    jump to a different image when locked. Fed one L*a*b* sample (P, 3) per frame.
    """

    def __init__(self, node, target_lab, before, start, settle_s=0.6, frames=6, max_steps=12):
        import numpy as np
        self.np = np
        self.node = node
        self.target = np.asarray(target_lab)
        self.settle_s = settle_s
        self.frames = frames
        self.max_steps = max_steps
        self.match_exposure = before.get("auto_exposure", {}).get("value", AUTO_EXPOSURE_MANUAL) != AUTO_EXPOSURE_MANUAL \
            and "exposure_time_absolute" in before
        self.match_wb = before.get("white_balance_automatic", {}).get("value", 0) == 1 \
            and "white_balance_temperature" in before
        self.before = before
        self.exposure_failed = False
        self.result = {}
        self.step = 0
        self.description = ""
        self._buffer = []
        self._settle_until = start
        self._search = self._plan()
        self._request = None
        self.done = not (self.match_exposure or self.match_wb)
        if not self.done:
            self._advance(None, start)

    def _error_l(self, lab):
        return float(lab[:, 0].mean() - self.target[:, 0].mean())

    def _error_chroma(self, lab):
        """ΔE00 with lightness taken from the target, so only a*/b* (white balance) count."""
        from .colorimetry import delta_e2000
        lab = self.np.array(lab, dtype=float)
        lab[:, 0] = self.target[:, 0]
        return float(delta_e2000(lab, self.target).mean())

    def _plan(self):
        np = self.np
        if self.match_exposure:
            info = self.before["exposure_time_absolute"]
            lo, hi = np.log(max(1, info["min"])), np.log(info["max"])
            best = None
            for i in range(self.max_steps):
                value = int(round(np.exp((lo + hi) / 2)))
                self.description = f"dò exposure {i + 1}/{self.max_steps}: {value}"
                lab = yield ("exposure_time_absolute", value)
                error = self._error_l(lab)
                if best is None or abs(error) < abs(best[1]):
                    best = (value, error)
                if abs(error) < 0.5 or hi - lo < 0.02:
                    break
                if error > 0:
                    hi = np.log(value)
                else:
                    lo = np.log(value)
            matched = abs(best[1]) <= MAX_EXPOSURE_RESIDUAL_L
            self.result["exposure_time_absolute"] = dict(value=best[0], residual_dL=round(best[1], 3), matched=matched)
            if matched:
                yield ("exposure_time_absolute", best[0], False)
            else:
                # Auto exposure uses something manual mode cannot reach (often hidden gain): keep AE on.
                self.exposure_failed = True
                yield ("exposure_time_absolute", self.before["exposure_time_absolute"]["value"], False)
                yield ("auto_exposure", self.before["auto_exposure"]["value"], False)
                self.settle_s = max(self.settle_s, 1.5)
        if self.match_wb:
            info = self.before["white_balance_temperature"]
            step = 10
            a, b = info["min"], info["max"]
            ratio = (5 ** 0.5 - 1) / 2
            cache = {}
            for i in range(self.max_steps):
                if b - a <= 2 * step:
                    break
                c, d = round((b - ratio * (b - a)) / step) * step, round((a + ratio * (b - a)) / step) * step
                for value in (c, d):
                    if value not in cache:
                        self.description = f"dò nhiệt độ màu {i + 1}/{self.max_steps}: {value} K"
                        cache[value] = self._error_chroma((yield ("white_balance_temperature", value)))
                if cache[c] < cache[d]:
                    b = d
                else:
                    a = c
            best = min(cache, key=cache.get)
            self.result["white_balance_temperature"] = dict(value=best, residual_de00=round(cache[best], 3))
            yield ("white_balance_temperature", best, False)

    def _advance(self, lab, now):
        while True:
            try:
                request = self._search.send(lab) if self._request is not None else next(self._search)
            except StopIteration:
                self.done = True
                return
            lab = None
            set_controls(self.node, {request[0]: request[1]})
            self._request = request
            if len(request) == 2:  # needs a measurement
                self._settle_until = now + self.settle_s
                self._buffer = []
                return
            self._request = request  # final value set; continue to the next stage

    def add(self, now, lab):
        """Return True when the search has finished."""
        if self.done:
            return True
        if now < self._settle_until:
            return False
        self._buffer.append(lab)
        if len(self._buffer) >= self.frames:
            self.step += 1
            self._advance(self.np.mean(self._buffer, axis=0), now)
        return self.done
