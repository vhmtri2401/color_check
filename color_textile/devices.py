"""Discover V4L2 capture nodes and the modes advertised by each driver."""

import glob
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CameraMode:
    fourcc: str
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class CameraDevice:
    node: str
    identity: str
    name: str
    modes: tuple

    @property
    def label(self):
        return f"{self.name}  ·  {self.node}"


def parse_formats(output):
    """Keep every discrete size/interval, removing duplicate driver entries."""
    modes = []
    fourcc = None
    size = None
    for line in output.splitlines():
        fmt = re.search(r"\[\d+\]:\s*'([^']+)'", line)
        if fmt:
            fourcc, size = fmt[1].strip(), None
            continue
        match = re.search(r"Size: Discrete (\d+)x(\d+)", line)
        if match:
            size = int(match[1]), int(match[2])
            continue
        if "Size:" in line:
            size = None
        fps = re.search(r"Interval: Discrete.*\(([\d.]+) fps\)", line)
        if fps and size and fourcc in ("MJPG", "YUYV"):
            value = CameraMode(fourcc, *size, float(fps[1]))
            if value.fps > 0 and value.width % 2 == 0 and value not in modes:
                modes.append(value)
    return tuple(sorted(modes, key=lambda m: (m.fourcc, m.width * m.height, m.width, m.height, -m.fps)))


def _query(*args):
    try:
        result = subprocess.run(["v4l2-ctl", *args], capture_output=True, text=True, timeout=4)
    except FileNotFoundError as exc:
        raise RuntimeError("Cần v4l2-ctl để đọc đúng các mode camera. Cài gói v4l-utils.") from exc
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Không truy vấn được camera.")
    return result.stdout


def discover_cameras():
    """Metadata nodes have no image modes and are excluded without opening them."""
    identities = {}
    for folder in ("/dev/v4l/by-path", "/dev/v4l/by-id"):
        for entry in sorted(glob.glob(folder + "/*")):
            identities[os.path.realpath(entry)] = entry
    cameras, failures = [], []
    nodes = sorted(glob.glob("/dev/video*"), key=lambda p: int(re.search(r"\d+$", p)[0]))
    for node in nodes:
        try:
            modes = parse_formats(_query("-d", node, "--list-formats-ext"))
        except subprocess.TimeoutExpired:
            failures.append(f"{node}: quá thời gian truy vấn")
            continue
        except RuntimeError as exc:
            failures.append(f"{node}: {exc}")
            continue
        if not modes:
            continue
        name_file = Path("/sys/class/video4linux") / Path(node).name / "name"
        try:
            name = name_file.read_text().strip()
            parts = name.split(": ", 1)
            if len(parts) == 2 and parts[0] == parts[1]:
                name = parts[0]
        except OSError:
            name = Path(node).name
        cameras.append(CameraDevice(node, identities.get(node, node), name, modes))
    return cameras, failures


def find_device(cameras, identity):
    return next((c for c in cameras if identity in (c.identity, c.node)), None)


def select_mode(camera, fourcc, width, height, fps):
    """Preserve the current size/rate where available, otherwise choose a real mode."""
    candidates = [m for m in camera.modes if m.fourcc == fourcc] or list(camera.modes)
    if not candidates:
        raise ValueError("Camera không có mode MJPG/YUYV được hỗ trợ.")
    return min(candidates, key=lambda m: (abs(m.width * m.height - width * height),
                                        abs(m.width - width) + abs(m.height - height), abs(m.fps - fps)))
