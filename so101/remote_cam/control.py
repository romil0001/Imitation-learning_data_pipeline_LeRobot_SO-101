"""Digital camera control applied to frames after they leave the Omen.

The wrist D405 and overhead webcam are fixed. Pan, zoom, and rotation crop
the picture that viewers receive. Quality and frame rate limit upload size.
Nothing here talks to RealSense or the Omen process.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_VIEW = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
_ROTATIONS = {0: 0, 90: 3, 180: 2, 270: 1}


@dataclass(frozen=True)
class CameraControl:
    view: str = "wrist"
    quality: int = 70
    max_fps: float = 10.0
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    rotate: int = 0
    paused: bool = False

    def to_dict(self) -> dict:
        return {
            "view": self.view,
            "quality": self.quality,
            "max_fps": self.max_fps,
            "zoom": self.zoom,
            "pan_x": self.pan_x,
            "pan_y": self.pan_y,
            "rotate": self.rotate,
            "paused": self.paused,
        }


def sanitize_view(name: str) -> str:
    cleaned = str(name).strip()
    if not _VIEW.fullmatch(cleaned):
        raise ValueError(f"invalid camera name {name!r}")
    return cleaned


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0", "yes", "no"}:
        return value.strip().lower() in {"true", "1", "yes"}
    raise ValueError("paused must be a boolean")


def merge_control(current: CameraControl, patch: dict) -> CameraControl:
    """Apply a partial control update. Unknown keys are ignored."""
    if not isinstance(patch, dict):
        raise ValueError("control must be an object")
    data = current.to_dict()
    if "view" in patch:
        data["view"] = sanitize_view(str(patch["view"]))
    if "quality" in patch:
        data["quality"] = int(patch["quality"])
    if "max_fps" in patch:
        data["max_fps"] = float(patch["max_fps"])
    if "zoom" in patch:
        data["zoom"] = float(patch["zoom"])
    if "pan_x" in patch:
        data["pan_x"] = float(patch["pan_x"])
    if "pan_y" in patch:
        data["pan_y"] = float(patch["pan_y"])
    if "rotate" in patch:
        rotate = int(patch["rotate"])
        if rotate not in _ROTATIONS:
            raise ValueError("rotate must be 0, 90, 180, or 270")
        data["rotate"] = rotate
    if "paused" in patch:
        data["paused"] = _as_bool(patch["paused"])
    return CameraControl(
        view=data["view"],
        quality=min(95, max(20, int(data["quality"]))),
        max_fps=min(30.0, max(0.5, float(data["max_fps"]))),
        zoom=min(4.0, max(1.0, float(data["zoom"]))),
        pan_x=min(1.0, max(-1.0, float(data["pan_x"]))),
        pan_y=min(1.0, max(-1.0, float(data["pan_y"]))),
        rotate=int(data["rotate"]),
        paused=bool(data["paused"]),
    )


def apply_frame(bgr: np.ndarray, control: CameraControl) -> bytes:
    """Rotate, crop, and JPEG-encode one BGR frame. Zoom uses nearest pixels."""
    import cv2

    image = np.asarray(bgr)
    turns = _ROTATIONS[int(control.rotate)]
    if turns:
        image = np.ascontiguousarray(np.rot90(image, k=turns))
    zoom = float(control.zoom)
    if zoom > 1.001:
        height, width = image.shape[:2]
        crop_w = max(1, int(round(width / zoom)))
        crop_h = max(1, int(round(height / zoom)))
        max_x = width - crop_w
        max_y = height - crop_h
        origin_x = int(round((width - crop_w) / 2 + control.pan_x * (max_x / 2)))
        origin_y = int(round((height - crop_h) / 2 + control.pan_y * (max_y / 2)))
        origin_x = min(max(origin_x, 0), max_x)
        origin_y = min(max(origin_y, 0), max_y)
        image = image[origin_y : origin_y + crop_h, origin_x : origin_x + crop_w]
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(control.quality)],
    )
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()
