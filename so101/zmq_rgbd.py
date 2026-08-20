"""Decode RGB + uint16 depth from the Omen ZMQ publisher."""

from __future__ import annotations

import base64
import json
import threading
import time
from typing import Any

import cv2
import numpy as np
import zmq


def depth_png_b64(depth_u16: np.ndarray) -> str:
    z = np.asarray(depth_u16, dtype=np.uint16)
    ok, buf = cv2.imencode(".png", z)
    if not ok:
        raise RuntimeError("depth PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_depth_png_b64(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    z = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if z is None:
        raise ValueError("failed to decode depth PNG")
    return np.asarray(z, dtype=np.uint16)


def depth_colormap_bgr(depth_u16: np.ndarray, max_mm: int = 2000) -> np.ndarray:
    z = np.asarray(depth_u16, dtype=np.float32)
    z[z <= 0] = np.nan
    scaled = np.nan_to_num(z, nan=0.0)
    scaled = np.clip(scaled, 0, max_mm)
    gray = (scaled * (255.0 / max(max_mm, 1))).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def parse_zmq_payload(data: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return RGB uint8 HxWx3 and depth uint16 HxW dicts keyed by camera name."""
    rgb: dict[str, np.ndarray] = {}
    depth: dict[str, np.ndarray] = {}
    for name, b64 in (data.get("images") or {}).items():
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    for name, b64 in (data.get("depths") or {}).items():
        depth[name] = decode_depth_png_b64(b64)
    return rgb, depth


class LatestZMQRGBD:
    """Background subscriber; keeps latest RGB JPEG bytes and depth uint16 per key."""

    def __init__(self, host: str, port: int = 5555) -> None:
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._jpegs: dict[str, bytes] = {}
        self._depths: dict[str, np.ndarray] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.setsockopt(zmq.CONFLATE, True)
        sock.setsockopt(zmq.RCVTIMEO, 2000)
        sock.connect(f"tcp://{self.host}:{self.port}")
        while not self._stop.is_set():
            try:
                data = json.loads(sock.recv_string())
            except zmq.Again:
                continue
            except Exception:
                time.sleep(0.2)
                continue
            jpegs: dict[str, bytes] = {}
            for name, b64 in (data.get("images") or {}).items():
                try:
                    jpegs[name] = base64.b64decode(b64)
                except Exception:
                    continue
            depths: dict[str, np.ndarray] = {}
            for name, b64 in (data.get("depths") or {}).items():
                try:
                    depths[name] = decode_depth_png_b64(b64)
                except Exception:
                    continue
            if jpegs or depths:
                with self._lock:
                    self._jpegs.update(jpegs)
                    self._depths.update(depths)
        sock.close(0)

    def latest_depth(self, name: str) -> np.ndarray | None:
        with self._lock:
            z = self._depths.get(name)
            return None if z is None else z.copy()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
