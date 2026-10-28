"""Push the Omen ZMQ stream to a relay without changing the Omen.

The publisher is a normal ZMQ subscriber, the same role the Mac dashboard
already has. Control (which view, zoom, quality, pause) is applied here so
only the frame the operator asked for crosses the internet.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

import numpy as np

from so101.remote_cam.control import CameraControl, apply_frame, sanitize_view


class FramePacer:
    """Allow at most ``max_fps`` encodes per second."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._next = 0.0

    def allow(self, max_fps: float) -> bool:
        now = self._clock()
        if now < self._next:
            return False
        fps = min(30.0, max(0.5, float(max_fps)))
        self._next = now + 1.0 / fps
        return True


def frames_from_zmq_payload(data: dict[str, Any]) -> dict[str, np.ndarray]:
    """Decode the Omen message into BGR frames, including ``<name>_depth`` views."""
    import cv2

    from so101.zmq_rgbd import decode_depth_png_b64, depth_colormap_bgr

    frames: dict[str, np.ndarray] = {}
    for name, b64 in (data.get("images") or {}).items():
        try:
            key = sanitize_view(str(name))
            raw = _b64_bytes(b64)
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            continue
        if image is not None:
            frames[key] = image
    for name, b64 in (data.get("depths") or {}).items():
        try:
            key = sanitize_view(f"{name}_depth")
            frames[key] = depth_colormap_bgr(decode_depth_png_b64(b64))
        except Exception:
            continue
    return frames


def _b64_bytes(value: Any) -> bytes:
    import base64

    if isinstance(value, bytes):
        text = value.decode("ascii")
    else:
        text = str(value)
    return base64.b64decode(text)


def select_and_encode(
    frames: dict[str, np.ndarray],
    control: CameraControl,
) -> tuple[str, bytes, list[str]] | None:
    """Encode the requested view. Falls back to the first available camera."""
    if control.paused or not frames:
        return None
    known = list(frames)
    view = control.view if control.view in frames else known[0]
    return view, apply_frame(frames[view], control), known


def http_post_frame(
    relay: str,
    token: str,
    camera: str,
    jpeg: bytes,
    known: list[str],
    timeout: float = 10.0,
) -> CameraControl:
    headers = {
        "Content-Type": "image/jpeg",
        "X-Camera": camera,
        "X-Known-Cameras": ",".join(known),
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        relay.rstrip("/") + "/api/frame",
        data=jpeg,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode())
    return _control_from(payload)


def http_fetch_control(relay: str, token: str, timeout: float = 10.0) -> CameraControl:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(relay.rstrip("/") + "/api/control", headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode())
    return _control_from(payload)


def _control_from(payload: dict) -> CameraControl:
    from so101.remote_cam.control import merge_control

    return merge_control(CameraControl(), payload.get("control") or payload)


def publish_loop(
    source: Callable[[], dict | None],
    fetch_control: Callable[[], CameraControl],
    post_frame: Callable[[str, bytes, list[str]], CameraControl],
    stop: threading.Event,
    idle_s: float = 0.05,
    pacer: FramePacer | None = None,
) -> None:
    """Read payloads, apply the relay's control, and upload one JPEG at a time."""
    pacer = pacer or FramePacer()
    last_error = ""
    quiet_until = 0.0

    def note(message: str) -> None:
        nonlocal last_error
        if message != last_error:
            print(message, file=sys.stderr, flush=True)
            last_error = message

    while not stop.is_set():
        try:
            control = fetch_control()
        except (OSError, urllib.error.URLError, ValueError, KeyError, json.JSONDecodeError) as exc:
            note(f"Camera relay unreachable: {exc}")
            stop.wait(0.5)
            continue
        if control.paused:
            stop.wait(max(idle_s, 0.05))
            continue
        if not pacer.allow(control.max_fps):
            stop.wait(min(idle_s, 0.02))
            continue
        try:
            payload = source()
        except Exception as exc:
            note(f"Camera read failed: {exc}")
            stop.wait(idle_s)
            continue
        if not payload:
            now = time.monotonic()
            if now >= quiet_until:
                note("No camera frames yet. Leave the Omen publisher running.")
                quiet_until = now + 5.0
            stop.wait(idle_s)
            continue
        quiet_until = 0.0
        try:
            selected = select_and_encode(frames_from_zmq_payload(payload), control)
        except Exception as exc:
            note(f"Camera frame decode failed: {exc}")
            stop.wait(idle_s)
            continue
        if selected is None:
            stop.wait(idle_s)
            continue
        camera, jpeg, known = selected
        try:
            post_frame(camera, jpeg, known)
        except (OSError, urllib.error.URLError, ValueError, RuntimeError) as exc:
            note(f"Camera relay rejected a frame: {exc}")
            stop.wait(0.5)
            continue
        last_error = ""


def make_zmq_source(host: str, port: int, stop: threading.Event):
    """Latest-frame ZMQ subscriber. ``source.close()`` releases the socket."""
    import zmq

    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.RCVTIMEO, 500)
    socket.connect(f"tcp://{host}:{port}")

    def pull() -> dict | None:
        if stop.is_set():
            return None
        try:
            return json.loads(socket.recv_string())
        except zmq.Again:
            return None
        except Exception:
            return None

    def close() -> None:
        socket.close(0)

    pull.close = close  # type: ignore[attr-defined]
    return pull
