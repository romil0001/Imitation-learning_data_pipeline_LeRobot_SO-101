"""Remote camera relay: off-LAN viewing and digital camera control.

The Omen ZMQ publisher is not involved. These tests drive the relay with
JPEG bytes and a fake frame source.
"""

from __future__ import annotations

import base64
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar

import cv2
import numpy as np
import pytest

from so101.cli import build_parser
from so101.remote_cam.control import CameraControl, apply_frame, merge_control
from so101.remote_cam.publish import (
    FramePacer,
    frames_from_zmq_payload,
    publish_loop,
    select_and_encode,
)
from so101.remote_cam.relay import resolve_token, serve_background
from so101.zmq_rgbd import depth_png_b64


def _jpeg(color: tuple[int, int, int], size: int = 100) -> bytes:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = color
    return apply_frame(img, CameraControl(quality=90))


def _b64_jpeg(color: tuple[int, int, int]) -> str:
    return base64.b64encode(_jpeg(color)).decode("ascii")


def test_merge_control_clamps_and_ignores_unknown_keys() -> None:
    merged = merge_control(
        CameraControl(),
        {"quality": 5, "zoom": 9, "pan_x": -3, "max_fps": 100, "nope": True, "paused": False},
    )
    assert merged.quality == 20
    assert merged.zoom == 4.0
    assert merged.pan_x == -1.0
    assert merged.max_fps == 30.0
    assert merged.paused is False
    assert not hasattr(merged, "nope")


def test_merge_control_rejects_bad_rotate_and_false_string() -> None:
    with pytest.raises(ValueError, match="rotate"):
        merge_control(CameraControl(), {"rotate": 45})
    paused = merge_control(CameraControl(paused=True), {"paused": "false"})
    assert paused.paused is False


def test_apply_frame_digital_pan_zoom_and_rotate() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[0:10, 0:10] = 255
    zoomed = cv2.imdecode(
        np.frombuffer(
            apply_frame(img, CameraControl(zoom=2, pan_x=-1, pan_y=-1, quality=95)),
            dtype=np.uint8,
        ),
        cv2.IMREAD_COLOR,
    )
    centered = cv2.imdecode(
        np.frombuffer(apply_frame(img, CameraControl(zoom=2, quality=95)), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert zoomed[0, 0, 0] > 200
    assert centered[0, 0, 0] < 20

    wide = np.zeros((40, 80, 3), dtype=np.uint8)
    turned = cv2.imdecode(
        np.frombuffer(apply_frame(wide, CameraControl(rotate=90, quality=90)), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert turned.shape[:2] == (80, 40)


def test_frames_from_zmq_payload_includes_depth_view() -> None:
    depth = np.zeros((20, 30), dtype=np.uint16)
    depth[4, 6] = 800
    frames = frames_from_zmq_payload(
        {"images": {"wrist": _b64_jpeg((0, 0, 255))}, "depths": {"wrist": depth_png_b64(depth)}}
    )
    assert "wrist" in frames
    assert frames["wrist_depth"].shape == (20, 30, 3)


def test_select_and_encode_falls_back_when_view_missing() -> None:
    frames = {"overhead": np.zeros((16, 16, 3), dtype=np.uint8)}
    frames["overhead"][:, :] = (0, 255, 0)
    selected = select_and_encode(frames, CameraControl(view="wrist"))
    assert selected is not None
    name, _jpeg_bytes, known = selected
    assert name == "overhead"
    assert known == ["overhead"]
    assert select_and_encode(frames, CameraControl(paused=True)) is None


def test_frame_pacer_respects_interval() -> None:
    now = [0.0]
    pacer = FramePacer(clock=lambda: now[0])
    assert pacer.allow(10)
    assert not pacer.allow(10)
    now[0] = 0.1
    assert pacer.allow(10)


def test_resolve_token_generates_one_for_public_bind() -> None:
    assert resolve_token("127.0.0.1", "") == ""
    generated = resolve_token("0.0.0.0", "")
    assert len(generated) >= 16
    assert resolve_token("0.0.0.0", "lab-token") == "lab-token"
    with pytest.raises(ValueError):
        resolve_token("127.0.0.1", "bad;token")


def test_remote_cam_cli_parses_serve_and_publish() -> None:
    serve = build_parser().parse_args(
        ["remote-cam", "serve", "--host", "0.0.0.0", "--port", "8091", "--token", "abc", "--zmq-host", "10.0.0.8"]
    )
    assert serve.remote_cmd == "serve"
    assert serve.port == 8091
    assert serve.zmq_host == "10.0.0.8"
    publish = build_parser().parse_args(
        ["remote-cam", "publish", "--relay", "http://203.0.113.5:8090", "--token", "abc"]
    )
    assert publish.remote_cmd == "publish"
    assert publish.relay.startswith("http://203.0.113.5")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["remote-cam"])


@pytest.fixture
def relay():
    httpd, thread = serve_background("127.0.0.1", 0, "secret")
    host, port = httpd.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        yield base
    finally:
        httpd.shutdown()
        thread.join(timeout=3)
        httpd.server_close()


def _request(url: str, *, data: bytes | None = None, headers: dict | None = None, method: str | None = None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def test_relay_requires_token_and_serves_selected_camera(relay: str) -> None:
    wrist = _jpeg((0, 0, 255))
    overhead = _jpeg((0, 180, 0))
    code, _, _ = _request(
        relay + "/api/frame",
        data=wrist,
        headers={"Content-Type": "image/jpeg", "X-Camera": "wrist"},
        method="POST",
    )
    assert code == 401

    headers = {
        "Authorization": "Bearer secret",
        "Content-Type": "image/jpeg",
        "X-Camera": "wrist",
        "X-Known-Cameras": "wrist,overhead",
    }
    code, _, body = _request(relay + "/api/frame", data=wrist, headers=headers, method="POST")
    assert code == 200
    assert json.loads(body)["control"]["view"] == "wrist"
    headers["X-Camera"] = "overhead"
    code, _, _ = _request(relay + "/api/frame", data=overhead, headers=headers, method="POST")
    assert code == 200

    code, _, snap = _request(relay + "/snapshot.jpg", headers={"Authorization": "Bearer secret"})
    assert code == 200
    assert snap == wrist

    code, _, raw = _request(
        relay + "/api/control",
        data=json.dumps({"view": "overhead", "zoom": 2, "paused": False}).encode(),
        headers={"Authorization": "Bearer secret", "Content-Type": "application/json"},
        method="POST",
    )
    assert code == 200
    control = json.loads(raw)["control"]
    assert control["view"] == "overhead"
    assert control["zoom"] == 2
    code, _, snap = _request(relay + "/snapshot.jpg", headers={"Authorization": "Bearer secret"})
    assert snap == overhead
    parsed = urllib.parse.urlparse(relay)
    with socket.create_connection((parsed.hostname, parsed.port), timeout=2) as sock:
        sock.sendall(
            b"GET /mjpeg HTTP/1.0\r\nAuthorization: Bearer secret\r\nHost: localhost\r\n\r\n"
        )
        blob = b""
        while b"\xff\xd8" not in blob and len(blob) < 200_000:
            part = sock.recv(4096)
            if not part:
                break
            blob += part
    assert b"multipart/x-mixed-replace" in blob
    assert b"\xff\xd8" in blob and b"\xff\xd9" in blob

    code, _, status = _request(relay + "/api/status", headers={"Authorization": "Bearer secret"})
    payload = json.loads(status)
    assert payload["publisher_online"] is True
    assert payload["cameras"] == ["wrist", "overhead"]
    assert payload["frames"] == 2


def test_login_cookie_opens_viewer(relay: str) -> None:
    jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    bad = urllib.parse.urlencode({"token": "nope"}).encode()
    with pytest.raises(urllib.error.HTTPError) as exc:
        opener.open(urllib.request.Request(relay + "/login", data=bad), timeout=3)
    assert exc.value.code == 401

    good = urllib.parse.urlencode({"token": "secret"}).encode()
    page = opener.open(urllib.request.Request(relay + "/login", data=good), timeout=3).read().decode()
    assert "Digital pan" in page
    snap = opener.open(relay + "/snapshot.jpg", timeout=3)
    assert snap.status == 200


def test_zmq_subscription_delivers_omen_shaped_payload(relay: str) -> None:
    """A ZMQ PUB in the Omen message shape must arrive as a viewable JPEG."""
    import zmq

    from so101.remote_cam.publish import http_fetch_control, http_post_frame, make_zmq_source

    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    port = socket.bind_to_random_port("tcp://127.0.0.1")
    stop = threading.Event()
    source = make_zmq_source("127.0.0.1", port, stop)
    payload = json.dumps(
        {
            "images": {"wrist": _b64_jpeg((30, 40, 210)), "overhead": _b64_jpeg((30, 210, 40))},
            "depths": {},
        }
    )

    def send_frames() -> None:
        for _ in range(40):
            if stop.is_set():
                break
            socket.send_string(payload)
            time.sleep(0.05)

    threading.Thread(target=send_frames, daemon=True).start()
    thread = threading.Thread(
        target=publish_loop,
        args=(
            source,
            lambda: http_fetch_control(relay, "secret"),
            lambda camera, jpeg, known: http_post_frame(relay, "secret", camera, jpeg, known),
            stop,
        ),
        kwargs={"idle_s": 0.02},
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 4
    status: dict = {}
    while time.monotonic() < deadline:
        _, _, raw = _request(relay + "/api/status", headers={"Authorization": "Bearer secret"})
        status = json.loads(raw)
        if status.get("frames", 0) >= 1 and "wrist" in status.get("cameras", []):
            break
        time.sleep(0.05)
    stop.set()
    thread.join(timeout=3)
    source.close()
    socket.close(0)
    assert status.get("frames", 0) >= 1
    assert "wrist" in status["cameras"]
    _, _, snap = _request(relay + "/snapshot.jpg", headers={"Authorization": "Bearer secret"})
    image = cv2.imdecode(np.frombuffer(snap, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None and image.size > 0


def test_publish_loop_obeys_view_pause_and_catalog(relay: str) -> None:
    payload = {
        "images": {
            "wrist": _b64_jpeg((40, 40, 220)),
            "overhead": _b64_jpeg((40, 200, 40)),
        }
    }
    code, _, _ = _request(
        relay + "/api/control",
        data=json.dumps({"view": "overhead", "max_fps": 20}).encode(),
        headers={"Authorization": "Bearer secret", "Content-Type": "application/json"},
        method="POST",
    )
    assert code == 200

    from so101.remote_cam.publish import http_fetch_control, http_post_frame

    stop = threading.Event()
    posts = {"n": 0}

    def post(camera, jpeg, known):
        posts["n"] += 1
        return http_post_frame(relay, "secret", camera, jpeg, known)

    thread = threading.Thread(
        target=publish_loop,
        args=(lambda: payload, lambda: http_fetch_control(relay, "secret"), post, stop),
        kwargs={"idle_s": 0.02},
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 3
    status = {}
    while time.monotonic() < deadline:
        _, _, raw = _request(relay + "/api/status", headers={"Authorization": "Bearer secret"})
        status = json.loads(raw)
        if status.get("frames", 0) >= 1 and status.get("active_view") == "overhead":
            break
        time.sleep(0.05)
    assert status.get("active_view") == "overhead"
    assert "wrist" in status["cameras"] and "overhead" in status["cameras"]
    _, _, snap = _request(relay + "/snapshot.jpg", headers={"Authorization": "Bearer secret"})
    image = cv2.imdecode(np.frombuffer(snap, dtype=np.uint8), cv2.IMREAD_COLOR)
    mean = image.mean(axis=(0, 1))
    assert mean[1] > mean[2]

    _request(
        relay + "/api/control",
        data=json.dumps({"paused": True}).encode(),
        headers={"Authorization": "Bearer secret", "Content-Type": "application/json"},
        method="POST",
    )
    seen = posts["n"]
    time.sleep(0.4)
    assert posts["n"] <= seen + 1
    stop.set()
    thread.join(timeout=3)
    assert not thread.is_alive()
