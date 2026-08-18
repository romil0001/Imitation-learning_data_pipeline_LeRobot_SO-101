"""Local site: live D405 + birdseye JPEGs + teleop/record log. Run on the Mac."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import zmq
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from zmq_jpeg import jpegs_from_zmq_payload

ROOT = Path(__file__).resolve().parents[1]
SESSION_LOG = ROOT / "outputs" / "session.log"

_latest_jpegs: dict[str, bytes] = {}
_jpeg_lock = threading.Lock()
_stop = threading.Event()


def load_camera_env() -> dict[str, str]:
    path = Path(os.environ.get("CAMERA_FILE", ROOT / "config" / "camera.env"))
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"')
    return out


def jpeg_loop_zmq(host: str, port: int) -> None:
    global _latest_jpegs
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.CONFLATE, True)
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    sock.connect(f"tcp://{host}:{port}")
    while not _stop.is_set():
        try:
            data = json.loads(sock.recv_string())
        except zmq.Again:
            continue
        except Exception:
            time.sleep(0.2)
            continue
        decoded = jpegs_from_zmq_payload(data)
        if not decoded:
            continue
        with _jpeg_lock:
            _latest_jpegs.update(decoded)
    sock.close(0)


def jpeg_loop_opencv(index: int, width: int, height: int, fps: int) -> None:
    global _latest_jpegs
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    delay = 1.0 / max(fps, 1)
    while not _stop.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.2)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            with _jpeg_lock:
                _latest_jpegs["front"] = buf.tobytes()
        time.sleep(delay)
    cap.release()


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>SO-101 site</title>
  <style>
    :root { color-scheme: dark; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, sans-serif; background: #111; color: #eee; }
    header { padding: 12px 16px; background: #1b1b1b; border-bottom: 1px solid #333; }
    main { display: grid; grid-template-columns: minmax(320px, 2fr) minmax(280px, 1fr); gap: 12px; padding: 12px; }
    .cams { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    img { width: 100%; background: #000; border: 1px solid #333; }
    h2 { margin: 0 0 8px; font-size: 16px; }
    pre { height: 70vh; overflow: auto; background: #0b0b0b; border: 1px solid #333; padding: 8px; font-size: 12px; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } .cams { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>SO-101 · live camera · command log</header>
  <main>
    <div class="cams">
      <section>
        <h2>front</h2>
        <img src="/mjpeg/front" alt="front"/>
      </section>
      <section>
        <h2>birdseye</h2>
        <img src="/mjpeg/birdseye" alt="birdseye"/>
      </section>
    </div>
    <section>
      <h2>Log</h2>
      <pre id="log">waiting for outputs/session.log …</pre>
    </section>
  </main>
  <script>
    async function poll() {
      try {
        const r = await fetch('/api/log');
        const t = await r.text();
        const el = document.getElementById('log');
        el.textContent = t || '(empty)';
        el.scrollTop = el.scrollHeight;
      } catch (e) {}
      setTimeout(poll, 1000);
    }
    poll();
  </script>
</body>
</html>
"""

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.get("/api/log")
def api_log() -> str:
    if not SESSION_LOG.is_file():
        return "(no session log yet — run teleop/record on the Mac)"
    text = SESSION_LOG.read_text(errors="replace")
    return text[-80_000:]


def mjpeg_iter(name: str):
    boundary = b"--frame"
    while True:
        with _jpeg_lock:
            frame = _latest_jpegs.get(name, b"")
        if not frame:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            ok, buf = cv2.imencode(".jpg", blank)
            frame = buf.tobytes() if ok else b""
        yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.05)


@app.get("/mjpeg")
def mjpeg_front_alias():
    return StreamingResponse(mjpeg_iter("front"), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/mjpeg/{name}")
def mjpeg_named(name: str):
    return StreamingResponse(mjpeg_iter(name), media_type="multipart/x-mixed-replace; boundary=frame")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    env = load_camera_env()
    backend = env.get("CAMERA_BACKEND", "zmq")
    if backend == "zmq":
        host = env.get("ZMQ_HOST", "")
        port = int(env.get("ZMQ_PORT", "5555"))
        if not host or host.startswith("CHANGE_ME"):
            raise SystemExit("Set ZMQ_HOST in config/camera.env to the Omen LAN IP")
        t = threading.Thread(target=jpeg_loop_zmq, args=(host, port), daemon=True)
    else:
        t = threading.Thread(
            target=jpeg_loop_opencv,
            args=(
                int(env.get("CAMERA_INDEX", "0")),
                int(env.get("CAMERA_WIDTH", "640")),
                int(env.get("CAMERA_HEIGHT", "480")),
                int(env.get("CAMERA_FPS", "15")),
            ),
            daemon=True,
        )
    t.start()
    import uvicorn

    print(f"Open http://{args.host}:{args.port}", flush=True)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        _stop.set()


if __name__ == "__main__":
    main()
