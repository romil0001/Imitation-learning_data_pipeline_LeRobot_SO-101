"""Local site: live D405 JPEG + teleop/record log. Run on the Mac."""

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

ROOT = Path(__file__).resolve().parents[1]
SESSION_LOG = ROOT / "outputs" / "session.log"

_latest_jpeg = b""
_last_frame_at = 0.0
_frames_received = 0
_camera_error = ""
_jpeg_lock = threading.Lock()
_stop = threading.Event()
_camera_config: dict[str, str | int | float | bool] = {}


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


def _note_frame() -> None:
    global _last_frame_at, _frames_received
    _last_frame_at = time.time()
    _frames_received += 1


def jpeg_loop_zmq(host: str, port: int, camera_name: str) -> None:
    global _latest_jpeg, _camera_error
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.CONFLATE, True)
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    sock.connect(f"tcp://{host}:{port}")
    while not _stop.is_set():
        try:
            data = json.loads(sock.recv_string())
            _camera_error = ""
        except zmq.Again:
            _camera_error = f"No frames from tcp://{host}:{port} (is the publisher running?)"
            continue
        except Exception as exc:
            _camera_error = str(exc)
            time.sleep(0.2)
            continue
        images = data.get("images") or {}
        b64 = images.get(camera_name) or (next(iter(images.values())) if images else None)
        if not b64:
            _camera_error = f"No image key '{camera_name}' in ZMQ payload"
            continue
        import base64

        raw = base64.b64decode(b64)
        with _jpeg_lock:
            _latest_jpeg = raw
        _note_frame()
    sock.close(0)


def jpeg_loop_opencv(index: int, width: int, height: int, fps: int) -> None:
    global _latest_jpeg, _camera_error
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        _camera_error = f"OpenCV cannot open camera index {index}"
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    delay = 1.0 / max(fps, 1)
    while not _stop.is_set():
        ok, frame = cap.read()
        if not ok:
            _camera_error = f"OpenCV read failed on index {index}"
            time.sleep(0.2)
            continue
        _camera_error = ""
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            with _jpeg_lock:
                _latest_jpeg = buf.tobytes()
            _note_frame()
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
    header { padding: 12px 16px; background: #1b1b1b; border-bottom: 1px solid #333; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
    header nav a { color: #8cf; text-decoration: none; margin-right: 12px; }
    header nav a.active { color: #fff; font-weight: 600; }
    main { display: grid; grid-template-columns: minmax(320px, 1fr) minmax(280px, 1fr); gap: 12px; padding: 12px; }
    img { width: 100%; background: #000; border: 1px solid #333; min-height: 240px; object-fit: contain; }
    pre { height: 70vh; overflow: auto; background: #0b0b0b; border: 1px solid #333; padding: 8px; font-size: 12px; }
    .status { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; padding: 4px 10px; border-radius: 999px; background: #222; border: 1px solid #444; }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: #666; }
    .dot.live { background: #3d3; box-shadow: 0 0 6px #3d3; }
    .dot.wait { background: #da3; }
    .meta { font-size: 12px; color: #aaa; margin: 8px 0; line-height: 1.5; }
    .hint { font-size: 13px; color: #ccc; background: #1a1a1a; border: 1px solid #333; padding: 10px; margin-top: 8px; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <strong>SO-101</strong>
    <nav>
      <a href="/" class="active">Cameras</a>
      <a href="/#log">Log</a>
    </nav>
    <span class="status"><span id="dot" class="dot wait"></span><span id="status-text">connecting…</span></span>
  </header>
  <main>
    <section>
      <h2>Live camera</h2>
      <img src="/mjpeg" alt="live camera feed"/>
      <div id="meta" class="meta"></div>
      <div id="hint" class="hint" hidden></div>
    </section>
    <section id="log">
      <h2>Command log</h2>
      <pre id="log-pre">waiting for outputs/session.log …</pre>
    </section>
  </main>
  <script>
    async function pollStatus() {
      try {
        const r = await fetch('/api/camera/status');
        const s = await r.json();
        const dot = document.getElementById('dot');
        const text = document.getElementById('status-text');
        const meta = document.getElementById('meta');
        const hint = document.getElementById('hint');
        dot.className = 'dot ' + (s.live ? 'live' : 'wait');
        text.textContent = s.live ? 'live' : (s.error || 'waiting for frames');
        meta.textContent =
          `backend=${s.backend}  name=${s.name}  ` +
          (s.backend === 'zmq' ? `zmq=${s.zmq_host}:${s.zmq_port}  ` : `index=${s.index}  `) +
          `${s.width}x${s.height}@${s.fps}  frames=${s.frames}`;
        if (!s.live && s.setup_hint) {
          hint.hidden = false;
          hint.textContent = s.setup_hint;
        } else {
          hint.hidden = true;
        }
      } catch (e) {}
      setTimeout(pollStatus, 1000);
    }
    async function pollLog() {
      try {
        const r = await fetch('/api/log');
        const t = await r.text();
        const el = document.getElementById('log-pre');
        el.textContent = t || '(empty)';
        el.scrollTop = el.scrollHeight;
      } catch (e) {}
      setTimeout(pollLog, 1000);
    }
    pollStatus();
    pollLog();
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


@app.get("/api/camera/status")
def api_camera_status() -> dict:
    with _jpeg_lock:
        has_frame = bool(_latest_jpeg)
    age = time.time() - _last_frame_at if _last_frame_at else None
    live = has_frame and age is not None and age < 3.0
    backend = str(_camera_config.get("backend", "unknown"))
    setup_hint = ""
    if not live:
        if backend == "zmq":
            host = _camera_config.get("zmq_host", "?")
            setup_hint = (
                f"Start the publisher on the Omen (run_publish.bat) or locally: "
                f"python3 scripts/mock_zmq_camera.py — then set ZMQ_HOST={host} in config/camera.env"
            )
        elif backend == "opencv":
            setup_hint = (
                "Plug in the D405 and run bash scripts/07_find_cameras.sh to set CAMERA_INDEX, "
                "or use ZMQ from the Omen for full RealSense support."
            )
        elif not _camera_config:
            setup_hint = "Run bash scripts/setup_cameras_view.sh to create config/camera.env"
    return {
        "live": live,
        "backend": backend,
        "name": _camera_config.get("name", "front"),
        "zmq_host": _camera_config.get("zmq_host", ""),
        "zmq_port": _camera_config.get("zmq_port", 5555),
        "index": _camera_config.get("index", 0),
        "width": _camera_config.get("width", 640),
        "height": _camera_config.get("height", 480),
        "fps": _camera_config.get("fps", 15),
        "frames": _frames_received,
        "last_frame_age_s": round(age, 2) if age is not None else None,
        "error": _camera_error or None,
        "setup_hint": setup_hint or None,
    }


def mjpeg_iter():
    boundary = b"--frame"
    while True:
        with _jpeg_lock:
            frame = _latest_jpeg
        if not frame:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                "no signal",
                (70, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (80, 80, 80),
                2,
                cv2.LINE_AA,
            )
            ok, buf = cv2.imencode(".jpg", blank)
            frame = buf.tobytes() if ok else b""
        yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.05)


@app.get("/mjpeg")
def mjpeg():
    return StreamingResponse(mjpeg_iter(), media_type="multipart/x-mixed-replace; boundary=frame")


def main() -> None:
    global _camera_config
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    env = load_camera_env()
    backend = env.get("CAMERA_BACKEND", "zmq")
    name = env.get("CAMERA_NAME", "front")
    _camera_config = {
        "backend": backend,
        "name": name,
        "width": int(env.get("CAMERA_WIDTH", "640")),
        "height": int(env.get("CAMERA_HEIGHT", "480")),
        "fps": int(env.get("CAMERA_FPS", "15")),
    }
    if backend == "zmq":
        host = env.get("ZMQ_HOST", "")
        port = int(env.get("ZMQ_PORT", "5555"))
        if not host or host.startswith("CHANGE_ME"):
            raise SystemExit("Set ZMQ_HOST in config/camera.env to the Omen LAN IP (or 127.0.0.1 for mock)")
        _camera_config["zmq_host"] = host
        _camera_config["zmq_port"] = port
        t = threading.Thread(target=jpeg_loop_zmq, args=(host, port, name), daemon=True)
    else:
        index = int(env.get("CAMERA_INDEX", "0"))
        _camera_config["index"] = index
        t = threading.Thread(
            target=jpeg_loop_opencv,
            args=(
                index,
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
