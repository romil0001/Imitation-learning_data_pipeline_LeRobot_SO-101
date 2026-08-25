"""Mac operator site: D405 over ZMQ, arm ports, live logs. No ROS.

Architected by romil0001. See docs/architecture.md.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import zmq
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STATIC = Path(__file__).resolve().parent / "static"
SESSION_LOG = ROOT / "outputs" / "session.log"
BANNER_RE = re.compile(r"===== (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\S+) =====")

CFG: dict[str, str] = {
    "backend": "zmq",
    "zmq_host": "",
    "zmq_port": "5555",
    "camera_name": "wrist",
    "camera_names": "wrist,overhead",
    "opencv_index": "0",
    "width": "640",
    "height": "480",
    "fps": "15",
}

_latest_jpeg = b""
_latest_jpegs: dict[str, bytes] = {}
_jpeg_lock = threading.Lock()
_stop = threading.Event()
_stats = {"last_ts": 0.0, "fps": 0.0, "n": 0, "depth_ok": False}


def load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def apply_camera_env() -> None:
    env = load_env_file(Path(os.environ.get("CAMERA_FILE", ROOT / "config" / "camera.env")))
    CFG["backend"] = env.get("CAMERA_BACKEND", "zmq")
    CFG["zmq_host"] = env.get("ZMQ_HOST", "")
    CFG["zmq_port"] = env.get("ZMQ_PORT", "5555")
    names = env.get("CAMERA_NAMES", "").strip()
    if names:
        CFG["camera_names"] = names
        CFG["camera_name"] = names.split(",")[0].strip() or "wrist"
    else:
        CFG["camera_name"] = env.get("CAMERA_NAME", "front")
        CFG["camera_names"] = CFG["camera_name"]
    CFG["opencv_index"] = env.get("CAMERA_INDEX", "0")
    CFG["width"] = env.get("CAMERA_WIDTH", "640")
    CFG["height"] = env.get("CAMERA_HEIGHT", "480")
    CFG["fps"] = env.get("CAMERA_FPS", "15")
    CFG["use_depth"] = env.get("CAMERA_USE_DEPTH", "false").strip().lower() in {"1", "true", "yes", "on"}


def note_frame() -> None:
    now = time.time()
    last = _stats["last_ts"]
    if last > 0:
        dt = now - last
        inst = 1.0 / dt if dt > 0 else 0.0
        prev = _stats["fps"]
        _stats["fps"] = inst if prev <= 0 else (0.25 * inst + 0.75 * prev)
    _stats["last_ts"] = now
    _stats["n"] += 1


def jpeg_loop_zmq(host: str, port: int, camera_name: str) -> None:
    global _latest_jpeg
    from so101.zmq_rgbd import decode_depth_png_b64, depth_colormap_bgr

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
        images = data.get("images") or {}
        if not images:
            continue
        decoded: dict[str, bytes] = {}
        for name, b64 in images.items():
            try:
                decoded[name] = base64.b64decode(b64)
            except Exception:
                continue
        for name, b64 in (data.get("depths") or {}).items():
            try:
                z = decode_depth_png_b64(b64)
                vis = depth_colormap_bgr(z)
                ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    decoded[f"{name}_depth"] = buf.tobytes()
                    _stats["depth_ok"] = True
            except Exception:
                continue
        if not decoded:
            continue
        preferred = decoded.get(camera_name) or next(iter(decoded.values()))
        with _jpeg_lock:
            _latest_jpegs.update(decoded)
            _latest_jpeg = preferred
            note_frame()
    sock.close(0)


def jpeg_loop_opencv(index: int, width: int, height: int, fps: int) -> None:
    global _latest_jpeg
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
                _latest_jpeg = buf.tobytes()
                _latest_jpegs[CFG["camera_name"]] = _latest_jpeg
                note_frame()
        time.sleep(delay)
    cap.release()


def start_camera_thread() -> None:
    backend = CFG["backend"]
    if backend == "zmq":
        host = CFG["zmq_host"]
        if not host or host.startswith("CHANGE_ME"):
            return
        args = (host, int(CFG["zmq_port"]), CFG["camera_name"])
        threading.Thread(target=jpeg_loop_zmq, args=args, daemon=True).start()
        return
    threading.Thread(
        target=jpeg_loop_opencv,
        args=(int(CFG["opencv_index"]), int(CFG["width"]), int(CFG["height"]), int(CFG["fps"])),
        daemon=True,
    ).start()


def tcp_ok(host: str, port: int, timeout: float = 0.35) -> bool:
    if not host or host.startswith("CHANGE_ME"):
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def process_running(needle: str) -> bool:
    try:
        r = subprocess.run(["pgrep", "-f", needle], capture_output=True, text=True, check=False)
    except OSError:
        return False
    return bool(r.stdout.strip())


def last_banner() -> tuple[str | None, str | None]:
    if not SESSION_LOG.is_file():
        return None, None
    text = SESSION_LOG.read_text(errors="replace")
    matches = BANNER_RE.findall(text)
    if not matches:
        return None, None
    ts, label = matches[-1]
    return ts, label


def session_state() -> str:
    recording = process_running("so101 record") or process_running("lerobot-record")
    teleop = process_running("lerobot-teleoperate")
    age = time.time() - _stats["last_ts"] if _stats["last_ts"] else 1e9
    omen_up = tcp_ok(CFG["zmq_host"], int(CFG.get("zmq_port") or "5555")) if CFG["backend"] == "zmq" else age < 2
    if recording:
        return "RECORDING"
    if teleop:
        return "TELEOP"
    if not omen_up and age > 3:
        return "DISCONNECTED"
    return "IDLE"


def elapsed_s(state: str) -> float:
    if state not in {"TELEOP", "RECORDING"}:
        return 0.0
    ts, _ = last_banner()
    if not ts:
        return 0.0
    try:
        started = time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return 0.0
    return max(0.0, time.time() - started)


def arm_status() -> dict:
    ports = load_env_file(ROOT / "config" / "ports.env")
    follower = ports.get("FOLLOWER_PORT", "")
    leader = ports.get("LEADER_PORT", "")
    return {
        "follower_port": follower,
        "leader_port": leader,
        "follower_id": ports.get("FOLLOWER_ID", ""),
        "leader_id": ports.get("LEADER_ID", ""),
        "follower_ok": bool(follower) and not follower.startswith("CHANGE_ME") and Path(follower).exists(),
        "leader_ok": bool(leader) and not leader.startswith("CHANGE_ME") and Path(leader).exists(),
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    apply_camera_env()
    start_camera_thread()
    yield
    _stop.set()


app = FastAPI(title="SO-101 Visuomotor Manipulation Lab", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def api_status():
    apply_camera_env()
    state = session_state()
    age = None if _stats["last_ts"] <= 0 else round(time.time() - _stats["last_ts"], 2)
    host, port = CFG["zmq_host"], int(CFG.get("zmq_port") or "5555")
    ts, label = last_banner()
    return {
        "state": state,
        "elapsed_s": round(elapsed_s(state), 1),
        "last_banner": label,
        "last_banner_ts": ts,
        "camera": {
            "backend": CFG["backend"],
            "name": CFG["camera_name"],
            "names": [n.strip() for n in CFG.get("camera_names", CFG["camera_name"]).split(",") if n.strip()],
            "host": host,
            "port": port,
            "reachable": tcp_ok(host, port) if CFG["backend"] == "zmq" else age is not None and age < 3,
            "last_frame_age_s": age,
            "fps": round(_stats["fps"], 1),
            "frames": _stats["n"],
            "depth_ok": bool(_stats.get("depth_ok")),
        },
        "arms": arm_status(),
        "v1": True,
        "hint": "Dashboard is the live viz — cameras/depth here; joints while recording",
    }


@app.get("/api/joints")
def api_joints():
    from so101.live_viz import read_live_joints

    data = read_live_joints()
    if data is None:
        return {"fresh": False, "names": [], "state": [], "action": [], "age_s": None}
    return data


@app.get("/api/log")
def api_log():
    if not SESSION_LOG.is_file():
        return {"text": "(no session log yet — run bash scripts/06_teleoperate.sh or 08_record.sh)"}
    return {"text": SESSION_LOG.read_text(errors="replace")[-80_000:]}


@app.get("/api/log/stream")
async def log_stream():
    async def gen():
        pos = 0
        if SESSION_LOG.is_file():
            text = SESSION_LOG.read_text(errors="replace")[-80_000:]
            pos = SESSION_LOG.stat().st_size
            yield f"data: {json.dumps({'reset': True, 'chunk': text})}\n\n"
        else:
            yield f"data: {json.dumps({'reset': True, 'chunk': '(no session log yet)'})}\n\n"
        while True:
            await asyncio.sleep(0.4)
            if not SESSION_LOG.is_file():
                continue
            size = SESSION_LOG.stat().st_size
            if size < pos:
                pos = 0
            if size > pos:
                chunk = SESSION_LOG.read_bytes()[pos:].decode(errors="replace")
                pos = size
                yield f"data: {json.dumps({'reset': False, 'chunk': chunk})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def mjpeg_iter(camera_name: str | None = None):
    boundary = b"--frame"
    while True:
        with _jpeg_lock:
            if camera_name:
                frame = _latest_jpegs.get(camera_name, b"")
            else:
                frame = _latest_jpeg
        if not frame:
            label = camera_name or "camera"
            blank = np.zeros((270, 480, 3), dtype=np.uint8)
            cv2.putText(blank, f"waiting for {label}", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
            ok, buf = cv2.imencode(".jpg", blank)
            frame = buf.tobytes() if ok else b""
        yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.05)


@app.get("/mjpeg")
def mjpeg():
    return StreamingResponse(mjpeg_iter(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/mjpeg/{camera_name}")
def mjpeg_named(camera_name: str):
    return StreamingResponse(
        mjpeg_iter(camera_name), media_type="multipart/x-mixed-replace; boundary=frame"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    apply_camera_env()
    import uvicorn

    print(f"Operator site  http://{args.host}:{args.port}", flush=True)
    print(f"Camera {CFG['backend']} {CFG['zmq_host']}:{CFG['zmq_port']} names={CFG.get('camera_names', CFG['camera_name'])}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
