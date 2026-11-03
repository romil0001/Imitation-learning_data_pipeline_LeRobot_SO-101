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
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


def _zmq():
    import zmq

    return zmq

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
_frame_ts: dict[str, float] = {}
_jpeg_lock = threading.Lock()
_stop = threading.Event()
# Tiles / MJPEG treat a stream as live only while frames stay this fresh.
STALE_S = 2.0
_stats = {
    "last_ts": 0.0,
    "fps": 0.0,
    "n": 0,
    "depth_ok": False,
    "stream_ids": [],
    "depth_ids": [],
}


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


def camera_names_from_cfg() -> list[str]:
    return [n.strip() for n in CFG.get("camera_names", CFG["camera_name"]).split(",") if n.strip()]


def depth_camera_names(rgb_names: list[str] | None = None) -> list[str]:
    """D405 views that normally carry depth (wrist/base); fall back to first cam."""
    names = list(rgb_names or camera_names_from_cfg())
    if not CFG.get("use_depth"):
        return []
    preferred = [n for n in names if n in {"wrist", "base"}]
    if preferred:
        return preferred
    return [names[0]] if names else []


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
    CFG["use_depth"] = env.get("CAMERA_USE_DEPTH", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def stream_catalog(live_ids: list[str] | None = None) -> list[dict]:
    """RGB + depth tiles the UI should show (config order, plus any surprise live keys)."""
    rgb_names = camera_names_from_cfg()
    depth_names = depth_camera_names(rgb_names)
    live = set(live_ids or [])
    out: list[dict] = []
    seen: set[str] = set()

    def add(stream_id: str, label: str, kind: str) -> None:
        if stream_id in seen:
            return
        seen.add(stream_id)
        out.append(
            {
                "id": stream_id,
                "label": label,
                "kind": kind,
                "url": f"/mjpeg/{stream_id}",
                "live": stream_id in live,
            }
        )

    for name in rgb_names:
        hint = "D405" if name in {"wrist", "base"} else ("USB" if name == "overhead" else "RGB")
        add(name, f"{name} · {hint}", "rgb")
    for name in depth_names:
        add(f"{name}_depth", f"{name} depth · Z16", "depth")
    for sid in sorted(live):
        if sid.endswith("_depth"):
            add(sid, sid.replace("_", " "), "depth")
        else:
            add(sid, sid, "rgb")
    return out


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


def frame_is_fresh(stream_id: str, max_age_s: float = STALE_S, now: float | None = None) -> bool:
    ts = _frame_ts.get(stream_id)
    if ts is None or ts <= 0:
        return False
    return ((now if now is not None else time.time()) - ts) <= max_age_s


def fresh_stream_ids(max_age_s: float = STALE_S, now: float | None = None) -> list[str]:
    t = now if now is not None else time.time()
    return sorted(sid for sid, ts in _frame_ts.items() if ts > 0 and (t - ts) <= max_age_s)


def jpeg_loop_zmq(host: str, port: int, camera_name: str) -> None:
    global _latest_jpeg
    from so101.zmq_rgbd import decode_depth_png_b64, depth_colormap_bgr

    zmq = _zmq()
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
        depth_ids = [k for k in decoded if k.endswith("_depth")]
        now = time.time()
        with _jpeg_lock:
            _latest_jpegs.update(decoded)
            _latest_jpeg = preferred
            for sid in decoded:
                _frame_ts[sid] = now
            live = fresh_stream_ids(now=now)
            _stats["stream_ids"] = live
            _stats["depth_ids"] = [k for k in live if k.endswith("_depth")]
            _stats["depth_ok"] = any(k.endswith("_depth") for k in live)
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
            name = CFG["camera_name"]
            now = time.time()
            with _jpeg_lock:
                _latest_jpeg = buf.tobytes()
                _latest_jpegs[name] = _latest_jpeg
                _frame_ts[name] = now
                _stats["stream_ids"] = fresh_stream_ids(now=now)
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
    with _jpeg_lock:
        live_ids = fresh_stream_ids()
        depth_live = [k for k in live_ids if k.endswith("_depth")]
        _stats["stream_ids"] = live_ids
        _stats["depth_ids"] = depth_live
        _stats["depth_ok"] = bool(depth_live)
    names = camera_names_from_cfg()
    streams = stream_catalog(live_ids)
    return {
        "state": state,
        "elapsed_s": round(elapsed_s(state), 1),
        "last_banner": label,
        "last_banner_ts": ts,
        "camera": {
            "backend": CFG["backend"],
            "name": CFG["camera_name"],
            "names": names,
            "depth_names": depth_camera_names(names),
            "host": host,
            "port": port,
            "reachable": tcp_ok(host, port) if CFG["backend"] == "zmq" else age is not None and age < 3,
            "last_frame_age_s": age,
            "fps": round(_stats["fps"], 1),
            "frames": _stats["n"],
            "depth_ok": bool(_stats.get("depth_ok")),
            "live_ids": live_ids,
            "depth_live": depth_live,
            "use_depth": bool(CFG.get("use_depth")),
        },
        "streams": streams,
        "arms": arm_status(),
        "v1": True,
        "hint": (
            "Dashboard streams every RGB/depth view from camera.env; joints while record/eval. "
            "Offline tiles mean that key is missing from the Omen ZMQ packet "
            "(restart omen\\run_publish.bat with WRIST_SERIAL + BASE_SERIAL set)."
        ),
    }


@app.get("/api/joints")
def api_joints():
    from so101.live_viz import read_live_joints

    data = read_live_joints()
    if data is None:
        return {"fresh": False, "names": [], "state": [], "action": [], "age_s": None}
    return data


def _open_browse_dataset(name: str):
    from dashboard.browse import resolve_dataset_dir
    from so101.dataset import SO101Dataset

    try:
        path = resolve_dataset_dir(ROOT, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return SO101Dataset(path)


@app.get("/api/datasets")
def api_datasets():
    from dashboard.browse import list_datasets

    return {"datasets": list_datasets(ROOT)}


@app.get("/api/datasets/{name}")
def api_dataset(name: str):
    from dashboard.browse import dataset_detail

    return dataset_detail(_open_browse_dataset(name))


@app.get("/api/datasets/{name}/episodes/{index}")
def api_episode(name: str, index: int):
    from dashboard.browse import episode_detail

    ds = _open_browse_dataset(name)
    try:
        return episode_detail(ds, index, dataset_name=name)
    except (IndexError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/api/datasets/{name}/episodes/{index}/joints")
def api_episode_joints(name: str, index: int):
    from dashboard.browse import episode_joints

    ds = _open_browse_dataset(name)
    try:
        return episode_joints(ds, index)
    except (IndexError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/api/datasets/{name}/episodes/{index}/files")
def api_episode_files(name: str, index: int):
    from dashboard.browse import episode_file_inventory
    from so101.dataset.reader import episode_dir

    ds = _open_browse_dataset(name)
    if index < 0 or index >= ds.meta.episode_count:
        raise HTTPException(status_code=404, detail=f"episode {index} out of range")
    folder = episode_dir(ds.root, index)
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail=f"episode folder missing: {index}")
    return {"index": index, "files": episode_file_inventory(folder)}


@app.get("/api/datasets/{name}/episodes/{index}/video/{camera_name}")
def api_episode_video(name: str, index: int, camera_name: str):
    from dashboard.browse import video_path

    ds = _open_browse_dataset(name)
    try:
        path = video_path(ds, index, camera_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/api/datasets/{name}/episodes/{index}/depth/{camera_name}/frame/{frame}")
def api_episode_depth_frame(name: str, index: int, camera_name: str, frame: int):
    from dashboard.browse import depth_frame_png
    from fastapi.responses import Response

    ds = _open_browse_dataset(name)
    try:
        png = depth_frame_png(ds, index, camera_name, frame)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except IndexError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return Response(content=png, media_type="image/png")


class KeyBody(BaseModel):
    key: str = Field(..., min_length=1, max_length=8)


@app.post("/api/key")
def api_key(body: KeyBody):
    """Forward an operator key to the running record/eval/dagger process."""
    from so101.keys import DASHBOARD_KEYS, push_live_key

    raw = body.key
    ch = " " if raw in {" ", "Space", "space"} else raw.lower()
    if ch not in DASHBOARD_KEYS:
        raise HTTPException(status_code=400, detail=f"unsupported key {body.key!r}")
    try:
        pushed = push_live_key(ch)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    state = session_state()
    listening = (
        process_running("so101 record")
        or process_running("so101 dagger")
        or process_running("so101 eval")
        or process_running("lerobot-record")
    )
    return {
        "ok": True,
        "key": pushed,
        "state": state,
        "listening": listening,
        "hint": None if listening else "No record/eval/dagger process is listening yet",
    }


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
                frame = _latest_jpegs.get(camera_name, b"") if frame_is_fresh(camera_name) else b""
            else:
                # Preferred cam: only serve while any stream is still fresh.
                frame = _latest_jpeg if fresh_stream_ids() else b""
        if not frame:
            label = camera_name or "camera"
            blank = np.zeros((270, 480, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                f"offline: {label}",
                (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 180),
                1,
            )
            cv2.putText(
                blank,
                "not in Omen ZMQ packet",
                (40, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (140, 140, 140),
                1,
            )
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
