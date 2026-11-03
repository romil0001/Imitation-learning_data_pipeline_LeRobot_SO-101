"""Web UI Server for LeRobot SO-101 Visuomotor Lab.

Bridges local hardware (2x RealSense D405 + 1x Overview RGB + SO-101 Follower Arm)
to the RoboticsUI WebSocket frontend (Port 8090) using FastAPI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import struct
import threading
import time
from typing import Any

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import psutil
import uvicorn

# ----------------- CONFIGURATION & STATE -----------------

CAMERAS_CONFIG = {
    0: {"id": "cam_d405_left", "title": "D405 Left (Wrist)", "index": 0, "w": 848, "h": 480, "fps": 30},
    1: {"id": "cam_d405_right", "title": "D405 Right (Wrist/Depth)", "index": 2, "w": 848, "h": 480, "fps": 30},
    2: {"id": "cam_overview", "title": "Overview Workspace", "index": 4, "w": 1280, "h": 720, "fps": 30},
}

ws_clients: set[WebSocket] = set()
latest_frames: dict[int, dict[str, Any]] = {}
latest_frames_lock = asyncio.Lock()
frame_event = asyncio.Event()

stop_signal = threading.Event()
target_fps = 0.0
last_frame_sent_time: dict[int, float] = {}

# ----------------- CAMERA CAPTURE WORKER -----------------

def camera_capture_worker(sensor_id: int, config: dict[str, Any]):
    """Captures frames in a background thread to prevent blocking the event loop."""
    cap = cv2.VideoCapture(config["index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["w"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["h"])
    cap.set(cv2.CAP_PROP_FPS, config["fps"])
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    delay = 1.0 / max(config["fps"], 1)

    while not stop_signal.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue

        ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ret:
            latest_frames[sensor_id] = {
                "data": buf.tobytes(),
                "cols": frame.shape[1],
                "rows": frame.shape[0],
            }
            frame_event.set()

        time.sleep(delay)

    cap.release()

# ----------------- BINARY BROADCASTER LOOP -----------------

async def frame_broadcaster_loop():
    """Broadcasts packed binary JPEG frames to WebSocket clients."""
    while True:
        await frame_event.wait()
        frame_event.clear()

        now = time.time()
        frames_to_send = {}

        async with latest_frames_lock:
            for sensor_id, frame in list(latest_frames.items()):
                if target_fps > 0:
                    last_sent = last_frame_sent_time.get(sensor_id, 0.0)
                    if now - last_sent < (1.0 / target_fps) - 0.005:
                        continue
                frames_to_send[sensor_id] = frame
                last_frame_sent_time[sensor_id] = now
                del latest_frames[sensor_id]

        if not frames_to_send or not ws_clients:
            continue

        for sensor_id, frame in frames_to_send.items():
            # RoboticsUI binary protocol: 1-byte type (0x01) + 3x uint32 (sensor_id, cols, rows) + JPEG bytes
            header = struct.pack(">BIII", 0x01, sensor_id, frame["cols"], frame["rows"])
            payload = header + frame["data"]

            for ws in list(ws_clients):
                try:
                    await ws.send_bytes(payload)
                except Exception:
                    ws_clients.discard(ws)

# ----------------- SYSTEM METRICS BROADCASTER -----------------

def get_system_metrics() -> dict[str, Any]:
    try:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/").percent
        return {"cpu": cpu, "ram": ram, "disk": disk}
    except Exception:
        return {"cpu": 0, "ram": 0, "disk": 0}

async def system_metrics_broadcaster():
    while True:
        await asyncio.sleep(2.0)
        if ws_clients:
            metrics = get_system_metrics()
            payload = json.dumps({"type": "system_metrics", "metrics": metrics})
            for ws in list(ws_clients):
                try:
                    await ws.send_text(payload)
                except Exception:
                    ws_clients.discard(ws)

# ----------------- FASTAPI LIFESPAN & APP -----------------

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start camera capture threads
    for s_id, cfg in CAMERAS_CONFIG.items():
        t = threading.Thread(target=camera_capture_worker, args=(s_id, cfg), daemon=True)
        t.start()

    # Start async broadcaster tasks
    frame_task = asyncio.create_task(frame_broadcaster_loop())
    metrics_task = asyncio.create_task(system_metrics_broadcaster())

    yield

    stop_signal.set()
    frame_task.cancel()
    metrics_task.cancel()

app = FastAPI(title="SO-101 Robotics Web UI Server", lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = Path(__file__).resolve().parent / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found. Place it in the same directory.</h1>")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.add(websocket)

    # 1. Notify client that backend is connected
    await websocket.send_text(json.dumps({"type": "backend_connected"}))

    # 2. Spawn the 3 camera windows in the browser canvas
    # Layout positions: [x, y, width, height] normalized (0.0 to 1.0)
    window_specs = [
        {"sensor_id": 0, "title": "D405 Left (Wrist)", "win_id": "win_cam_left", "x": 0.22, "y": 0.72, "w": 0.42, "h": 0.48},
        {"sensor_id": 1, "title": "D405 Right (Wrist/Depth)", "win_id": "win_cam_right", "x": 0.65, "y": 0.72, "w": 0.42, "h": 0.48},
        {"sensor_id": 2, "title": "Overview Workspace", "win_id": "win_cam_overview", "x": 0.44, "y": 0.25, "w": 0.52, "h": 0.45},
    ]

    for spec in window_specs:
        msg = {
            "type": "create_camera_window",
            "window_id": spec["win_id"],
            "title": spec["title"],
            "sensor_id": spec["sensor_id"],
            "x": spec["x"],
            "y": spec["y"],
            "w": spec["w"],
            "h": spec["h"],
            "mode": 0,
            "minimized": False,
        }
        await websocket.send_text(json.dumps(msg))

    # 3. Create Status / Control Buttons
    await websocket.send_text(json.dumps({
        "type": "create_button",
        "button_id": "btn_record",
        "label": "🔴 Record Episode [SPACE]",
        "shortcuts": [" "],
        "x": 0.90, "y": 0.90, "w": 0.16, "h": 0.08,
        "mode": 0
    }))

    try:
        while True:
            raw_text = await websocket.receive_text()
            data = json.loads(raw_text)
            event_type = data.get("type")

            if event_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "ts": data.get("ts")}))
            elif event_type == "button_pressed":
                btn_id = data.get("button_id")
                print(f"[UI Event] Button pressed: {btn_id}")
            elif event_type == "set_target_fps":
                global target_fps
                target_fps = float(data.get("fps", 0.0))
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.discard(websocket)

# ----------------- MAIN RUNNER -----------------

def main():
    parser = argparse.ArgumentParser(description="SO-101 Robotics Web UI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind to")
    parser.add_argument("--web_port", type=int, default=8090, help="Web UI port (default 8090)")
    args = parser.parse_args()

    print(f"==================================================")
    print(f"🚀 SO-101 Robotics Web UI Server running at:")
    print(f"👉 http://localhost:{args.web_port}/")
    print(f"==================================================")

    uvicorn.run(app, host=args.host, port=args.web_port, log_level="info")

if __name__ == "__main__":
    main()
