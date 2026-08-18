# Dual-camera birdseye webcam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the Omen USB webcam as a separate `birdseye` JPEG next to D405 `front` on the existing ZMQ port so the Mac dashboard shows both panes and LeRobot records both streams.

**Architecture:** One `publish_d405.py` process grabs RealSense + OpenCV webcam and sends `{"images": {"front": "...", "birdseye": "..."}, "ts": ...}` on TCP 5555. Mac `robot_cameras_json` emits two `type: zmq` cameras when `BIRDSEYE_NAME` is set. Dashboard uses one SUB and two MJPEG routes.

**Tech Stack:** Python 3.12, OpenCV, pyrealsense2, pyzmq, FastAPI (dashboard), bash helpers in `scripts/_common.sh`

## Global Constraints

- Cameras stay on the Omen; the Mac never opens the webcam
- One ZMQ PUB on TCP **5555** (no second port, no second Windows process)
- Message keys: `front` (D405) and `birdseye` (webcam)
- Webcam OpenCV index is configurable (Windows often also exposes the D405 as UVC)
- Empty `BIRDSEYE_NAME` on the Mac keeps today’s front-only LeRobot config
- If the webcam is missing, D405 publishing must continue
- If birdseye is enabled but a webcam frame drops, republish the last good JPEG under `birdseye` (never omit the key)
- No new Python dependencies; OpenCV and pyzmq are already in `omen/requirements.txt`
- Do not restructure unrelated bring-up scripts
- Spec: `docs/superpowers/specs/2026-08-18-birdseye-webcam-design.md`

## File Structure

| Path | Responsibility |
|------|----------------|
| `scripts/_common.sh` | Load `BIRDSEYE_NAME`; emit one or two ZMQ camera dicts |
| `scripts/test_common.sh` | Assert front-only and dual-camera ZMQ JSON |
| `config/camera.env.example` | Default `BIRDSEYE_NAME=birdseye` |
| `omen/publish_d405.py` | D405 + webcam capture; payload helpers; `--list-webcams` |
| `omen/test_publish_d405.py` | Unit tests for JPEG/base64 payload helpers (no hardware) |
| `dashboard/zmq_jpeg.py` | Decode `images{}` from one ZMQ JSON dict |
| `dashboard/test_zmq_jpeg.py` | Unit tests for that decoder |
| `dashboard/app.py` | Two MJPEG routes and two labeled panes |
| `scripts/06_teleoperate.sh` | Print birdseye name when set |
| `scripts/08_record.sh` | Print birdseye name when set |
| `scripts/09_dashboard.sh` | Print birdseye name when set |
| `README.md` | How to pick `--webcam-index` and enable `BIRDSEYE_NAME` |
| `docs/checklist.md` | Plug in webcam + list indices |
| `omen/setup_omen.bat` | Point at `--list-webcams` |

---

### Task 1: Dual ZMQ camera JSON for LeRobot

**Files:**
- Modify: `scripts/test_common.sh`
- Modify: `scripts/_common.sh`
- Modify: `config/camera.env.example`

**Interfaces:**
- Consumes: existing `load_camera` / `robot_cameras_json` (ZMQ single-camera printf)
- Produces: `BIRDSEYE_NAME` env (empty string default); `robot_cameras_json` returns `{ front: {type: zmq, ...}, birdseye: {type: zmq, ...} }` when `BIRDSEYE_NAME` is non-empty, otherwise today’s single-camera fragment

- [ ] **Step 1: Write the failing dual-camera assertion**

Append this block to `scripts/test_common.sh` **before** the existing `echo "PASS: test_common.sh"` line. Keep the current zmq test (no `BIRDSEYE_NAME`) unchanged.

```bash
# zmq camera json without BIRDSEYE_NAME must stay single-camera
echo "$json" | grep -q 'camera_name: "front"'
if echo "$json" | grep -q birdseye; then
  echo "FAIL: birdseye must be absent when BIRDSEYE_NAME is unset"
  exit 1
fi

# zmq camera json with BIRDSEYE_NAME must list both cameras on the same host/port
cat > "$TMP/camera.env" <<EOF
CAMERA_BACKEND=zmq
ZMQ_HOST=192.168.1.50
ZMQ_PORT=5555
CAMERA_NAME=front
BIRDSEYE_NAME=birdseye
EOF
CAMERA_FILE="$TMP/camera.env"
load_camera
test "$BIRDSEYE_NAME" = "birdseye"
json="$(robot_cameras_json)"
echo "$json" | grep -q 'front: {type: zmq'
echo "$json" | grep -q 'birdseye: {type: zmq'
echo "$json" | grep -q 'camera_name: "front"'
echo "$json" | grep -q 'camera_name: "birdseye"'
echo "$json" | grep -q 'server_address: "192.168.1.50"'
echo "$json" | grep -q 'port: 5555'
```

- [ ] **Step 2: Run test to verify it fails**

From repo root `lerobot-so101-setup`:

```bash
bash scripts/test_common.sh
```

Expected: FAIL (`BIRDSEYE_NAME` empty/unset and/or JSON has no `birdseye` key). Do not implement `_common.sh` until this fails.

- [ ] **Step 3: Implement `BIRDSEYE_NAME` load + dual ZMQ JSON**

In `scripts/_common.sh` `load_camera`, after `ZMQ_PORT="${ZMQ_PORT:-5555}"` add:

```bash
  BIRDSEYE_NAME="${BIRDSEYE_NAME:-}"
```

Replace the ZMQ branch of `robot_cameras_json` with:

```bash
  elif [[ "$backend" == "zmq" ]]; then
    local json
    json="$(printf '{ %s: {type: zmq, server_address: "%s", port: %s, camera_name: "%s"}' \
      "${CAMERA_NAME:-front}" \
      "${ZMQ_HOST}" \
      "${ZMQ_PORT:-5555}" \
      "${CAMERA_NAME:-front}")"
    if [[ -n "${BIRDSEYE_NAME:-}" ]]; then
      json+="$(printf ', %s: {type: zmq, server_address: "%s", port: %s, camera_name: "%s"}' \
        "${BIRDSEYE_NAME}" \
        "${ZMQ_HOST}" \
        "${ZMQ_PORT:-5555}" \
        "${BIRDSEYE_NAME}")"
    fi
    printf '%s }' "$json"
```

Add this line to `config/camera.env.example` immediately after `CAMERA_NAME=front`:

```bash
BIRDSEYE_NAME=birdseye
```

Leave opencv / intelrealsense branches unchanged.

- [ ] **Step 4: Run tests and make sure they pass**

```bash
bash scripts/test_common.sh
```

Expected: `PASS: test_common.sh`

- [ ] **Step 5: Commit**

```bash
git add scripts/test_common.sh scripts/_common.sh config/camera.env.example
git commit -m "$(cat <<'EOF'
feat: emit front and birdseye ZMQ cameras for LeRobot

EOF
)"
```

---

### Task 2: Omen publisher helpers + webcam capture

**Files:**
- Create: `omen/test_publish_d405.py`
- Modify: `omen/publish_d405.py`
- Modify: `omen/setup_omen.bat`

**Interfaces:**
- Consumes: existing D405 pipeline and ZMQ PUB bind on `--port` (default 5555)
- Produces:
  - `bgr_to_jpeg_b64(bgr: np.ndarray, quality: int) -> str | None`
  - `zmq_payload(images: dict[str, str], ts: float) -> str` returning `json.dumps({"images": images, "ts": ts})`
  - `list_webcams(max_index: int = 8) -> list[tuple[int, int, int]]`
  - `open_webcam(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture | None`
  - CLI: `--birdseye-name` (default `birdseye`), `--webcam-index` (default `0`), `--no-birdseye`, `--list-webcams`
  - Live payload always includes `front`; includes `birdseye` only after a webcam has produced at least one JPEG; after that, dropped reads reuse `last_birdseye_b64` and still set the key

- [ ] **Step 1: Write the failing publisher unit tests**

Create `omen/test_publish_d405.py`:

```python
from __future__ import annotations

import json
import unittest

import cv2
import numpy as np

from publish_d405 import bgr_to_jpeg_b64, zmq_payload


class PayloadTests(unittest.TestCase):
    def test_zmq_payload_includes_both_keys(self) -> None:
        raw = zmq_payload({"front": "aaa", "birdseye": "bbb"}, 1.5)
        data = json.loads(raw)
        self.assertEqual(data["images"]["front"], "aaa")
        self.assertEqual(data["images"]["birdseye"], "bbb")
        self.assertEqual(data["ts"], 1.5)

    def test_bgr_to_jpeg_b64_roundtrip(self) -> None:
        bgr = np.zeros((16, 24, 3), dtype=np.uint8)
        bgr[:] = (0, 255, 0)
        b64 = bgr_to_jpeg_b64(bgr, 70)
        self.assertIsInstance(b64, str)
        self.assertGreater(len(b64), 10)
        buf = np.frombuffer(__import__("base64").b64decode(b64), dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape[2], 3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

From `lerobot-so101-setup` with the Omen env (or the project venv that has opencv):

```bash
cd omen
python -m unittest test_publish_d405.py -v
```

Expected: FAIL with `ImportError` for `bgr_to_jpeg_b64` / `zmq_payload` (or `cannot import name`).

- [ ] **Step 3: Implement helpers, webcam open, and the publish loop**

Replace `omen/publish_d405.py` with:

```python
"""Publish D405 RGB (and optional webcam birdseye) over ZMQ for the Mac.

LeRobot on the Mac subscribes with type=zmq. Message shape:
  {"images": {"front": "<base64 JPEG>", "birdseye": "<base64 JPEG>"}, "ts": ...}

Usage (Anaconda Prompt on the Omen):
  conda activate omen-d405
  python omen/publish_d405.py --bind 0.0.0.0 --port 5555
  python omen/publish_d405.py --list-webcams
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import zmq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D405 + optional webcam → ZMQ JPEG publisher (Omen)")
    p.add_argument("--bind", default="0.0.0.0", help="Bind address")
    p.add_argument("--port", type=int, default=5555, help="ZMQ PUB port")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--camera-name", default="front", help="Key in images{} for the D405")
    p.add_argument("--birdseye-name", default="birdseye", help="Key in images{} for the webcam")
    p.add_argument("--webcam-index", type=int, default=0, help="OpenCV index for the overhead webcam")
    p.add_argument("--no-birdseye", action="store_true", help="D405 only")
    p.add_argument("--list-webcams", action="store_true", help="Print OpenCV camera indices and exit")
    p.add_argument("--serial", default="", help="D405 serial; empty = first device")
    p.add_argument("--no-depth", action="store_true", help="RGB only (saves RAM)")
    p.add_argument(
        "--dump-dir",
        default="",
        help="If set, also write rgb_*.jpg and depth_*.png here for later copy to Mac",
    )
    p.add_argument("--dump-every", type=int, default=1, help="Save every Nth frame to dump-dir")
    return p.parse_args()


def bgr_to_jpeg_b64(bgr: np.ndarray, quality: int) -> str | None:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def zmq_payload(images: dict[str, str], ts: float) -> str:
    return json.dumps({"images": images, "ts": ts})


def list_webcams(max_index: int = 8) -> list[tuple[int, int, int]]:
    found: list[tuple[int, int, int]] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            found.append((i, w, h))
        cap.release()
    return found


def open_webcam(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def main() -> int:
    args = parse_args()
    if args.list_webcams:
        cams = list_webcams()
        if not cams:
            print("No OpenCV cameras found.", flush=True)
            return 1
        print("index  width  height", flush=True)
        for i, w, h in cams:
            print(f"{i:5d}  {w:5d}  {h:6d}", flush=True)
        print("Pick --webcam-index for the overhead USB webcam (not the D405 UVC device).", flush=True)
        return 0

    dump = Path(args.dump_dir) if args.dump_dir else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    if not args.no_depth:
        cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    print(f"Starting D405 pipeline {args.width}x{args.height}@{args.fps} …", flush=True)
    try:
        profile = pipeline.start(cfg)
    except Exception as e:
        print("Failed to start RealSense pipeline:", e, file=sys.stderr)
        print("Plug D405 into USB 3, close RealSense Viewer, then retry.", file=sys.stderr)
        return 1

    device = profile.get_device()
    serial = device.get_info(rs.camera_info.serial_number)
    print(f"Device serial: {serial}", flush=True)

    webcam = None
    last_birdseye_b64: str | None = None
    if not args.no_birdseye:
        webcam = open_webcam(args.webcam_index, args.width, args.height, args.fps)
        if webcam is None:
            print(
                f"Webcam index {args.webcam_index} failed to open; publishing {args.camera_name} only. "
                "Run --list-webcams and retry with --webcam-index.",
                file=sys.stderr,
            )
        else:
            print(f"Webcam index {args.webcam_index} → images.{args.birdseye_name}", flush=True)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://{args.bind}:{args.port}")
    print(f"ZMQ PUB tcp://{args.bind}:{args.port}  camera_name={args.camera_name}", flush=True)
    print("On the Mac, set ZMQ_HOST to this PC's LAN IP.", flush=True)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    frame_i = 0
    t0 = time.time()
    n = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            bgr = np.asanyarray(color.get_data())
            front_b64 = bgr_to_jpeg_b64(bgr, args.jpeg_quality)
            if not front_b64:
                continue
            images = {args.camera_name: front_b64}

            if webcam is not None:
                ok, wframe = webcam.read()
                if ok and wframe is not None:
                    bird = bgr_to_jpeg_b64(wframe, args.jpeg_quality)
                    if bird:
                        last_birdseye_b64 = bird
                if last_birdseye_b64:
                    images[args.birdseye_name] = last_birdseye_b64

            sock.send_string(zmq_payload(images, time.time()))

            if dump and frame_i % max(1, args.dump_every) == 0:
                ok, buf = cv2.imencode(".jpg", bgr, encode_params)
                if ok:
                    (dump / f"rgb_{frame_i:06d}.jpg").write_bytes(buf.tobytes())
                if not args.no_depth:
                    depth = frames.get_depth_frame()
                    if depth:
                        z = np.asanyarray(depth.get_data())
                        cv2.imwrite(str(dump / f"depth_{frame_i:06d}.png"), z)

            frame_i += 1
            n += 1
            if n % 30 == 0:
                dt = time.time() - t0
                print(f"published {n} frames ({n / max(dt, 1e-3):.1f} fps)", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        if webcam is not None:
            webcam.release()
        pipeline.stop()
        sock.close(0)
        ctx.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

In `omen/setup_omen.bat`, after the existing `python omen\publish_d405.py` hint, add:

```bat
echo Optional overhead webcam:
echo   python omen\publish_d405.py --list-webcams
echo   python omen\publish_d405.py --webcam-index N
```

- [ ] **Step 4: Run the unit tests and make sure they pass**

```bash
cd omen
python -m unittest test_publish_d405.py -v
```

Expected: `OK` (2 tests). Hardware is not required for this step.

Manual check after tests (not a unit test): with the webcam plugged in, `python omen\publish_d405.py --list-webcams` prints at least one index. Then `python omen\publish_d405.py --webcam-index N` logs `Webcam index N → images.birdseye`.

- [ ] **Step 5: Commit**

```bash
git add omen/publish_d405.py omen/test_publish_d405.py omen/setup_omen.bat
git commit -m "$(cat <<'EOF'
feat: publish webcam as birdseye next to D405 front

EOF
)"
```

---

### Task 3: Dashboard two panes from one ZMQ message

**Files:**
- Create: `dashboard/zmq_jpeg.py`
- Create: `dashboard/test_zmq_jpeg.py`
- Modify: `dashboard/app.py`

**Interfaces:**
- Consumes: ZMQ JSON with `images` dict of base64 JPEG strings (same payload as Task 2)
- Produces:
  - `jpegs_from_zmq_payload(data: dict) -> dict[str, bytes]`
  - Routes: `GET /mjpeg` (alias of front), `GET /mjpeg/{name}`
  - Home page: labeled **front** and **birdseye** panes plus the existing log

- [ ] **Step 1: Write the failing decoder tests**

Create `dashboard/test_zmq_jpeg.py`:

```python
from __future__ import annotations

import base64
import unittest

from zmq_jpeg import jpegs_from_zmq_payload


class ZmqJpegTests(unittest.TestCase):
    def test_decodes_front_and_birdseye(self) -> None:
        payload = {
            "images": {
                "front": base64.b64encode(b"FRONTJPEG").decode("ascii"),
                "birdseye": base64.b64encode(b"BIRDJPEG").decode("ascii"),
            },
            "ts": 1.0,
        }
        out = jpegs_from_zmq_payload(payload)
        self.assertEqual(out["front"], b"FRONTJPEG")
        self.assertEqual(out["birdseye"], b"BIRDJPEG")

    def test_skips_empty_and_missing_images(self) -> None:
        self.assertEqual(jpegs_from_zmq_payload({}), {})
        self.assertEqual(jpegs_from_zmq_payload({"images": {"front": ""}}), {})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd dashboard
python -m unittest test_zmq_jpeg.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'zmq_jpeg'` (or import error for `jpegs_from_zmq_payload`).

- [ ] **Step 3: Implement decoder and dual MJPEG dashboard**

Create `dashboard/zmq_jpeg.py`:

```python
"""Decode LeRobot-style ZMQ JSON camera payloads."""

from __future__ import annotations

import base64


def jpegs_from_zmq_payload(data: dict) -> dict[str, bytes]:
    images = data.get("images") or {}
    out: dict[str, bytes] = {}
    for name, b64 in images.items():
        if not isinstance(b64, str) or not b64:
            continue
        out[str(name)] = base64.b64decode(b64)
    return out
```

Replace `dashboard/app.py` with:

```python
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
```

- [ ] **Step 4: Run tests and make sure they pass**

```bash
cd dashboard
python -m unittest test_zmq_jpeg.py -v
```

Expected: `OK` (2 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/zmq_jpeg.py dashboard/test_zmq_jpeg.py dashboard/app.py
git commit -m "$(cat <<'EOF'
feat: show front and birdseye panes on the dashboard

EOF
)"
```

---

### Task 4: Operator docs and script banners

**Files:**
- Modify: `README.md`
- Modify: `docs/checklist.md`
- Modify: `scripts/06_teleoperate.sh`
- Modify: `scripts/08_record.sh`
- Modify: `scripts/09_dashboard.sh`

**Interfaces:**
- Consumes: `BIRDSEYE_NAME` from `load_camera` (Task 1)
- Produces: operators know to plug the webcam into the Omen, run `--list-webcams`, pass `--webcam-index`, and keep `BIRDSEYE_NAME=birdseye` in `camera.env`

- [ ] **Step 1: Update README Omen + camera sections**

In `README.md`, change the machines table / flow so the Omen hosts D405 **and** the overhead webcam. Replace the Omen setup steps 2–4 with:

```markdown
2. Plug D405 into a **USB 3** port (not a USB 2 hub). Plug the overhead USB webcam into any USB port on the Omen.
3. Double-click `omen/setup_omen.bat` (or run it from Anaconda Prompt)
4. List OpenCV indices, pick the webcam (not the D405 UVC device), then publish:
   `python omen\publish_d405.py --list-webcams`
   `python omen\publish_d405.py --webcam-index N`
   (`omen/run_publish.bat` still works; pass `--webcam-index N` after it)
```

After the camera.env copy step on the Mac, add: keep `BIRDSEYE_NAME=birdseye` so teleop/record subscribe to both keys. Set `BIRDSEYE_NAME=` empty for front-only.

In the teleop/site section, state that the dashboard shows **front** and **birdseye** side by side.

- [ ] **Step 2: Update the physical checklist**

In `docs/checklist.md` under Two machines, add:

```markdown
- [ ] **HP Omen:** overhead USB webcam plugged in (in addition to D405 USB 3)
- [ ] Ran `python omen\publish_d405.py --list-webcams` and set `--webcam-index` to the webcam
```

Under Mac software order step 7, add: leave `BIRDSEYE_NAME=birdseye` in `camera.env`.

- [ ] **Step 3: Print birdseye in teleop, record, and dashboard scripts**

In `scripts/06_teleoperate.sh`, change the Camera echo to:

```bash
  echo "Camera:        $CAMERA_NAME birdseye=${BIRDSEYE_NAME:-off} backend=$CAMERA_BACKEND zmq=${ZMQ_HOST:-}:${ZMQ_PORT:-} index=${CAMERA_INDEX:-} ${CAMERA_WIDTH}x${CAMERA_HEIGHT}@${CAMERA_FPS}"
```

In `scripts/08_record.sh`, change the Camera echo to:

```bash
echo "Camera:   $CAMERA_NAME birdseye=${BIRDSEYE_NAME:-off} backend=$CAMERA_BACKEND zmq=${ZMQ_HOST:-}:${ZMQ_PORT:-} index=${CAMERA_INDEX:-} depth=$CAMERA_USE_DEPTH"
```

In `scripts/09_dashboard.sh`, change the ZMQ echo to:

```bash
    echo "ZMQ: $ZMQ_HOST:${ZMQ_PORT:-5555} name=$CAMERA_NAME birdseye=${BIRDSEYE_NAME:-off}"
```

- [ ] **Step 4: Re-run the bash camera test so docs edits did not break helpers**

```bash
bash scripts/test_common.sh
```

Expected: `PASS: test_common.sh`

- [ ] **Step 5: Commit**

```bash
git add README.md docs/checklist.md scripts/06_teleoperate.sh scripts/08_record.sh scripts/09_dashboard.sh
git commit -m "$(cat <<'EOF'
docs: document Omen webcam birdseye stream

EOF
)"
```

---

## Plan self-review

1. **Spec coverage:** Dual JSON (`Task 1`), publisher + last-frame key (`Task 2`), dashboard two panes (`Task 3`), docs/index selection (`Task 4`). `--list-webcams`, `--no-birdseye`, missing webcam continues D405, no second port — all in Task 2.
2. **Placeholders:** None. Commands and file bodies are complete.
3. **Types:** `bgr_to_jpeg_b64` / `zmq_payload` / `jpegs_from_zmq_payload` names match across tasks. Keys stay `front` and `birdseye`. Port stays 5555.
