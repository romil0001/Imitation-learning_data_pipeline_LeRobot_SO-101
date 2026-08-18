# Dual-camera ZMQ: D405 `front` + webcam `birdseye`

**Date:** 2026-08-18  
**Status:** Approved for implementation planning  
**Host:** HP Omen (Windows) publishes; Mac subscribes  
**Scope:** Second USB webcam as a separate `birdseye` stream next to existing D405 `front`

## Goal

Publish the overhead USB webcam as `birdseye` in the same ZMQ message as the D405 `front` camera so the Mac dashboard shows both live panes and LeRobot teleop/record stores both streams.

## Constraints

- Cameras stay on the Omen; the Mac never opens the webcam
- One ZMQ PUB on TCP **5555** (no second port, no second Windows process)
- Message keys: `front` (D405) and `birdseye` (webcam)
- Webcam OpenCV index is configurable (Windows often also exposes the D405 as UVC)
- Empty `BIRDSEYE_NAME` on the Mac keeps today’s front-only LeRobot config
- If the webcam is missing, D405 publishing must continue
- If birdseye is enabled but a webcam frame drops, republish the last good JPEG under `birdseye` (never omit the key). LeRobot’s ZMQ camera falls back to the first image in `images{}` when the named key is missing, which would record `front` as `birdseye`
- No new Python dependencies; OpenCV and pyzmq are already in `omen/requirements.txt`
- Do not restructure unrelated bring-up scripts

## Architecture

One Omen process (`omen/publish_d405.py`) opens the RealSense D405 and, unless `--no-birdseye`, an OpenCV webcam. Each loop encodes both frames as JPEG/base64 and publishes:

```json
{"images": {"front": "<jpeg>", "birdseye": "<jpeg>"}, "ts": <float>}
```

The Mac dashboard uses one ZMQ SUB and displays `/mjpeg/front` and `/mjpeg/birdseye`. Teleop and record pass two `type: zmq` cameras in `--robot.cameras` (same `ZMQ_HOST` / `ZMQ_PORT`, different `camera_name`).

## Components

| Unit | Responsibility |
|------|----------------|
| `omen/publish_d405.py` | D405 + optional webcam; `--list-webcams`, `--webcam-index`, `--no-birdseye`, `--birdseye-name` |
| `dashboard/zmq_jpeg.py` | Decode `images{}` base64 JPEGs from one ZMQ JSON dict (no FastAPI import) |
| `dashboard/app.py` | One SUB; two MJPEG routes; two labeled panes + existing log |
| `scripts/_common.sh` `robot_cameras_json` | Emit two ZMQ camera entries when `BIRDSEYE_NAME` is set |
| `config/camera.env.example` | Document `BIRDSEYE_NAME=birdseye` |

## Data flow

1. Omen: D405 color frame → JPEG → `images.front`
2. Omen: webcam frame → JPEG → `images.birdseye` (or last good frame)
3. Omen: `sock.send_string` JSON on `tcp://0.0.0.0:5555`
4. Mac dashboard: SUB + CONFLATE → `jpegs_from_zmq_payload` → two MJPEG iterators
5. Mac LeRobot: two `ZMQCamera` instances, `camera_name=front` and `camera_name=birdseye`

## Error handling

- `--list-webcams` prints OpenCV indices (CAP_DSHOW then default backend) and exits
- Webcam open failure at startup: stderr warning, publish `front` only
- Webcam `read()` failure after a successful open: keep last `birdseye` JPEG in the payload
- D405 start failure: exit 1 (unchanged)
- Dashboard missing key: that pane shows a black JPEG, other pane keeps updating

## Testing

- `scripts/test_common.sh`: ZMQ JSON still works with only `CAMERA_NAME=front`; with `BIRDSEYE_NAME=birdseye` the fragment contains both names and two `type: zmq` blocks
- `omen/test_publish_d405.py`: `bgr_to_jpeg_b64` and `zmq_payload` without hardware
- `dashboard/test_zmq_jpeg.py`: decode both keys; ignore empty values

## Success criteria

- `run_publish.bat` (webcam plugged in, correct `--webcam-index`) publishes both keys
- http://127.0.0.1:8080 shows **front** and **birdseye** side by side
- `scripts/06_teleoperate.sh` and `scripts/08_record.sh` pass two ZMQ cameras so datasets contain both image keys
- Unplugged webcam does not stop the D405 publisher
