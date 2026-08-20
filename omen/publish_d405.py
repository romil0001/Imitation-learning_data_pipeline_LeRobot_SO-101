"""Publish D405 wrist RGB + USB webcam overhead over ZMQ for the Mac.

LeRobot on the Mac subscribes with type=zmq. Message shape:
  {"images": {"wrist": "<base64 JPEG>", "overhead": "<base64 JPEG>"}}

Usage (Anaconda Prompt on the Omen):
  conda activate omen-d405
  python omen/publish_d405.py --list-webcams
  python omen/publish_d405.py --webcam-index 2
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
    p = argparse.ArgumentParser(description="D405 wrist + USB webcam overhead → ZMQ JPEG (Omen)")
    p.add_argument("--bind", default="0.0.0.0", help="Bind address")
    p.add_argument("--port", type=int, default=5555, help="ZMQ PUB port")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--camera-name", default="wrist", help="D405 key in images{}")
    p.add_argument("--overhead-name", default="overhead", help="Webcam key in images{}")
    p.add_argument(
        "--webcam-index",
        type=int,
        default=-1,
        help="OpenCV index for USB overhead cam (-1 = auto-pick highest-res non-laptop)",
    )
    p.add_argument("--no-overhead", action="store_true", help="Publish D405 only")
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


def opencv_backend() -> int:
    return cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


def list_webcams(max_index: int = 8) -> list[tuple[int, int, int]]:
    found: list[tuple[int, int, int]] = []
    backend = opencv_backend()
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            found.append((i, int(frame.shape[1]), int(frame.shape[0])))
        cap.release()
    return found


def _webcam_hint(width: int, height: int) -> str:
    if width == 1280 and height == 720:
        return "likely laptop built-in (HP Wide Vision)"
    if width == 1920 and height == 1080:
        return "likely USB overhead (XWF-1080P)"
    if width == 640 and height == 480:
        return "likely D405 RGB duplicate — do not use for overhead"
    return ""


def resolve_webcam_index(explicit: int) -> int:
    """Pick overhead USB cam. Index 1 is often empty on Windows; XWF-1080P is often index 2."""
    if explicit >= 0:
        return explicit
    found = list_webcams()
    if not found:
        raise RuntimeError("No OpenCV webcams found")
    # Drop typical laptop preview (1280x720) and low-res D405 duplicate (640x480).
    usb = [(i, w, h) for i, w, h in found if (w, h) not in {(1280, 720), (640, 480)}]
    pool = usb or found
    index, width, height = max(pool, key=lambda row: row[1] * row[2])
    print(
        f"Auto-picked overhead webcam index={index} ({width}x{height})",
        flush=True,
    )
    return index


def open_webcam(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, opencv_backend())
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {index}")
    return cap


def jpeg_b64(bgr: np.ndarray, quality: int) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def main() -> int:
    args = parse_args()
    if args.list_webcams:
        found = list_webcams()
        if not found:
            print("No OpenCV webcams opened. Plug in the USB camera and retry.", file=sys.stderr)
            return 1
        print("OpenCV camera indices (use --webcam-index N, or omit for auto-pick):")
        for idx, w, h in found:
            hint = _webcam_hint(w, h)
            extra = f"  — {hint}" if hint else ""
            print(f"  {idx}: {w}x{h}{extra}")
        try:
            auto = resolve_webcam_index(-1)
            print(f"Auto overhead pick: index {auto}")
        except RuntimeError:
            pass
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
    print(f"Device serial: {serial}  name={args.camera_name}", flush=True)

    webcam = None
    webcam_index = -1
    if not args.no_overhead:
        try:
            webcam_index = resolve_webcam_index(args.webcam_index)
            webcam = open_webcam(webcam_index, args.width, args.height, args.fps)
        except Exception as e:
            pipeline.stop()
            label = args.webcam_index if args.webcam_index >= 0 else "auto"
            print(f"Failed to open overhead webcam ({label}): {e}", file=sys.stderr)
            print("Run: python omen\\publish_d405.py --list-webcams", file=sys.stderr)
            print("Then: python omen\\publish_d405.py --webcam-index N  (XWF-1080P is often 2)", file=sys.stderr)
            return 1
        print(
            f"Overhead webcam index={webcam_index}  name={args.overhead_name}",
            flush=True,
        )

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://{args.bind}:{args.port}")
    keys = [args.camera_name] + ([] if webcam is None else [args.overhead_name])
    print(f"ZMQ PUB tcp://{args.bind}:{args.port}  cameras={keys}", flush=True)
    print("On the Mac, set ZMQ_HOST to this PC's LAN IP and CAMERA_NAMES=wrist,overhead.", flush=True)

    last_web: np.ndarray | None = None
    frame_i = 0
    t0 = time.time()
    n = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            wrist_bgr = np.asanyarray(color.get_data())
            images = {args.camera_name: jpeg_b64(wrist_bgr, args.jpeg_quality)}
            if webcam is not None:
                ok, web_bgr = webcam.read()
                if ok and web_bgr is not None:
                    if web_bgr.shape[1] != args.width or web_bgr.shape[0] != args.height:
                        web_bgr = cv2.resize(web_bgr, (args.width, args.height))
                    last_web = web_bgr
                if last_web is None:
                    continue
                images[args.overhead_name] = jpeg_b64(last_web, args.jpeg_quality)
            payload = {"images": images, "ts": time.time()}
            sock.send_string(json.dumps(payload))

            if dump and frame_i % max(1, args.dump_every) == 0:
                wrist_jpg = base64.b64decode(images[args.camera_name])
                (dump / f"rgb_{frame_i:06d}.jpg").write_bytes(wrist_jpg)
                if args.overhead_name in images:
                    over_jpg = base64.b64decode(images[args.overhead_name])
                    (dump / f"overhead_{frame_i:06d}.jpg").write_bytes(over_jpg)
                if not args.no_depth:
                    depth = frames.get_depth_frame()
                    if depth:
                        z = np.asanyarray(depth.get_data())
                        cv2.imwrite(str(dump / f"depth_{frame_i:06d}.png"), z)

            frame_i += 1
            n += 1
            if n % 30 == 0:
                dt = time.time() - t0
                print(f"published {n} frames ({n / max(dt, 1e-3):.1f} fps) keys={list(images)}", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        pipeline.stop()
        if webcam is not None:
            webcam.release()
        sock.close(0)
        ctx.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
