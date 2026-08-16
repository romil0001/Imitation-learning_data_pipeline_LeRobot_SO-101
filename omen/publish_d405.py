"""Publish D405 RGB (and optional depth dump) over ZMQ for the Mac.

LeRobot on the Mac subscribes with type=zmq. Message shape:
  {"images": {"front": "<base64 JPEG>"}}

Usage (Anaconda Prompt on the Omen):
  conda activate omen-d405
  python omen/publish_d405.py --bind 0.0.0.0 --port 5555
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
    p = argparse.ArgumentParser(description="D405 → ZMQ JPEG publisher (Omen)")
    p.add_argument("--bind", default="0.0.0.0", help="Bind address")
    p.add_argument("--port", type=int, default=5555, help="ZMQ PUB port")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--camera-name", default="front", help="Key in images{} for LeRobot")
    p.add_argument("--serial", default="", help="D405 serial; empty = first device")
    p.add_argument("--no-depth", action="store_true", help="RGB only (saves RAM)")
    p.add_argument(
        "--dump-dir",
        default="",
        help="If set, also write rgb_*.jpg and depth_*.png here for later copy to Mac",
    )
    p.add_argument("--dump-every", type=int, default=1, help="Save every Nth frame to dump-dir")
    return p.parse_args()


def main() -> int:
    args = parse_args()
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
            ok, buf = cv2.imencode(".jpg", bgr, encode_params)
            if not ok:
                continue
            payload = {
                "images": {args.camera_name: base64.b64encode(buf.tobytes()).decode("ascii")},
                "ts": time.time(),
            }
            sock.send_string(json.dumps(payload))

            if dump and frame_i % max(1, args.dump_every) == 0:
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
        pipeline.stop()
        sock.close(0)
        ctx.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
