#!/usr/bin/env python3
"""Publish synthetic JPEG frames over ZMQ for dashboard / teleop testing without hardware."""

from __future__ import annotations

import argparse
import base64
import json
import time

import cv2
import numpy as np
import zmq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mock D405 ZMQ camera publisher")
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--camera-name", default="front")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://{args.bind}:{args.port}")
    delay = 1.0 / max(args.fps, 1)
    print(
        f"Mock camera publishing to tcp://{args.bind}:{args.port} "
        f"({args.width}x{args.height}@{args.fps}) name={args.camera_name}",
        flush=True,
    )
    frame_i = 0
    try:
        while True:
            img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            hue = (frame_i * 3) % 180
            img[:] = cv2.cvtColor(
                np.full((1, 1, 3), (hue, 200, 200), dtype=np.uint8), cv2.COLOR_HSV2BGR
            )[0, 0]
            cv2.putText(
                img,
                f"mock D405  frame {frame_i}",
                (20, args.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                payload = {
                    "images": {args.camera_name: base64.b64encode(buf.tobytes()).decode("ascii")},
                    "ts": time.time(),
                }
                sock.send_string(json.dumps(payload))
            frame_i += 1
            time.sleep(delay)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        sock.close(0)
        ctx.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
