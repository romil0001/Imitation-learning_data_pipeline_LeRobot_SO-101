"""Publish wrist D405 + optional base D405 + USB overhead over ZMQ for the Mac.

LeRobot / so101 on the Mac subscribes with type=zmq. Message shape:
  {"images": {"wrist": "<JPEG b64>", "overhead": "...", "base": "..."},
   "depths": {"wrist": "<uint16 PNG b64>", "base": "..."}}

Usage (Anaconda Prompt on the Omen):
  conda activate omen-d405
  python omen/publish_d405.py --list-realsense
  python omen/publish_d405.py --list-webcams
  # single D405 (legacy):
  python omen/publish_d405.py --webcam-index 2
  # dual D405 + overhead:
  python omen/publish_d405.py --wrist-serial <S1> --base-serial <S2> --webcam-index 2
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import zmq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D405 wrist (+ base) + USB overhead → ZMQ")
    p.add_argument("--bind", default="0.0.0.0", help="Bind address")
    p.add_argument("--port", type=int, default=5555, help="ZMQ PUB port")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--camera-name", default="wrist", help="Primary D405 key in images{}")
    p.add_argument("--base-name", default="base", help="Second D405 key when --base-serial set")
    p.add_argument("--overhead-name", default="overhead", help="Webcam key in images{}")
    p.add_argument(
        "--webcam-index",
        type=int,
        default=-1,
        help="OpenCV index for USB overhead cam (-1 = auto-pick)",
    )
    p.add_argument("--no-overhead", action="store_true", help="Publish D405 only")
    p.add_argument("--list-webcams", action="store_true", help="Print OpenCV camera indices and exit")
    p.add_argument("--list-realsense", action="store_true", help="Print RealSense serials and exit")
    p.add_argument("--serial", default="", help="Legacy alias for --wrist-serial")
    p.add_argument("--wrist-serial", default="", help="Wrist D405 serial; empty = first device")
    p.add_argument("--base-serial", default="", help="Base D405 serial (enables second depth cam)")
    p.add_argument("--no-depth", action="store_true", help="RGB only on all D405s (saves RAM/USB)")
    p.add_argument(
        "--base-no-depth",
        action="store_true",
        help="RGB-only on base D405 (keeps wrist depth; helps dual-D405 USB bandwidth)",
    )
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


def list_realsense() -> list[tuple[str, str]]:
    ctx = rs.context()
    out: list[tuple[str, str]] = []
    for dev in ctx.query_devices():
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        out.append((serial, name))
    return out


def _webcam_hint(width: int, height: int) -> str:
    if width == 1280 and height == 720:
        return "likely laptop built-in (HP Wide Vision)"
    if width == 1920 and height == 1080:
        return "likely USB overhead (XWF-1080P)"
    if width == 640 and height == 480:
        return "likely D405 RGB duplicate — do not use for overhead"
    return ""


def resolve_webcam_index(explicit: int) -> int:
    if explicit >= 0:
        return explicit
    found = list_webcams()
    if not found:
        raise RuntimeError("No OpenCV webcams found")
    usb = [(i, w, h) for i, w, h in found if (w, h) not in {(1280, 720), (640, 480)}]
    pool = usb or found
    index, width, height = max(pool, key=lambda row: row[1] * row[2])
    print(f"Auto-picked overhead webcam index={index} ({width}x{height})", flush=True)
    return index


def open_webcam(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, opencv_backend())
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {index}")
    return cap


def depth_png_b64(depth_u16: np.ndarray) -> str:
    z = np.asarray(depth_u16, dtype=np.uint16)
    ok, buf = cv2.imencode(".png", z)
    if not ok:
        raise RuntimeError("depth PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def jpeg_b64(bgr: np.ndarray, quality: int) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


@dataclass
class D405Stream:
    name: str
    serial: str
    pipeline: rs.pipeline
    align: rs.align | None


def start_d405(
    name: str,
    serial: str,
    width: int,
    height: int,
    fps: int,
    *,
    with_depth: bool,
) -> D405Stream:
    pipeline = rs.pipeline()
    cfg = rs.config()
    if serial:
        cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    if with_depth:
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(cfg)
    device = profile.get_device()
    got = device.get_info(rs.camera_info.serial_number)
    print(f"D405 {name!r} serial={got} depth={'on' if with_depth else 'off'}", flush=True)
    align = rs.align(rs.stream.color) if with_depth else None
    return D405Stream(name=name, serial=got, pipeline=pipeline, align=align)


def grab_d405(stream: D405Stream) -> tuple[np.ndarray, np.ndarray | None]:
    frames = stream.pipeline.wait_for_frames(timeout_ms=5000)
    if stream.align is not None:
        frames = stream.align.process(frames)
    color = frames.get_color_frame()
    if not color:
        raise RuntimeError(f"No color frame from {stream.name}")
    bgr = np.asanyarray(color.get_data())
    depth_u16 = None
    if stream.align is not None:
        depth = frames.get_depth_frame()
        if depth:
            depth_u16 = np.asanyarray(depth.get_data())
    return bgr, depth_u16


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

    if args.list_realsense:
        devices = list_realsense()
        if not devices:
            print("No RealSense devices. Plug D405(s) into USB 3.", file=sys.stderr)
            return 1
        print("RealSense devices (pass --wrist-serial / --base-serial):")
        for serial, name in devices:
            print(f"  {serial}  {name}")
        return 0

    dump = Path(args.dump_dir) if args.dump_dir else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)

    wrist_serial = (args.wrist_serial or args.serial or "").strip()
    base_serial = (args.base_serial or "").strip()
    with_depth = not args.no_depth

    if not base_serial:
        print(
            "WARNING: no --base-serial — publishing a single D405 as "
            f"{args.camera_name!r}. Dashboard base tile will stay offline. "
            "Set BASE_SERIAL in omen\\camera_serials.bat and restart.",
            flush=True,
        )
    streams: list[D405Stream] = []
    try:
        streams.append(
            start_d405(
                args.camera_name,
                wrist_serial,
                args.width,
                args.height,
                args.fps,
                with_depth=with_depth,
            )
        )
        if base_serial:
            if not wrist_serial:
                print(
                    "Warning: --base-serial set without --wrist-serial; "
                    "pin both serials so wrist/base do not swap after replug.",
                    flush=True,
                )
            if wrist_serial and base_serial == wrist_serial:
                raise RuntimeError(
                    f"--wrist-serial and --base-serial are the same ({wrist_serial})"
                )
            base_depth = with_depth and not args.base_no_depth
            streams.append(
                start_d405(
                    args.base_name,
                    base_serial,
                    args.width,
                    args.height,
                    args.fps,
                    with_depth=base_depth,
                )
            )
    except Exception as e:
        for s in streams:
            try:
                s.pipeline.stop()
            except Exception:
                pass
        print("Failed to start RealSense pipeline:", e, file=sys.stderr)
        print("Plug D405 into USB 3, close RealSense Viewer, then retry.", file=sys.stderr)
        print("List devices: python omen\\publish_d405.py --list-realsense", file=sys.stderr)
        return 1

    webcam = None
    webcam_index = -1
    if not args.no_overhead:
        try:
            webcam_index = resolve_webcam_index(args.webcam_index)
            webcam = open_webcam(webcam_index, args.width, args.height, args.fps)
        except Exception as e:
            for s in streams:
                s.pipeline.stop()
            label = args.webcam_index if args.webcam_index >= 0 else "auto"
            print(f"Failed to open overhead webcam ({label}): {e}", file=sys.stderr)
            print("Run: python omen\\publish_d405.py --list-webcams", file=sys.stderr)
            return 1
        print(f"Overhead webcam index={webcam_index}  name={args.overhead_name}", flush=True)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    try:
        sock.bind(f"tcp://{args.bind}:{args.port}")
    except zmq.ZMQError as e:
        for s in streams:
            try:
                s.pipeline.stop()
            except Exception:
                pass
        if webcam is not None:
            webcam.release()
        sock.close(0)
        ctx.term()
        print(
            f"ZMQ bind failed ({e}). Port {args.port} is already in use — "
            "stop the other publish_d405.py (or change --port) and retry.",
            file=sys.stderr,
        )
        return 1
    cam_keys = [s.name for s in streams] + ([] if webcam is None else [args.overhead_name])
    print(f"ZMQ PUB tcp://{args.bind}:{args.port}  cameras={cam_keys}", flush=True)
    print(
        "On the Mac set CAMERA_NAMES=" + ",".join(cam_keys) + " and ZMQ_HOST to this PC's LAN IP.",
        flush=True,
    )

    last_web: np.ndarray | None = None
    frame_i = 0
    t0 = time.time()
    n = 0
    try:
        while True:
            images: dict[str, str] = {}
            depths: dict[str, str] = {}
            for stream in streams:
                bgr, depth_u16 = grab_d405(stream)
                images[stream.name] = jpeg_b64(bgr, args.jpeg_quality)
                if depth_u16 is not None:
                    depths[stream.name] = depth_png_b64(depth_u16)
            if webcam is not None:
                ok, web_bgr = webcam.read()
                if ok and web_bgr is not None:
                    if web_bgr.shape[1] != args.width or web_bgr.shape[0] != args.height:
                        web_bgr = cv2.resize(web_bgr, (args.width, args.height))
                    last_web = web_bgr
                if last_web is None:
                    continue
                images[args.overhead_name] = jpeg_b64(last_web, args.jpeg_quality)
            payload: dict[str, object] = {"images": images, "ts": time.time()}
            if depths:
                payload["depths"] = depths
            sock.send_string(json.dumps(payload))

            if dump and frame_i % max(1, args.dump_every) == 0:
                for name, b64 in images.items():
                    (dump / f"{name}_{frame_i:06d}.jpg").write_bytes(base64.b64decode(b64))
                for name, b64 in depths.items():
                    raw = base64.b64decode(b64)
                    z = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    if z is not None:
                        cv2.imwrite(str(dump / f"{name}_depth_{frame_i:06d}.png"), z)

            frame_i += 1
            n += 1
            if n % 30 == 0:
                dt = time.time() - t0
                keys = list(images) + [f"depth:{k}" for k in depths]
                print(f"published {n} frames ({n / max(dt, 1e-3):.1f} fps) keys={keys}", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        for s in streams:
            try:
                s.pipeline.stop()
            except Exception:
                pass
        if webcam is not None:
            webcam.release()
        sock.close(0)
        ctx.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
