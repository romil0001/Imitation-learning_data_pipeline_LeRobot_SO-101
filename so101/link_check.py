"""Probe the Omen ZMQ publisher from the Mac — prove RGB (+ depth) is arriving."""

from __future__ import annotations

import base64
import json
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import zmq

from so101.config import load_cameras, project_root
from so101.zmq_rgbd import depth_colormap_bgr, parse_zmq_payload


@dataclass
class DepthStats:
    shape: tuple[int, ...]
    min_mm: int
    max_mm: int
    valid_frac: float


@dataclass
class LinkReport:
    host: str
    port: int
    tcp_ok: bool
    duration_s: float
    frames: int
    fps: float
    image_keys: list[str] = field(default_factory=list)
    depth_keys: list[str] = field(default_factory=list)
    image_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    depth_stats: dict[str, DepthStats] = field(default_factory=dict)
    expected_images: list[str] = field(default_factory=list)
    expect_depth: bool = False
    depth_name: str = "wrist"
    dump_dir: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.tcp_ok and self.frames > 0 and not self.errors


def tcp_reachable(host: str, port: int, timeout_s: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _depth_stats(z: np.ndarray) -> DepthStats:
    arr = np.asarray(z, dtype=np.uint16)
    valid = arr[arr > 0]
    if valid.size == 0:
        return DepthStats(shape=tuple(arr.shape), min_mm=0, max_mm=0, valid_frac=0.0)
    return DepthStats(
        shape=tuple(arr.shape),
        min_mm=int(valid.min()),
        max_mm=int(valid.max()),
        valid_frac=float(valid.size) / float(arr.size),
    )


def _save_dump(
    dump_dir: Path,
    rgb: dict[str, np.ndarray],
    depth: dict[str, np.ndarray],
) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    for name, img in rgb.items():
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(dump_dir / f"{name}.jpg"), bgr)
    for name, z in depth.items():
        cv2.imwrite(str(dump_dir / f"{name}_depth.png"), z)
        cv2.imwrite(str(dump_dir / f"{name}_depth_vis.jpg"), depth_colormap_bgr(z))


def probe_zmq_stream(
    host: str,
    port: int,
    *,
    seconds: float = 3.0,
    expected_images: list[str] | None = None,
    expect_depth: bool = True,
    depth_name: str = "wrist",
    dump_dir: Path | None = None,
    min_frames: int = 3,
) -> LinkReport:
    """Subscribe for ``seconds`` and verify expected image/depth keys arrive."""
    expected = list(expected_images or ["wrist", "overhead"])
    report = LinkReport(
        host=host,
        port=port,
        tcp_ok=False,
        duration_s=0.0,
        frames=0,
        fps=0.0,
        expected_images=expected,
        expect_depth=expect_depth,
        depth_name=depth_name,
    )

    report.tcp_ok = tcp_reachable(host, port)
    if not report.tcp_ok:
        report.errors.append(
            f"TCP {host}:{port} unreachable. On the Omen: omen\\run_publish.bat running, "
            f"omen\\open_firewall.bat once as Admin, same Wi-Fi, ZMQ_HOST correct."
        )
        return report

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.CONFLATE, True)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    sock.connect(f"tcp://{host}:{port}")

    image_keys: set[str] = set()
    depth_keys: set[str] = set()
    image_shapes: dict[str, tuple[int, ...]] = {}
    depth_stats: dict[str, DepthStats] = {}
    last_rgb: dict[str, np.ndarray] = {}
    last_depth: dict[str, np.ndarray] = {}
    frames = 0
    t0 = time.monotonic()
    deadline = t0 + max(0.5, seconds)

    try:
        while time.monotonic() < deadline:
            try:
                raw = sock.recv_string()
            except zmq.Again:
                continue
            except Exception as e:
                report.warnings.append(f"recv error: {e}")
                time.sleep(0.05)
                continue
            try:
                data: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                report.warnings.append("non-JSON ZMQ frame skipped")
                continue
            rgb, depth = parse_zmq_payload(data)
            if not rgb and not depth:
                continue
            frames += 1
            image_keys.update(rgb)
            depth_keys.update(depth)
            for name, img in rgb.items():
                image_shapes[name] = tuple(img.shape)
                last_rgb[name] = img
            for name, z in depth.items():
                depth_stats[name] = _depth_stats(z)
                last_depth[name] = z
    finally:
        sock.close(0)

    elapsed = max(time.monotonic() - t0, 1e-6)
    report.duration_s = elapsed
    report.frames = frames
    report.fps = frames / elapsed
    report.image_keys = sorted(image_keys)
    report.depth_keys = sorted(depth_keys)
    report.image_shapes = image_shapes
    report.depth_stats = depth_stats

    if frames < min_frames:
        report.errors.append(
            f"Only received {frames} frame(s) in {elapsed:.1f}s (need ≥{min_frames}). "
            "Publisher may be stuck or cameras failed to open on the Omen."
        )

    missing = [name for name in expected if name not in image_keys]
    if missing:
        report.errors.append(
            f"Missing RGB keys {missing}. Seen={sorted(image_keys)}. "
            "On Omen check --list-webcams / --webcam-index; Mac CAMERA_NAMES must match."
        )

    if expect_depth:
        if depth_name not in depth_keys:
            report.errors.append(
                f"Missing depth key {depth_name!r}. Seen={sorted(depth_keys)}. "
                "Restart Omen without --no-depth (normal: omen\\run_publish.bat)."
            )
        else:
            stats = depth_stats[depth_name]
            if stats.valid_frac < 0.01:
                report.errors.append(
                    f"Depth {depth_name!r} has almost no valid pixels "
                    f"(valid_frac={stats.valid_frac:.3f}). Cover lens / check USB 3."
                )
            elif stats.valid_frac < 0.05:
                report.warnings.append(
                    f"Depth {depth_name!r} valid_frac={stats.valid_frac:.3f} is low — "
                    "point the wrist at the scene."
                )

    if dump_dir is not None and (last_rgb or last_depth):
        _save_dump(dump_dir, last_rgb, last_depth)
        report.dump_dir = str(dump_dir)

    return report


def format_report(report: LinkReport) -> str:
    lines = [
        "=== Omen → Mac link check ===",
        f"Target: tcp://{report.host}:{report.port}",
        f"TCP: {'ok' if report.tcp_ok else 'FAIL'}",
        f"Frames: {report.frames} in {report.duration_s:.1f}s  (~{report.fps:.1f} fps)",
        f"RGB keys: {report.image_keys or '—'}",
        f"Depth keys: {report.depth_keys or '—'}",
    ]
    for name, shape in sorted(report.image_shapes.items()):
        lines.append(f"  image[{name}] shape={shape}")
    for name, st in sorted(report.depth_stats.items()):
        lines.append(
            f"  depth[{name}] shape={st.shape} min={st.min_mm}mm max={st.max_mm}mm "
            f"valid={st.valid_frac:.1%}"
        )
    lines.append(
        f"Expected RGB: {report.expected_images}  depth={'on' if report.expect_depth else 'off'} "
        f"({report.depth_name})"
    )
    if report.dump_dir:
        lines.append(f"Saved sample frames → {report.dump_dir}")
    for w in report.warnings:
        lines.append(f"WARN: {w}")
    if report.errors:
        lines.append("RESULT: FAIL")
        for e in report.errors:
            lines.append(f"  - {e}")
    else:
        lines.append("RESULT: PASS — wrist/overhead/depth (as configured) are transmitting.")
    return "\n".join(lines) + "\n"


def link_check_from_env(
    *,
    seconds: float = 3.0,
    host: str = "",
    port: int | None = None,
    dump: bool = True,
    min_frames: int = 3,
) -> LinkReport:
    cams = load_cameras()
    primary = cams[0]
    names = [c.name for c in cams]
    zmq_host = host or primary.zmq_host
    zmq_port = int(port if port is not None else primary.zmq_port)
    expect_depth = bool(primary.use_depth)
    depth_name = names[0] if names else "wrist"
    dump_dir = None
    if dump:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        dump_dir = project_root() / "outputs" / "link_check" / stamp
    return probe_zmq_stream(
        zmq_host,
        zmq_port,
        seconds=seconds,
        expected_images=names,
        expect_depth=expect_depth,
        depth_name=depth_name,
        dump_dir=dump_dir,
        min_frames=min_frames,
    )


def encode_fixture_payload(
    *,
    width: int = 64,
    height: int = 48,
    with_depth: bool = True,
) -> str:
    """Build one publisher-shaped JSON string for offline unit tests."""
    wrist = np.zeros((height, width, 3), dtype=np.uint8)
    wrist[:, :] = (40, 80, 160)
    overhead = np.zeros((height, width, 3), dtype=np.uint8)
    overhead[:, :] = (20, 160, 40)
    images = {}
    for name, bgr in (("wrist", wrist), ("overhead", overhead)):
        ok, buf = cv2.imencode(".jpg", bgr)
        assert ok
        images[name] = base64.b64encode(buf.tobytes()).decode("ascii")
    payload: dict[str, Any] = {"images": images, "ts": time.time()}
    if with_depth:
        z = np.full((height, width), 800, dtype=np.uint16)
        z[: height // 4, :] = 0
        ok, buf = cv2.imencode(".png", z)
        assert ok
        payload["depths"] = {"wrist": base64.b64encode(buf.tobytes()).decode("ascii")}
    return json.dumps(payload)
