"""Capture the follower's current joints as protocol home_qpos."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from so101.dataset.schema import JOINT_NAMES
from so101.hardware import qpos_from_obs


def format_home_qpos_line(qpos: np.ndarray | list[float] | tuple[float, ...]) -> str:
    vals = ", ".join(f"{float(x):.3f}" for x in np.asarray(qpos, dtype=np.float64).reshape(-1))
    return f"home_qpos: [{vals}]"


def write_home_qpos_to_protocol(path: str | Path, qpos: np.ndarray | list[float] | tuple[float, ...]) -> str:
    """Insert or replace ``home_qpos`` in a protocol YAML. Returns the written line."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    line = format_home_qpos_line(qpos)
    text = path.read_text()
    if re.search(r"^home_qpos:\s*", text, flags=re.MULTILINE):
        text = re.sub(r"^home_qpos:\s*.*$", line, text, count=1, flags=re.MULTILINE)
    elif re.search(r"^home_steps:\s*", text, flags=re.MULTILINE):
        text = re.sub(
            r"(^home_steps:\s*.*$)",
            line + r"\n\1",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        text = line + "\n" + text
    path.write_text(text)
    return line


def capture_follower_qpos(follower: Any) -> np.ndarray:
    return np.asarray(qpos_from_obs(follower.get_observation()), dtype=np.float32)


def capture_home_from_env(
    *,
    protocol: str = "config/eval/bag_clip_in_bowl.yaml",
    confirm: bool = True,
    write: bool = True,
) -> np.ndarray:
    """Read live follower joints; optionally write them into the protocol YAML."""
    from so101.config import load_ports, project_root
    from so101.hardware import make_follower

    root = project_root()
    proto_path = Path(protocol)
    if not proto_path.is_file():
        proto_path = root / protocol
    ports = load_ports()
    print("CAPTURE HOME: FOLLOWER USB + 12V ON. Leader can be OFF.")
    print("Move the follower to the episode start pose you want, then continue.")
    print(f"Joints: {list(JOINT_NAMES)}")
    print(f"Protocol: {proto_path}")
    if confirm:
        input("Pose the arm, then press Enter to capture (Ctrl-C to abort) ")
    follower = make_follower(ports, camera=None, with_cameras=False)
    follower.connect()
    try:
        q = capture_follower_qpos(follower)
    finally:
        try:
            follower.disconnect()
        except Exception as exc:  # noqa: BLE001
            print(f"disconnect: {exc}", flush=True)
    print("Captured qpos:")
    for name, val in zip(JOINT_NAMES, q, strict=True):
        print(f"  {name:16s} {float(val):8.3f}")
    print(format_home_qpos_line(q))
    if write:
        line = write_home_qpos_to_protocol(proto_path, q)
        print(f"Wrote {line} → {proto_path}")
    return q
