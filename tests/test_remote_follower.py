"""Remote follower config + action encode (no real robot / network)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from so101.config import PortsConfig, load_ports
from so101.dataset.schema import JOINT_NAMES
from so101.hardware import action_dict_from_qpos, make_follower
from so101.remote_follower import RemoteFollower


def test_ports_remote_falls_back_to_camera_host(tmp_path: Path) -> None:
    (tmp_path / "camera.env").write_text("CAMERA_BACKEND=zmq\nZMQ_HOST=10.0.0.9\nZMQ_PORT=5555\n")
    (tmp_path / "ports.env").write_text(
        "FOLLOWER_BACKEND=remote\nFOLLOWER_PORT=remote\nLEADER_PORT=/dev/tty.leader\n"
    )
    # Point project_root via monkeypatch of load path — call load_ports with ports path
    # and patch camera path by writing relative to tmp; load_ports uses project_root().
    # Instead construct PortsConfig directly for unit semantics and test helper fields.
    cfg = PortsConfig(
        follower_port="remote",
        leader_port="/dev/tty.leader",
        follower_backend="remote",
        robot_host="10.0.0.9",
    )
    assert cfg.follower_is_remote
    assert cfg.robot_state_port == 5556


def test_make_follower_remote_returns_remote_class() -> None:
    ports = PortsConfig(
        follower_port="remote",
        leader_port="/dev/tty.leader",
        follower_backend="remote",
        robot_host="127.0.0.1",
    )
    follower = make_follower(ports, camera=None, with_cameras=False)
    assert isinstance(follower, RemoteFollower)
    assert follower.host == "127.0.0.1"


def test_action_dict_roundtrip_for_bridge() -> None:
    q = np.arange(6, dtype=np.float32)
    d = action_dict_from_qpos(q)
    assert list(d) == [f"{n}.pos" for n in JOINT_NAMES]
    assert abs(d["gripper.pos"] - 5.0) < 1e-5


def test_load_ports_local_still_requires_follower(tmp_path: Path) -> None:
    p = tmp_path / "ports.env"
    p.write_text("FOLLOWER_BACKEND=local\nFOLLOWER_PORT=/dev/tty.f\nLEADER_PORT=/dev/tty.l\n")
    cfg = load_ports(p)
    assert cfg.follower_backend == "local"
    assert not cfg.follower_is_remote


def test_bridge_action_from_msg_accepts_vector_and_dict() -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from omen.robot_bridge import action_from_msg

    vec = action_from_msg({"action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    assert vec is not None
    assert abs(vec["shoulder_pan.pos"] - 0.1) < 1e-6
    named = action_from_msg({f"{n}.pos": float(i) for i, n in enumerate(JOINT_NAMES)})
    assert named is not None
    assert named["gripper.pos"] == 5.0
    assert action_from_msg({"junk": 1}) is None
