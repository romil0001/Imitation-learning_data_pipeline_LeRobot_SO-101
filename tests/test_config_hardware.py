"""Tests for env loading and hardware factory (no robot connect)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from so101.config import CameraSettings, PortsConfig, load_camera, load_cameras, load_env_file, load_ports, project_root
from so101.dataset.schema import JOINT_NAMES
from so101.hardware import action_dict_from_qpos, camera_config_dict, qpos_from_obs, rgb_from_obs

ROOT = Path(__file__).resolve().parents[1]


def test_project_root_is_repo() -> None:
    assert (project_root() / "pyproject.toml").is_file()
    assert (project_root() / "config" / "ports.env.example").is_file()


def test_load_env_file_skips_comments(tmp_path: Path) -> None:
    path = tmp_path / "x.env"
    path.write_text("# hi\nFOO=bar\n\nBAZ='qux'\n")
    assert load_env_file(path) == {"FOO": "bar", "BAZ": "qux"}


def test_load_ports_rejects_equal(tmp_path: Path) -> None:
    path = tmp_path / "ports.env"
    path.write_text("FOLLOWER_PORT=/dev/tty.A\nLEADER_PORT=/dev/tty.A\n")
    with pytest.raises(ValueError, match="different"):
        load_ports(path)


def test_load_ports_rejects_placeholders() -> None:
    with pytest.raises(ValueError, match="CHANGE_ME"):
        load_ports(ROOT / "config" / "ports.env.example")


def test_load_ports_ok(tmp_path: Path) -> None:
    path = tmp_path / "ports.env"
    path.write_text(
        "FOLLOWER_PORT=/dev/tty.usbmodemA\n"
        "LEADER_PORT=/dev/tty.usbmodemB\n"
        "FOLLOWER_ID=my_follower_arm\n"
        "LEADER_ID=my_leader_arm\n"
        "FOLLOWER_P=10\nFOLLOWER_I=0\nFOLLOWER_D=48\nFOLLOWER_BUS_RETRIES=8\n"
    )
    ports = load_ports(path)
    assert isinstance(ports, PortsConfig)
    assert ports.follower_port == "/dev/tty.usbmodemA"
    assert ports.leader_port == "/dev/tty.usbmodemB"
    assert ports.follower_p == 10
    assert ports.follower_d == 48
    assert ports.follower_bus_retries == 8


def test_load_camera_zmq_rejects_placeholder() -> None:
    with pytest.raises(ValueError, match="ZMQ_HOST"):
        load_camera(ROOT / "config" / "camera.env.example")


def test_load_camera_zmq_ok(tmp_path: Path) -> None:
    path = tmp_path / "camera.env"
    path.write_text(
        "CAMERA_BACKEND=zmq\n"
        "ZMQ_HOST=192.168.1.50\n"
        "ZMQ_PORT=5555\n"
        "CAMERA_NAME=front\n"
        "CAMERA_WIDTH=640\n"
        "CAMERA_HEIGHT=480\n"
        "CAMERA_FPS=15\n"
        "CAMERA_USE_DEPTH=false\n"
    )
    cam = load_camera(path)
    assert isinstance(cam, CameraSettings)
    assert cam.backend == "zmq"
    assert cam.zmq_host == "192.168.1.50"
    assert cam.zmq_port == 5555
    assert cam.name == "front"
    assert cam.width == 640
    assert cam.height == 480
    assert cam.fps == 15


def test_load_cameras_wrist_and_overhead(tmp_path: Path) -> None:
    path = tmp_path / "camera.env"
    path.write_text(
        "CAMERA_BACKEND=zmq\n"
        "ZMQ_HOST=192.168.1.50\n"
        "ZMQ_PORT=5555\n"
        "CAMERA_NAMES=wrist,overhead\n"
        "CAMERA_WIDTH=640\n"
        "CAMERA_HEIGHT=480\n"
        "CAMERA_FPS=15\n"
    )
    cams = load_cameras(path)
    assert [c.name for c in cams] == ["wrist", "overhead"]
    assert all(c.zmq_host == "192.168.1.50" for c in cams)
    assert all(c.width == 640 and c.height == 480 for c in cams)


def test_camera_config_dict_zmq() -> None:
    cam = CameraSettings(
        backend="zmq",
        name="front",
        width=640,
        height=480,
        fps=15,
        zmq_host="192.168.1.50",
        zmq_port=5555,
        opencv_index=0,
        serial="",
        use_depth=False,
    )
    cfg = camera_config_dict(cam)
    assert cfg["type"] == "zmq"
    assert cfg["server_address"] == "192.168.1.50"
    assert cfg["port"] == 5555
    assert cfg["camera_name"] == "front"
    assert cfg["width"] == 640
    assert cfg["height"] == 480
    assert cfg["fps"] == 15


def test_qpos_from_obs_and_action_dict() -> None:
    obs = {f"{n}.pos": float(i) for i, n in enumerate(JOINT_NAMES)}
    obs["front"] = np.zeros((48, 64, 3), dtype=np.uint8)
    q = qpos_from_obs(obs)
    assert q.shape == (6,)
    np.testing.assert_allclose(q, np.arange(6, dtype=np.float32))
    d = action_dict_from_qpos(q)
    assert d["gripper.pos"] == 5.0
    rgb = rgb_from_obs(obs, "front")
    assert rgb.shape == (48, 64, 3)


def test_make_follower_and_leader_construct_without_connect() -> None:
    from so101.hardware import make_follower, make_leader

    ports = PortsConfig(follower_port="/dev/tty.usbmodemA", leader_port="/dev/tty.usbmodemB")
    cam = CameraSettings(
        backend="zmq",
        name="front",
        width=640,
        height=480,
        fps=15,
        zmq_host="192.168.1.50",
        zmq_port=5555,
        opencv_index=0,
        serial="",
        use_depth=False,
    )
    follower = make_follower(ports, cam, with_cameras=True)
    leader = make_leader(ports)
    assert follower.config.port == ports.follower_port
    assert "front" in follower.config.cameras
    assert leader.config.port == ports.leader_port


def test_make_follower_two_zmq_cameras() -> None:
    from so101.hardware import make_follower

    ports = PortsConfig(follower_port="/dev/tty.usbmodemA", leader_port="/dev/tty.usbmodemB")
    cams = [
        CameraSettings(
            backend="zmq",
            name="wrist",
            width=640,
            height=480,
            fps=15,
            zmq_host="192.168.1.50",
            zmq_port=5555,
        ),
        CameraSettings(
            backend="zmq",
            name="overhead",
            width=640,
            height=480,
            fps=15,
            zmq_host="192.168.1.50",
            zmq_port=5555,
        ),
    ]
    follower = make_follower(ports, cams, with_cameras=True)
    assert set(follower.config.cameras) == {"wrist", "overhead"}
