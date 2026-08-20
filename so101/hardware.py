"""SO-101 follower/leader/camera construction. LeRobot imports live here only."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from so101.config import CameraSettings, PortsConfig
from so101.dataset.schema import JOINT_NAMES


def camera_config_dict(cam: CameraSettings) -> dict[str, Any]:
    if cam.backend == "zmq":
        return {
            "type": "zmq",
            "server_address": cam.zmq_host,
            "port": cam.zmq_port,
            "camera_name": cam.name,
            "width": cam.width,
            "height": cam.height,
            "fps": cam.fps,
        }
    if cam.backend == "opencv":
        return {
            "type": "opencv",
            "index_or_path": cam.opencv_index,
            "width": cam.width,
            "height": cam.height,
            "fps": cam.fps,
        }
    if cam.backend == "intelrealsense":
        return {
            "type": "intelrealsense",
            "serial_number_or_name": cam.serial,
            "width": cam.width,
            "height": cam.height,
            "fps": cam.fps,
            "use_depth": cam.use_depth,
            "use_rgb": True,
        }
    raise ValueError(f"Unknown CAMERA_BACKEND {cam.backend}")


def _one_lerobot_camera(cam: CameraSettings) -> dict[str, Any]:
    if cam.backend == "zmq":
        from lerobot.cameras.zmq import ZMQCameraConfig

        return {
            cam.name: ZMQCameraConfig(
                server_address=cam.zmq_host,
                port=cam.zmq_port,
                camera_name=cam.name,
                width=cam.width,
                height=cam.height,
                fps=cam.fps,
            )
        }
    if cam.backend == "opencv":
        from lerobot.cameras.opencv import OpenCVCameraConfig

        return {
            cam.name: OpenCVCameraConfig(
                index_or_path=cam.opencv_index,
                width=cam.width,
                height=cam.height,
                fps=cam.fps,
            )
        }
    if cam.backend == "intelrealsense":
        from lerobot.cameras.realsense import RealSenseCameraConfig

        return {
            cam.name: RealSenseCameraConfig(
                serial_number_or_name=cam.serial,
                width=cam.width,
                height=cam.height,
                fps=cam.fps,
                use_depth=cam.use_depth,
            )
        }
    raise ValueError(f"Unknown CAMERA_BACKEND {cam.backend}")


def make_lerobot_cameras(cameras: CameraSettings | Sequence[CameraSettings]) -> dict[str, Any]:
    """Build LeRobot CameraConfig objects keyed by camera name."""
    if isinstance(cameras, CameraSettings):
        cameras = [cameras]
    out: dict[str, Any] = {}
    for cam in cameras:
        out.update(_one_lerobot_camera(cam))
    return out


def make_follower(
    ports: PortsConfig,
    camera: CameraSettings | Sequence[CameraSettings] | None = None,
    *,
    with_cameras: bool = True,
) -> Any:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    cameras = make_lerobot_cameras(camera) if with_cameras and camera is not None else {}
    cfg = SO101FollowerConfig(
        port=ports.follower_port,
        id=ports.follower_id,
        cameras=cameras,
        position_p_coefficient=ports.follower_p,
        position_i_coefficient=ports.follower_i,
        position_d_coefficient=ports.follower_d,
        num_read_retries=ports.follower_bus_retries,
    )
    return SO101Follower(cfg)


def make_leader(ports: PortsConfig) -> Any:
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    cfg = SO101LeaderConfig(port=ports.leader_port, id=ports.leader_id)
    return SO101Leader(cfg)


def qpos_from_obs(obs: dict[str, Any]) -> np.ndarray:
    missing = [name for name in JOINT_NAMES if f"{name}.pos" not in obs]
    if missing:
        raise KeyError(f"Observation missing joints {missing}")
    return np.array([float(obs[f"{name}.pos"]) for name in JOINT_NAMES], dtype=np.float32)


def action_dict_from_qpos(qpos: np.ndarray) -> dict[str, float]:
    q = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if q.shape[0] != len(JOINT_NAMES):
        raise ValueError(f"qpos length {q.shape[0]} != {len(JOINT_NAMES)}")
    return {f"{name}.pos": float(q[i]) for i, name in enumerate(JOINT_NAMES)}


def rgb_from_obs(obs: dict[str, Any], camera_name: str) -> np.ndarray:
    if camera_name not in obs:
        raise KeyError(f"Observation missing camera '{camera_name}'")
    img = np.asarray(obs[camera_name])
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB for {camera_name}, got {img.shape}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def rgbs_from_obs(obs: dict[str, Any], camera_names: Sequence[str]) -> list[np.ndarray]:
    return [rgb_from_obs(obs, name) for name in camera_names]
