"""Load ports.env and camera.env (same keys as scripts/_common.sh)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        raise FileNotFoundError(f"Missing env file {path}")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def _as_bool(value: str, default: bool = False) -> bool:
    if value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PortsConfig:
    follower_port: str
    leader_port: str
    follower_id: str = "my_follower_arm"
    leader_id: str = "my_leader_arm"
    follower_p: int = 10
    follower_i: int = 0
    follower_d: int = 48
    follower_bus_retries: int = 8
    # local = USB on Mac; remote = follower USB on Omen via omen/robot_bridge.py
    follower_backend: str = "local"
    robot_host: str = ""
    robot_state_port: int = 5556
    robot_cmd_port: int = 5557

    @property
    def follower_is_remote(self) -> bool:
        return self.follower_backend.strip().lower() in {"remote", "zmq", "network"}


@dataclass(frozen=True)
class CameraSettings:
    backend: str
    name: str
    width: int
    height: int
    fps: int
    zmq_host: str = ""
    zmq_port: int = 5555
    opencv_index: int = 0
    serial: str = ""
    use_depth: bool = False


def load_ports(path: Path | None = None) -> PortsConfig:
    path = path or Path(project_root() / "config" / "ports.env")
    env = load_env_file(path)
    backend = (env.get("FOLLOWER_BACKEND", "local") or "local").strip().lower()
    remote = backend in {"remote", "zmq", "network"}
    follower = env.get("FOLLOWER_PORT", "")
    leader = env.get("LEADER_PORT", "")
    robot_host = (env.get("ROBOT_HOST", "") or "").strip()
    if not leader:
        raise ValueError(f"LEADER_PORT must be set in {path}")
    if leader.startswith("CHANGE_ME"):
        raise ValueError(f"Replace CHANGE_ME placeholders in {path}")
    if remote:
        if not robot_host or robot_host.startswith("CHANGE_ME"):
            try:
                cam_env = load_env_file(project_root() / "config" / "camera.env")
                robot_host = (cam_env.get("ZMQ_HOST", "") or "").strip()
            except FileNotFoundError:
                robot_host = ""
        if not robot_host or robot_host.startswith("CHANGE_ME"):
            raise ValueError(
                f"Remote follower needs ROBOT_HOST in {path} (or ZMQ_HOST in camera.env)"
            )
        if not follower:
            follower = "remote"
    else:
        if not follower:
            raise ValueError(f"FOLLOWER_PORT must be set in {path}")
        if follower.startswith("CHANGE_ME"):
            raise ValueError(f"Replace CHANGE_ME placeholders in {path}")
        if follower == leader:
            raise ValueError(f"FOLLOWER_PORT and LEADER_PORT must be different (got {follower})")
    return PortsConfig(
        follower_port=follower,
        leader_port=leader,
        follower_id=env.get("FOLLOWER_ID", "my_follower_arm"),
        leader_id=env.get("LEADER_ID", "my_leader_arm"),
        follower_p=int(env.get("FOLLOWER_P", "10")),
        follower_i=int(env.get("FOLLOWER_I", "0")),
        follower_d=int(env.get("FOLLOWER_D", "48")),
        follower_bus_retries=int(env.get("FOLLOWER_BUS_RETRIES", "8")),
        follower_backend=backend,
        robot_host=robot_host,
        robot_state_port=int(env.get("ROBOT_STATE_PORT", "5556")),
        robot_cmd_port=int(env.get("ROBOT_CMD_PORT", "5557")),
    )


def camera_name_list(env: dict[str, str]) -> list[str]:
    raw = env.get("CAMERA_NAMES", "").strip()
    if raw:
        names = [part.strip() for part in raw.split(",") if part.strip()]
        if names:
            return names
    return [env.get("CAMERA_NAME", "front")]


def load_cameras(path: Path | None = None) -> list[CameraSettings]:
    path = path or Path(project_root() / "config" / "camera.env")
    env = load_env_file(path)
    backend = env.get("CAMERA_BACKEND", "opencv")
    host = env.get("ZMQ_HOST", "")
    serial = env.get("CAMERA_SERIAL", "")
    if backend == "zmq" and (not host or host.startswith("CHANGE_ME")):
        raise ValueError(f"Set ZMQ_HOST in {path} to the HP Omen LAN IP")
    if backend == "intelrealsense" and (not serial or serial.startswith("CHANGE_ME")):
        raise ValueError(f"Set CAMERA_SERIAL in {path}")
    if backend == "opencv" and not env.get("CAMERA_INDEX"):
        # default index 0 is allowed if key missing; example files omit it when using zmq
        pass
    shared = dict(
        backend=backend,
        width=int(env.get("CAMERA_WIDTH", "640")),
        height=int(env.get("CAMERA_HEIGHT", "480")),
        fps=int(env.get("CAMERA_FPS", "15")),
        zmq_host=host,
        zmq_port=int(env.get("ZMQ_PORT", "5555")),
        opencv_index=int(env.get("CAMERA_INDEX", "0")),
        serial=serial,
        use_depth=_as_bool(env.get("CAMERA_USE_DEPTH", "false")),
    )
    return [CameraSettings(name=name, **shared) for name in camera_name_list(env)]


def load_camera(path: Path | None = None) -> CameraSettings:
    cameras = load_cameras(path)
    if not cameras:
        raise ValueError("No cameras configured")
    return cameras[0]
