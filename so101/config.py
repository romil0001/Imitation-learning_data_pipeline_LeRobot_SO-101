"""Load ports.env and camera.env (same keys as scripts/_common.sh)."""

from __future__ import annotations

import socket
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


def zmq_host_candidates(env: dict[str, str]) -> list[str]:
    """Ordered ZMQ hosts: ZMQ_HOSTS (comma list) else ZMQ_HOST. Skips CHANGE_ME."""
    raw = env.get("ZMQ_HOSTS", "").strip()
    if raw:
        hosts = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        single = env.get("ZMQ_HOST", "").strip()
        hosts = [single] if single else []
    return [h for h in hosts if h and not h.startswith("CHANGE_ME")]


def tcp_reachable(host: str, port: int, timeout: float = 0.35) -> bool:
    if not host or host.startswith("CHANGE_ME"):
        return False
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def resolve_zmq_host(
    env: dict[str, str],
    *,
    port: int | None = None,
    timeout: float = 0.35,
) -> str:
    """Pick first reachable host from ZMQ_HOSTS / ZMQ_HOST; else first candidate.

    Startup-only failover: Ethernet listed before Wi-Fi in ZMQ_HOSTS.
    """
    candidates = zmq_host_candidates(env)
    if not candidates:
        raise ValueError(
            "Set ZMQ_HOST or ZMQ_HOSTS (e.g. 192.168.2.2,192.168.0.59) to the HP Omen IP(s)"
        )
    p = int(port if port is not None else env.get("ZMQ_PORT", "5555") or "5555")
    for host in candidates:
        if tcp_reachable(host, p, timeout=timeout):
            return host
    return candidates[0]


def zmq_link_mode(host: str, env: dict[str, str] | None = None) -> str:
    """Classify active ZMQ path for the dashboard: ethernet | wifi | lan | unknown.

    Convention: first entry in ZMQ_HOSTS is the direct Ethernet link; later entries
    are Wi-Fi / other LAN. Falls back to 192.168.2.x ⇒ ethernet.
    """
    if not host or host.startswith("CHANGE_ME"):
        return "unknown"
    candidates = zmq_host_candidates(env or {})
    if candidates:
        if host == candidates[0]:
            return "ethernet"
        if host in candidates[1:]:
            return "wifi"
    if host.startswith("192.168.2."):
        return "ethernet"
    if host.startswith("192.168."):
        return "wifi"
    return "lan"


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
    follower = env.get("FOLLOWER_PORT", "")
    leader = env.get("LEADER_PORT", "")
    if not follower or not leader:
        raise ValueError(f"FOLLOWER_PORT and LEADER_PORT must be set in {path}")
    if follower.startswith("CHANGE_ME") or leader.startswith("CHANGE_ME"):
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
    serial = env.get("CAMERA_SERIAL", "")
    zmq_port = int(env.get("ZMQ_PORT", "5555"))
    if backend == "zmq":
        try:
            host = resolve_zmq_host(env, port=zmq_port)
        except ValueError as exc:
            raise ValueError(f"{exc} (in {path})") from exc
    else:
        host = env.get("ZMQ_HOST", "")
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
        zmq_port=zmq_port,
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
