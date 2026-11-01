"""Mac-side follower that talks to omen/robot_bridge.py over ZMQ."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import numpy as np

from so101.dataset.schema import JOINT_NAMES


def _zmq():
    import zmq

    return zmq


class RemoteFollower:
    """Drop-in for local SO101Follower when the arm USB is on the Omen.

    - SUB joint state from ROBOT_STATE_PORT (default 5556)
    - PUB actions to ROBOT_CMD_PORT (default 5557)
    - Optional RGB from the existing camera ZMQ publisher (5555)
    """

    def __init__(
        self,
        host: str,
        *,
        state_port: int = 5556,
        cmd_port: int = 5557,
        camera_host: str | None = None,
        camera_port: int = 5555,
        camera_names: list[str] | None = None,
        with_cameras: bool = True,
        state_timeout_s: float = 2.0,
    ) -> None:
        self.host = host
        self.state_port = int(state_port)
        self.cmd_port = int(cmd_port)
        self.camera_host = camera_host or host
        self.camera_port = int(camera_port)
        self.camera_names = list(camera_names or [])
        self.with_cameras = bool(with_cameras) and bool(self.camera_names)
        self.state_timeout_s = float(state_timeout_s)
        self._ctx: Any = None
        self._sub: Any = None
        self._pub: Any = None
        self._cam: Any = None
        self._lock = threading.Lock()
        self._state: np.ndarray | None = None
        self._state_ts: float = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.is_connected = False

    def connect(self) -> None:
        if self.is_connected:
            return
        zmq = _zmq()
        self._ctx = zmq.Context.instance()
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.connect(f"tcp://{self.host}:{self.state_port}")
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub.setsockopt(zmq.CONFLATE, True)
        self._sub.setsockopt(zmq.RCVTIMEO, 2000)
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{self.host}:{self.cmd_port}")
        # PUB needs a moment after connect before first sends are reliable.
        time.sleep(0.2)
        self._stop.clear()
        self._thread = threading.Thread(target=self._state_loop, daemon=True)
        self._thread.start()
        if self.with_cameras:
            from so101.zmq_rgbd import LatestZMQRGBD

            self._cam = LatestZMQRGBD(self.camera_host, self.camera_port)
        deadline = time.monotonic() + self.state_timeout_s
        while time.monotonic() < deadline:
            if self._latest_state() is not None:
                self.is_connected = True
                print(
                    f"RemoteFollower connected  state={self.host}:{self.state_port}  "
                    f"cmd={self.host}:{self.cmd_port}",
                    flush=True,
                )
                return
            time.sleep(0.05)
        self.disconnect()
        raise RuntimeError(
            f"No follower state from {self.host}:{self.state_port} within {self.state_timeout_s:.0f}s. "
            "On the Omen run: python omen\\robot_bridge.py --port COMx"
        )

    def _state_loop(self) -> None:
        zmq = _zmq()
        assert self._sub is not None
        while not self._stop.is_set():
            try:
                data = json.loads(self._sub.recv_string())
            except zmq.Again:
                continue
            except Exception:
                time.sleep(0.05)
                continue
            raw = data.get("state")
            if raw is None:
                continue
            q = np.asarray(raw, dtype=np.float32).reshape(-1)
            if q.shape[0] != len(JOINT_NAMES):
                continue
            with self._lock:
                self._state = q
                self._state_ts = float(data.get("ts") or time.time())

    def _latest_state(self) -> np.ndarray | None:
        with self._lock:
            return None if self._state is None else self._state.copy()

    def get_observation(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.state_timeout_s
        state = None
        while time.monotonic() < deadline:
            state = self._latest_state()
            if state is not None:
                break
            time.sleep(0.01)
        if state is None:
            raise RuntimeError("RemoteFollower: timed out waiting for joint state")
        obs: dict[str, Any] = {f"{name}.pos": float(state[i]) for i, name in enumerate(JOINT_NAMES)}
        if self._cam is not None:
            for name in self.camera_names:
                rgb = self._cam.latest_rgb(name)
                if rgb is not None:
                    obs[name] = rgb
        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if self._pub is None:
            raise RuntimeError("RemoteFollower not connected")
        # Prefer compact action vector; bridge accepts either form.
        try:
            q = [float(action[f"{name}.pos"]) for name in JOINT_NAMES]
            payload = {"action": q, "ts": time.time()}
        except KeyError:
            payload = {**action, "ts": time.time()}
        self._pub.send_string(json.dumps(payload))
        return action

    def disconnect(self) -> None:
        self.is_connected = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cam is not None:
            self._cam.close()
            self._cam = None
        for sock in (self._sub, self._pub):
            if sock is not None:
                try:
                    sock.close(0)
                except Exception:
                    pass
        self._sub = None
        self._pub = None
