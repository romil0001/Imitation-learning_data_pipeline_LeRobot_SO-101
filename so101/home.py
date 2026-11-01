"""Slow joint-space return to a start / home pose."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from so101.hardware import action_dict_from_qpos, qpos_from_obs


def mean_start_qpos(dataset: Any) -> np.ndarray:
    n = int(dataset.meta.episode_count)
    if n < 1:
        raise ValueError("Dataset has no episodes to take a start pose from")
    acc = np.zeros(6, dtype=np.float64)
    for i in range(n):
        acc += np.asarray(dataset.load_episode(i).state[0], dtype=np.float64)
    return (acc / n).astype(np.float32)


def interpolate_to_qpos(
    follower: Any,
    target: np.ndarray,
    *,
    n_steps: int = 80,
    dt: float = 0.04,
    sleep: Callable[[float], None],
) -> None:
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    current = qpos_from_obs(follower.get_observation())
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    if target.shape[0] != current.shape[0]:
        raise ValueError(f"home qpos length {target.shape[0]} != {current.shape[0]}")
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        q = (1.0 - alpha) * current + alpha * target
        follower.send_action(action_dict_from_qpos(q))
        if dt > 0:
            sleep(dt)


def safe_power_down(
    follower: Any,
    target: np.ndarray | None,
    *,
    n_steps: int = 100,
    dt: float = 0.05,
    sleep: Callable[[float], None],
    hold_s: float = 0.3,
) -> None:
    """Slowly park at ``target`` before torque-off disconnect (Ctrl-C / exit).

    Avoids slamming the arm onto the table when ``disconnect()`` drops torque
    mid-pose. A second Ctrl-C during the move aborts the interpolate and
    returns so the caller can still disconnect.
    """
    if follower is None:
        return
    print("Slow power-down — moving to rest pose (Ctrl-C again skips park)…", flush=True)
    try:
        if target is not None:
            interpolate_to_qpos(follower, target, n_steps=n_steps, dt=dt, sleep=sleep)
            if hold_s > 0:
                sleep(hold_s)
        else:
            # Hold current pose briefly so the bus isn't cut mid-command.
            q = qpos_from_obs(follower.get_observation())
            for _ in range(max(1, int(0.5 / max(dt, 1e-3)))):
                follower.send_action(action_dict_from_qpos(q))
                if dt > 0:
                    sleep(dt)
    except KeyboardInterrupt:
        print("Park skipped — disconnecting.", flush=True)
    except Exception as exc:  # noqa: BLE001 — best-effort shutdown
        print(f"Power-down failed ({exc}); disconnecting anyway.", flush=True)
