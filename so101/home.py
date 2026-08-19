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
