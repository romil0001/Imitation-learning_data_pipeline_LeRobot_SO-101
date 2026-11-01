"""Move the leader arm to the follower pose, then release torque.

Used on Pause during human-gated DAgger. The leader stays where the operator
left it until this is called. SO-101 leaders are powered STS3215 servos, so the
Feetech bus can track a goal; upstream SO101Leader.send_feedback is often
unimplemented, and this module falls back to bus.sync_write("Goal_Position").
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from so101.dataset.schema import JOINT_NAMES
from so101.hardware import action_dict_from_qpos, qpos_from_obs


def sync_leader_to_follower(
    leader: Any,
    follower: Any,
    *,
    n_steps: int = 40,
    dt: float = 0.04,
    sleep: Callable[[float], None] = time.sleep,
) -> np.ndarray:
    """Hold the follower, interpolate the leader onto that pose, then disable torque.

    Returns the held follower qpos (length 6). Torque is turned off even if a
    later step raises, so the operator can grab the leader.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    target = qpos_from_obs(follower.get_observation())
    follower.send_action(action_dict_from_qpos(target))
    start = qpos_from_obs(leader.get_action())
    set_leader_torque(leader, True)
    try:
        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            q = ((1.0 - alpha) * start + alpha * target).astype(np.float32)
            write_leader_goal(leader, q)
            follower.send_action(action_dict_from_qpos(target))
            if dt > 0:
                sleep(dt)
    finally:
        set_leader_torque(leader, False)
    return target


def set_leader_torque(leader: Any, enabled: bool) -> None:
    bus = getattr(leader, "bus", None)
    if bus is not None:
        if enabled and hasattr(bus, "enable_torque"):
            bus.enable_torque()
            return
        if not enabled and hasattr(bus, "disable_torque"):
            bus.disable_torque()
            return
    setter = getattr(leader, "set_torque", None)
    if callable(setter):
        setter(enabled)
        return
    raise RuntimeError("Leader has no bus.enable_torque/disable_torque and no set_torque()")


def write_leader_goal(leader: Any, qpos: np.ndarray) -> None:
    action = action_dict_from_qpos(qpos)
    send = getattr(leader, "send_feedback", None)
    if callable(send):
        try:
            send(action)
            return
        except NotImplementedError:
            pass
    bus = getattr(leader, "bus", None)
    if bus is None or not hasattr(bus, "sync_write"):
        raise RuntimeError(
            "Leader cannot track a goal pose. Need send_feedback() or bus.sync_write('Goal_Position')."
        )
    values = {name: float(action[f"{name}.pos"]) for name in JOINT_NAMES}
    bus.sync_write("Goal_Position", values)
