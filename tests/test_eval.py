"""Eval loop: policy.select_action -> follower (no real arms in tests)."""

from __future__ import annotations

import numpy as np

from so101.dataset.schema import JOINT_NAMES
from so101.eval import run_eval_loop
from so101.hardware import qpos_from_obs


class FakeFollower:
    def __init__(self) -> None:
        self.t = 0
        self.sent: list[np.ndarray] = []

    def get_observation(self) -> dict:
        q = np.zeros(6, dtype=np.float32)
        q[0] = float(self.t)
        img = np.zeros((24, 32, 3), dtype=np.uint8)
        obs = {f"{n}.pos": float(q[i]) for i, n in enumerate(JOINT_NAMES)}
        obs["front"] = img
        self.t += 1
        return obs

    def send_action(self, action: dict) -> dict:
        self.sent.append(qpos_from_obs(action))
        return action


class StubPolicy:
    def __init__(self) -> None:
        self.resets = 0
        self.seen_states: list[np.ndarray] = []

    def reset(self) -> None:
        self.resets += 1

    def select_action(self, image_rgb: np.ndarray, state: np.ndarray, depth_u16=None) -> np.ndarray:
        assert image_rgb.shape == (24, 32, 3)
        self.seen_states.append(state.copy())
        out = state.copy()
        out[1] = 42.0
        return out


def test_eval_loop_sends_policy_actions() -> None:
    follower = FakeFollower()
    policy = StubPolicy()
    n = run_eval_loop(
        follower=follower,
        policy=policy,
        camera_name="front",
        fps=15,
        duration_s=999,
        max_steps=5,
        sleep=lambda _s: None,
    )
    assert n == 5
    assert policy.resets == 1
    assert len(follower.sent) == 5
    assert follower.sent[0][1] == 42.0
    assert policy.seen_states[3][0] == 3.0
