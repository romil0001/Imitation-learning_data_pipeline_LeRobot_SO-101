"""Phase-mix depth gating (offline, no robot)."""

from __future__ import annotations

import numpy as np

from so101.depth_grasp import DepthGraspConfig, DepthGraspController
from so101.rl.dense_reward import DenseDepthReward, DenseDepthRewardConfig
from so101.rl.phase_mix_policy import PhaseMixPolicy
from so101.rl.residual_sac import ResidualSAC, ResidualSACConfig
from so101.rl.train_loop import _should_learn


def _depth(h: int, w: int, fill_mm: int) -> np.ndarray:
    return np.full((h, w), fill_mm, dtype=np.uint16)


class _Base:
    cfg = type("C", (), {"use_depth": False, "depth_camera_name": "wrist"})()

    def select_action(self, image_rgb, state, depth_u16=None):
        del image_rgb, depth_u16
        out = np.asarray(state, dtype=np.float32).copy()
        out[:5] += 5.0
        out[5] = 55.0
        return out


class _FixedAgent:
    """Stand-in ResidualSAC with a fixed residual."""

    def __init__(self, delta: float = 1.0) -> None:
        self.cfg = ResidualSACConfig(hidden=8, device="cpu", max_delta_deg=2.0)
        self._delta = float(delta)

    def act(self, feat, deterministic: bool = True):
        del feat, deterministic
        return np.full(6, self._delta, dtype=np.float32)


def _policy(*, far: float = 280.0, near: float = 180.0, residual: float = 1.5) -> PhaseMixPolicy:
    reward = DenseDepthReward(DenseDepthRewardConfig(target_mm=95.0))
    grasp = DepthGraspController(
        DepthGraspConfig(
            enabled=True,
            enter_mm=near,
            exit_mm=far,
            target_mm=95.0,
            close_gripper=True,
            gripper_closed=2.0,
            track_enabled=False,
        )
    )
    return PhaseMixPolicy(
        _Base(),
        _FixedAgent(residual),  # type: ignore[arg-type]
        reward,
        grasp=grasp,
        explore=False,
        far_enter_mm=far,
        near_enter_mm=near,
        base_scale=1.0,
        residual_scale=1.0,
        low_pass_alpha=0.0,
        approach_dampen_mm=0.0,  # disable for unit tests
        approach_dampen_scale=1.0,
        auto_close=False,
    )


def test_phase_far_no_residual() -> None:
    policy = _policy()
    state = np.zeros(6, dtype=np.float32)
    out = policy.select_action(np.zeros((8, 8, 3), dtype=np.uint8), state, depth_u16=_depth(32, 32, 800))
    assert policy.last_phase == "far"
    assert policy.last_residual is not None
    np.testing.assert_allclose(policy.last_residual, 0.0, atol=1e-6)
    np.testing.assert_allclose(out[:5], 5.0, atol=1e-5)


def test_phase_mid_adds_residual() -> None:
    policy = _policy(residual=1.5)
    state = np.zeros(6, dtype=np.float32)
    out = policy.select_action(np.zeros((8, 8, 3), dtype=np.uint8), state, depth_u16=_depth(32, 32, 220))
    assert policy.last_phase == "mid"
    assert policy.last_residual is not None
    np.testing.assert_allclose(policy.last_residual, 1.5, atol=1e-5)
    np.testing.assert_allclose(out[:5], 5.0 + 1.5, atol=1e-4)


def test_phase_near_uses_grasp() -> None:
    policy = _policy()
    state = np.zeros(6, dtype=np.float32)
    state[5] = 55.0
    out = policy.select_action(np.zeros((8, 8, 3), dtype=np.uint8), state, depth_u16=_depth(32, 32, 100))
    assert policy.last_phase == "near"
    assert policy.last_residual is not None
    np.testing.assert_allclose(policy.last_residual, 0.0, atol=1e-6)
    # Grasp close_gripper engages in enter band → jaw commanded closed.
    assert float(out[5]) <= 10.0


def test_should_learn_mid_only() -> None:
    policy = _policy()
    assert _should_learn(policy, "mid") is True
    assert _should_learn(policy, "far") is False
    assert _should_learn(policy, "near") is False
    plain = type("P", (), {})()
    assert _should_learn(plain, None) is True


def test_phase_mix_rejects_bad_thresholds() -> None:
    try:
        PhaseMixPolicy(
            _Base(),
            ResidualSAC(ResidualSACConfig(hidden=8, device="cpu")),
            far_enter_mm=100.0,
            near_enter_mm=200.0,
        )
    except ValueError as exc:
        assert "near_enter_mm" in str(exc)
    else:
        raise AssertionError("expected ValueError")
