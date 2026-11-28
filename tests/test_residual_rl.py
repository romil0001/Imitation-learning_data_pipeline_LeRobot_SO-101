"""Dense depth reward + residual SAC (offline, no robot)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from so101.rl.dense_reward import DenseDepthReward, DenseDepthRewardConfig
from so101.rl.residual_sac import ResidualSAC, ResidualSACConfig


def _depth(h: int, w: int, fill_mm: int) -> np.ndarray:
    return np.full((h, w), fill_mm, dtype=np.uint16)


def test_reward_prefers_target_depth() -> None:
    rw = DenseDepthReward(DenseDepthRewardConfig(target_mm=100.0, w_time=0.0, w_center=0.0, w_close=0.0))
    state = np.zeros(6, dtype=np.float32)
    far = rw.step(state, _depth(48, 64, 400))
    near = rw.step(state, _depth(48, 64, 110))
    assert near.reward > far.reward
    assert near.depth_mm is not None


def test_reward_prefers_closed_gripper_when_near() -> None:
    rw = DenseDepthReward(
        DenseDepthRewardConfig(target_mm=100.0, w_time=0.0, w_center=0.0, w_depth=0.0, w_close=2.0, w_grasp=0.0)
    )
    open_g = np.zeros(6, dtype=np.float32)
    open_g[5] = 55.0
    closed_g = np.zeros(6, dtype=np.float32)
    closed_g[5] = 2.0
    z = _depth(48, 64, 100)
    assert rw.step(closed_g, z).reward > rw.step(open_g, z).reward


def test_reward_features_dim() -> None:
    rw = DenseDepthReward()
    feat, _, _ = rw.features(np.arange(6, dtype=np.float32), _depth(32, 32, 200))
    assert feat.shape == (DenseDepthReward.feat_dim,)


def test_sac_update_and_save_load(tmp_path: Path) -> None:
    cfg = ResidualSACConfig(
        obs_dim=9,
        act_dim=6,
        hidden=32,
        batch_size=8,
        start_learning_after=8,
        buffer_size=64,
        device="cpu",
        max_delta_deg=3.0,
    )
    agent = ResidualSAC(cfg)
    rng = np.random.default_rng(0)
    for _ in range(16):
        o = rng.normal(size=9).astype(np.float32)
        a = rng.normal(size=6).astype(np.float32) * 0.5
        no = rng.normal(size=9).astype(np.float32)
        agent.push(o, a, float(rng.normal()), no, done=False)
    stats = agent.update()
    assert "q_loss" in stats
    agent.save(tmp_path)
    assert (tmp_path / "residual_sac.pt").is_file()
    loaded = ResidualSAC.load(tmp_path, device="cpu")
    o = np.zeros(9, dtype=np.float32)
    a = loaded.act(o, deterministic=True)
    assert a.shape == (6,)
    assert float(np.max(np.abs(a))) <= cfg.max_delta_deg + 1e-3


def test_residual_policy_adds_delta() -> None:
    from so101.rl.residual_policy import ResidualRLPolicy

    class _Base:
        cfg = type("C", (), {"use_depth": False, "depth_camera_name": "wrist"})()

        def select_action(self, image_rgb, state, depth_u16=None):
            del image_rgb, depth_u16
            return np.asarray(state, dtype=np.float32) + 10.0

    agent = ResidualSAC(ResidualSACConfig(hidden=16, device="cpu", max_delta_deg=2.0))
    # Far depth so auto_close does not override gripper.
    policy = ResidualRLPolicy(_Base(), agent, explore=False, auto_close=False)
    state = np.zeros(6, dtype=np.float32)
    out = policy.select_action(np.zeros((8, 8, 3), dtype=np.uint8), state, depth_u16=_depth(16, 16, 800))
    assert out.shape == (6,)
    assert policy.last_residual is not None
    # base wanted +10; residual bounded
    assert float(np.max(np.abs(out - 10.0))) <= 2.0 + 1e-2


def test_residual_policy_base_scale_slows() -> None:
    from so101.rl.residual_policy import ResidualRLPolicy

    class _Base:
        cfg = type("C", (), {"use_depth": False, "depth_camera_name": "wrist"})()

        def select_action(self, image_rgb, state, depth_u16=None):
            del image_rgb, depth_u16
            out = np.asarray(state, dtype=np.float32).copy()
            out[:5] += 40.0
            return out

    agent = ResidualSAC(ResidualSACConfig(hidden=16, device="cpu", max_delta_deg=0.01))
    policy = ResidualRLPolicy(
        _Base(), agent, explore=False, base_scale=0.25, residual_scale=0.0, low_pass_alpha=0.0
    )
    state = np.zeros(6, dtype=np.float32)
    out = policy.select_action(np.zeros((4, 4, 3), dtype=np.uint8), state, depth_u16=None)
    np.testing.assert_allclose(out[:5], np.full(5, 10.0), atol=0.05)
