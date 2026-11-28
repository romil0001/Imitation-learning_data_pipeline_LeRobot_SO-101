"""Base policy (ACT) + learned residual δa from ResidualSAC."""

from __future__ import annotations

from typing import Any

import numpy as np

from so101.rl.dense_reward import DenseDepthReward
from so101.rl.residual_sac import ResidualSAC


class ResidualRLPolicy:
    """``action = base(obs) + residual(features)`` with optional slowing / EMA."""

    def __init__(
        self,
        base: Any,
        agent: ResidualSAC,
        reward_fn: DenseDepthReward | None = None,
        *,
        explore: bool = True,
        base_scale: float = 1.0,
        residual_scale: float = 1.0,
        low_pass_alpha: float = 0.0,
        auto_close: bool = True,
        auto_close_tol_mm: float = 40.0,
        gripper_closed: float = 2.0,
    ) -> None:
        self.base = base
        self.agent = agent
        self.reward_fn = reward_fn or DenseDepthReward()
        self.explore = bool(explore)
        self.base_scale = float(np.clip(base_scale, 0.05, 1.0))
        self.residual_scale = float(np.clip(residual_scale, 0.0, 1.0))
        self.low_pass_alpha = float(np.clip(low_pass_alpha, 0.0, 1.0))
        self.auto_close = bool(auto_close)
        self.auto_close_tol_mm = float(auto_close_tol_mm)
        self.gripper_closed = float(gripper_closed)
        self.cfg = type("Cfg", (), {"use_depth": True, "depth_camera_name": "wrist"})()
        base_cfg = getattr(base, "cfg", None)
        if base_cfg is not None and hasattr(base_cfg, "depth_camera_name"):
            self.cfg.depth_camera_name = base_cfg.depth_camera_name
        self.last_residual: np.ndarray | None = None
        self.last_features: np.ndarray | None = None
        self.last_base: np.ndarray | None = None
        self._prev_out: np.ndarray | None = None

    def reset(self) -> None:
        if hasattr(self.base, "reset"):
            self.base.reset()
        self.reward_fn.reset()
        self.last_residual = None
        self.last_features = None
        self.last_base = None
        self._prev_out = None

    def select_action(
        self,
        image_rgb: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
        state: np.ndarray,
        depth_u16: np.ndarray | None = None,
    ) -> np.ndarray:
        try:
            base_a = self.base.select_action(image_rgb, state, depth_u16=depth_u16)
        except TypeError:
            base_a = self.base.select_action(image_rgb, state)
        base_a = np.asarray(base_a, dtype=np.float32).reshape(-1)
        st = np.asarray(state, dtype=np.float32).reshape(-1)
        # Slow ACT: only take a fraction of the commanded delta.
        if self.base_scale < 0.999:
            n_arm = min(5, base_a.shape[0], st.shape[0])
            base_a = base_a.copy()
            base_a[:n_arm] = st[:n_arm] + self.base_scale * (base_a[:n_arm] - st[:n_arm])

        feat, _, _ = self.reward_fn.features(state, depth_u16)
        residual = self.agent.act(feat, deterministic=not self.explore)
        residual = residual * self.residual_scale
        n = min(base_a.shape[0], residual.shape[0])
        out = base_a.copy()
        out[:n] = base_a[:n] + residual[:n]

        # When wrist depth is in the grasp band, force close (ACT often nudges forever).
        if self.auto_close and depth_u16 is not None and out.shape[0] > 5:
            _, med, _ = self.reward_fn.features(state, depth_u16)
            target = float(self.reward_fn.cfg.target_mm)
            if med is not None and abs(med - target) <= self.auto_close_tol_mm:
                out[5] = self.gripper_closed

        if self.low_pass_alpha > 1e-6:
            if self._prev_out is None:
                self._prev_out = out.copy()
            else:
                a = self.low_pass_alpha
                out = a * out + (1.0 - a) * self._prev_out
                self._prev_out = out.astype(np.float32, copy=True)

        self.last_base = base_a
        self.last_residual = residual
        self.last_features = feat
        return out.astype(np.float32)
