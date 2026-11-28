"""Depth-gated mix: far=ACT, mid=ACT+residual SAC, near=depth-grasp."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from so101.depth_grasp import DepthGraspController, DepthGraspConfig
from so101.rl.dense_reward import DenseDepthReward
from so101.rl.residual_sac import ResidualSAC

PhaseName = Literal["far", "mid", "near"]


class PhaseMixPolicy:
    """``far → ACT``, ``mid → ACT + δ``, ``near → depth_grasp.override(ACT)``."""

    def __init__(
        self,
        base: Any,
        agent: ResidualSAC,
        reward_fn: DenseDepthReward | None = None,
        grasp: DepthGraspController | None = None,
        *,
        grasp_cfg: DepthGraspConfig | None = None,
        explore: bool = True,
        far_enter_mm: float = 280.0,
        near_enter_mm: float = 180.0,
        base_scale: float = 1.0,
        residual_scale: float = 1.0,
        low_pass_alpha: float = 0.0,
        approach_dampen_mm: float = 260.0,
        approach_dampen_scale: float = 0.35,
        auto_close: bool = True,
        auto_close_tol_mm: float = 40.0,
        gripper_closed: float = 2.0,
    ) -> None:
        self.base = base
        self.agent = agent
        self.reward_fn = reward_fn or DenseDepthReward()
        self.grasp = grasp or DepthGraspController(grasp_cfg or DepthGraspConfig())
        self.explore = bool(explore)
        self.far_enter_mm = float(far_enter_mm)
        self.near_enter_mm = float(near_enter_mm)
        if self.near_enter_mm >= self.far_enter_mm:
            raise ValueError(
                f"near_enter_mm ({self.near_enter_mm}) must be < far_enter_mm ({self.far_enter_mm})"
            )
        self.base_scale = float(np.clip(base_scale, 0.05, 1.0))
        self.residual_scale = float(np.clip(residual_scale, 0.0, 1.0))
        self.low_pass_alpha = float(np.clip(low_pass_alpha, 0.0, 1.0))
        self.approach_dampen_mm = float(approach_dampen_mm)
        self.approach_dampen_scale = float(np.clip(approach_dampen_scale, 0.05, 1.0))
        self.auto_close = bool(auto_close)
        self.auto_close_tol_mm = float(auto_close_tol_mm)
        self.gripper_closed = float(gripper_closed)
        self.cfg = type("Cfg", (), {"use_depth": True, "depth_camera_name": "wrist"})()
        base_cfg = getattr(base, "cfg", None)
        if base_cfg is not None and hasattr(base_cfg, "depth_camera_name"):
            self.cfg.depth_camera_name = base_cfg.depth_camera_name
        self.last_phase: PhaseName = "far"
        self.last_residual: np.ndarray | None = None
        self.last_features: np.ndarray | None = None
        self.last_base: np.ndarray | None = None
        self.last_depth_mm: float | None = None
        self._prev_out: np.ndarray | None = None

    def reset(self) -> None:
        if hasattr(self.base, "reset"):
            self.base.reset()
        self.reward_fn.reset()
        self.grasp.reset()
        self.last_phase = "far"
        self.last_residual = None
        self.last_features = None
        self.last_base = None
        self.last_depth_mm = None
        self._prev_out = None

    def _phase_for_depth(self, med: float | None) -> PhaseName:
        if med is None or med > self.far_enter_mm:
            return "far"
        if med <= self.near_enter_mm:
            return "near"
        return "mid"

    def _scaled_base(
        self,
        image_rgb: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
        state: np.ndarray,
        depth_u16: np.ndarray | None,
    ) -> np.ndarray:
        try:
            base_a = self.base.select_action(image_rgb, state, depth_u16=depth_u16)
        except TypeError:
            base_a = self.base.select_action(image_rgb, state)
        base_a = np.asarray(base_a, dtype=np.float32).reshape(-1)
        st = np.asarray(state, dtype=np.float32).reshape(-1)
        if self.base_scale < 0.999:
            n_arm = min(5, base_a.shape[0], st.shape[0])
            base_a = base_a.copy()
            base_a[:n_arm] = st[:n_arm] + self.base_scale * (base_a[:n_arm] - st[:n_arm])
        return base_a

    def _maybe_low_pass(self, out: np.ndarray) -> np.ndarray:
        if self.low_pass_alpha <= 1e-6:
            return out
        if self._prev_out is None:
            self._prev_out = out.copy()
            return out
        a = self.low_pass_alpha
        smoothed = a * out + (1.0 - a) * self._prev_out
        self._prev_out = smoothed.astype(np.float32, copy=True)
        return self._prev_out.copy()

    def select_action(
        self,
        image_rgb: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
        state: np.ndarray,
        depth_u16: np.ndarray | None = None,
    ) -> np.ndarray:
        feat, med, _ = self.reward_fn.features(state, depth_u16)
        phase = self._phase_for_depth(med)
        base_a = self._scaled_base(image_rgb, state, depth_u16)
        act_dim = int(self.agent.cfg.act_dim)
        zero_res = np.zeros(act_dim, dtype=np.float32)

        if phase == "far":
            out = base_a
            residual = zero_res
        elif phase == "mid":
            # Near the contact band: shrink ACT+δ arm motion so we don't drive through.
            mid_base = base_a
            mid_res_scale = self.residual_scale
            if med is not None and med < self.approach_dampen_mm:
                st = np.asarray(state, dtype=np.float32).reshape(-1)
                mid_base = base_a.copy()
                n_arm = min(5, mid_base.shape[0], st.shape[0])
                s = self.approach_dampen_scale
                mid_base[:n_arm] = st[:n_arm] + s * (mid_base[:n_arm] - st[:n_arm])
                mid_res_scale = self.residual_scale * s
            residual = self.agent.act(feat, deterministic=not self.explore)
            residual = residual * mid_res_scale
            n = min(mid_base.shape[0], residual.shape[0])
            out = mid_base.copy()
            out[:n] = mid_base[:n] + residual[:n]
            if self.auto_close and depth_u16 is not None and out.shape[0] > 5:
                target = float(self.reward_fn.cfg.target_mm)
                if med is not None and abs(med - target) <= self.auto_close_tol_mm:
                    out[5] = self.gripper_closed
        else:
            # near: classical depth-grasp on ACT (no residual credit)
            residual = zero_res
            out = self.grasp.override(base_a, state, depth_u16)

        out = self._maybe_low_pass(np.asarray(out, dtype=np.float32).reshape(-1))
        self.last_phase = phase
        self.last_base = base_a
        self.last_residual = residual
        self.last_features = feat
        self.last_depth_mm = med
        return out.astype(np.float32)
