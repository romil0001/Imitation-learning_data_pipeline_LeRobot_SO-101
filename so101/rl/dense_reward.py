"""Dense rewards from wrist depth blob (centering + distance + optional grasp)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml

from so101.depth_grasp import nearest_depth_blob
from so101.dataset.schema import JOINT_NAMES


@dataclass
class DenseDepthRewardConfig:
    """Shaped reward for bag-clip approach / grasp (no learned model)."""

    target_mm: float = 120.0
    approach_max_mm: float = 1100.0
    min_mm: float = 40.0
    max_mm: float = 2000.0
    track_roi_frac: float = 0.75
    track_band_mm: float = 50.0
    # Weights
    w_center: float = 1.0
    w_depth: float = 1.0
    w_time: float = 0.01
    w_grasp: float = 2.0
    depth_scale_mm: float = 200.0  # |d-d*| / scale → ~1 at scale
    grasp_depth_tol_mm: float = 30.0
    gripper_closed: float = 2.0
    gripper_open: float = 55.0
    stall_gap: float = 2.5
    # Encourage closing when near (not just being close in depth).
    w_close: float = 1.5
    # Terminal bonus when depth near target and jaw closed / held
    success_bonus: float = 5.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DenseDepthRewardConfig:
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in fields})

    @classmethod
    def from_yaml(cls, path: str) -> DenseDepthRewardConfig:
        from pathlib import Path

        raw = yaml.safe_load(Path(path).read_text()) or {}
        reward = raw.get("reward", raw)
        if not isinstance(reward, dict):
            raise ValueError(f"reward config must be a mapping: {path}")
        return cls.from_dict(reward)


@dataclass
class RewardStep:
    reward: float
    depth_mm: float | None
    offset_xy: tuple[float, float] | None
    grasped: bool
    features: np.ndarray  # float32 feature vector for the residual critic/actor


def _gripper_closedness(state: np.ndarray, closed: float, opened: float) -> float:
    """1 = at closed pose, 0 = fully open (or beyond)."""
    gi = JOINT_NAMES.index("gripper")
    g = float(state[gi])
    span = max(1e-3, float(opened) - float(closed))
    return float(np.clip((float(opened) - g) / span, 0.0, 1.0))


def _gripper_held(state: np.ndarray, closed: float, stall_gap: float) -> bool:
    """True when jaw is near-closed or stopped short of closed (object in fingers)."""
    gi = JOINT_NAMES.index("gripper")
    g = float(state[gi])
    # Fully / nearly closed
    if abs(g - float(closed)) <= float(stall_gap):
        return True
    # Commanded closed region but residual (thin object): slightly open of closed
    return float(closed) < g <= float(closed) + 15.0


class DenseDepthReward:
    """r_t from wrist depth blob + optional grasp detect."""

    feat_dim: int = 9  # depth_n, ox, oy, qpos[6]

    def __init__(self, cfg: DenseDepthRewardConfig | None = None) -> None:
        self.cfg = cfg or DenseDepthRewardConfig()

    def reset(self) -> None:
        return None

    def features(
        self,
        state: np.ndarray,
        depth_u16: np.ndarray | None,
    ) -> tuple[np.ndarray, float | None, tuple[float, float] | None]:
        st = np.asarray(state, dtype=np.float32).reshape(-1)
        q = st[:6].copy() if st.size >= 6 else np.pad(st, (0, max(0, 6 - st.size)))[:6]
        depth_n = 1.0
        ox = oy = 0.0
        med: float | None = None
        offset: tuple[float, float] | None = None
        if depth_u16 is not None:
            blob = nearest_depth_blob(
                depth_u16,
                roi_frac=self.cfg.track_roi_frac,
                min_mm=self.cfg.min_mm,
                max_mm=self.cfg.max_mm,
                band_mm=self.cfg.track_band_mm,
            )
            if blob is not None:
                ox, oy, med = blob
                offset = (float(ox), float(oy))
                depth_n = float(
                    np.clip((med - self.cfg.target_mm) / self.cfg.depth_scale_mm, -3.0, 3.0)
                )
        feat = np.array([depth_n, ox, oy, *q.tolist()], dtype=np.float32)
        return feat, med, offset

    def step(
        self,
        state: np.ndarray,
        depth_u16: np.ndarray | None,
        *,
        done: bool = False,
    ) -> RewardStep:
        feat, med, offset = self.features(state, depth_u16)
        r = -float(self.cfg.w_time)
        grasped = False
        if med is None or offset is None:
            r -= 0.5  # no blob → mild penalty
            return RewardStep(reward=r, depth_mm=None, offset_xy=None, grasped=False, features=feat)

        ox, oy = offset
        r -= float(self.cfg.w_center) * abs(ox)
        r -= float(self.cfg.w_depth) * abs(med - self.cfg.target_mm) / self.cfg.depth_scale_mm

        near = abs(med - self.cfg.target_mm) <= self.cfg.grasp_depth_tol_mm
        close_frac = _gripper_closedness(state, self.cfg.gripper_closed, self.cfg.gripper_open)
        if near:
            # Prefer closing when in the grasp band (stops "nudge forever, never pinch").
            r += float(self.cfg.w_close) * close_frac
            # Penalty for staying open while already near the object.
            r -= float(self.cfg.w_close) * 0.5 * (1.0 - close_frac)
        held = _gripper_held(state, self.cfg.gripper_closed, self.cfg.stall_gap)
        if near and held and close_frac > 0.5:
            grasped = True
            r += float(self.cfg.w_grasp)
            if done:
                r += float(self.cfg.success_bonus)
        return RewardStep(
            reward=float(r),
            depth_mm=float(med),
            offset_xy=offset,
            grasped=grasped,
            features=feat,
        )
