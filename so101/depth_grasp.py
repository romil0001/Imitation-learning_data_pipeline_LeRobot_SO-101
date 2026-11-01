"""Metric wrist-depth grasp assist + blob tracking + gripper stall detect."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from so101.dataset.schema import JOINT_NAMES


@dataclass
class DepthGraspConfig:
    """Tune for keys-on-table with D405 wrist cam (mm)."""

    enabled: bool = True
    roi_frac: float = 0.3
    enter_mm: float = 180.0
    target_mm: float = 95.0
    exit_mm: float = 280.0
    tol_mm: float = 15.0
    min_mm: float = 40.0
    max_mm: float = 2000.0
    gain_deg_per_mm: float = 0.08
    vertical_joints: tuple[str, ...] = ("shoulder_lift", "elbow_flex")
    # Positive sign: increase joint when object is farther than target (depth > target).
    joint_signs: dict[str, float] = field(
        default_factory=lambda: {"shoulder_lift": 1.0, "elbow_flex": 1.0}
    )
    max_delta_deg: float = 4.0
    close_gripper: bool = True
    gripper_closed: float = 15.0
    hold_close_frames: int = 8
    # Nearest-depth blob tracking (image X → lateral joints).
    track_enabled: bool = True
    track_roi_frac: float = 0.7
    track_band_mm: float = 40.0
    gain_deg_per_norm: float = 8.0
    lateral_joints: tuple[str, ...] = ("shoulder_pan",)
    lateral_signs: dict[str, float] = field(default_factory=lambda: {"shoulder_pan": 1.0})
    # Grasp held? gripper position residual after close (no extra sensor).
    detect_grasp: bool = True
    stall_gap: float = 4.0
    gripper_open: float = 55.0
    retry_on_miss: int = 1
    depth_confirm: bool = False
    depth_confirm_tol_mm: float = 25.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DepthGraspConfig:
        fields = cls.__dataclass_fields__
        kwargs: dict[str, Any] = {}
        for k, v in data.items():
            if k not in fields:
                continue
            if k in {"vertical_joints", "lateral_joints"}:
                kwargs[k] = tuple(str(x) for x in v)
            elif k in {"joint_signs", "lateral_signs"}:
                kwargs[k] = {str(a): float(b) for a, b in dict(v).items()}
            else:
                kwargs[k] = v
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "roi_frac": self.roi_frac,
            "enter_mm": self.enter_mm,
            "target_mm": self.target_mm,
            "exit_mm": self.exit_mm,
            "tol_mm": self.tol_mm,
            "min_mm": self.min_mm,
            "max_mm": self.max_mm,
            "gain_deg_per_mm": self.gain_deg_per_mm,
            "vertical_joints": list(self.vertical_joints),
            "joint_signs": dict(self.joint_signs),
            "max_delta_deg": self.max_delta_deg,
            "close_gripper": self.close_gripper,
            "gripper_closed": self.gripper_closed,
            "hold_close_frames": self.hold_close_frames,
            "track_enabled": self.track_enabled,
            "track_roi_frac": self.track_roi_frac,
            "track_band_mm": self.track_band_mm,
            "gain_deg_per_norm": self.gain_deg_per_norm,
            "lateral_joints": list(self.lateral_joints),
            "lateral_signs": dict(self.lateral_signs),
            "detect_grasp": self.detect_grasp,
            "stall_gap": self.stall_gap,
            "gripper_open": self.gripper_open,
            "retry_on_miss": self.retry_on_miss,
            "depth_confirm": self.depth_confirm,
            "depth_confirm_tol_mm": self.depth_confirm_tol_mm,
        }


def median_roi_depth_mm(
    depth_u16: np.ndarray,
    *,
    roi_frac: float = 0.3,
    min_mm: float = 40.0,
    max_mm: float = 2000.0,
) -> float | None:
    """Median depth (mm) in a centered ROI; None if no valid pixels."""
    z = np.asarray(depth_u16)
    if z.ndim != 2:
        raise ValueError(f"depth must be HxW, got {z.shape}")
    h, w = z.shape
    rf = float(np.clip(roi_frac, 0.05, 1.0))
    rh, rw = max(1, int(h * rf)), max(1, int(w * rf))
    y0 = (h - rh) // 2
    x0 = (w - rw) // 2
    patch = z[y0 : y0 + rh, x0 : x0 + rw].astype(np.float32)
    valid = patch[(patch >= min_mm) & (patch <= max_mm)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def nearest_depth_blob(
    depth_u16: np.ndarray,
    *,
    roi_frac: float = 0.7,
    min_mm: float = 40.0,
    max_mm: float = 2000.0,
    band_mm: float = 40.0,
) -> tuple[float, float, float] | None:
    """Nearest valid depth cluster in a centered ROI.

    Returns ``(offset_x, offset_y, depth_mm)`` where offsets are normalized to
    roughly [-1, 1] relative to the full image center (right/down positive),
    or ``None`` if no valid pixels.
    """
    z = np.asarray(depth_u16)
    if z.ndim != 2:
        raise ValueError(f"depth must be HxW, got {z.shape}")
    h, w = z.shape
    rf = float(np.clip(roi_frac, 0.05, 1.0))
    rh, rw = max(1, int(h * rf)), max(1, int(w * rf))
    y0 = (h - rh) // 2
    x0 = (w - rw) // 2
    patch = z[y0 : y0 + rh, x0 : x0 + rw].astype(np.float32)
    valid = (patch >= min_mm) & (patch <= max_mm)
    if not np.any(valid):
        return None
    z_min = float(patch[valid].min())
    blob = valid & (patch <= z_min + max(float(band_mm), 1.0))
    ys, xs = np.nonzero(blob)
    if ys.size == 0:
        return None
    cy = float(ys.mean()) + y0
    cx = float(xs.mean()) + x0
    depth = float(np.median(patch[blob]))
    # Normalize by half-width/height so ±1 ≈ image edge.
    offset_x = (cx - (w - 1) * 0.5) / max(w * 0.5, 1.0)
    offset_y = (cy - (h - 1) * 0.5) / max(h * 0.5, 1.0)
    return float(offset_x), float(offset_y), depth


def gripper_stall_held(
    measured: float,
    commanded_closed: float,
    *,
    stall_gap: float,
) -> bool:
    """True when the jaw stops short of the closed command (object in fingers)."""
    return abs(float(measured) - float(commanded_closed)) >= float(stall_gap)


class DepthGraspController:
    """State machine: idle → grasping → closing → done (until reset)."""

    def __init__(self, cfg: DepthGraspConfig | None = None) -> None:
        self.cfg = cfg or DepthGraspConfig()
        self.phase: str = "idle"
        self._close_left: int = 0
        self._retries_left: int = max(0, int(self.cfg.retry_on_miss))
        self.last_depth_mm: float | None = None
        self.last_offset_xy: tuple[float, float] | None = None
        self.grasp_held: bool | None = None
        self.last_gripper_residual: float | None = None

    def reset(self) -> None:
        self.phase = "idle"
        self._close_left = 0
        self._retries_left = max(0, int(self.cfg.retry_on_miss))
        self.last_depth_mm = None
        self.last_offset_xy = None
        self.grasp_held = None
        self.last_gripper_residual = None

    def override(
        self,
        action: np.ndarray,
        state: np.ndarray,
        depth_u16: np.ndarray | None,
    ) -> np.ndarray:
        out = np.asarray(action, dtype=np.float32).copy()
        if not self.cfg.enabled or depth_u16 is None or self.phase == "done":
            return out

        offset_xy: tuple[float, float] | None = None
        med: float | None
        if self.cfg.track_enabled:
            blob = nearest_depth_blob(
                depth_u16,
                roi_frac=self.cfg.track_roi_frac,
                min_mm=self.cfg.min_mm,
                max_mm=self.cfg.max_mm,
                band_mm=self.cfg.track_band_mm,
            )
            if blob is not None:
                ox, oy, med = blob
                offset_xy = (ox, oy)
            else:
                med = None
        else:
            med = median_roi_depth_mm(
                depth_u16,
                roi_frac=self.cfg.roi_frac,
                min_mm=self.cfg.min_mm,
                max_mm=self.cfg.max_mm,
            )
        self.last_depth_mm = med
        self.last_offset_xy = offset_xy

        if self.phase == "closing":
            return self._force_close(out, state)

        if med is None:
            return out

        if self.phase == "idle":
            if med <= self.cfg.enter_mm:
                self.phase = "grasping"
            else:
                # Still nudge laterally while approaching if we see a near-ish blob.
                if offset_xy is not None and med <= self.cfg.exit_mm:
                    self._apply_lateral(out, state, offset_xy[0])
                return out

        if self.phase == "grasping" and med >= self.cfg.exit_mm:
            self.phase = "idle"
            return out

        if self.phase != "grasping":
            return out

        if offset_xy is not None:
            self._apply_lateral(out, state, offset_xy[0])

        error = med - self.cfg.target_mm  # >0 → too far
        for name in self.cfg.vertical_joints:
            if name not in JOINT_NAMES:
                continue
            i = JOINT_NAMES.index(name)
            sign = float(self.cfg.joint_signs.get(name, 1.0))
            delta = sign * self.cfg.gain_deg_per_mm * error
            delta = float(np.clip(delta, -self.cfg.max_delta_deg, self.cfg.max_delta_deg))
            # Correct around current state so we don't fight stale chunk targets too hard.
            out[i] = float(state[i]) + delta

        if abs(error) <= self.cfg.tol_mm and self.cfg.close_gripper:
            self.phase = "closing"
            self._close_left = max(1, int(self.cfg.hold_close_frames))
            return self._force_close(out, state)
        return out

    def _apply_lateral(self, out: np.ndarray, state: np.ndarray, offset_x: float) -> None:
        """Steer so the nearest blob moves toward image center (offset_x → 0)."""
        for name in self.cfg.lateral_joints:
            if name not in JOINT_NAMES:
                continue
            i = JOINT_NAMES.index(name)
            sign = float(self.cfg.lateral_signs.get(name, 1.0))
            # Object right of center (offset_x > 0) → signed pan toward it.
            delta = sign * self.cfg.gain_deg_per_norm * float(offset_x)
            delta = float(np.clip(delta, -self.cfg.max_delta_deg, self.cfg.max_delta_deg))
            out[i] = float(state[i]) + delta

    def _force_close(self, action: np.ndarray, state: np.ndarray) -> np.ndarray:
        gi = JOINT_NAMES.index("gripper")
        action[gi] = float(self.cfg.gripper_closed)
        self._close_left -= 1
        if self._close_left > 0:
            return action

        if not self.cfg.detect_grasp:
            self.phase = "done"
            return action

        measured = float(np.asarray(state, dtype=np.float32)[gi])
        residual = abs(measured - float(self.cfg.gripper_closed))
        self.last_gripper_residual = residual
        held_stall = gripper_stall_held(
            measured, self.cfg.gripper_closed, stall_gap=self.cfg.stall_gap
        )
        # Thin objects (keys) often close almost fully — residual stays small.
        # Optional depth confirm: object still near target_mm after close ⇒ held.
        held_depth = False
        d = self.last_depth_mm
        if self.cfg.depth_confirm and d is not None:
            held_depth = abs(d - self.cfg.target_mm) <= self.cfg.depth_confirm_tol_mm
        if self.cfg.depth_confirm:
            held = held_stall or held_depth
        else:
            held = held_stall
        self.grasp_held = held
        if held:
            why = []
            if held_stall:
                why.append(f"stall residual={residual:.2f}>={self.cfg.stall_gap}")
            if held_depth:
                why.append(f"depth={d:.0f}mm~{self.cfg.target_mm:.0f}")
            print(f"[depth-grasp] HELD ({', '.join(why)})", flush=True)
            self.phase = "done"
            return action

        depth_msg = f" depth={d:.0f}mm" if d is not None else ""
        print(
            f"[depth-grasp] MISS residual={residual:.2f} (stall>={self.cfg.stall_gap}){depth_msg}",
            flush=True,
        )
        if self._retries_left > 0:
            self._retries_left -= 1
            self.phase = "grasping"
            action[gi] = float(self.cfg.gripper_open)
            print(
                f"[depth-grasp] retry ({self._retries_left} left) — reopening gripper",
                flush=True,
            )
            return action

        self.phase = "done"
        return action
