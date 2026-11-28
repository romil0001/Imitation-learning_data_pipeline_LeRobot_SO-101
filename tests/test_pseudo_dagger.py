"""Depth pseudo-DAgger servo + logging threshold."""

from __future__ import annotations

import numpy as np

from so101.dataset.schema import JOINT_NAMES
from so101.pseudo_dagger import PseudoDaggerConfig, PseudoDaggerPolicy, PseudoDaggerServo


def _depth_with_blob(
    h: int,
    w: int,
    *,
    far_mm: int = 800,
    near_mm: int = 200,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
) -> np.ndarray:
    z = np.full((h, w), far_mm, dtype=np.uint16)
    z[y0:y1, x0:x1] = near_mm
    return z


class _Hold:
    def __init__(self) -> None:
        self.cfg = type("Cfg", (), {"use_depth": False, "depth_camera_name": "wrist"})()
        self.calls = 0

    def reset(self) -> None:
        return None

    def select_action(self, image_rgb, state, depth_u16=None):
        del image_rgb, depth_u16
        self.calls += 1
        return np.asarray(state, dtype=np.float32).copy()


def test_config_from_dict_signs() -> None:
    cfg = PseudoDaggerConfig.from_dict(
        {
            "lateral_signs": {"shoulder_pan": -1.0},
            "approach_max_mm": 400,
            "log_delta_deg": 0.5,
        }
    )
    assert cfg.lateral_signs["shoulder_pan"] == -1.0
    assert cfg.approach_max_mm == 400
    assert cfg.log_delta_deg == 0.5


def test_no_correction_when_far() -> None:
    cfg = PseudoDaggerConfig(approach_min_mm=80, approach_max_mm=300, log_delta_deg=0.2)
    servo = PseudoDaggerServo(cfg)
    state = np.zeros(6, dtype=np.float32)
    action = np.zeros(6, dtype=np.float32)
    # Blob at 500 mm is outside approach band.
    z = _depth_with_blob(48, 64, near_mm=500, x0=20, x1=40, y0=10, y1=30)
    out, info = servo.correct(action, state, z)
    assert not info.active
    assert not info.logged
    np.testing.assert_array_equal(out, action)


def test_lateral_nudge_when_blob_off_center() -> None:
    cfg = PseudoDaggerConfig(
        approach_min_mm=50,
        approach_max_mm=400,
        target_mm=200,
        gain_deg_per_norm=10.0,
        gain_deg_per_mm=0.0,  # isolate lateral
        vertical_joints=(),
        max_delta_deg=5.0,
        log_delta_deg=0.2,
        blend=1.0,
        lateral_signs={"shoulder_pan": 1.0},
    )
    servo = PseudoDaggerServo(cfg)
    state = np.zeros(6, dtype=np.float32)
    action = np.zeros(6, dtype=np.float32)
    # Put nearest blob on the right half of a 64-wide image.
    z = _depth_with_blob(48, 64, near_mm=200, x0=48, x1=60, y0=16, y1=32)
    out, info = servo.correct(action, state, z)
    assert info.active
    assert info.offset_xy is not None and info.offset_xy[0] > 0
    pi = JOINT_NAMES.index("shoulder_pan")
    assert out[pi] > action[pi]
    assert info.logged


def test_policy_wraps_base_and_sets_use_depth() -> None:
    base = _Hold()
    policy = PseudoDaggerPolicy(base, PseudoDaggerServo(PseudoDaggerConfig(enabled=False)))
    assert policy.cfg.use_depth is True
    state = np.arange(6, dtype=np.float32)
    out = policy.select_action(np.zeros((8, 8, 3), dtype=np.uint8), state, depth_u16=None)
    np.testing.assert_array_equal(out, state)
    assert base.calls == 1


class _Leap:
    def __init__(self) -> None:
        self.cfg = type("Cfg", (), {"use_depth": False, "depth_camera_name": "wrist"})()

    def reset(self) -> None:
        return None

    def select_action(self, image_rgb, state, depth_u16=None):
        del image_rgb, depth_u16
        st = np.asarray(state, dtype=np.float32)
        out = st.copy()
        out[:5] = st[:5] + 40.0
        return out


def test_action_scale_slows_base_arm_deltas() -> None:
    cfg = PseudoDaggerConfig(enabled=False, action_scale=0.25)
    policy = PseudoDaggerPolicy(_Leap(), PseudoDaggerServo(cfg))
    state = np.zeros(6, dtype=np.float32)
    out = policy.select_action(np.zeros((8, 8, 3), dtype=np.uint8), state, depth_u16=None)
    np.testing.assert_allclose(out[:5], np.full(5, 10.0), atol=1e-4)
    assert float(out[5]) == 0.0
