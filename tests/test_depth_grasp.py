"""Depth grasp servo: metric wrist depth overrides vertical joints near contact."""

from __future__ import annotations

import numpy as np

from so101.dataset.schema import JOINT_NAMES
from so101.depth_grasp import (
    DepthGraspConfig,
    DepthGraspController,
    median_roi_depth_mm,
    nearest_depth_blob,
)


def _depth(h: int, w: int, fill_mm: int, *, invalid: int = 0) -> np.ndarray:
    z = np.full((h, w), fill_mm, dtype=np.uint16)
    z[:2, :] = invalid
    return z


def test_median_roi_depth_center_patch() -> None:
    z = np.zeros((100, 100), dtype=np.uint16)
    z[40:60, 40:60] = 120
    med = median_roi_depth_mm(z, roi_frac=0.2, min_mm=1.0, max_mm=2000.0)
    assert med is not None
    assert abs(med - 120.0) < 1.0


def test_median_roi_ignores_invalid() -> None:
    z = np.zeros((40, 40), dtype=np.uint16)
    assert median_roi_depth_mm(z, roi_frac=0.5, min_mm=1.0, max_mm=2000.0) is None


def test_idle_passthrough_when_far() -> None:
    cfg = DepthGraspConfig(enter_mm=150.0, target_mm=90.0)
    ctrl = DepthGraspController(cfg)
    action = np.array([1, 2, 3, 4, 5, 50], dtype=np.float32)
    state = action.copy()
    out = ctrl.override(action, state, _depth(48, 64, 400))
    np.testing.assert_array_equal(out, action)
    assert ctrl.phase == "idle"


def test_enters_grasp_and_pulls_closer_when_too_far() -> None:
    cfg = DepthGraspConfig(
        enter_mm=200.0,
        target_mm=100.0,
        tol_mm=10.0,
        gain_deg_per_mm=0.1,
        vertical_joints=("elbow_flex",),
        joint_signs={"elbow_flex": 1.0},
        close_gripper=False,
    )
    ctrl = DepthGraspController(cfg)
    state = np.zeros(6, dtype=np.float32)
    action = np.zeros(6, dtype=np.float32)
    # 150 mm > target 100 → positive error → +elbow_flex with sign +1
    out = ctrl.override(action, state, _depth(48, 64, 150))
    assert ctrl.phase == "grasping"
    ei = JOINT_NAMES.index("elbow_flex")
    assert out[ei] > action[ei]


def test_closes_gripper_when_in_tolerance() -> None:
    cfg = DepthGraspConfig(
        enter_mm=200.0,
        target_mm=100.0,
        tol_mm=15.0,
        close_gripper=True,
        gripper_closed=12.0,
        hold_close_frames=3,
        detect_grasp=False,
        track_enabled=False,
    )
    ctrl = DepthGraspController(cfg)
    state = np.zeros(6, dtype=np.float32)
    action = np.array([0, 0, 0, 0, 0, 80], dtype=np.float32)
    out = ctrl.override(action, state, _depth(48, 64, 105))
    gi = JOINT_NAMES.index("gripper")
    assert out[gi] == 12.0
    assert ctrl.phase in {"closing", "done"}


def test_done_passthrough_after_hold() -> None:
    cfg = DepthGraspConfig(
        enter_mm=200.0,
        target_mm=100.0,
        tol_mm=20.0,
        close_gripper=True,
        gripper_closed=10.0,
        hold_close_frames=2,
        gain_deg_per_mm=0.0,
        detect_grasp=False,
        track_enabled=False,
    )
    ctrl = DepthGraspController(cfg)
    state = np.zeros(6, dtype=np.float32)
    action = np.array([1, 2, 3, 4, 5, 90], dtype=np.float32)
    ctrl.override(action, state, _depth(48, 64, 100))
    ctrl.override(action, state, _depth(48, 64, 100))
    out = ctrl.override(action, state, _depth(48, 64, 100))
    assert ctrl.phase == "done"
    np.testing.assert_array_equal(out, action)


def test_grasp_held_when_jaw_stalls_short() -> None:
    cfg = DepthGraspConfig(
        enter_mm=200.0,
        target_mm=100.0,
        tol_mm=20.0,
        close_gripper=True,
        gripper_closed=10.0,
        hold_close_frames=2,
        gain_deg_per_mm=0.0,
        detect_grasp=True,
        stall_gap=4.0,
        retry_on_miss=0,
        track_enabled=False,
    )
    ctrl = DepthGraspController(cfg)
    # Measured jaw stuck open-ish while commanding closed → held.
    state = np.array([0, 0, 0, 0, 0, 25], dtype=np.float32)
    action = np.zeros(6, dtype=np.float32)
    z = _depth(48, 64, 100)
    ctrl.override(action, state, z)  # enter closing
    out = ctrl.override(action, state, z)  # finish hold
    assert ctrl.phase == "done"
    assert ctrl.grasp_held is True
    assert ctrl.last_gripper_residual is not None
    assert ctrl.last_gripper_residual >= 4.0
    assert out[JOINT_NAMES.index("gripper")] == 10.0


def test_thin_object_held_via_depth_confirm() -> None:
    cfg = DepthGraspConfig(
        enter_mm=200.0,
        target_mm=100.0,
        tol_mm=20.0,
        close_gripper=True,
        gripper_closed=10.0,
        hold_close_frames=1,
        gain_deg_per_mm=0.0,
        detect_grasp=True,
        stall_gap=4.0,
        retry_on_miss=0,
        track_enabled=False,
        depth_confirm=True,
        depth_confirm_tol_mm=30.0,
    )
    ctrl = DepthGraspController(cfg)
    # Empty-looking residual but depth still at target → thin object held.
    state = np.array([0, 0, 0, 0, 0, 10.2], dtype=np.float32)
    out = ctrl.override(np.zeros(6, np.float32), state, _depth(48, 64, 105))
    assert ctrl.phase == "done"
    assert ctrl.grasp_held is True
    assert out[JOINT_NAMES.index("gripper")] == 10.0


def test_grasp_miss_retries_then_opens() -> None:
    cfg = DepthGraspConfig(
        enter_mm=200.0,
        target_mm=100.0,
        tol_mm=20.0,
        close_gripper=True,
        gripper_closed=10.0,
        gripper_open=60.0,
        hold_close_frames=1,
        gain_deg_per_mm=0.0,
        detect_grasp=True,
        stall_gap=4.0,
        retry_on_miss=1,
        track_enabled=False,
    )
    ctrl = DepthGraspController(cfg)
    # Empty close: measured ≈ commanded.
    state = np.array([0, 0, 0, 0, 0, 10.5], dtype=np.float32)
    action = np.zeros(6, dtype=np.float32)
    z = _depth(48, 64, 100)
    out = ctrl.override(action, state, z)
    assert ctrl.grasp_held is False
    assert ctrl.phase == "grasping"  # retry
    assert out[JOINT_NAMES.index("gripper")] == 60.0
    # Second empty close → done miss.
    out2 = ctrl.override(action, state, z)
    assert ctrl.phase == "done"
    assert ctrl.grasp_held is False
    assert out2[JOINT_NAMES.index("gripper")] == 10.0


def test_reset_returns_to_idle() -> None:
    cfg = DepthGraspConfig(enter_mm=200.0, target_mm=100.0, close_gripper=False)
    ctrl = DepthGraspController(cfg)
    ctrl.override(np.zeros(6, np.float32), np.zeros(6, np.float32), _depth(48, 64, 120))
    assert ctrl.phase == "grasping"
    ctrl.reset()
    assert ctrl.phase == "idle"


def test_none_depth_passthrough() -> None:
    ctrl = DepthGraspController(DepthGraspConfig())
    action = np.arange(6, dtype=np.float32)
    out = ctrl.override(action, action, None)
    np.testing.assert_array_equal(out, action)


def test_nearest_blob_prefers_closer_off_center() -> None:
    z = np.full((100, 100), 800, dtype=np.uint16)
    # Near object on the right half of the image.
    z[40:55, 70:90] = 120
    blob = nearest_depth_blob(z, roi_frac=0.9, min_mm=40.0, max_mm=2000.0, band_mm=30.0)
    assert blob is not None
    ox, oy, depth = blob
    assert ox > 0.2
    assert abs(depth - 120.0) < 5.0
    assert abs(oy) < 0.3


def test_track_pans_toward_off_center_blob() -> None:
    cfg = DepthGraspConfig(
        enter_mm=200.0,
        target_mm=100.0,
        tol_mm=5.0,
        close_gripper=False,
        track_enabled=True,
        track_roi_frac=0.9,
        track_band_mm=30.0,
        gain_deg_per_norm=10.0,
        lateral_joints=("shoulder_pan",),
        lateral_signs={"shoulder_pan": 1.0},
        gain_deg_per_mm=0.0,
        max_delta_deg=5.0,
    )
    ctrl = DepthGraspController(cfg)
    z = np.full((80, 80), 500, dtype=np.uint16)
    z[30:50, 55:75] = 140  # right of center, near enough to enter
    state = np.zeros(6, dtype=np.float32)
    action = np.zeros(6, dtype=np.float32)
    out = ctrl.override(action, state, z)
    assert ctrl.phase == "grasping"
    assert ctrl.last_offset_xy is not None
    assert ctrl.last_offset_xy[0] > 0
    pi = JOINT_NAMES.index("shoulder_pan")
    assert out[pi] > 0

    # Flip sign → pan the other way.
    cfg.lateral_signs = {"shoulder_pan": -1.0}
    ctrl = DepthGraspController(cfg)
    out2 = ctrl.override(action, state, z)
    assert out2[pi] < 0


def test_grasp_config_roundtrip_track_fields(tmp_path) -> None:
    from so101.grasp_config import load_grasp_config

    p = tmp_path / "g.yaml"
    p.write_text(
        "track_enabled: true\ntrack_roi_frac: 0.55\nlateral_joints: [shoulder_pan]\n"
        "lateral_signs: {shoulder_pan: -1.0}\ngain_deg_per_norm: 6.5\n"
    )
    cfg = load_grasp_config(p)
    assert cfg.track_enabled is True
    assert cfg.track_roi_frac == 0.55
    assert cfg.lateral_joints == ("shoulder_pan",)
    assert cfg.lateral_signs["shoulder_pan"] == -1.0
    assert cfg.gain_deg_per_norm == 6.5
