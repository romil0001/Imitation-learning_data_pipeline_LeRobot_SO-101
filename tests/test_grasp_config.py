"""Grasp config YAML loader."""

from __future__ import annotations

from pathlib import Path

from so101.grasp_config import load_grasp_config


def test_load_grasp_config_defaults() -> None:
    cfg = load_grasp_config(None)
    assert cfg.enabled is True
    assert cfg.target_mm > 0


def test_load_grasp_config_yaml(tmp_path: Path) -> None:
    p = tmp_path / "g.yaml"
    p.write_text("target_mm: 77.0\ngripper_closed: 9.5\nvertical_joints: [elbow_flex]\n")
    cfg = load_grasp_config(p)
    assert cfg.target_mm == 77.0
    assert cfg.gripper_closed == 9.5
    assert cfg.vertical_joints == ("elbow_flex",)
