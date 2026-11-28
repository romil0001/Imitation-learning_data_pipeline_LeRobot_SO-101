"""SmolVLA checkpoint path resolution (reject ACT run dirs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from so101.algorithms.loaders import _normalize_smolvla_checkpoint, _resolve_smolvla_checkpoint


def test_normalize_rejects_act_run_config(tmp_path: Path) -> None:
    act = tmp_path / "act_run"
    act.mkdir()
    (act / "config.json").write_text(json.dumps({"n_joints": 6, "chunk_size": 100}))
    assert _normalize_smolvla_checkpoint(act) is None


def test_normalize_accepts_smolvla_pretrained(tmp_path: Path) -> None:
    smol = tmp_path / "pretrained_model"
    smol.mkdir()
    (smol / "config.json").write_text(json.dumps({"type": "smolvla", "n_obs_steps": 1}))
    assert _normalize_smolvla_checkpoint(smol) == smol


def test_normalize_train_run_last(tmp_path: Path) -> None:
    run = tmp_path / "smolvla_job"
    nested = run / "checkpoints" / "last" / "pretrained_model"
    nested.mkdir(parents=True)
    (nested / "config.json").write_text(json.dumps({"type": "smolvla"}))
    assert _normalize_smolvla_checkpoint(run) == nested.resolve()


def test_resolve_errors_on_act_path(tmp_path: Path) -> None:
    act = tmp_path / "act_so101_bag_clip_bowl"
    act.mkdir()
    (act / "config.json").write_text(json.dumps({"n_joints": 6}))
    with pytest.raises(FileNotFoundError, match="not a SmolVLA"):
        _resolve_smolvla_checkpoint(tmp_path, str(act), "so101_bag_clip_bowl")
