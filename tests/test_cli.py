"""CLI inspect does not need arms."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from so101.cli import build_parser, main
from so101.dataset import JOINT_NAMES, DatasetMeta, EpisodeWriter, SO101Dataset, compute_stats, write_stats


def test_inspect_prints_episode_count(tmp_path: Path, capsys) -> None:
    ds = SO101Dataset.create(
        tmp_path,
        DatasetMeta(
            name="unit_demo",
            fps=15,
            joint_names=list(JOINT_NAMES),
            camera_name="front",
            camera_width=32,
            camera_height=24,
            task="task",
            episode_count=0,
        ),
    )
    w = EpisodeWriter(ds, 0)
    img = np.zeros((24, 32, 3), dtype=np.uint8)
    w.add_frame(np.zeros(6, np.float32), np.ones(6, np.float32), 0.0, img)
    w.save()
    write_stats(tmp_path, compute_stats(SO101Dataset(tmp_path)))
    assert main(["inspect", "--dataset-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "episode_count" in out
    assert "000000" in out
    assert "state_mean" in out


def test_train_parser_gdm_flags() -> None:
    args = build_parser().parse_args(
        ["train", "--val-frac", "0.2", "--seed", "0", "--eval-every", "1000"]
    )
    assert args.val_frac == 0.2
    assert args.seed == 0
    assert args.eval_every == 1000


def test_eval_parser_hybrid_flags() -> None:
    args = build_parser().parse_args(
        [
            "eval",
            "--policy",
            "hybrid",
            "--policy-base",
            "act",
            "--grasp-config",
            "config/hybrid_grasp.yaml",
            "--task",
            "Place keys into metal cup",
        ]
    )
    assert args.policy == "hybrid"
    assert args.policy_base == "act"
    assert args.grasp_config.endswith("hybrid_grasp.yaml")
    assert "keys" in args.task


def test_convert_parser() -> None:
    args = build_parser().parse_args(
        ["convert", "--dataset", "so101_d405_demo", "--max-episodes", "2"]
    )
    assert args.dataset == "so101_d405_demo"
    assert args.max_episodes == 2
