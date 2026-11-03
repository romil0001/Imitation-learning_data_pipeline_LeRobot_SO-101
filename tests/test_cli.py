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


def test_replay_capture_parser() -> None:
    args = build_parser().parse_args(
        [
            "replay-capture",
            "--source",
            "so101_d405_demo",
            "--dest",
            "so101_keys_in_cup",
            "--task",
            "keys in the cup",
            "--episode",
            "0",
            "-y",
        ]
    )
    assert args.source == "so101_d405_demo"
    assert args.dest == "so101_keys_in_cup"
    assert args.task == "keys in the cup"
    assert args.episode == 0
    defaults = build_parser().parse_args(["replay-capture"])
    assert defaults.source == "so101_bag_clip_bowl"
    assert defaults.dest == "so101_bag_clip_bowl_capture"
    assert defaults.task == "Place bag clip in bowl"


def test_train_parser_gdm_flags() -> None:
    args = build_parser().parse_args(
        ["train", "--algo", "act-jax", "--val-frac", "0.2", "--seed", "0", "--eval-every", "1000"]
    )
    assert args.algo == "act-jax"
    assert args.val_frac == 0.2
    assert args.seed == 0
    assert args.eval_every == 1000


def test_train_parser_dagger_flags() -> None:
    args = build_parser().parse_args(
        [
            "train",
            "--corrections",
            "so101_d405_demo_dagger",
            "--init-checkpoint",
            "outputs/train/act_so101_d405_demo",
            "--steps",
            "8000",
        ]
    )
    assert args.corrections == "so101_d405_demo_dagger"
    assert args.init_checkpoint.endswith("act_so101_d405_demo")
    assert args.steps == 8000


def test_dagger_and_compare_eval_parsers() -> None:
    dg = build_parser().parse_args(
        ["dagger", "--dataset", "so101_d405_demo", "--num-episodes", "3", "--policy", "act", "-y"]
    )
    assert dg.num_episodes == 3
    assert dg.policy == "act"
    assert dg.yes is True
    cmp = build_parser().parse_args(
        ["compare-eval", "--baseline", "a/summary.json", "--candidate", "b/summary.json"]
    )
    assert cmp.baseline.endswith("a/summary.json")
    assert cmp.candidate.endswith("b/summary.json")


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
            "Place bag clip in bowl",
        ]
    )
    assert args.policy == "hybrid"
    assert args.policy_base == "act"
    assert args.grasp_config.endswith("hybrid_grasp.yaml")
    assert "bag clip" in args.task


def test_record_defaults_bag_clip_task() -> None:
    args = build_parser().parse_args(["record"])
    assert args.dataset == "so101_bag_clip_bowl"
    assert args.task == "Place bag clip in bowl"


def test_eval_parser_recipe_flags() -> None:
    args = build_parser().parse_args(
        [
            "eval",
            "--policy",
            "gemfuse",
            "--recipe",
            "config/recipes/gemfuse_1p3b_r18.json",
            "--n-action-steps",
            "5",
            "--async-replan",
            "--replan-threshold",
            "25",
            "--low-pass",
            "0.3",
            "--freespace-scale",
            "2.0",
        ]
    )
    assert args.policy == "gemfuse"
    assert args.recipe.endswith("gemfuse_1p3b_r18.json")
    assert args.n_action_steps == 5
    assert args.async_replan is True
    assert args.replan_threshold == 25
    assert args.low_pass == 0.3
    assert args.freespace_scale == 2.0


def test_eval_parser_no_async_replan() -> None:
    args = build_parser().parse_args(["eval", "--no-async-replan"])
    assert args.async_replan is False


def test_eval_parser_accepts_new_policy_name() -> None:
    args = build_parser().parse_args(["eval", "--policy", "my_policy", "--policy-base", "my_policy"])
    assert args.policy == "my_policy"
    assert args.policy_base == "my_policy"


def test_algorithms_command_prints_both_families(capsys) -> None:
    assert main(["algorithms"]) == 0
    out = capsys.readouterr().out
    assert "SETUP" in out
    assert "ML" in out
    assert "depth-grasp" in out
    assert "act-jax" in out
    assert "gemfuse" in out
    assert "my_policy" not in out


def test_convert_parser() -> None:
    args = build_parser().parse_args(
        ["convert", "--dataset", "so101_d405_demo", "--max-episodes", "2"]
    )
    assert args.dataset == "so101_d405_demo"
    assert args.max_episodes == 2
