"""python -m so101 {record,dagger,train,eval,replay,replay-capture,inspect,convert,algorithms,capture-home}"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _dataset_dir(name: str) -> Path:
    from so101.config import project_root

    return project_root() / "data" / "datasets" / name


def cmd_inspect(args: argparse.Namespace) -> int:
    from so101.dataset import SO101Dataset

    root = Path(args.dataset_dir) if args.dataset_dir else _dataset_dir(args.dataset)
    ds = SO101Dataset(root)
    print(json.dumps(ds.meta.to_dict(), indent=2))
    stats_path = root / "stats.json"
    if stats_path.is_file():
        print("stats.json:")
        print(stats_path.read_text())
    for i in range(ds.meta.episode_count):
        ep = ds.load_episode(i)
        print(
            f"episode {i:06d}  frames={ep.num_frames}  "
            f"source={ep.source}  intervention={ep.intervention}  "
            f"videos={sorted((ep.video_paths or {ep.camera_name: ep.video_path}).keys())}"
            + (f"  depth={ep.depth_path.name}" if ep.depth_path else "")
        )
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    from so101.record import record_from_env

    record_from_env(
        dataset_name=args.dataset,
        num_episodes=args.num_episodes,
        task=args.task,
        confirm=not args.yes,
    )
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from so101.replay import replay_from_env

    replay_from_env(dataset_name=args.dataset, episode=args.episode, confirm=not args.yes)
    return 0


def cmd_replay_capture(args: argparse.Namespace) -> int:
    from so101.replay_capture import replay_capture_from_env

    saved = replay_capture_from_env(
        source_dataset=args.source,
        dest_dataset=args.dest,
        task=args.task,
        episode=args.episode,
        max_episodes=args.max_episodes,
        confirm=not args.yes,
    )
    print(f"Captured {saved} episode(s) → {args.dest}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from so101.algorithms import TrainContext, train_algorithm

    train_algorithm(
        TrainContext(
            algo=args.algo,
            dataset_name=args.dataset,
            steps=args.steps,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            val_frac=args.val_frac,
            seed=args.seed,
            eval_every=args.eval_every,
            backend=args.backend,
            corrections_name=args.corrections,
            init_checkpoint=args.init_checkpoint,
        )
    )
    return 0


def cmd_dagger(args: argparse.Namespace) -> int:
    from so101.dagger import dagger_from_env

    dagger_from_env(
        dataset_name=args.dataset,
        corrections_name=args.corrections,
        checkpoint=args.checkpoint,
        num_episodes=args.num_episodes,
        confirm=not args.yes,
        policy=args.policy,
        policy_base=args.policy_base,
        grasp_config=args.grasp_config,
        task=args.task,
        sync_steps=args.sync_steps,
        sync_dt=args.sync_dt,
    )
    return 0


def cmd_compare_eval(args: argparse.Namespace) -> int:
    from so101.compare_eval import compare_summaries, format_comparison, load_summary

    report = compare_summaries(load_summary(args.baseline), load_summary(args.candidate))
    text = format_comparison(report)
    print(text, end="")
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {args.output}")
    return 0


def cmd_algorithms(args: argparse.Namespace) -> int:
    from so101.algorithms import format_catalog

    print(format_catalog(args.family), end="")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from so101.eval import eval_from_env

    eval_from_env(
        checkpoint=args.checkpoint,
        dataset_name=args.dataset,
        duration_s=args.duration,
        confirm=not args.yes,
        protocol=args.protocol,
        n_trials=args.n_trials,
        policy=args.policy,
        policy_base=args.policy_base,
        grasp_config=args.grasp_config,
        task=args.task,
        recipe=args.recipe,
        n_action_steps=args.n_action_steps,
        async_replan=args.async_replan,
        replan_threshold=args.replan_threshold,
        low_pass_alpha=args.low_pass,
        freespace_scale=args.freespace_scale,
    )
    return 0


def cmd_capture_home(args: argparse.Namespace) -> int:
    from so101.capture_home import capture_home_from_env

    capture_home_from_env(
        protocol=args.protocol,
        confirm=not args.yes,
        write=not args.print_only,
    )
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    from so101.config import project_root
    from so101.convert_lerobot import convert_so101_to_lerobot

    root = project_root()
    src = Path(args.dataset_dir) if args.dataset_dir else _dataset_dir(args.dataset)
    dest = Path(args.output_dir) if args.output_dir else root / "data" / "lerobot" / args.dataset
    meta = convert_so101_to_lerobot(
        src,
        dest,
        repo_id=args.repo_id or f"local/{args.dataset}",
        max_episodes=args.max_episodes,
    )
    print(json.dumps(meta, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="so101",
        description=(
            "SO-101 visuomotor manipulation — record, DAgger, ACT/SmolVLA train, "
            "hybrid eval, replay"
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="Collect teleop episodes")
    rec.add_argument("--dataset", default="so101_d405_demo")
    rec.add_argument("--num-episodes", type=int, default=50)
    rec.add_argument("--task", default="Place keys into metal cup")
    rec.add_argument("-y", "--yes", action="store_true", help="Skip hardware confirm prompt")
    rec.set_defaults(func=cmd_record)

    tr = sub.add_parser("train", help="Train a registered ML algorithm (default: act)")
    tr.add_argument(
        "--algo",
        default="act",
        help="ML algorithm name (python -m so101 algorithms --family ml). act-jax forces the Flax trainer.",
    )
    tr.add_argument("--dataset", default="so101_d405_demo")
    tr.add_argument("--steps", type=int, default=40_000)
    tr.add_argument("--batch-size", type=int, default=0, help="0 = auto (2 on mps/cpu, 8 on cuda)")
    tr.add_argument("--output-dir", default="")
    tr.add_argument("--val-frac", type=float, default=0.2, help="Episode-level holdout fraction")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--eval-every", type=int, default=1000, help="Offline val L1 interval (steps)")
    tr.add_argument(
        "--backend",
        choices=("torch", "jax"),
        default="torch",
        help="torch (PyTorch/MPS/CUDA, resnet18) or jax (Flax, tiny backbone)",
    )
    tr.add_argument(
        "--corrections",
        default="",
        help="DAgger dataset name under data/datasets/ (merged into train; val stays on demos)",
    )
    tr.add_argument(
        "--init-checkpoint",
        default="",
        help="ACT weights to fine-tune (file or directory with best_val/model.pt)",
    )
    tr.set_defaults(func=cmd_train)

    dg = sub.add_parser(
        "dagger",
        help="Human-gated DAgger: pause, match leader to follower, record corrections",
    )
    dg.add_argument("--dataset", default="so101_d405_demo", help="Demo dataset (cameras, home pose, task)")
    dg.add_argument(
        "--corrections",
        default="",
        help="Where to store corrections (default <dataset>_dagger)",
    )
    dg.add_argument("--checkpoint", default="", help="ACT / SmolVLA / hybrid checkpoint directory")
    dg.add_argument("--num-episodes", type=int, default=10)
    dg.add_argument("--policy", choices=("act", "smolvla", "hybrid"), default="act")
    dg.add_argument("--policy-base", choices=("act", "smolvla"), default="smolvla")
    dg.add_argument("--grasp-config", default="")
    dg.add_argument("--task", default="")
    dg.add_argument(
        "--sync-steps",
        type=int,
        default=40,
        help="Leader interpolation steps on pause (~1.6s at dt=0.04)",
    )
    dg.add_argument("--sync-dt", type=float, default=0.04)
    dg.add_argument("-y", "--yes", action="store_true")
    dg.set_defaults(func=cmd_dagger)

    cmp = sub.add_parser("compare-eval", help="Diff two protocol summary.json files")
    cmp.add_argument("--baseline", required=True, help="summary.json or eval run directory")
    cmp.add_argument("--candidate", required=True)
    cmp.add_argument("--output", default="", help="Optional JSON report path")
    cmp.set_defaults(func=cmd_compare_eval)

    ev = sub.add_parser("eval", help="Run a registered ML policy, or hybrid (ML + depth grasp)")
    ev.add_argument("--dataset", default="so101_d405_demo")
    ev.add_argument("--checkpoint", default="")
    ev.add_argument("--duration", type=float, default=60.0, help="Unscored timed run if --protocol is omitted")
    ev.add_argument(
        "--protocol",
        default="",
        help="YAML/JSON protocol: condition grid, repeated trials, binary y/n success (see docs/outcomes.md)",
    )
    ev.add_argument("--n-trials", type=int, default=None, help="Override protocol n_trials_per_condition")
    ev.add_argument(
        "--policy",
        default="act",
        help="ML algorithm, or hybrid. Names: python -m so101 algorithms",
    )
    ev.add_argument(
        "--policy-base",
        default="smolvla",
        help="For --policy hybrid: which ML algorithm to wrap",
    )
    ev.add_argument(
        "--grasp-config",
        default="",
        help="YAML/JSON for depth grasp (default: built-in; recommend config/hybrid_grasp.yaml)",
    )
    ev.add_argument("--task", default="", help="Language task for SmolVLA (defaults to protocol.task)")
    ev.add_argument(
        "--recipe",
        default="",
        help="Inference recipe YAML/JSON (act5, async replan, low-pass, freepace). "
        "gemfuse defaults to config/recipes/gemfuse_1p3b_r18.yaml when omitted.",
    )
    ev.add_argument("--n-action-steps", type=int, default=None, help="Override recipe actN (e.g. 5)")
    ev.set_defaults(async_replan=None)
    ev.add_argument("--async-replan", action="store_true", dest="async_replan", help="Enable async chunk replan")
    ev.add_argument("--no-async-replan", action="store_false", dest="async_replan", help="Disable async replan")
    ev.add_argument("--replan-threshold", type=int, default=None, help="Replan when queue length ≤ N (e.g. 25)")
    ev.add_argument("--low-pass", type=float, default=None, dest="low_pass", help="EMA alpha in (0,1]; 0 disables")
    ev.add_argument("--freespace-scale", type=float, default=None, help="Scale free-space joint deltas (e.g. 2.0)")
    ev.add_argument("-y", "--yes", action="store_true")
    ev.set_defaults(func=cmd_eval)

    conv = sub.add_parser("convert", help="Convert local SO101 dataset → LeRobot (for SmolVLA)")
    conv.add_argument("--dataset", default="so101_d405_demo")
    conv.add_argument("--dataset-dir", default="")
    conv.add_argument("--output-dir", default="", help="Default: data/lerobot/<dataset>")
    conv.add_argument("--repo-id", default="", help="LeRobot repo_id (default local/<dataset>)")
    conv.add_argument("--max-episodes", type=int, default=None)
    conv.set_defaults(func=cmd_convert)

    rp = sub.add_parser("replay", help="Replay one episode on the follower")
    rp.add_argument("--dataset", default="so101_d405_demo")
    rp.add_argument("--episode", type=int, default=0)
    rp.add_argument("-y", "--yes", action="store_true")
    rp.set_defaults(func=cmd_replay)

    rc = sub.add_parser(
        "replay-capture",
        help="Replay source actions while recording live multi-cam (+depth) into a new dataset",
    )
    rc.add_argument("--source", default="so101_d405_demo", help="Existing demo dataset to replay")
    rc.add_argument("--dest", default="so101_keys_in_cup", help="New dataset name to write")
    rc.add_argument("--task", default="keys in the cup", help="Language/task string for dest meta")
    rc.add_argument("--episode", type=int, default=None, help="Replay only this source episode index")
    rc.add_argument("--max-episodes", type=int, default=None, help="Cap number of source episodes")
    rc.add_argument("-y", "--yes", action="store_true")
    rc.set_defaults(func=cmd_replay_capture)

    cap = sub.add_parser(
        "capture-home",
        help="Read follower joints and write home_qpos into an eval protocol YAML",
    )
    cap.add_argument(
        "--protocol",
        default="config/eval/pen_to_keys.yaml",
        help="Protocol YAML to update (default: config/eval/pen_to_keys.yaml)",
    )
    cap.add_argument(
        "--print-only",
        action="store_true",
        help="Print the pose but do not modify the YAML",
    )
    cap.add_argument("-y", "--yes", action="store_true", help="Skip the pose-confirm prompt")
    cap.set_defaults(func=cmd_capture_home)

    insp = sub.add_parser("inspect", help="Print local dataset meta")
    insp.add_argument("--dataset", default="so101_d405_demo")
    insp.add_argument("--dataset-dir", default="")
    insp.set_defaults(func=cmd_inspect)

    alg = sub.add_parser("algorithms", help="List setup algorithms and ML algorithms")
    alg.add_argument(
        "--family",
        choices=("setup", "ml"),
        default=None,
        help="Show one side of the split",
    )
    alg.set_defaults(func=cmd_algorithms)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
