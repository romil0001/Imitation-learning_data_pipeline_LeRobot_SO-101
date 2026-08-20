"""python -m so101 {record,train,eval,replay,inspect}"""

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
        print(f"episode {i:06d}  frames={ep.num_frames}  videos={sorted((ep.video_paths or {ep.camera_name: ep.video_path}).keys())}" + (
            f"  depth={ep.depth_path.name}" if ep.depth_path else ""
        ))
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


def cmd_train(args: argparse.Namespace) -> int:
    from so101.train import train_from_env

    train_from_env(
        dataset_name=args.dataset,
        steps=args.steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        val_frac=args.val_frac,
        seed=args.seed,
        eval_every=args.eval_every,
    )
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
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="so101", description="Local SO-101 record / ACT train / eval / replay")
    sub = p.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="Collect teleop episodes")
    rec.add_argument("--dataset", default="so101_d405_demo")
    rec.add_argument("--num-episodes", type=int, default=50)
    rec.add_argument("--task", default="Pick pen and place it next to the keys")
    rec.add_argument("-y", "--yes", action="store_true", help="Skip hardware confirm prompt")
    rec.set_defaults(func=cmd_record)

    tr = sub.add_parser("train", help="Train ACT on a local dataset (Mac)")
    tr.add_argument("--dataset", default="so101_d405_demo")
    tr.add_argument("--steps", type=int, default=40_000)
    tr.add_argument("--batch-size", type=int, default=0, help="0 = auto (2 on mps/cpu, 8 on cuda)")
    tr.add_argument("--output-dir", default="")
    tr.add_argument("--val-frac", type=float, default=0.2, help="Episode-level holdout fraction")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--eval-every", type=int, default=1000, help="Offline val L1 interval (steps)")
    tr.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval", help="Run a trained ACT policy on the follower")
    ev.add_argument("--dataset", default="so101_d405_demo")
    ev.add_argument("--checkpoint", default="")
    ev.add_argument("--duration", type=float, default=60.0, help="Unscored timed run if --protocol is omitted")
    ev.add_argument("--protocol", default="", help="YAML/JSON scored grid (GDM-style y/n trials)")
    ev.add_argument("--n-trials", type=int, default=None, help="Override protocol n_trials_per_condition")
    ev.add_argument("-y", "--yes", action="store_true")
    ev.set_defaults(func=cmd_eval)

    rp = sub.add_parser("replay", help="Replay one episode on the follower")
    rp.add_argument("--dataset", default="so101_d405_demo")
    rp.add_argument("--episode", type=int, default=0)
    rp.add_argument("-y", "--yes", action="store_true")
    rp.set_defaults(func=cmd_replay)

    insp = sub.add_parser("inspect", help="Print local dataset meta")
    insp.add_argument("--dataset", default="so101_d405_demo")
    insp.add_argument("--dataset-dir", default="")
    insp.set_defaults(func=cmd_inspect)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
