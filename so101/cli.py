"""python -m so101 {record,train,eval,replay,inspect,convert,algorithms,link-check}"""

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
            f"episode {i:06d}  frames={ep.num_frames}  videos={sorted((ep.video_paths or {ep.camera_name: ep.video_path}).keys())}"
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
        )
    )
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


def cmd_link_check(args: argparse.Namespace) -> int:
    from so101.link_check import format_report, link_check_from_env, probe_zmq_stream

    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    if args.host:
        report = probe_zmq_stream(
            args.host,
            args.port,
            seconds=args.seconds,
            expected_images=[n.strip() for n in args.expect.split(",") if n.strip()],
            expect_depth=not args.no_depth,
            depth_name=args.depth_name,
            dump_dir=dump_dir,
            min_frames=args.min_frames,
        )
    else:
        report = link_check_from_env(
            seconds=args.seconds,
            dump=not args.no_dump and dump_dir is None,
            min_frames=args.min_frames,
        )
        if dump_dir is not None and report.tcp_ok:
            # User asked for a specific dump path; re-sample once into it.
            report = probe_zmq_stream(
                report.host,
                report.port,
                seconds=args.seconds,
                expected_images=report.expected_images,
                expect_depth=report.expect_depth,
                depth_name=report.depth_name,
                dump_dir=dump_dir,
                min_frames=args.min_frames,
            )
    print(format_report(report), end="")
    return 0 if report.ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="so101",
        description="SO-101 visuomotor manipulation — cell setup, learned policies, eval, replay",
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
    tr.set_defaults(func=cmd_train)

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

    link = sub.add_parser(
        "link-check",
        help="Verify Omen→Mac ZMQ: wrist RGB, overhead RGB, wrist depth transmitting",
    )
    link.add_argument("--seconds", type=float, default=3.0, help="How long to sample the stream")
    link.add_argument("--host", default="", help="Override ZMQ_HOST (default: config/camera.env)")
    link.add_argument("--port", type=int, default=5555, help="ZMQ port when using --host")
    link.add_argument(
        "--expect",
        default="wrist,overhead",
        help="Comma-separated RGB keys required when using --host",
    )
    link.add_argument("--depth-name", default="wrist", help="Depth key required unless --no-depth")
    link.add_argument("--no-depth", action="store_true", help="Do not require depth frames")
    link.add_argument("--min-frames", type=int, default=3, help="Minimum frames for PASS")
    link.add_argument("--no-dump", action="store_true", help="Do not write sample frames")
    link.add_argument(
        "--dump-dir",
        default="",
        help="With --host: directory for sample JPEGs/PNGs (default: no dump unless set)",
    )
    link.set_defaults(func=cmd_link_check)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
