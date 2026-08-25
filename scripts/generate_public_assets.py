#!/usr/bin/env python3
"""Build public README assets: demo camera frames and eval success-rate graphs.

With no local dataset, writes illustrative synthetic frames under assets/frames/.
When given --dataset-dir or --eval-dir, exports real frames / summary plots instead.

Usage:
  python scripts/generate_public_assets.py
  python scripts/generate_public_assets.py --dataset-dir data/datasets/so101_d405_demo
  python scripts/generate_public_assets.py --eval-dir outputs/eval/act_so101_d405_demo/latest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
FRAMES = ASSETS / "frames"
GRAPHS = ASSETS / "graphs"
EVAL_FIXTURES = ASSETS / "eval"


def _draw_overhead(keys_x: float, label: str) -> np.ndarray:
    """Synthetic overhead RGB (640×480) — illustrative layout."""
    h, w = 480, 640
    img = np.full((h, w, 3), 48, dtype=np.uint8)
    # table
    cv2.rectangle(img, (40, 60), (w - 40, h - 40), (72, 68, 62), -1)
    cv2.rectangle(img, (40, 60), (w - 40, h - 40), (110, 105, 98), 2)
    # metal cup
    cx, cy = int(w * 0.62), int(h * 0.52)
    cv2.circle(img, (cx, cy), 42, (140, 145, 155), -1)
    cv2.circle(img, (cx, cy), 28, (90, 95, 105), -1)
    # keys blob
    kx, ky = int(w * keys_x), int(h * 0.48)
    cv2.ellipse(img, (kx, ky), (36, 18), -15, 0, 360, (200, 170, 60), -1)
    cv2.putText(img, label, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(img, "overhead", (w - 130, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _draw_wrist(phase: str) -> np.ndarray:
    """Synthetic wrist RGB — gripper approaching cup."""
    h, w = 480, 640
    img = np.full((h, w, 3), 32, dtype=np.uint8)
    # depth-like gradient floor
    for y in range(h // 2, h):
        v = int(40 + 80 * (y - h // 2) / (h // 2))
        img[y, :, :] = (v, v - 8, v - 16)
    # cup rim
    cv2.ellipse(img, (320, 300), (90, 35), 0, 0, 360, (160, 165, 175), 3)
    cv2.ellipse(img, (320, 295), (70, 22), 0, 0, 360, (100, 105, 115), -1)
    # gripper fingers
    gy = 180 if phase == "approach" else 240
    cv2.rectangle(img, (270, gy), (290, gy + 80), (210, 210, 215), -1)
    cv2.rectangle(img, (350, gy), (370, gy + 80), (210, 210, 215), -1)
    cv2.putText(img, f"wrist · {phase}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _draw_depth() -> np.ndarray:
    """Pseudo-colormap wrist depth preview."""
    h, w = 480, 640
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - 320) ** 2 + (yy - 280) ** 2)
    depth = np.clip(800 - dist * 1.2, 200, 900).astype(np.float32)
    norm = ((depth - 200) / 700 * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    cv2.putText(
        cv2.cvtColor(colored, cv2.COLOR_RGB2BGR),
        "wrist depth (mm)",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return colored


def write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write {path}")


def synthetic_frames(out: Path) -> list[Path]:
    specs = [
        ("overhead_keys_left.png", _draw_overhead(0.28, "keys=left")),
        ("overhead_keys_center.png", _draw_overhead(0.50, "keys=center")),
        ("overhead_keys_right.png", _draw_overhead(0.72, "keys=right")),
        ("wrist_approach.png", _draw_wrist("approach")),
        ("wrist_grasp.png", _draw_wrist("grasp")),
        ("wrist_depth.png", _draw_depth()),
    ]
    written: list[Path] = []
    for name, rgb in specs:
        p = out / name
        write_png(p, rgb)
        written.append(p)
    return written


def extract_frames_from_dataset(dataset_dir: Path, out: Path, indices: tuple[int, ...] = (0, 10, 20)) -> list[Path]:
    from so101.dataset import SO101Dataset

    ds = SO101Dataset(dataset_dir)
    if ds.meta.episode_count < 1:
        raise ValueError(f"No episodes in {dataset_dir}")
    ep = ds.load_episode(0)
    videos = ep.video_paths or {ep.camera_name: ep.video_path}
    written: list[Path] = []
    for cam, vpath in videos.items():
        from so101.dataset.video import read_mp4

        frames = read_mp4(vpath)
        for fi in indices:
            fi = min(fi, len(frames) - 1)
            name = f"{cam}_ep000_f{fi:04d}.png"
            p = out / name
            write_png(p, frames[fi])
            written.append(p)
    if ep.depth_path and ep.depth_path.is_file():
        depth = np.load(ep.depth_path)
        fi = min(indices[0], len(depth) - 1)
        d = depth[fi]
        norm = np.clip((d.astype(np.float32) - 200) / 700 * 255, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        p = out / f"wrist_depth_ep000_f{fi:04d}.png"
        write_png(p, cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
        written.append(p)
    return written


def example_summaries() -> dict[str, dict]:
    """Illustrative eval outcomes (pen-to-keys protocol, 5 trials × 3 conditions)."""
    return {
        "act": {
            "method": "ACT (RGB + depth)",
            "pc_success": 53.3,
            "n_trials": 15,
            "n_success": 8,
            "per_condition": {
                "keys=left": {"n": 5, "n_success": 3, "pc_success": 60.0},
                "keys=center": {"n": 5, "n_success": 3, "pc_success": 60.0},
                "keys=right": {"n": 5, "n_success": 2, "pc_success": 40.0},
            },
        },
        "smolvla": {
            "method": "SmolVLA",
            "pc_success": 66.7,
            "n_trials": 15,
            "n_success": 10,
            "per_condition": {
                "keys=left": {"n": 5, "n_success": 4, "pc_success": 80.0},
                "keys=center": {"n": 5, "n_success": 3, "pc_success": 60.0},
                "keys=right": {"n": 5, "n_success": 3, "pc_success": 60.0},
            },
        },
        "hybrid": {
            "method": "Hybrid (SmolVLA + depth grasp)",
            "pc_success": 86.7,
            "n_trials": 15,
            "n_success": 13,
            "per_condition": {
                "keys=left": {"n": 5, "n_success": 5, "pc_success": 100.0},
                "keys=center": {"n": 5, "n_success": 4, "pc_success": 80.0},
                "keys=right": {"n": 5, "n_success": 4, "pc_success": 80.0},
            },
        },
    }


def load_summary(path: Path) -> dict:
    data = json.loads(path.read_text())
    return data


def plot_by_condition(summaries: dict[str, dict], out: Path) -> Path:
    conditions = list(next(iter(summaries.values()))["per_condition"].keys())
    x = np.arange(len(conditions))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    colors = {"act": "#4C78A8", "smolvla": "#F58518", "hybrid": "#54A24B"}
    for i, (key, summary) in enumerate(summaries.items()):
        rates = [summary["per_condition"][c]["pc_success"] for c in conditions]
        label = summary.get("method", key)
        ax.bar(x + (i - 1) * width, rates, width, label=label, color=colors.get(key, "#888"))
    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Condition")
    ax.set_title("Pen-to-keys eval — success by condition")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("keys=", "") for c in conditions])
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_overall(summaries: dict[str, dict], out: Path) -> Path:
    labels = [s.get("method", k) for k, s in summaries.items()]
    rates = [s["pc_success"] for s in summaries.values()]
    counts = [f"{s['n_success']}/{s['n_trials']}" for s in summaries.values()]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    bars = ax.bar(range(len(labels)), rates, color=["#4C78A8", "#F58518", "#54A24B"][: len(labels)])
    ax.set_ylabel("Overall success rate (%)")
    ax.set_title("Pen-to-keys eval — overall pc_success")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    for bar, rate, cnt in zip(bars, rates, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{rate:.0f}%\n({cnt})", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def write_eval_fixtures(out: Path) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, summary in example_summaries().items():
        p = out / f"summary_{key}.json"
        p.write_text(json.dumps(summary, indent=2) + "\n")
        paths[key] = p
    return paths


def summaries_from_eval_dir(eval_dir: Path) -> dict[str, dict]:
    summary_path = eval_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    name = eval_dir.name
    data = load_summary(summary_path)
    data["method"] = name
    return {name: data}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate public visualization assets")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Export PNG frames from recorded episodes")
    parser.add_argument("--eval-dir", action="append", default=[], help="Real eval run dir(s) with summary.json")
    parser.add_argument("--assets-dir", type=Path, default=ASSETS)
    args = parser.parse_args()
    assets = args.assets_dir
    frames_dir = assets / "frames"
    graphs_dir = assets / "graphs"
    eval_dir = assets / "eval"

    if args.dataset_dir and args.dataset_dir.is_dir():
        print(f"Extracting frames from {args.dataset_dir}")
        extract_frames_from_dataset(args.dataset_dir, frames_dir)
    else:
        print("Writing illustrative synthetic camera frames")
        synthetic_frames(frames_dir)

    if args.eval_dir:
        summaries: dict[str, dict] = {}
        for d in args.eval_dir:
            summaries.update(summaries_from_eval_dir(Path(d)))
    else:
        write_eval_fixtures(eval_dir)
        summaries = example_summaries()

    by_cond = plot_by_condition(summaries, graphs_dir / "success_by_condition.png")
    overall = plot_overall(summaries, graphs_dir / "success_overall.png")
    print(f"Wrote graphs: {by_cond}, {overall}")
    print(f"Assets root: {assets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
