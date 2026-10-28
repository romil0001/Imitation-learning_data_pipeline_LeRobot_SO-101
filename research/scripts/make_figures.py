#!/usr/bin/env python3
"""Regenerate the quantitative figures in research/paper/figures/.

Real-robot photos are copied from assets/frames/. Everything else is computed
here so the figures can be rebuilt after a real evaluation run by pointing
--eval-dir at outputs/eval/<checkpoint>/<run_id>/ summaries.

Usage:
  python research/scripts/make_figures.py
  python research/scripts/make_figures.py --eval-json outputs/eval/a/summary.json --eval-json outputs/eval/b/summary.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from math import comb, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FIG = ROOT / "research" / "paper" / "figures"
FRAMES = ROOT / "assets" / "frames"
FIXTURES = ROOT / "assets" / "eval"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = {"act": "#4C72B0", "smolvla": "#DD8452", "hybrid": "#55A868", "grey": "#7f7f7f"}


# ----------------------------------------------------------------------------
# statistics helpers
# ----------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fisher_two_sided(a: int, b: int, n: int) -> float:
    tot = a + b

    def pr(x: int) -> float:
        return comb(n, x) * comb(n, tot - x) / comb(2 * n, tot)

    obs = pr(a)
    return sum(pr(x) for x in range(max(0, tot - n), min(n, tot) + 1) if pr(x) <= obs + 1e-12)


def fisher_power(p1: float, p2: float, n: int, alpha: float = 0.05) -> float:
    pw = 0.0
    for a in range(n + 1):
        pa = comb(n, a) * p1**a * (1 - p1) ** (n - a)
        for b in range(n + 1):
            pb = comb(n, b) * p2**b * (1 - p2) ** (n - b)
            if fisher_two_sided(a, b, n) < alpha:
                pw += pa * pb
    return pw


def beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    from math import lgamma

    logc = lgamma(a + b) - lgamma(a) - lgamma(b)
    return np.exp(logc + (a - 1) * np.log(x) + (b - 1) * np.log(1 - x))


def prob_a_gt_b(ka: int, na: int, kb: int, nb: int, n_mc: int = 200_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    a = rng.beta(ka + 1, na - ka + 1, n_mc)
    b = rng.beta(kb + 1, nb - kb + 1, n_mc)
    return float(np.mean(a > b))


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def fig_ci_width() -> None:
    ns = np.arange(5, 121)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for p, ls in [(0.5, "-"), (0.7, "--"), (0.9, ":")]:
        half = []
        for n in ns:
            k = int(round(p * n))
            lo, hi = wilson(k, int(n))
            half.append(100 * (hi - lo) / 2)
        ax.plot(ns, half, ls, color="k", label=f"$\\hat p$ = {p:.1f}")
    for n, lab in [(15, "default\nprotocol"), (30, ""), (60, "")]:
        ax.axvline(n, color=COLORS["grey"], lw=0.6, alpha=0.6)
        if lab:
            ax.text(n + 1, 38, lab, fontsize=7, color=COLORS["grey"], va="top")
    ax.set_xlabel("trials $n$")
    ax.set_ylabel("95% Wilson half-width (pp)")
    ax.set_ylim(0, 40)
    ax.legend(frameon=False)
    fig.savefig(FIG / "ci_width.pdf")
    plt.close(fig)


def fig_power() -> None:
    ns = list(range(5, 76, 5))
    pairs = [(0.4, 0.8), (0.5, 0.8), (0.6, 0.8), (0.5, 0.7)]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for (p1, p2), ls in zip(pairs, ["-", "--", "-.", ":"]):
        ax.plot(ns, [100 * fisher_power(p1, p2, n) for n in ns], ls, color="k", label=f"{p1:.0%} vs {p2:.0%}")
    ax.axhline(80, color=COLORS["hybrid"], lw=0.8, alpha=0.8)
    ax.text(5, 82, "80% power", fontsize=7, color=COLORS["hybrid"])
    ax.axvline(15, color=COLORS["grey"], lw=0.6, alpha=0.6)
    ax.set_xlabel("trials per policy")
    ax.set_ylabel("power of two-sided Fisher test (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, title="true success rates", title_fontsize=7)
    fig.savefig(FIG / "power.pdf")
    plt.close(fig)


def fig_controller_response() -> None:
    """Drive the real DepthGraspController against a toy 1-D plant."""
    from so101.depth_grasp import DepthGraspConfig, DepthGraspController
    from so101.dataset.schema import JOINT_NAMES

    cfg = DepthGraspConfig()
    ctrl = DepthGraspController(cfg)
    rng = np.random.default_rng(1)

    # Toy plant: a positive change on either vertical joint lowers the wrist
    # camera toward the table by mm_per_deg millimetres. The base policy keeps
    # commanding a slow descent toward a height that is 20 mm too high (the
    # grasp-height error the controller is meant to remove).
    mm_per_deg = 2.0
    state = np.zeros(6, dtype=np.float32)
    depth_mm = 260.0
    base_target_mm = cfg.target_mm + 20.0
    jl = [JOINT_NAMES.index(j) for j in cfg.vertical_joints]
    gi = JOINT_NAMES.index("gripper")

    T = 90
    log = {k: [] for k in ("depth", "phase", "delta", "gripper_cmd", "base_depth")}
    base_depth = depth_mm
    for _t in range(T):
        # base policy: descend 1.2 mm/step until its own (too high) target
        base_desc = 1.2 if base_depth > base_target_mm else 0.0
        base_depth -= base_desc
        base_action = state.copy()
        for j in jl:
            base_action[j] = state[j] + base_desc / (mm_per_deg * len(jl))
        base_action[gi] = 40.0  # open

        img = np.full((48, 64), depth_mm + rng.normal(0, 1.5), dtype=np.float32)
        img[img < 0] = 0
        out = ctrl.override(base_action, state, img.astype(np.uint16))

        delta = float(sum(out[j] - state[j] for j in jl))
        state[:] = out  # servo reaches its target within one step
        depth_mm -= mm_per_deg * delta
        log["depth"].append(depth_mm)
        log["base_depth"].append(base_depth)
        log["phase"].append(ctrl.phase)
        log["delta"].append(delta)
        log["gripper_cmd"].append(float(out[gi]))

    t = np.arange(T)
    phases = ["idle", "grasping", "closing", "done"]
    ph_idx = np.array([phases.index(p) for p in log["phase"]])

    fig, axes = plt.subplots(3, 1, figsize=(3.4, 3.6), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.3]})
    ax = axes[0]
    ax.plot(t, log["depth"], color="k", label="wrist depth (hybrid)")
    ax.plot(t, log["base_depth"], color=COLORS["grey"], ls="--", label="base policy alone")
    ax.axhline(cfg.enter_mm, color=COLORS["smolvla"], lw=0.7, ls=":")
    ax.axhline(cfg.target_mm, color=COLORS["hybrid"], lw=0.7, ls=":")
    ax.text(T - 1, cfg.enter_mm + 3, "enter", ha="right", fontsize=7, color=COLORS["smolvla"])
    ax.text(T - 1, cfg.target_mm + 3, "target $d^*$", ha="right", fontsize=7, color=COLORS["hybrid"])
    ax.set_ylabel("ROI median depth (mm)")
    ax.legend(frameon=False, loc="lower left")
    ax = axes[1]
    ax.plot(t, log["delta"], color="k")
    ax.axhline(cfg.max_delta_deg * len(jl), color=COLORS["grey"], lw=0.6, ls=":")
    ax.axhline(-cfg.max_delta_deg * len(jl), color=COLORS["grey"], lw=0.6, ls=":")
    ax.set_ylabel("$\\sum_j \\Delta q_j$ (deg)")
    ax = axes[2]
    ax.step(t, ph_idx, where="post", color="k")
    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels(phases)
    ax.set_xlabel("control step (30 Hz)")
    for a in axes:
        for i in range(1, T):
            if ph_idx[i] != ph_idx[i - 1]:
                a.axvline(i, color=COLORS["grey"], lw=0.5, alpha=0.5)
    fig.savefig(FIG / "controller_response.pdf")
    plt.close(fig)


def _load_summaries(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        d = json.loads(p.read_text())
        d.setdefault("method", p.stem.replace("summary_", ""))
        rows.append(d)
    return rows


def fig_success_with_ci(summaries: list[dict], illustrative: bool) -> None:
    conds = list(summaries[0]["per_condition"].keys())
    groups = ["overall"] + conds
    x = np.arange(len(groups))
    w = 0.8 / len(summaries)
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    for i, s in enumerate(summaries):
        ks = [s["n_success"]] + [s["per_condition"][c]["n_success"] for c in conds]
        ns = [s["n_trials"]] + [s["per_condition"][c]["n"] for c in conds]
        rates = np.array([100 * k / n for k, n in zip(ks, ns)])
        los, his = zip(*[wilson(k, n) for k, n in zip(ks, ns)])
        err = np.vstack([rates - 100 * np.array(los), 100 * np.array(his) - rates])
        name = s["method"].lower()
        color = COLORS["hybrid"] if "hybrid" in name else COLORS["smolvla"] if "smolvla" in name else COLORS["act"]
        ax.bar(x + (i - (len(summaries) - 1) / 2) * w, rates, w, yerr=err, capsize=2, color=color, label=s["method"], alpha=0.9)
        for xi, k, n in zip(x + (i - (len(summaries) - 1) / 2) * w, ks, ns):
            ax.text(xi, 2, f"{k}/{n}", ha="center", fontsize=6, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("keys=", "") for g in groups])
    ax.set_ylabel("success rate (%)  ±95% Wilson")
    ax.set_ylim(0, 110)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    if illustrative:
        ax.set_title("Illustrative fixture data (not a real evaluation)", color="firebrick", fontsize=8)
    fig.savefig(FIG / "success_with_ci.pdf")
    plt.close(fig)


def fig_posteriors(summaries: list[dict], illustrative: bool) -> None:
    xs = np.linspace(0.001, 0.999, 600)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    by_name = {}
    for s in summaries:
        k, n = s["n_success"], s["n_trials"]
        name = s["method"].lower()
        color = COLORS["hybrid"] if "hybrid" in name else COLORS["smolvla"] if "smolvla" in name else COLORS["act"]
        ax.plot(100 * xs, beta_pdf(xs, k + 1, n - k + 1), color=color, label=f"{s['method']}  ({k}/{n})")
        by_name[name] = (k, n)
    ax.set_xlabel("true success rate $p$ (%)")
    ax.set_ylabel("posterior density")
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    hyb = next((v for k, v in by_name.items() if "hybrid" in k), None)
    base = next((v for k, v in by_name.items() if "smolvla" in k and "hybrid" not in k), None)
    if hyb and base:
        pr = prob_a_gt_b(hyb[0], hyb[1], base[0], base[1])
        ax.text(0.02, 0.45, f"P(hybrid > SmolVLA | data) = {pr:.2f}", transform=ax.transAxes, fontsize=7, va="top")
    if illustrative:
        ax.set_title("Illustrative fixture data (not a real evaluation)", color="firebrick", fontsize=8)
    fig.savefig(FIG / "posteriors.pdf")
    plt.close(fig)


def fig_chunk_override_timeline() -> None:
    """Schematic of the action-chunk / override interaction (gap G3)."""
    fig, ax = plt.subplots(figsize=(5.4, 1.9))
    K = 100
    t_enter, t_exit = 50, 75
    # chunk planned at t=0 executes open-loop to t=K; replanned at K
    ax.broken_barh([(0, K)], (2.0, 0.6), color=COLORS["act"], alpha=0.85)
    ax.broken_barh([(K, 60)], (2.0, 0.6), color=COLORS["act"], alpha=0.5)
    ax.text(K / 2, 2.3, "chunk planned at $t_0$, popped open-loop", ha="center", va="center", fontsize=6.5, color="white")
    ax.text(K + 30, 2.3, "next chunk", ha="center", va="center", fontsize=6.5, color="white")
    ax.broken_barh([(t_enter, t_exit - t_enter)], (1.0, 0.6), color=COLORS["hybrid"], alpha=0.9)
    ax.text((t_enter + t_exit) / 2, 1.3, "depth override", ha="center", va="center", fontsize=6.5, color="white")
    ax.broken_barh([(t_exit, K - t_exit)], (0.0, 0.6), color="firebrick", alpha=0.8)
    ax.text(t_exit + (K - t_exit) / 2, 0.3, "stale plan", ha="center", va="center", fontsize=6.5, color="white")
    ax.annotate("", xy=(t_exit, 0.9), xytext=(t_exit, 1.0), arrowprops=dict(arrowstyle="-", color="k", lw=0.6))
    ax.set_yticks([0.3, 1.3, 2.3])
    ax.set_yticklabels(["executed from\nold plan", "controller", "base policy\nqueue"], fontsize=7)
    ax.set_xlabel("control step")
    ax.set_xlim(0, K + 60)
    ax.set_ylim(-0.3, 2.9)
    ax.spines["left"].set_visible(False)
    fig.savefig(FIG / "chunk_override.pdf")
    plt.close(fig)


def fig_roi_overlay(roi_frac: float = 0.3) -> None:
    """Draw the controller's centred ROI on the wrist-depth placeholder frame."""
    import matplotlib.patches as patches
    from PIL import Image

    src = FRAMES / "wrist_depth.png"
    if not src.is_file():
        return
    im = np.asarray(Image.open(src).convert("RGB"))
    h, w = im.shape[:2]
    rh, rw = int(h * roi_frac), int(w * roi_frac)
    y0, x0 = (h - rh) // 2, (w - rw) // 2
    fig, ax = plt.subplots(figsize=(2.6, 1.95))
    ax.imshow(im)
    ax.add_patch(patches.Rectangle((x0, y0), rw, rh, fill=False, lw=1.5, ec="white", ls="--"))
    ax.text(x0 + 4, y0 - 6, f"ROI  $\\rho$={roi_frac}", color="white", fontsize=7, va="bottom")
    ax.text(w // 2, h // 2, "$d_t$ = median", color="white", fontsize=7, ha="center", va="center")
    ax.set_axis_off()
    fig.savefig(FIG / "roi_overlay.pdf")
    plt.close(fig)


def copy_photos() -> None:
    for name in ("overhead_real.jpg", "wrist_real.jpg", "wrist_depth.png", "wrist_approach.png", "wrist_grasp.png"):
        src = FRAMES / name
        if src.is_file():
            shutil.copy2(src, FIG / name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", action="append", default=[], help="summary.json from a real protocol run (repeatable)")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    if args.eval_json:
        summaries = _load_summaries([Path(p) for p in args.eval_json])
        illustrative = False
    else:
        summaries = _load_summaries(sorted(FIXTURES.glob("summary_*.json")))
        illustrative = True

    copy_photos()
    fig_ci_width()
    fig_power()
    fig_controller_response()
    fig_success_with_ci(summaries, illustrative)
    fig_posteriors(summaries, illustrative)
    fig_chunk_override_timeline()
    fig_roi_overlay()
    print(f"figures written to {FIG}")


if __name__ == "__main__":
    main()
