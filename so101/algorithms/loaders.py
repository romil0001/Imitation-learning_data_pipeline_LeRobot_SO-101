"""Load and train the built-in learned policies. Imports of torch/LeRobot stay inside functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from so101.algorithms.registry import LoadContext, TrainContext


def load_act(ctx: LoadContext) -> Any:
    from so101.models.act import ACTPolicy, find_policy_weights
    from so101.models.lerobot_act_wrap import LeRobotACTPolicyWrap, resolve_lerobot_act_dir

    ckpt = Path(ctx.checkpoint) if ctx.checkpoint else ctx.root / "outputs" / "train" / f"act_{ctx.dataset_name}"
    lerobot_dir = resolve_lerobot_act_dir(ckpt)
    if lerobot_dir is not None:
        return LeRobotACTPolicyWrap.load(
            lerobot_dir,
            device=ctx.device,
            task=ctx.task,
            camera_names=ctx.camera_names,
            image_height=ctx.image_height,
            image_width=ctx.image_width,
        )
    find_policy_weights(ckpt)
    return ACTPolicy.load(ckpt, device=ctx.device)


def _is_smolvla_pretrained_dir(path: Path) -> bool:
    """True if ``path/config.json`` is a LeRobot SmolVLA config (has ``type: smolvla``)."""
    cfg = path / "config.json"
    if not cfg.is_file():
        return False
    try:
        import json

        data = json.loads(cfg.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return str(data.get("type") or "").lower() == "smolvla"


def _normalize_smolvla_checkpoint(path: Path) -> Path | None:
    """Map a train-run dir or checkpoint root to the HF ``pretrained_model`` folder.

    Accepts:
    - ``.../pretrained_model`` (already loadable)
    - ``.../checkpoints/<step>`` (has ``pretrained_model/``)
    - ``.../smolvla_<job>`` train output (has ``checkpoints/last`` or numbered steps)

    Rejects native ACT run dirs (they also have ``config.json`` but no ``type`` field).
    """
    if not path.is_dir():
        return None
    if _is_smolvla_pretrained_dir(path):
        return path
    nested = path / "pretrained_model"
    if _is_smolvla_pretrained_dir(nested):
        return nested
    last = path / "checkpoints" / "last" / "pretrained_model"
    if _is_smolvla_pretrained_dir(last):
        return last.resolve()
    ckpts = sorted(
        p for p in path.glob("checkpoints/*/pretrained_model") if _is_smolvla_pretrained_dir(p)
    )
    if ckpts:
        # Prefer numeric step dirs over a dangling ``last`` name when both appear in the glob.
        numeric = [p for p in ckpts if p.parent.name.isdigit()]
        return (numeric[-1] if numeric else ckpts[-1]).resolve()
    return None


def _resolve_smolvla_checkpoint(root: Path, checkpoint: str, dataset_name: str) -> Path:
    if checkpoint:
        explicit = Path(checkpoint)
        resolved = _normalize_smolvla_checkpoint(explicit)
        if resolved is not None:
            return resolved
        raise FileNotFoundError(
            f"Path {explicit} is not a SmolVLA pretrained_model "
            "(ACT run dirs are rejected). Pick outputs/train/smolvla_* "
            f"(e.g. smolvla_{dataset_name}_v2)."
        )
    candidates = [
        root / "outputs" / "train" / f"smolvla_{dataset_name}_v2",
        root / "outputs" / "train" / f"smolvla_{dataset_name}",
        root
        / "outputs"
        / "train"
        / f"smolvla_{dataset_name}_v2"
        / "checkpoints"
        / "last"
        / "pretrained_model",
        root
        / "outputs"
        / "train"
        / f"smolvla_{dataset_name}"
        / "checkpoints"
        / "last"
        / "pretrained_model",
    ]
    for candidate in candidates:
        resolved = _normalize_smolvla_checkpoint(candidate)
        if resolved is not None:
            return resolved
    raise FileNotFoundError(
        f"No SmolVLA checkpoint under outputs/train/smolvla_{dataset_name}[_v2]. "
        "Convert + train first: bash scripts/13_convert_lerobot.sh && bash scripts/14_train_smolvla.sh"
    )


def load_smolvla(ctx: LoadContext) -> Any:
    from so101.models.smolvla_wrap import SmolVLAPolicyWrap

    ckpt = _resolve_smolvla_checkpoint(ctx.root, ctx.checkpoint, ctx.dataset_name)
    return SmolVLAPolicyWrap.load(
        ckpt,
        device=ctx.device,
        task=ctx.task,
        camera_names=ctx.camera_names,
        image_height=ctx.image_height,
        image_width=ctx.image_width,
    )


def load_hybrid(ctx: LoadContext) -> Any:
    from so101.algorithms.registry import get
    from so101.grasp_config import load_grasp_config
    from so101.hybrid_policy import HybridGraspPolicy

    base_name = ctx.policy_base.strip().lower()
    if base_name == "hybrid":
        raise ValueError("--policy-base cannot be hybrid")
    try:
        base_algo = get(base_name)
    except ValueError as exc:
        from so101.algorithms.registry import names

        raise ValueError(
            f"Unknown --policy-base {ctx.policy_base!r}. ML algorithms: {', '.join(names('ml'))}"
        ) from exc
    if base_algo.family != "ml" or base_algo.load is None:
        from so101.algorithms.registry import names

        raise ValueError(
            f"--policy-base {ctx.policy_base!r} is not an ML policy. "
            f"ML algorithms: {', '.join(names('ml'))}"
        )
    base = base_algo.load(ctx)
    return HybridGraspPolicy(base, grasp_cfg=load_grasp_config(ctx.grasp_config or None))


def _resolve_act_like_base_name(ctx: LoadContext, *, policy_label: str) -> str:
    """Pick ML base for residual / phase-mix; force ACT when checkpoint looks like ACT."""
    from so101.algorithms.registry import get

    base_name = (ctx.policy_base or "act").strip().lower()
    if base_name in {
        "residual-rl",
        "residual_rl",
        "hybrid",
        "phase-mix",
        "phase_mix",
    }:
        base_name = "act"
    ckpt = Path(ctx.checkpoint) if ctx.checkpoint else Path()
    act_like = (
        (ckpt / "best_val" / "model.pt").is_file()
        or (ckpt / "last" / "model.pt").is_file()
        or (ckpt / "model.pt").is_file()
        or (ckpt / "best_val" / "model.msgpack").is_file()
    )
    if base_name == "smolvla" and act_like:
        base_name = "act"
    base_algo = get(base_name)
    if base_algo.family != "ml" or base_algo.load is None:
        raise ValueError(f"{policy_label} needs an ML --policy-base, got {ctx.policy_base!r}")
    return base_name


def load_residual_rl(ctx: LoadContext) -> Any:
    """ACT (or policy_base) + trained residual SAC (deterministic at eval)."""
    from so101.algorithms.registry import get
    from so101.rl.dense_reward import DenseDepthReward
    from so101.rl.residual_policy import ResidualRLPolicy
    from so101.rl.residual_sac import ResidualSAC

    base_name = _resolve_act_like_base_name(ctx, policy_label="residual-rl")
    base = get(base_name).load(ctx)
    res_path = (ctx.residual_checkpoint or "").strip()
    if not res_path:
        default = ctx.root / "outputs" / "train" / "residual_rl_act"
        res_path = str(default)
    dev = ctx.device if ctx.device in {"cpu", "cuda", "mps"} else "cpu"
    agent = ResidualSAC.load(res_path, device=dev)
    return ResidualRLPolicy(base, agent, DenseDepthReward(), explore=False)


def load_phase_mix(ctx: LoadContext) -> Any:
    """ACT + residual SAC mid-band + depth-grasp near contact (deterministic at eval)."""
    import yaml

    from so101.algorithms.registry import get
    from so101.grasp_config import load_grasp_config
    from so101.rl.dense_reward import DenseDepthReward, DenseDepthRewardConfig
    from so101.rl.phase_mix_policy import PhaseMixPolicy
    from so101.rl.residual_sac import ResidualSAC

    base_name = _resolve_act_like_base_name(ctx, policy_label="phase-mix")
    base = get(base_name).load(ctx)

    mix_cfg_path = ctx.root / "config" / "phase_mix.yaml"
    raw: dict[str, Any] = {}
    if mix_cfg_path.is_file():
        loaded = yaml.safe_load(mix_cfg_path.read_text()) or {}
        if isinstance(loaded, dict):
            raw = loaded
    phases = raw.get("phases") or {}
    ctrl = raw.get("control") or {}
    reward_cfg = DenseDepthRewardConfig.from_dict(raw.get("reward") or {})

    grasp_path = (ctx.grasp_config or "").strip() or str(raw.get("grasp_config") or "")
    if grasp_path and not Path(grasp_path).is_file():
        alt = ctx.root / grasp_path
        if alt.is_file():
            grasp_path = str(alt)
    grasp_cfg = load_grasp_config(grasp_path or None)

    res_path = (ctx.residual_checkpoint or "").strip()
    if not res_path:
        res_path = str(ctx.root / "outputs" / "train" / "phase_mix_act")
    dev = ctx.device if ctx.device in {"cpu", "cuda", "mps"} else "cpu"
    agent = ResidualSAC.load(res_path, device=dev)

    return PhaseMixPolicy(
        base,
        agent,
        DenseDepthReward(reward_cfg),
        grasp_cfg=grasp_cfg,
        explore=False,
        far_enter_mm=float(phases.get("far_enter_mm", 280.0)),
        near_enter_mm=float(phases.get("near_enter_mm", 180.0)),
        base_scale=float(ctrl.get("base_scale", 1.0)),
        residual_scale=float(ctrl.get("residual_scale", 1.0)),
        low_pass_alpha=float(ctrl.get("low_pass_alpha", 0.0)),
        approach_dampen_mm=float(ctrl.get("approach_dampen_mm", 260.0)),
        approach_dampen_scale=float(ctrl.get("approach_dampen_scale", 0.35)),
        auto_close=bool(ctrl.get("auto_close", True)),
        auto_close_tol_mm=float(ctrl.get("auto_close_tol_mm", 40.0)),
        gripper_closed=float(
            ctrl.get("gripper_closed", (raw.get("reward") or {}).get("gripper_closed", 2.0))
        ),
    )


def resolve_act_backend(algo: str, backend: str) -> str:
    """``act-jax`` always trains the Flax tiny backbone. ``--backend jax`` on ``act`` does the same."""
    chosen = "jax" if algo == "act-jax" else backend
    if chosen not in ("torch", "jax"):
        raise ValueError(f"ACT backend must be 'torch' or 'jax', got {chosen!r}")
    return chosen


def train_act(ctx: TrainContext) -> Path:
    from so101.train import train_from_env

    backend = resolve_act_backend(ctx.algo, ctx.backend)
    return train_from_env(
        dataset_name=ctx.dataset_name,
        steps=ctx.steps,
        batch_size=ctx.batch_size,
        output_dir=ctx.output_dir,
        val_frac=ctx.val_frac,
        seed=ctx.seed,
        eval_every=ctx.eval_every,
        backend=backend,
        corrections_name=ctx.corrections_name,
        init_checkpoint=ctx.init_checkpoint,
    )
