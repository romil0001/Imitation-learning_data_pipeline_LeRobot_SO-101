"""Load and train the built-in learned policies. Imports of torch/LeRobot stay inside functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from so101.algorithms.registry import LoadContext, TrainContext


def load_act(ctx: LoadContext) -> Any:
    from so101.models.act import ACTPolicy, find_policy_weights

    ckpt = Path(ctx.checkpoint) if ctx.checkpoint else ctx.root / "outputs" / "train" / f"act_{ctx.dataset_name}"
    find_policy_weights(ckpt)
    return ACTPolicy.load(ckpt, device=ctx.device)


def _normalize_smolvla_checkpoint(path: Path) -> Path | None:
    """Map a train-run dir or checkpoint root to the HF ``pretrained_model`` folder.

    Accepts:
    - ``.../pretrained_model`` (already loadable)
    - ``.../checkpoints/<step>`` (has ``pretrained_model/``)
    - ``.../smolvla_<job>`` train output (has ``checkpoints/last`` or numbered steps)
    """
    if not path.is_dir():
        return None
    if (path / "config.json").is_file():
        return path
    nested = path / "pretrained_model"
    if (nested / "config.json").is_file():
        return nested
    last = path / "checkpoints" / "last" / "pretrained_model"
    if (last / "config.json").is_file():
        return last.resolve()
    ckpts = sorted(
        p for p in path.glob("checkpoints/*/pretrained_model") if (p / "config.json").is_file()
    )
    if ckpts:
        # Prefer numeric step dirs over a dangling ``last`` name when both appear in the glob.
        numeric = [p for p in ckpts if p.parent.name.isdigit()]
        return (numeric[-1] if numeric else ckpts[-1]).resolve()
    return None


def _resolve_smolvla_checkpoint(root: Path, checkpoint: str, dataset_name: str) -> Path:
    candidates: list[Path] = []
    if checkpoint:
        candidates.append(Path(checkpoint))
    candidates.extend(
        [
            root / "outputs" / "train" / f"smolvla_{dataset_name}",
            root
            / "outputs"
            / "train"
            / f"smolvla_{dataset_name}"
            / "checkpoints"
            / "last"
            / "pretrained_model",
        ]
    )
    for candidate in candidates:
        resolved = _normalize_smolvla_checkpoint(candidate)
        if resolved is not None:
            return resolved
    raise FileNotFoundError(
        f"No SmolVLA checkpoint under outputs/train/smolvla_{dataset_name}. "
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
