"""Gemfuse recipe policy: user-supplied checkpoint + inference post-processors.

Until native ~1.3B gemfuse weights ship, ``--checkpoint`` may point at an ACT or
SmolVLA directory. The recipe (async replan / act5 / low-pass / freepace2x) still
applies through ``load_policy`` after this loader returns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from so101.algorithms.registry import Algorithm, LoadContext, register
from so101.recipe import RecipeConfig, load_recipe, resolve_gemfuse_checkpoint


def _detect_base(checkpoint: Path, recipe_base: str) -> str:
    """Prefer recipe.base; sniff SmolVLA vs ACT when the directory is ambiguous."""
    base = (recipe_base or "act").strip().lower()
    if base in {"act", "act-jax", "smolvla"}:
        text_paths = list(checkpoint.glob("**/config.json"))[:3]
        for cfg_path in text_paths:
            try:
                text = cfg_path.read_text().lower()
            except OSError:
                continue
            if "smolvla" in text or "vlm" in text or "tokenizer" in text:
                return "smolvla"
        if (checkpoint / "model.pt").is_file() or (checkpoint / "best_val" / "model.pt").is_file():
            return "act" if base != "smolvla" else base
        return base
    return "act"


def _load(ctx: LoadContext) -> Any:
    from so101.algorithms.loaders import load_act, load_smolvla

    recipe = (
        load_recipe(ctx.recipe)
        if ctx.recipe
        else RecipeConfig(name="gemfuse", base="act", post_id="303653304")
    )
    ckpt = resolve_gemfuse_checkpoint(
        root=Path(ctx.root),
        dataset_name=ctx.dataset_name,
        checkpoint=ctx.checkpoint,
        recipe=recipe,
    )
    base_name = _detect_base(ckpt, recipe.base or ctx.policy_base or "act")
    inner = LoadContext(
        checkpoint=str(ckpt),
        dataset_name=ctx.dataset_name,
        root=ctx.root,
        device=ctx.device,
        task=ctx.task,
        camera_names=ctx.camera_names,
        image_height=ctx.image_height,
        image_width=ctx.image_width,
        grasp_config=ctx.grasp_config,
        policy_base=base_name,
        recipe="",  # wrappers applied once by load_policy
        n_action_steps=None,
        async_replan=False,
        replan_threshold=ctx.replan_threshold,
        low_pass_alpha=0.0,
        freespace_scale=1.0,
    )
    if base_name == "smolvla":
        return load_smolvla(inner)
    return load_act(inner)


register(
    Algorithm(
        name="gemfuse",
        family="ml",
        summary=(
            "Gemfuse recipe stub (~1.3B when weights exist): loads a user checkpoint "
            "(ACT/SmolVLA today) and pairs with async/act5/low-pass/freespace recipes."
        ),
        run=(
            "eval:  python -m so101 eval --policy gemfuse "
            "--recipe config/recipes/gemfuse_1p3b_r18.yaml --checkpoint <path>\n"
            "train: use ACT/SmolVLA (or drop native gemfuse weights under "
            "outputs/train/gemfuse_<dataset>/ / post_<id>/)"
        ),
        module="so101.algorithms.plugins.gemfuse",
        load=_load,
        train=None,
    )
)
