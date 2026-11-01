"""Inference recipe YAML: act5, async replan, low-pass, freepace, gemfuse post id."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                f"Reading {path} requires PyYAML. Install with: pip install pyyaml"
            ) from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Recipe {path} must be a mapping")
    return data


@dataclass
class RecipeConfig:
    """Closed-loop inference knobs (training steps are informational)."""

    name: str = "default"
    base: str = "act"
    backbone: str = "resnet18"
    post_id: str = ""
    steps: int = 45_000
    n_action_steps: int | None = 5
    async_replan: bool = True
    replan_threshold: int = 25
    low_pass_alpha: float = 0.3
    freespace_scale: float = 2.0
    freespace_enter_mm: float = 180.0
    checkpoint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecipeConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def default_gemfuse_recipe_path(root: Path | None = None) -> Path:
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    return base / "config" / "recipes" / "gemfuse_1p3b_r18.yaml"


def load_recipe(path: str | Path | None = None) -> RecipeConfig:
    if not path:
        return RecipeConfig()
    p = Path(path)
    cfg = RecipeConfig.from_dict(_load_mapping(p))
    if not cfg.name or cfg.name == "default":
        cfg.name = p.stem
    return cfg


def merge_recipe_with_ctx(
    recipe: RecipeConfig,
    *,
    n_action_steps: int | None = None,
    async_replan: bool | None = None,
    replan_threshold: int | None = None,
    low_pass_alpha: float | None = None,
    freespace_scale: float | None = None,
    checkpoint: str = "",
) -> RecipeConfig:
    """CLI / LoadContext overrides win over the YAML recipe when set."""
    out = RecipeConfig(**recipe.to_dict())
    if n_action_steps is not None:
        out.n_action_steps = int(n_action_steps)
    if async_replan is not None:
        out.async_replan = bool(async_replan)
    if replan_threshold is not None:
        out.replan_threshold = int(replan_threshold)
    if low_pass_alpha is not None:
        out.low_pass_alpha = float(low_pass_alpha)
    if freespace_scale is not None:
        out.freespace_scale = float(freespace_scale)
    if checkpoint:
        out.checkpoint = checkpoint
    return out


def looks_like_checkpoint(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.suffix in {".pt", ".msgpack", ".safetensors", ".bin"}
    markers = (
        "config.json",
        "model.pt",
        "model.msgpack",
        "stats.json",
        "pretrained_model/config.json",
    )
    if any((path / m).is_file() for m in markers):
        return True
    if (path / "best_val").is_dir() or (path / "last").is_dir():
        return True
    return any(path.glob("**/config.json")) or any(path.glob("**/model.pt"))


def resolve_gemfuse_checkpoint(
    *,
    root: Path,
    dataset_name: str,
    checkpoint: str = "",
    recipe: RecipeConfig | None = None,
) -> Path:
    """Resolve gemfuse weights from --checkpoint, recipe fields, or outputs/train."""
    recipe = recipe or RecipeConfig()
    candidates: list[Path] = []
    if checkpoint:
        candidates.append(Path(checkpoint))
    if recipe.checkpoint:
        candidates.append(Path(recipe.checkpoint))
    post = (recipe.post_id or "").strip()
    if post:
        candidates.extend(
            [
                root / "outputs" / "train" / f"post_{post}",
                root / "outputs" / "train" / f"gemfuse_post_{post}",
                root / "outputs" / "train" / f"gemfuse_{post}",
            ]
        )
    candidates.extend(
        [
            root / "outputs" / "train" / f"gemfuse_{dataset_name}",
            root / "outputs" / "train" / f"gemfuse_{dataset_name}" / "best_val",
            root / "outputs" / "train" / f"gemfuse_{dataset_name}" / "last",
        ]
    )
    for path in candidates:
        if looks_like_checkpoint(path):
            return path
    searched = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "gemfuse checkpoint not found. Pass --checkpoint PATH to ACT or SmolVLA "
        "weights (or a future native gemfuse ~1.3B export), or place them under "
        f"outputs/train/gemfuse_{dataset_name}/ or outputs/train/post_<id>/.\n"
        f"Searched:\n  {searched}"
    )
