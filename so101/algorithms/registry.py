"""Registry that keeps robot-cell setup separate from learned policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Family = Literal["setup", "ml"]

_ALGOS: dict[str, "Algorithm"] = {}
_BUILTINS = False
_PLUGINS = False


@dataclass(frozen=True)
class LoadContext:
    """Everything eval knows about the robot when it asks a policy for actions."""

    checkpoint: str
    dataset_name: str
    root: Path
    device: str
    task: str
    camera_names: tuple[str, ...]
    image_height: int
    image_width: int
    grasp_config: str
    policy_base: str


@dataclass(frozen=True)
class TrainContext:
    """Arguments from `python -m so101 train`. Plugins may ignore fields they do not use."""

    algo: str
    dataset_name: str
    steps: int
    batch_size: int
    output_dir: str
    val_frac: float
    seed: int
    eval_every: int
    backend: str


@dataclass(frozen=True)
class Algorithm:
    """One named way to run the cell (setup) or to map observations to joint actions (ml)."""

    name: str
    family: Family
    summary: str
    run: str
    module: str
    load: Callable[[LoadContext], Any] | None = None
    train: Callable[[TrainContext], Any] | None = None


def register(algo: Algorithm) -> Algorithm:
    if algo.family not in ("setup", "ml"):
        raise ValueError(f"{algo.name}: family must be 'setup' or 'ml', got {algo.family!r}")
    key = algo.name.strip().lower()
    if not key:
        raise ValueError("Algorithm name is empty")
    existing = _ALGOS.get(key)
    if existing is not None and existing is not algo:
        raise ValueError(f"Algorithm {key!r} is already registered ({existing.module})")
    stored = algo if algo.name == key else Algorithm(
        name=key,
        family=algo.family,
        summary=algo.summary,
        run=algo.run,
        module=algo.module,
        load=algo.load,
        train=algo.train,
    )
    _ALGOS[key] = stored
    return stored


def unregister(name: str) -> None:
    _ALGOS.pop(name.strip().lower(), None)


def ensure_registered() -> None:
    """Register built-ins, then any plugin file that does not start with '_'."""
    global _BUILTINS, _PLUGINS
    if not _BUILTINS:
        from so101.algorithms.catalog import builtin_algorithms

        for algo in builtin_algorithms():
            if algo.name not in _ALGOS:
                register(algo)
        _BUILTINS = True
    if not _PLUGINS:
        _PLUGINS = True
        from so101.algorithms.plugins_loader import load_plugins

        load_plugins()


def get(name: str) -> Algorithm:
    ensure_registered()
    key = name.strip().lower()
    algo = _ALGOS.get(key)
    if algo is None:
        known = ", ".join(_ALGOS)
        raise ValueError(f"Unknown algorithm {name!r}. Known: {known}")
    return algo


def list_algorithms(family: str | None = None) -> list[Algorithm]:
    ensure_registered()
    if family is not None and family not in ("setup", "ml"):
        raise ValueError(f"family must be 'setup' or 'ml', got {family!r}")
    algos = list(_ALGOS.values())
    if family is None:
        return algos
    return [algo for algo in algos if algo.family == family]


def names(family: str | None = None) -> list[str]:
    return [algo.name for algo in list_algorithms(family)]


def runnable_names() -> list[str]:
    """Algorithms eval can execute on the follower (learned policies, plus hybrid)."""
    return [algo.name for algo in list_algorithms() if algo.load is not None]


def load_policy(name: str, ctx: LoadContext) -> Any:
    algo = get(name)
    if algo.load is None:
        raise ValueError(
            f"{algo.name} is a setup algorithm ({algo.summary}). "
            f"Eval runs: {', '.join(runnable_names())}.\n{algo.run}"
        )
    return algo.load(ctx)


def train_algorithm(ctx: TrainContext) -> Any:
    algo = get(ctx.algo)
    if algo.family != "ml":
        raise ValueError(
            f"{algo.name} is a setup algorithm ({algo.summary}). "
            f"Train an ML algorithm: {', '.join(names('ml'))}.\n{algo.run}"
        )
    if algo.train is None:
        raise RuntimeError(
            f"{algo.name} is not trained by `python -m so101 train`.\n{algo.run}"
        )
    if ctx.algo != algo.name:
        ctx = TrainContext(
            algo=algo.name,
            dataset_name=ctx.dataset_name,
            steps=ctx.steps,
            batch_size=ctx.batch_size,
            output_dir=ctx.output_dir,
            val_frac=ctx.val_frac,
            seed=ctx.seed,
            eval_every=ctx.eval_every,
            backend=ctx.backend,
        )
    return algo.train(ctx)


def format_catalog(family: str | None = None) -> str:
    """Human-readable split between cell setup and learned policies."""
    if family is not None and family not in ("setup", "ml"):
        raise ValueError(f"family must be 'setup' or 'ml', got {family!r}")
    sections: list[str] = []
    if family in (None, "setup"):
        sections.append(
            _section(
                "SETUP",
                "Robot-cell routines. Motors, demos, home, depth grasp, and scoring live here. "
                "Leave this list alone when you try a new model.",
                list_algorithms("setup"),
            )
        )
    if family in (None, "ml"):
        sections.append(
            _section(
                "ML",
                "Learned policies. Each one maps cameras and the 6 joint positions to a 6-joint "
                "action on the follower. Swap the name to try another model. "
                "Depth-grasp contact override: python -m so101 eval --policy hybrid --policy-base <name>.",
                list_algorithms("ml"),
            )
        )
    sections.append(
        "Add an ML algorithm: copy so101/algorithms/plugins/_template.py to "
        "so101/algorithms/plugins/<name>.py (no leading underscore), implement select_action, "
        "and uncomment register(). Then run: python -m so101 algorithms"
    )
    return "\n\n".join(sections) + "\n"


def _section(title: str, intro: str, algos: list[Algorithm]) -> str:
    if not algos:
        return f"{title}\n{intro}\n\n  (none)"
    width = max(len(algo.name) for algo in algos)
    lines = [title, intro, ""]
    for algo in algos:
        lines.append(f"  {algo.name.ljust(width)}  {algo.summary}")
        lines.append(f"  {' ' * width}  {algo.module}")
        for line in algo.run.splitlines():
            lines.append(f"  {' ' * width}  {line}")
    return "\n".join(lines)
