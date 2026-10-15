"""Setup algorithms stay listed apart from learned policies, and a new policy can register."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from so101.algorithms import (
    Algorithm,
    LoadContext,
    TrainContext,
    format_catalog,
    get,
    list_algorithms,
    load_policy,
    names,
    register,
    runnable_names,
    train_algorithm,
    unregister,
)
from so101.algorithms.plugins_loader import discover_plugin_files


def _ctx(**overrides: object) -> LoadContext:
    data: dict[str, object] = dict(
        checkpoint="",
        dataset_name="so101_d405_demo",
        root=Path("/tmp/so101"),
        device="cpu",
        task="Place keys into metal cup",
        camera_names=("wrist", "overhead"),
        image_height=48,
        image_width=64,
        grasp_config="",
        policy_base="act",
    )
    data.update(overrides)
    return LoadContext(**data)  # type: ignore[arg-type]


def _train_ctx(algo: str) -> TrainContext:
    return TrainContext(
        algo=algo,
        dataset_name="so101_d405_demo",
        steps=1,
        batch_size=1,
        output_dir="",
        val_frac=0.2,
        seed=0,
        eval_every=1,
        backend="torch",
    )


def test_catalog_splits_setup_and_ml() -> None:
    setup = names("setup")
    ml = names("ml")
    assert set(setup).isdisjoint(ml)
    for name in (
        "find-port",
        "record",
        "replay",
        "home",
        "depth-grasp",
        "hybrid",
        "protocol",
    ):
        assert name in setup
    for name in ("act", "act-jax", "smolvla"):
        assert name in ml
    assert get("hybrid").load is not None
    assert get("depth-grasp").load is None
    assert get("act").train is not None
    assert get("smolvla").train is None
    assert "my_policy" not in ml


def test_every_ml_policy_can_load_for_eval() -> None:
    for algo in list_algorithms("ml"):
        assert algo.load is not None, algo.name
    assert "hybrid" in runnable_names()
    assert "record" not in runnable_names()


def test_catalog_text_keeps_the_families_apart() -> None:
    both = format_catalog()
    assert both.index("SETUP") < both.index("ML")
    ml_only = format_catalog("ml")
    assert "find-port" not in ml_only
    assert "act-jax" in ml_only
    setup_only = format_catalog("setup")
    assert "smolvla" not in setup_only
    assert "depth-grasp" in setup_only


def test_setup_algorithm_is_not_an_eval_policy() -> None:
    with pytest.raises(ValueError, match="setup"):
        load_policy("record", _ctx())


def test_unknown_algorithm_names_the_catalog() -> None:
    with pytest.raises(ValueError, match="act"):
        get("not-a-policy")


def test_train_rejects_setup_and_points_smolvla_at_its_script() -> None:
    with pytest.raises(ValueError, match="setup"):
        train_algorithm(_train_ctx("record"))
    with pytest.raises(RuntimeError, match="14_train_smolvla"):
        train_algorithm(_train_ctx("smolvla"))


def test_registered_policy_runs_through_eval_and_hybrid() -> None:
    class _Hold:
        def __init__(self) -> None:
            self.cfg = type("Cfg", (), {"use_depth": False, "depth_camera_name": "wrist"})()

        def reset(self) -> None:
            return None

        def select_action(self, image_rgb, state, depth_u16=None):
            del image_rgb, depth_u16
            return np.asarray(state, dtype=np.float32).copy()

    def _load(ctx: LoadContext) -> _Hold:
        assert ctx.dataset_name == "so101_d405_demo"
        return _Hold()

    register(
        Algorithm(
            name="unit-hold",
            family="ml",
            summary="Test policy.",
            run="eval: python -m so101 eval --policy unit-hold",
            module="tests.test_algorithms",
            load=_load,
        )
    )
    try:
        state = np.arange(6, dtype=np.float32)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        direct = load_policy("unit-hold", _ctx())
        assert direct.select_action(image, state).tolist() == state.tolist()
        wrapped = load_policy("hybrid", _ctx(policy_base="unit-hold"))
        assert wrapped.select_action(image, state, depth_u16=None).tolist() == state.tolist()
        with pytest.raises(ValueError, match="ML algorithms"):
            load_policy("hybrid", _ctx(policy_base="record"))
    finally:
        unregister("unit-hold")


def test_plugin_discovery_skips_underscore_files(tmp_path: Path) -> None:
    (tmp_path / "_template.py").write_text("register = None\n")
    (tmp_path / "bc.py").write_text("VALUE = 1\n")
    (tmp_path / "__init__.py").write_text("")
    found = discover_plugin_files(tmp_path)
    assert [path.name for path in found] == ["bc.py"]


def test_act_jax_selects_the_flax_backend() -> None:
    from so101.algorithms.loaders import resolve_act_backend

    assert resolve_act_backend("act", "torch") == "torch"
    assert resolve_act_backend("act", "jax") == "jax"
    assert resolve_act_backend("act-jax", "torch") == "jax"
    with pytest.raises(ValueError, match="backend"):
        resolve_act_backend("act", "cpu")


def test_names_are_case_insensitive() -> None:
    assert get("ACT").name == "act"
    assert get("Act-Jax").module.endswith("act_jax")
