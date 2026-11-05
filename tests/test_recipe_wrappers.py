"""Inference recipe: low-pass, freepace, act5/async, gemfuse resolve."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pytest

from so101.algorithms import LoadContext, get, load_policy, names
from so101.policy_wrappers import (
    AsyncReplanPolicy,
    FreespacePolicy,
    LowPassPolicy,
    apply_inference_recipe,
)
from so101.recipe import RecipeConfig, load_recipe, merge_recipe_with_ctx, resolve_gemfuse_checkpoint


class _Hold:
    def __init__(self, use_depth: bool = False) -> None:
        self.cfg = type("Cfg", (), {"use_depth": use_depth, "depth_camera_name": "wrist"})()

    def reset(self) -> None:
        return None

    def select_action(self, image_rgb, state, depth_u16=None):
        del image_rgb, depth_u16
        return np.asarray(state, dtype=np.float32) + 2.0


class _ChunkHold:
    """Minimal ACT-like queue policy for async replan tests."""

    def __init__(self) -> None:
        self.cfg = type(
            "Cfg",
            (),
            {"use_depth": False, "depth_camera_name": "wrist", "n_action_steps": 5, "chunk_size": 20},
        )()
        self._action_queue: deque[np.ndarray] = deque()
        self.predict_calls = 0

    def reset(self) -> None:
        self._action_queue = deque()

    def _predict_chunk(self, image_rgb, state, depth_u16=None):
        del image_rgb, depth_u16
        self.predict_calls += 1
        st = np.asarray(state, dtype=np.float32)
        return np.stack([st + float(i) for i in range(20)], axis=0)

    def select_action(self, image_rgb, state, depth_u16=None):
        if not self._action_queue:
            chunk = self._predict_chunk(image_rgb, state, depth_u16=depth_u16)
            n = min(int(self.cfg.n_action_steps), chunk.shape[0])
            self._action_queue.extend(chunk[:n])
        return np.asarray(self._action_queue.popleft(), dtype=np.float32)


def test_load_recipe_json() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_recipe(root / "config" / "recipes" / "gemfuse_1p3b_r18.json")
    assert cfg.name == "gemfuse_1p3b_r18"
    assert cfg.n_action_steps == 5
    assert cfg.replan_threshold == 25
    assert cfg.freespace_scale == 2.0
    assert cfg.post_id == "303653304"


def test_merge_recipe_cli_overrides() -> None:
    base = RecipeConfig(n_action_steps=5, async_replan=True, low_pass_alpha=0.3)
    merged = merge_recipe_with_ctx(base, n_action_steps=8, async_replan=False, low_pass_alpha=0.0)
    assert merged.n_action_steps == 8
    assert merged.async_replan is False
    assert merged.low_pass_alpha == 0.0


def test_low_pass_ema() -> None:
    policy = LowPassPolicy(_Hold(), alpha=0.5)
    policy.reset()
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    st = np.zeros(6, dtype=np.float32)
    a0 = policy.select_action(img, st)
    assert np.allclose(a0, 2.0)
    a1 = policy.select_action(img, st)
    assert np.allclose(a1, 0.5 * 2.0 + 0.5 * a0)


def test_freespace_scales_arm_keeps_gripper() -> None:
    policy = FreespacePolicy(_Hold(), scale=2.0, enter_mm=180.0)
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    st = np.zeros(6, dtype=np.float32)
    out = policy.select_action(img, st, depth_u16=None)
    assert np.allclose(out[:5], 4.0)
    assert np.isclose(out[5], 2.0)


def test_freespace_identity_near_contact() -> None:
    policy = FreespacePolicy(_Hold(), scale=2.0, enter_mm=180.0)
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    st = np.zeros(6, dtype=np.float32)
    depth = np.full((8, 8), 50, dtype=np.uint16)
    out = policy.select_action(img, st, depth_u16=depth)
    assert np.allclose(out, 2.0)


def test_async_replan_refills_before_queue_empty() -> None:
    base = _ChunkHold()
    wrapped = AsyncReplanPolicy(base, replan_threshold=15, n_action_steps=5, async_replan=True)
    wrapped.reset()
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    st = np.zeros(6, dtype=np.float32)
    # First call sync-fills full chunk (async path uses chunk_size).
    wrapped.select_action(img, st)
    assert base.predict_calls == 1
    assert len(base._action_queue) == 19
    # Drain until at/under threshold; a background predict should start.
    while len(base._action_queue) > 15:
        wrapped.select_action(img, st)
    wrapped.select_action(img, st)
    # Allow the worker to finish.
    import time

    for _ in range(50):
        if base.predict_calls >= 2:
            break
        time.sleep(0.01)
        wrapped.select_action(img, st)
    assert base.predict_calls >= 2
    wrapped.close()


def test_apply_recipe_stacks_wrappers() -> None:
    recipe = RecipeConfig(
        n_action_steps=5,
        async_replan=True,
        replan_threshold=25,
        freespace_scale=2.0,
        low_pass_alpha=0.5,
    )
    out = apply_inference_recipe(_ChunkHold(), recipe)
    assert isinstance(out, LowPassPolicy)
    assert isinstance(out.base, FreespacePolicy)
    assert isinstance(out.base.base, AsyncReplanPolicy)
    out.base.base.close()


def test_n_action_steps_disables_temporal_ensemble() -> None:
    from so101.policy_wrappers import apply_inference_recipe

    class _Cfg:
        n_action_steps = 100
        ensemble_temp = 0.01
        use_depth = False
        depth_camera_name = "wrist"

    class _Ensembled:
        def __init__(self) -> None:
            self.cfg = _Cfg()
            self._ensembler = object()
            self._action_queue = deque([1, 2, 3])

    policy = _Ensembled()
    out = apply_inference_recipe(
        policy,
        RecipeConfig(n_action_steps=1, async_replan=False, low_pass_alpha=0.0, freespace_scale=1.0),
    )
    assert out is policy
    assert out.cfg.n_action_steps == 1
    assert out.cfg.ensemble_temp is None
    assert out._ensembler is None
    assert len(out._action_queue) == 0


def test_gemfuse_registered() -> None:
    assert "gemfuse" in names("ml")
    assert get("gemfuse").load is not None


def test_gemfuse_missing_checkpoint_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="gemfuse checkpoint"):
        resolve_gemfuse_checkpoint(
            root=tmp_path,
            dataset_name="missing_demo",
            recipe=RecipeConfig(post_id="303653304"),
        )


def test_gemfuse_resolves_post_dir(tmp_path: Path) -> None:
    ckpt = tmp_path / "outputs" / "train" / "post_303653304"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text('{"backbone": "resnet18", "chunk_size": 8}\n')
    (ckpt / "model.pt").write_text("fake")
    found = resolve_gemfuse_checkpoint(
        root=tmp_path,
        dataset_name="demo",
        recipe=RecipeConfig(post_id="303653304"),
    )
    assert found == ckpt


def test_load_policy_applies_cli_low_pass() -> None:
    from so101.algorithms import Algorithm, register, unregister

    def _load(ctx: LoadContext):
        del ctx
        return _Hold()

    register(
        Algorithm(
            name="unit-recipe-hold",
            family="ml",
            summary="test",
            run="eval",
            module="tests.test_recipe_wrappers",
            load=_load,
        )
    )
    try:
        ctx = LoadContext(
            checkpoint="",
            dataset_name="demo",
            root=Path("/tmp"),
            device="cpu",
            task="t",
            camera_names=("wrist",),
            image_height=8,
            image_width=8,
            grasp_config="",
            policy_base="act",
            low_pass_alpha=0.5,
            async_replan=False,
            freespace_scale=1.0,
            n_action_steps=None,
        )
        policy = load_policy("unit-recipe-hold", ctx)
        assert isinstance(policy, LowPassPolicy)
    finally:
        unregister("unit-recipe-hold")
