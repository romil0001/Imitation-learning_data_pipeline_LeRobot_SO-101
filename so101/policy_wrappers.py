"""Compose inference post-processors: act5 queue, async replan, freepace, low-pass."""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np

from so101.recipe import RecipeConfig


def _call_select(
    policy: Any,
    image_rgb: Any,
    state: np.ndarray,
    depth_u16: np.ndarray | None,
) -> np.ndarray:
    try:
        action = policy.select_action(image_rgb, state, depth_u16=depth_u16)
    except TypeError:
        action = policy.select_action(image_rgb, state)
    return np.asarray(action, dtype=np.float32).reshape(-1)


class LowPassPolicy:
    """EMA on commanded joints: a_t = α·raw + (1-α)·a_{t-1}."""

    def __init__(self, base: Any, alpha: float = 0.3) -> None:
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError(f"low_pass_alpha must be in (0, 1], got {alpha}")
        self.base = base
        self.alpha = float(alpha)
        self.cfg = getattr(base, "cfg", type("Cfg", (), {"use_depth": False, "depth_camera_name": "wrist"})())
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        if hasattr(self.base, "reset"):
            self.base.reset()
        self._prev = None

    def select_action(
        self,
        image_rgb: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
        state: np.ndarray,
        depth_u16: np.ndarray | None = None,
    ) -> np.ndarray:
        raw = _call_select(self.base, image_rgb, state, depth_u16)
        if self._prev is None:
            self._prev = raw.copy()
            return raw
        out = self.alpha * raw + (1.0 - self.alpha) * self._prev
        self._prev = out.astype(np.float32, copy=True)
        return self._prev.copy()


class FreespacePolicy:
    """Scale arm joint deltas by ``scale`` in free space; identity near contact depth."""

    def __init__(
        self,
        base: Any,
        scale: float = 2.0,
        *,
        enter_mm: float = 180.0,
    ) -> None:
        if float(scale) <= 0:
            raise ValueError(f"freespace_scale must be > 0, got {scale}")
        self.base = base
        self.scale = float(scale)
        self.enter_mm = float(enter_mm)
        base_cfg = getattr(base, "cfg", None)
        # Do not force depth: without a frame, treat motion as free space and apply scale.
        use_depth = bool(getattr(base_cfg, "use_depth", False))
        depth_name = getattr(base_cfg, "depth_camera_name", "wrist")
        self.cfg = type("Cfg", (), {"use_depth": use_depth, "depth_camera_name": depth_name})()

    def reset(self) -> None:
        if hasattr(self.base, "reset"):
            self.base.reset()

    def select_action(
        self,
        image_rgb: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
        state: np.ndarray,
        depth_u16: np.ndarray | None = None,
    ) -> np.ndarray:
        raw = _call_select(self.base, image_rgb, state, depth_u16)
        st = np.asarray(state, dtype=np.float32).reshape(-1)
        scale = self.scale
        if depth_u16 is not None:
            valid = np.asarray(depth_u16, dtype=np.float32)
            valid = valid[(valid > 0) & np.isfinite(valid)]
            if valid.size and float(np.median(valid)) < self.enter_mm:
                scale = 1.0
        if abs(scale - 1.0) < 1e-6:
            return raw
        out = st.copy()
        n = min(5, out.shape[0], raw.shape[0])
        out[:n] = st[:n] + scale * (raw[:n] - st[:n])
        if raw.shape[0] > 5 and out.shape[0] > 5:
            out[5] = raw[5]
        return out.astype(np.float32)


class AsyncReplanPolicy:
    """Overlap chunk (or single-step) prediction with execution.

    For ACT-like bases with ``_predict_chunk`` / ``_action_queue``, refill when the
    queue length drops to ``replan_threshold``. Otherwise prefetch the next
    ``select_action`` on a worker thread.
    """

    def __init__(
        self,
        base: Any,
        *,
        replan_threshold: int = 25,
        n_action_steps: int | None = None,
        async_replan: bool = True,
    ) -> None:
        self.base = base
        self.replan_threshold = max(1, int(replan_threshold))
        self.async_replan = bool(async_replan)
        self.cfg = getattr(base, "cfg", type("Cfg", (), {"use_depth": False, "depth_camera_name": "wrist"})())
        if n_action_steps is not None and hasattr(self.cfg, "n_action_steps"):
            self.cfg.n_action_steps = int(n_action_steps)
        self._chunk_mode = hasattr(base, "_predict_chunk") and hasattr(base, "_action_queue")
        self._pool: ThreadPoolExecutor | None = None
        self._future: Future[np.ndarray] | None = None
        self._lock = threading.Lock()
        self._prefetch: np.ndarray | None = None
        if self.async_replan:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="so101-replan")

    def reset(self) -> None:
        with self._lock:
            if self._future is not None and not self._future.done():
                self._future.cancel()
            self._future = None
            self._prefetch = None
        if hasattr(self.base, "reset"):
            self.base.reset()

    def close(self) -> None:
        self.reset()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def _enqueue_chunk(self, chunk: np.ndarray) -> None:
        cfg = self.cfg
        n_steps = int(getattr(cfg, "n_action_steps", chunk.shape[0]))
        n = min(n_steps, int(getattr(cfg, "chunk_size", chunk.shape[0])), chunk.shape[0])
        # Async keeps a longer open-loop buffer so replan_threshold can fire early.
        if self.async_replan:
            n = min(int(getattr(cfg, "chunk_size", chunk.shape[0])), chunk.shape[0])
            n = max(n, min(n_steps, chunk.shape[0]))
        queue: deque = self.base._action_queue
        queue.clear()
        queue.extend(np.asarray(row, dtype=np.float32) for row in chunk[:n])

    def _predict_chunk_locked(
        self,
        image_rgb: Any,
        state: np.ndarray,
        depth_u16: np.ndarray | None,
    ) -> np.ndarray:
        return np.asarray(
            self.base._predict_chunk(image_rgb, state, depth_u16=depth_u16),
            dtype=np.float32,
        )

    def _maybe_start_chunk_replan(self, image_rgb: Any, state: np.ndarray, depth_u16: np.ndarray | None) -> None:
        if not self.async_replan or self._pool is None:
            return
        queue: deque = self.base._action_queue
        if len(queue) > self.replan_threshold:
            return
        with self._lock:
            if self._future is not None and not self._future.done():
                return
            img = image_rgb
            st = np.asarray(state, dtype=np.float32).copy()
            depth = None if depth_u16 is None else np.asarray(depth_u16).copy()
            self._future = self._pool.submit(self._predict_chunk_locked, img, st, depth)

    def _absorb_chunk_future(self) -> None:
        with self._lock:
            fut = self._future
            if fut is None or not fut.done():
                return
            self._future = None
        try:
            chunk = fut.result()
        except Exception:
            return
        self._enqueue_chunk(chunk)

    def select_action(
        self,
        image_rgb: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
        state: np.ndarray,
        depth_u16: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._chunk_mode:
            self._absorb_chunk_future()
            queue: deque = self.base._action_queue
            if not queue:
                chunk = self._predict_chunk_locked(image_rgb, state, depth_u16)
                self._enqueue_chunk(chunk)
            else:
                self._maybe_start_chunk_replan(image_rgb, state, depth_u16)
            return np.asarray(queue.popleft(), dtype=np.float32)

        # Single-step policies: optional one-action prefetch.
        if self._prefetch is not None:
            action = self._prefetch
            self._prefetch = None
        else:
            action = _call_select(self.base, image_rgb, state, depth_u16)

        if self.async_replan and self._pool is not None:
            with self._lock:
                if self._future is None or self._future.done():
                    if self._future is not None and self._future.done():
                        try:
                            self._prefetch = np.asarray(self._future.result(), dtype=np.float32)
                        except Exception:
                            self._prefetch = None
                    img = image_rgb
                    st = np.asarray(state, dtype=np.float32).copy()
                    depth = None if depth_u16 is None else np.asarray(depth_u16).copy()
                    self._future = self._pool.submit(_call_select, self.base, img, st, depth)
        return action


def apply_n_action_steps(policy: Any, n_action_steps: int | None) -> None:
    if n_action_steps is None:
        return
    cfg = getattr(policy, "cfg", None)
    if cfg is not None and hasattr(cfg, "n_action_steps"):
        cfg.n_action_steps = int(n_action_steps)
    base = getattr(policy, "base", None)
    if base is not None:
        apply_n_action_steps(base, n_action_steps)


def disable_temporal_ensemble(policy: Any) -> None:
    """Turn off overlapping-chunk ensembling so short action queues can replan.

    With ``ensemble_temp`` set, ACT predicts every frame but blends with history,
    which delays corrections. Queue mode (``n_action_steps``) needs the ensembler
    off so open-loop steps are drained and a fresh chunk is requested.
    """
    cfg = getattr(policy, "cfg", None)
    if cfg is not None and hasattr(cfg, "ensemble_temp"):
        cfg.ensemble_temp = None
    if hasattr(policy, "_ensembler"):
        policy._ensembler = None
    if hasattr(policy, "_action_queue"):
        try:
            policy._action_queue.clear()
        except Exception:
            pass
    base = getattr(policy, "base", None)
    if base is not None:
        disable_temporal_ensemble(base)


def apply_inference_recipe(policy: Any, recipe: RecipeConfig) -> Any:
    """Order: actN cfg → drop temporal ensemble → async replan → freepace → low-pass."""
    apply_n_action_steps(policy, recipe.n_action_steps)
    # Short closed-loop horizons need queue mode, not temporal ensembling.
    if recipe.n_action_steps is not None:
        disable_temporal_ensemble(policy)

    out = policy
    if recipe.async_replan:
        # Prefer wrapping the ACT-like base under hybrid so chunk queues still work.
        inner = getattr(out, "base", None)
        if inner is not None and hasattr(inner, "_predict_chunk") and hasattr(inner, "_action_queue"):
            out.base = AsyncReplanPolicy(
                inner,
                replan_threshold=recipe.replan_threshold,
                n_action_steps=recipe.n_action_steps,
                async_replan=True,
            )
        else:
            out = AsyncReplanPolicy(
                out,
                replan_threshold=recipe.replan_threshold,
                n_action_steps=recipe.n_action_steps,
                async_replan=True,
            )

    if float(recipe.freespace_scale) != 1.0:
        out = FreespacePolicy(
            out,
            scale=recipe.freespace_scale,
            enter_mm=recipe.freespace_enter_mm,
        )
    if float(recipe.low_pass_alpha) > 0.0:
        out = LowPassPolicy(out, alpha=recipe.low_pass_alpha)
    return out
