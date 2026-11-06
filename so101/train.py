"""Train ACT on a local SO101Dataset. Mac MPS / CUDA / CPU. No Hub."""

from __future__ import annotations

from collections import OrderedDict
import json
import math
import os
import random
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from so101.dataset import SO101Dataset, compute_stats, write_stats
from so101.models.act import ACT, ACTConfig, preprocess_depth, preprocess_image


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_init_fn(worker_id: int) -> None:
    """Prevent OpenCV thread contention across PyTorch DataLoader workers and seed RNGs."""
    cv2.setNumThreads(0)
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def split_episodes(
    n_episodes: int,
    val_frac: float = 0.2,
    seed: int = 0,
    force_train: list[int] | tuple[int, ...] | None = None,
) -> tuple[list[int], list[int]]:
    """Hold out episodes for val. Indices in force_train always stay in train.

    DAgger corrections are forced into train so the val score stays a holdout
    of the original demonstrations.
    """
    if n_episodes < 1:
        raise ValueError("Need at least one episode to split")
    forced = sorted({int(i) for i in (force_train or [])})
    if any(i < 0 or i >= n_episodes for i in forced):
        raise ValueError(f"force_train indices {forced} outside 0..{n_episodes - 1}")
    pool = [i for i in range(n_episodes) if i not in set(forced)]
    if len(pool) <= 1:
        if len(pool) == 1 or n_episodes == 1:
            warnings.warn("Too few episodes for a val split; train-only", stacklevel=2)
        return sorted(pool + forced), []
    rng = np.random.default_rng(seed)
    indices = np.array(pool, dtype=np.int64)
    rng.shuffle(indices)
    n_val = int(round(len(pool) * val_frac))
    n_val = max(1, n_val)
    n_val = min(n_val, len(pool) - 1)
    val = sorted(int(i) for i in indices[:n_val])
    train = sorted(int(i) for i in list(indices[n_val:]) + forced)
    return train, val


def compute_normalization_stats(
    ds: SO101Dataset,
    episode_indices: list[int],
    norm_mode: str = "minmax",
) -> dict:
    """Compute normalization stats.

    In 'minmax' mode, maps [q_min, q_max] to [-1, 1] by setting:
      mean = (q_max + q_min) / 2
      std  = (q_max - q_min) / 2
    This makes (x - mean) / std map directly to [-1, 1] while keeping full backward
    compatibility with rollout scripts that compute `pred * action_std + action_mean`.
    """
    base_stats = compute_stats(ds, episode_indices=episode_indices)
    if norm_mode != "minmax":
        base_stats["norm_mode"] = "gaussian"
        return base_stats

    states = []
    actions = []
    for ep_i in episode_indices:
        ep = ds.load_episode(ep_i)
        states.append(np.asarray(ep.state, dtype=np.float32))
        actions.append(np.asarray(ep.action, dtype=np.float32))

    all_states = np.concatenate(states, axis=0)
    all_actions = np.concatenate(actions, axis=0)

    # Use 1st and 99th percentiles (or min/max) with epsilon floor to avoid div-by-zero
    s_min = np.min(all_states, axis=0)
    s_max = np.max(all_states, axis=0)
    a_min = np.min(all_actions, axis=0)
    a_max = np.max(all_actions, axis=0)

    s_center = 0.5 * (s_max + s_min)
    s_half_range = np.maximum(0.5 * (s_max - s_min), 1e-4)
    a_center = 0.5 * (a_max + a_min)
    a_half_range = np.maximum(0.5 * (a_max - a_min), 1e-4)

    return {
        **base_stats,
        "norm_mode": "minmax",
        "state_mean": s_center.tolist(),
        "state_std": s_half_range.tolist(),
        "action_mean": a_center.tolist(),
        "action_std": a_half_range.tolist(),
        "state_min": s_min.tolist(),
        "state_max": s_max.tolist(),
        "action_min": a_min.tolist(),
        "action_max": a_max.tolist(),
    }


def resolve_act_weights(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    for cand in (path / "best_val" / "model.pt", path / "last" / "model.pt", path / "model.pt"):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"No ACT weights under {path} (expected best_val/model.pt)")


def _load_init_weights(model: ACT, path: Path) -> Path:
    ckpt = resolve_act_weights(path)
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    return ckpt


class ACTChunkDataset(Dataset):
    """Streaming/LRU-cached dataset with on-the-fly frame resizing and augmentations."""

    def __init__(
        self,
        root: Path,
        cfg: ACTConfig,
        stats: dict,
        episode_indices: list[int] | None = None,
        augment: bool = False,
        max_cached_episodes: int = 12,
    ) -> None:
        self.root = Path(root)
        self.ds = SO101Dataset(self.root)
        self.cfg = cfg
        self.augment = augment
        self.max_cached_episodes = max_cached_episodes
        self.norm_mode = stats.get("norm_mode", "minmax")

        self.state_mean = np.asarray(stats["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(stats["state_std"], dtype=np.float32)
        self.action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(stats["action_std"], dtype=np.float32)

        self.camera_names = list(self.ds.meta.names())
        self.use_depth = bool(self.ds.meta.use_depth)

        if episode_indices is None:
            episode_indices = list(range(self.ds.meta.episode_count))
        self.episode_indices = list(episode_indices)

        # Store only lightweight state/action arrays in RAM; lazy-load heavy video frames
        self.traj_meta: list[tuple[int, np.ndarray, np.ndarray]] = []
        self.index: list[tuple[int, int]] = []
        self._frame_cache: OrderedDict[int, tuple[list[np.ndarray], np.ndarray | None]] = OrderedDict()

        for local_i, ep_i in enumerate(self.episode_indices):
            ep = self.ds.load_episode(ep_i)
            state_arr = np.asarray(ep.state, dtype=np.float32)
            action_arr = np.asarray(ep.action, dtype=np.float32)
            self.traj_meta.append((ep_i, state_arr, action_arr))
            for t in range(ep.num_frames):
                self.index.append((local_i, t))

    def _get_episode_media(self, ep_i: int) -> tuple[list[np.ndarray], np.ndarray | None]:
        """Per-worker bounded LRU cache for episode frames to avoid RAM exhaustion."""
        if ep_i in self._frame_cache:
            self._frame_cache.move_to_end(ep_i)
            return self._frame_cache[ep_i]

        ep = self.ds.load_episode(ep_i)
        cam_frames = [ep.load_frames(name) for name in self.camera_names]
        depth_frames = ep.load_depth() if self.use_depth else None

        self._frame_cache[ep_i] = (cam_frames, depth_frames)
        if len(self._frame_cache) > self.max_cached_episodes:
            self._frame_cache.popitem(last=False)
        return cam_frames, depth_frames

    @staticmethod
    def _augment_rgb_and_depth(
        rgb_list: list[np.ndarray],
        depth_frame: np.ndarray | None,
        target_size: int,
    ) -> tuple[list[np.ndarray], np.ndarray | None]:
        """Apply random crop + color jitter during training while keeping RGB & Depth aligned."""
        # Photometric jitter (shared across views to simulate room lighting changes)
        brightness = np.random.uniform(0.85, 1.15)
        contrast = np.random.uniform(0.85, 1.15)
        shift = np.random.uniform(-12.0, 12.0)

        out_rgbs: list[np.ndarray] = []
        crop_Scale = np.random.uniform(0.92, 1.0)
        u_y = np.random.uniform(0.0, 1.0)
        u_x = np.random.uniform(0.0, 1.0)

        for frame in rgb_list:
            h, w = frame.shape[:2]
            ch, cw = max(1, int(h * crop_Scale)), max(1, int(w * crop_Scale))
            y0 = int((h - ch) * u_y)
            x0 = int((w - cw) * u_x)
            cropped = frame[y0 : y0 + ch, x0 : x0 + cw]
            resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)

            jittered = resized.astype(np.float32) * contrast * brightness + shift
            out_rgbs.append(np.clip(jittered, 0, 255).astype(np.uint8))

        out_depth: np.ndarray | None = None
        if depth_frame is not None:
            dh, dw = depth_frame.shape[:2]
            dch, dcw = max(1, int(dh * crop_Scale)), max(1, int(dw * crop_Scale))
            dy0 = int((dh - dch) * u_y)
            dx0 = int((dw - dcw) * u_x)
            d_crop = depth_frame[dy0 : dy0 + dch, dx0 : dx0 + dcw]
            out_depth = cv2.resize(d_crop, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

        return out_rgbs, out_depth

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        local_i, t = self.index[i]
        ep_i, state, action = self.traj_meta[local_i]
        t_len = state.shape[0]

        cam_frames, depth_frames = self._get_episode_media(ep_i)
        raw_cams_t = [cam[t] for cam in cam_frames]
        raw_depth_t = depth_frames[t] if (self.use_depth and depth_frames is not None) else None

        if self.augment:
            resized_cams, resized_depth = self._augment_rgb_and_depth(
                raw_cams_t, raw_depth_t, self.cfg.image_size
            )
        else:
            resized_cams = [
                cv2.resize(f, (self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_AREA)
                for f in raw_cams_t
            ]
            resized_depth = (
                cv2.resize(
                    raw_depth_t,
                    (self.cfg.image_size, self.cfg.image_size),
                    interpolation=cv2.INTER_NEAREST,
                )
                if raw_depth_t is not None
                else None
            )

        imgs = [
            preprocess_image(cam_img, self.cfg.image_size, self.cfg.backbone)
            for cam_img in resized_cams
        ]
        img = torch.stack(imgs, dim=0)

        st = (state[t] - self.state_mean) / self.state_std
        chunk = np.zeros((self.cfg.chunk_size, self.cfg.n_joints), dtype=np.float32)
        mask = np.zeros((self.cfg.chunk_size,), dtype=np.float32)

        end = min(t + self.cfg.chunk_size, t_len)
        n = end - t
        norm_act = (action[t:end] - self.action_mean) / self.action_std
        if self.norm_mode == "minmax":
            norm_act = np.clip(norm_act, -1.0, 1.0)
            st = np.clip(st, -1.0, 1.0)

        chunk[:n] = norm_act
        mask[:n] = 1.0

        out: dict[str, torch.Tensor] = {
            "images": img,
            "state": torch.from_numpy(st.astype(np.float32)),
            "actions": torch.from_numpy(chunk),
            "action_mask": torch.from_numpy(mask),
        }
        if self.use_depth and resized_depth is not None:
            out["depth"] = preprocess_depth(resized_depth, self.cfg.image_size, self.cfg.depth_max_mm)
        return out


def _save_weights(model: ACT, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def _batch_depth(batch: dict, device: str) -> torch.Tensor | None:
    depth = batch.get("depth")
    if depth is None:
        return None
    return depth.to(device, non_blocking=True)


def _compute_step_losses(
    out: dict[str, torch.Tensor],
    target_actions: torch.Tensor,
    action_mask: torch.Tensor,
    kl_weight: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ensure L1 loss is strictly masked over valid (non-padded) chunk timesteps."""
    kl_loss = out["kl"]
    # If model returns raw predictions ('pred_actions' or 'a_hat'), enforce exact mask normalization
    pred = out.get("pred_actions", out.get("a_hat"))
    if pred is not None:
        mask_3d = action_mask.unsqueeze(-1)  # (B, chunk_size, 1)
        denom = (mask_3d.sum() * target_actions.shape[-1]).clamp(min=1.0)
        l1_loss = (torch.abs(pred - target_actions) * mask_3d).sum() / denom
        total_loss = l1_loss + kl_weight * kl_loss
        return total_loss, l1_loss, kl_loss
    return out["loss"], out["l1"], kl_loss


def _build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 500,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Linear warmup followed by cosine decay down to min_lr_ratio * base_lr."""
    warmup_steps = max(1, min(warmup_steps, total_steps // 5))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _validate(model: ACT, loader: DataLoader, device: str, kl_weight: float = 10.0) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    l1s: list[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            mask = batch["action_mask"].to(device, non_blocking=True)
            out = model(
                images=images,
                state=state,
                actions=actions,
                action_mask=mask,
                depth=_batch_depth(batch, device),
            )
            loss, l1, _ = _compute_step_losses(out, actions, mask, kl_weight=kl_weight)
            losses.append(float(loss.detach().cpu()))
            l1s.append(float(l1.detach().cpu()))
    model.train()
    return float(np.mean(losses)), float(np.mean(l1s))


def train(
    *,
    dataset_root: Path,
    output_dir: Path,
    steps: int = 40_000,
    batch_size: int = 8,
    config: ACTConfig | None = None,
    device: str | None = None,
    lr_backbone: float = 1e-5,
    lr_rest: float = 1e-4,  # Fixed from 1e-5 -> 1e-4 for Transformer layers
    weight_decay: float = 1e-4,
    warmup_steps: int = 500,
    norm_mode: str = "minmax",
    augment: bool = True,
    num_workers: int | None = None,
    val_frac: float = 0.2,
    seed: int = 0,
    eval_every: int = 1000,
    grad_clip: float = 1.0,
    backend: str = "torch",
    force_train: list[int] | tuple[int, ...] | None = None,
    init_checkpoint: str | Path | None = None,
) -> Path:
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or ACTConfig()
    cfg = ACTConfig(**{**cfg.to_dict(), "backend": backend})
    if backend == "jax":
        if force_train or init_checkpoint:
            raise ValueError(
                "DAgger fine-tune (--corrections / --init-checkpoint) is implemented for --backend torch"
            )
        from so101.jax_train import train_jax

        return train_jax(
            dataset_root=dataset_root,
            output_dir=output_dir,
            steps=steps,
            batch_size=batch_size,
            config=cfg,
            val_frac=val_frac,
            seed=seed,
            eval_every=eval_every,
            grad_clip=grad_clip,
            lr_backbone=lr_backbone,
            lr_rest=lr_rest,
            weight_decay=weight_decay,
        )

    device = device or detect_device()
    set_seed(seed)

    if num_workers is None:
        cpu_count = os.cpu_count() or 4
        num_workers = min(4, max(1, cpu_count // 2)) if device in ("cuda", "mps") else 0
    pin_memory = device == "cuda"

    ds = SO101Dataset(dataset_root)
    train_episodes, val_episodes = split_episodes(
        ds.meta.episode_count, val_frac=val_frac, seed=seed, force_train=force_train
    )
    if cfg.use_depth != ds.meta.use_depth:
        cfg = ACTConfig(
            **{
                **cfg.to_dict(),
                "use_depth": ds.meta.use_depth,
                "depth_camera_name": ds.meta.depth_camera_name,
            }
        )

    stats = compute_normalization_stats(ds, episode_indices=train_episodes, norm_mode=norm_mode)
    write_stats(output_dir, stats)
    (output_dir / "split.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "train_episodes": train_episodes,
                "val_episodes": val_episodes,
                "force_train": list(force_train or []),
            },
            indent=2,
        )
        + "\n"
    )

    data = ACTChunkDataset(
        dataset_root, cfg, stats, episode_indices=train_episodes, augment=augment
    )
    if len(data) < 1:
        raise ValueError(f"No frames in train split of {dataset_root}")

    bs = min(batch_size, len(data))
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        data,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        drop_last=len(data) > bs,
        generator=g,
    )

    val_loader: DataLoader | None = None
    if val_episodes:
        val_data = ACTChunkDataset(
            dataset_root, cfg, stats, episode_indices=val_episodes, augment=False
        )
        if len(val_data) >= 1:
            val_bs = min(batch_size, len(val_data))
            val_loader = DataLoader(
                val_data,
                batch_size=val_bs,
                shuffle=False,
                num_workers=max(0, num_workers // 2),
                pin_memory=pin_memory,
                persistent_workers=(num_workers // 2 > 0),
                worker_init_fn=_worker_init_fn if (num_workers // 2 > 0) else None,
            )

    run_config = {
        **cfg.to_dict(),
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "val_frac": val_frac,
        "eval_every": eval_every,
        "task": ds.meta.task,
        "camera_names": list(ds.meta.names()),
        "lr_backbone": lr_backbone,
        "lr_rest": lr_rest,
        "warmup_steps": warmup_steps,
        "norm_mode": norm_mode,
        "augment": augment,
        "num_workers": num_workers,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "backend": backend,
        "init_checkpoint": str(init_checkpoint or ""),
        "force_train": list(force_train or []),
    }
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    model = ACT(cfg).to(device)
    if init_checkpoint:
        loaded = _load_init_weights(model, Path(init_checkpoint))
        lr_backbone *= 0.1
        lr_rest *= 0.1
        print(f"Initialized from {loaded}  lr_backbone={lr_backbone}  lr_rest={lr_rest}", flush=True)
        model = model.to(device)

    backbone_params = list(model.backbone.parameters()) + list(model.proj.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    rest = [p for p in model.parameters() if id(p) not in backbone_ids]
    opt = torch.optim.AdamW(
        [
            {
                "params": [p for p in backbone_params if p.requires_grad],
                "lr": lr_backbone,
                "weight_decay": weight_decay,
            },
            {"params": rest, "lr": lr_rest, "weight_decay": weight_decay},
        ]
    )
    scheduler = _build_warmup_cosine_scheduler(opt, total_steps=steps, warmup_steps=warmup_steps)
    kl_weight = float(getattr(cfg, "kl_weight", 10.0))

    metrics_path = output_dir / "metrics.jsonl"
    model.train()
    it = iter(loader)
    t0 = time.time()
    best_val_l1 = float("inf")

    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        images = batch["images"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        mask = batch["action_mask"].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        out = model(
            images=images,
            state=state,
            actions=actions,
            action_mask=mask,
            depth=_batch_depth(batch, device),
        )
        loss, l1, kl = _compute_step_losses(out, actions, mask, kl_weight=kl_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        scheduler.step()

        current_lr = opt.param_groups[-1]["lr"]
        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "l1": float(l1.detach().cpu()),
            "kl": float(kl.detach().cpu()),
            "lr": float(current_lr),
        }

        do_val = val_loader is not None and (step % eval_every == 0 or step == steps)
        if do_val:
            assert val_loader is not None
            val_loss, val_l1 = _validate(model, val_loader, device, kl_weight=kl_weight)
            row["val_l1"] = val_l1
            row["val_loss"] = val_loss
            _save_weights(model, output_dir / "last" / "model.pt")
            if val_l1 < best_val_l1:
                best_val_l1 = val_l1
                _save_weights(model, output_dir / "best_val" / "model.pt")

        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        if step == 1 or step % 50 == 0 or step == steps or do_val:
            dt = time.time() - t0
            extra = f"  val_l1={row['val_l1']:.4f}" if "val_l1" in row else ""
            print(
                f"step {step}/{steps}  loss={row['loss']:.4f}  l1={row['l1']:.4f}  "
                f"kl={row['kl']:.4f}  lr={current_lr:.2e}{extra}  {dt:.1f}s  device={device}",
                flush=True,
            )

    _save_weights(model, output_dir / "last" / "model.pt")
    return output_dir


def train_from_env(
    *,
    dataset_name: str = "so101_bag_clip_bowl",
    steps: int = 40_000,
    batch_size: int = 0,
    output_dir: str = "",
    val_frac: float = 0.2,
    seed: int = 0,
    eval_every: int = 1000,
    backend: str = "torch",
    corrections_name: str = "",
    init_checkpoint: str = "",
) -> Path:
    from so101.config import project_root
    from so101.dataset.combine import combine_datasets

    root = project_root()
    dataset_root = root / "data" / "datasets" / dataset_name
    if not (dataset_root / "meta.json").is_file():
        raise FileNotFoundError(f"No local dataset at {dataset_root}. Record first: python -m so101 record")
    force_train: tuple[int, ...] | None = None
    init_path = init_checkpoint.strip()
    if corrections_name.strip():
        corr_root = root / "data" / "datasets" / corrections_name.strip()
        if not (corr_root / "meta.json").is_file():
            raise FileNotFoundError(
                f"No corrections at {corr_root}. Collect them with: python -m so101 dagger"
            )
        n_corr = SO101Dataset(corr_root).meta.episode_count
        if n_corr < 1:
            raise ValueError(f"Corrections dataset {corr_root} has no episodes")
        combined = root / "data" / "datasets" / f"{dataset_name}__with_dagger"
        merged = combine_datasets(dataset_root, corr_root, combined)
        print(
            f"Combined {merged.n_base} demos + {merged.n_corrections} corrections -> {combined}",
            flush=True,
        )
        dataset_root = merged.root
        force_train = merged.correction_indices
        if not init_path:
            guess = root / "outputs" / "train" / f"act_{dataset_name}"
            try:
                init_path = str(resolve_act_weights(guess))
                print(f"Fine-tune init: {init_path}", flush=True)
            except FileNotFoundError:
                print(
                    f"No baseline checkpoint at {guess}; training the combined set from scratch",
                    flush=True,
                )
        if not output_dir:
            output_dir = str(root / "outputs" / "train" / f"act_{dataset_name}_dagger")
    meta = SO101Dataset(dataset_root).meta
    device = detect_device() if backend == "torch" else "cpu"
    if batch_size <= 0:
        if backend == "jax":
            batch_size = 4
        else:
            batch_size = 8 if device == "cuda" else 4
    out = Path(output_dir) if output_dir else root / "outputs" / "train" / f"act_{dataset_name}"
    cfg = ACTConfig(
        use_depth=meta.use_depth,
        depth_camera_name=meta.depth_camera_name,
        backend=backend,
    )
    print(
        f"ACT train  dataset={dataset_root}  backend={backend}  device={device}  "
        f"batch={batch_size}  steps={steps}  depth={meta.use_depth}"
    )
    print(f"  seed={seed}  val_frac={val_frac}  eval_every={eval_every}")
    print(f"Checkpoints: {out}/last and {out}/best_val (pick best_val for protocol eval)")
    return train(
        dataset_root=dataset_root,
        output_dir=out,
        steps=steps,
        batch_size=batch_size,
        config=cfg,
        device=device,
        val_frac=val_frac,
        seed=seed,
        eval_every=eval_every,
        backend=backend,
        force_train=force_train,
        init_checkpoint=init_path or None,
    )
