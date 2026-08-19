"""Train ACT on a local SO101Dataset. Mac MPS / CUDA / CPU. No Hub."""

from __future__ import annotations

import json
import random
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from so101.dataset import SO101Dataset, compute_stats, write_stats
from so101.models.act import ACT, ACTConfig, preprocess_image


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


def split_episodes(
    n_episodes: int, val_frac: float = 0.2, seed: int = 0
) -> tuple[list[int], list[int]]:
    if n_episodes < 1:
        raise ValueError("Need at least one episode to split")
    if n_episodes == 1:
        warnings.warn("Only 1 episode; train-only (no val split)", stacklevel=2)
        return [0], []
    rng = np.random.default_rng(seed)
    indices = np.arange(n_episodes)
    rng.shuffle(indices)
    n_val = int(round(n_episodes * val_frac))
    n_val = max(1, n_val)
    n_val = min(n_val, n_episodes - 1)
    val = sorted(int(i) for i in indices[:n_val])
    train = sorted(int(i) for i in indices[n_val:])
    return train, val


class ACTChunkDataset(Dataset):
    def __init__(
        self,
        root: Path,
        cfg: ACTConfig,
        stats: dict,
        episode_indices: list[int] | None = None,
    ) -> None:
        self.ds = SO101Dataset(root)
        self.cfg = cfg
        self.state_mean = np.asarray(stats["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(stats["state_std"], dtype=np.float32)
        self.action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(stats["action_std"], dtype=np.float32)
        self.episodes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.index: list[tuple[int, int]] = []
        if episode_indices is None:
            episode_indices = list(range(self.ds.meta.episode_count))
        for local_i, ep_i in enumerate(episode_indices):
            ep = self.ds.load_episode(ep_i)
            frames = ep.load_frames()
            resized = np.stack(
                [
                    cv2.resize(f, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_AREA)
                    for f in frames
                ]
            )
            self.episodes.append((ep.state, ep.action, resized))
            for t in range(ep.num_frames):
                self.index.append((local_i, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ep_i, t = self.index[i]
        state, action, frames = self.episodes[ep_i]
        t_len = state.shape[0]
        img = preprocess_image(frames[t], self.cfg.image_size, self.cfg.backbone)
        st = (state[t] - self.state_mean) / self.state_std
        chunk = np.zeros((self.cfg.chunk_size, self.cfg.n_joints), dtype=np.float32)
        mask = np.zeros((self.cfg.chunk_size,), dtype=np.float32)
        end = min(t + self.cfg.chunk_size, t_len)
        n = end - t
        chunk[:n] = (action[t:end] - self.action_mean) / self.action_std
        mask[:n] = 1.0
        return {
            "images": img,
            "state": torch.from_numpy(st.astype(np.float32)),
            "actions": torch.from_numpy(chunk),
            "action_mask": torch.from_numpy(mask),
        }


def _save_weights(model: ACT, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def _validate(model: ACT, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    l1s: list[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            state = batch["state"].to(device)
            actions = batch["actions"].to(device)
            mask = batch["action_mask"].to(device)
            out = model(images=images, state=state, actions=actions, action_mask=mask)
            losses.append(float(out["loss"].detach().cpu()))
            l1s.append(float(out["l1"].detach().cpu()))
    model.train()
    return float(np.mean(losses)), float(np.mean(l1s))


def train(
    *,
    dataset_root: Path,
    output_dir: Path,
    steps: int = 40_000,
    batch_size: int = 2,
    config: ACTConfig | None = None,
    device: str | None = None,
    lr_backbone: float = 1e-5,
    lr_rest: float = 1e-4,
    val_frac: float = 0.2,
    seed: int = 0,
    eval_every: int = 1000,
    grad_clip: float = 1.0,
) -> Path:
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or ACTConfig()
    device = device or detect_device()
    set_seed(seed)

    ds = SO101Dataset(dataset_root)
    train_episodes, val_episodes = split_episodes(ds.meta.episode_count, val_frac=val_frac, seed=seed)
    stats = compute_stats(ds, episode_indices=train_episodes)
    write_stats(output_dir, stats)
    (output_dir / "split.json").write_text(
        json.dumps(
            {"seed": seed, "train_episodes": train_episodes, "val_episodes": val_episodes},
            indent=2,
        )
        + "\n"
    )

    data = ACTChunkDataset(dataset_root, cfg, stats, episode_indices=train_episodes)
    if len(data) < 1:
        raise ValueError(f"No frames in train split of {dataset_root}")
    bs = min(batch_size, len(data))
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        data, batch_size=bs, shuffle=True, num_workers=0, drop_last=len(data) > bs, generator=g
    )
    val_loader: DataLoader | None = None
    if val_episodes:
        val_data = ACTChunkDataset(dataset_root, cfg, stats, episode_indices=val_episodes)
        if len(val_data) >= 1:
            val_bs = min(batch_size, len(val_data))
            val_loader = DataLoader(val_data, batch_size=val_bs, shuffle=False, num_workers=0)

    run_config = {
        **cfg.to_dict(),
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "val_frac": val_frac,
        "eval_every": eval_every,
        "task": ds.meta.task,
        "lr_backbone": lr_backbone,
        "lr_rest": lr_rest,
        "grad_clip": grad_clip,
    }
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    model = ACT(cfg).to(device)
    backbone_params = list(model.backbone.parameters()) + list(model.proj.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    rest = [p for p in model.parameters() if id(p) not in backbone_ids]
    opt = torch.optim.AdamW(
        [
            {"params": [p for p in backbone_params if p.requires_grad], "lr": lr_backbone},
            {"params": rest, "lr": lr_rest},
        ]
    )
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
        images = batch["images"].to(device)
        state = batch["state"].to(device)
        actions = batch["actions"].to(device)
        mask = batch["action_mask"].to(device)
        opt.zero_grad(set_to_none=True)
        out = model(images=images, state=state, actions=actions, action_mask=mask)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        row = {
            "step": step,
            "loss": float(out["loss"].detach().cpu()),
            "l1": float(out["l1"].detach().cpu()),
            "kl": float(out["kl"].detach().cpu()),
        }
        do_val = val_loader is not None and (step % eval_every == 0 or step == steps)
        if do_val:
            assert val_loader is not None
            val_loss, val_l1 = _validate(model, val_loader, device)
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
                f"kl={row['kl']:.4f}{extra}  {dt:.1f}s  device={device}",
                flush=True,
            )
    _save_weights(model, output_dir / "last" / "model.pt")
    return output_dir


def train_from_env(
    *,
    dataset_name: str = "so101_d405_demo",
    steps: int = 40_000,
    batch_size: int = 0,
    output_dir: str = "",
    val_frac: float = 0.2,
    seed: int = 0,
    eval_every: int = 1000,
) -> Path:
    from so101.config import project_root

    root = project_root()
    dataset_root = root / "data" / "datasets" / dataset_name
    if not (dataset_root / "meta.json").is_file():
        raise FileNotFoundError(f"No local dataset at {dataset_root}. Record first: python -m so101 record")
    device = detect_device()
    if batch_size <= 0:
        batch_size = 8 if device == "cuda" else 2
    out = Path(output_dir) if output_dir else root / "outputs" / "train" / f"act_{dataset_name}"
    print(f"ACT train  dataset={dataset_root}  device={device}  batch={batch_size}  steps={steps}")
    print(f"  seed={seed}  val_frac={val_frac}  eval_every={eval_every}")
    print(f"Checkpoints: {out}/last and {out}/best_val (pick best_val for protocol eval)")
    return train(
        dataset_root=dataset_root,
        output_dir=out,
        steps=steps,
        batch_size=batch_size,
        config=ACTConfig(),
        device=device,
        val_frac=val_frac,
        seed=seed,
        eval_every=eval_every,
    )
