"""ACT: ResNet (or tiny) + CVAE transformer, action chunks. Zhao et al. 2023.

Defaults are aligned with Hugging Face LeRobot ACT (1 decoder layer, post-norm,
FrozenBatchNorm, lr 1e-5 + weight decay, temporal ensembling).
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ACTConfig:
    n_joints: int = 6
    chunk_size: int = 100
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    # LeRobot / original ACT effective depth (paper lists 7 but only layer 0 ran).
    n_decoder_layers: int = 1
    n_vae_encoder_layers: int = 4
    # LeRobot default: post-norm (pre_norm=False).
    pre_norm: bool = False
    dropout: float = 0.1
    latent_dim: int = 32
    kl_weight: float = 10.0
    backbone: str = "resnet18"
    image_size: int = 224
    # Temporal ensembling coeff (Zhao et al.). None disables and uses action queue.
    ensemble_temp: float | None = 0.01
    n_action_steps: int = 100
    use_depth: bool = False
    depth_camera_name: str = "wrist"
    depth_max_mm: float = 2000.0
    backend: str = "torch"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ACTConfig:
        """Load config; fill fields added in the LeRobot-alignment for older checkpoints."""
        fields = cls.__dataclass_fields__
        filtered = {k: v for k, v in data.items() if k in fields}
        # Pre-alignment checkpoints used n_encoder//2 VAE layers and pre-norm.
        if "n_vae_encoder_layers" not in data:
            n_enc = int(filtered.get("n_encoder_layers", cls.n_encoder_layers))
            filtered["n_vae_encoder_layers"] = max(1, n_enc // 2)
        if "pre_norm" not in data:
            filtered["pre_norm"] = True
        return cls(**filtered)

    def uses_temporal_ensemble(self) -> bool:
        return self.ensemble_temp is not None


class ACTTemporalEnsembler:
    """Online exponential temporal ensembling (LeRobot ACTTemporalEnsembler / ACT Algo 2)."""

    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = int(chunk_size)
        self.ensemble_weights = np.exp(
            -float(temporal_ensemble_coeff) * np.arange(self.chunk_size, dtype=np.float64)
        )
        self.ensemble_weights_cumsum = np.cumsum(self.ensemble_weights)
        self.reset()

    def reset(self) -> None:
        self.ensembled_actions: np.ndarray | None = None
        self.ensembled_actions_count: np.ndarray | None = None

    def update(self, actions: np.ndarray) -> np.ndarray:
        """Update ensemble with a (chunk_size, action_dim) chunk; return next action."""
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2:
            raise ValueError(f"expected (chunk, dim) actions, got shape {actions.shape}")
        if actions.shape[0] != self.chunk_size:
            raise ValueError(
                f"chunk length {actions.shape[0]} != ensemble chunk_size {self.chunk_size}"
            )
        actions_b = actions[None, ...]
        if self.ensembled_actions is None:
            self.ensembled_actions = actions_b.copy()
            self.ensembled_actions_count = np.ones((self.chunk_size, 1), dtype=np.int64)
        else:
            assert self.ensembled_actions_count is not None
            count = self.ensembled_actions_count
            w_prev = self.ensemble_weights_cumsum[count - 1]
            w_new = self.ensemble_weights[count]
            w_all = self.ensemble_weights_cumsum[count]
            self.ensembled_actions = (
                self.ensembled_actions * w_prev + actions_b[:, :-1] * w_new
            ) / w_all
            self.ensembled_actions_count = np.minimum(count + 1, self.chunk_size)
            self.ensembled_actions = np.concatenate(
                [self.ensembled_actions, actions_b[:, -1:]], axis=1
            )
            self.ensembled_actions_count = np.concatenate(
                [self.ensembled_actions_count, np.ones_like(self.ensembled_actions_count[-1:])]
            )
        assert self.ensembled_actions is not None and self.ensembled_actions_count is not None
        action = self.ensembled_actions[:, 0]
        self.ensembled_actions = self.ensembled_actions[:, 1:]
        self.ensembled_actions_count = self.ensembled_actions_count[1:]
        return np.asarray(action[0], dtype=np.float32)


class TinyBackbone(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_dim, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _resnet18_trunk() -> nn.Module:
    import torchvision
    from torchvision.ops.misc import FrozenBatchNorm2d

    try:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        net = torchvision.models.resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
    except Exception:
        net = torchvision.models.resnet18(weights=None, norm_layer=FrozenBatchNorm2d)
    return nn.Sequential(*list(net.children())[:-2])


class ACT(nn.Module):
    def __init__(self, cfg: ACTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.backbone == "tiny":
            self.backbone = TinyBackbone(cfg.dim_model)
            self.proj = nn.Identity()
            backbone_dim = cfg.dim_model
        else:
            self.backbone = _resnet18_trunk()
            backbone_dim = 512
            self.proj = nn.Conv2d(backbone_dim, cfg.dim_model, kernel_size=1)
        self.state_proj = nn.Linear(cfg.n_joints, cfg.dim_model)
        self.action_proj = nn.Linear(cfg.n_joints, cfg.dim_model)
        self.z_proj = nn.Linear(cfg.latent_dim, cfg.dim_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.dim_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="relu",
            norm_first=cfg.pre_norm,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=cfg.n_encoder_layers, enable_nested_tensor=False
        )
        self.latent_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.dim_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                batch_first=True,
                activation="relu",
                norm_first=cfg.pre_norm,
            ),
            num_layers=max(1, int(cfg.n_vae_encoder_layers)),
            enable_nested_tensor=False,
        )
        self.mu = nn.Linear(cfg.dim_model, cfg.latent_dim)
        self.logvar = nn.Linear(cfg.dim_model, cfg.latent_dim)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.dim_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="relu",
            norm_first=cfg.pre_norm,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.n_decoder_layers)
        self.action_head = nn.Linear(cfg.dim_model, cfg.n_joints)
        # Long enough for multi-cam ResNet maps; resized on the fly if needed.
        self.enc_pos = nn.Parameter(torch.randn(1, 512, cfg.dim_model) * 0.02)
        self.latent_pos = nn.Parameter(torch.randn(1, 1 + cfg.chunk_size, cfg.dim_model) * 0.02)
        self.dec_queries = nn.Parameter(torch.randn(1, cfg.chunk_size, cfg.dim_model) * 0.02)
        if cfg.use_depth:
            # Pool to the same spatial grid as the RGB backbone tokens (e.g. 7x7 for ResNet18@224).
            self.depth_encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, cfg.dim_model, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
        else:
            self.depth_encoder = None
        self._reset_transformer_parameters()

    def _reset_transformer_parameters(self) -> None:
        for module in (self.encoder, self.decoder, self.latent_encoder):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def encode_images(self, images: torch.Tensor, depth: torch.Tensor | None = None) -> torch.Tensor:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        b, n, _c, _h, _w = images.shape
        feat = self.backbone(images.reshape(b * n, images.shape[2], images.shape[3], images.shape[4]))
        feat = self.proj(feat)
        tokens = feat.flatten(2).transpose(1, 2)
        _bn, hw, d = tokens.shape
        img_tokens = tokens.reshape(b, n * hw, d)
        if depth is not None:
            if self.depth_encoder is None:
                raise ValueError("depth provided but ACTConfig.use_depth is false")
            import torch.nn.functional as F

            d_feat = self.depth_encoder(depth.unsqueeze(1))
            # Match one camera's spatial size so depth doesn't explode the sequence length.
            cam_h, cam_w = feat.shape[-2], feat.shape[-1]
            d_feat = F.adaptive_avg_pool2d(d_feat, (cam_h, cam_w))
            d_tokens = d_feat.flatten(2).transpose(1, 2)
            img_tokens = torch.cat([img_tokens, d_tokens], dim=1)
        return img_tokens

    @staticmethod
    def _add_pos(tokens: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        n = tokens.size(1)
        if n <= pos.size(1):
            return tokens + pos[:, :n]
        import torch.nn.functional as F

        # (1, L, D) -> (1, D, L) interpolate along length, then back.
        grown = F.interpolate(pos.transpose(1, 2), size=n, mode="linear", align_corners=False)
        return tokens + grown.transpose(1, 2)

    def encode_latent(self, state: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state_tok = self.state_proj(state).unsqueeze(1)
        act_tok = self.action_proj(actions)
        tokens = torch.cat([state_tok, act_tok], dim=1)
        tokens = self._add_pos(tokens, self.latent_pos)
        hid = self.latent_encoder(tokens)
        cls = hid[:, 0]
        return self.mu(cls), self.logvar(cls).clamp(-10, 10)

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b = state.shape[0]
        img_tokens = self.encode_images(images, depth=depth)
        state_tok = self.state_proj(state).unsqueeze(1)
        if actions is not None:
            mu, logvar = self.encode_latent(state, actions)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        else:
            z = torch.zeros(b, self.cfg.latent_dim, device=state.device, dtype=state.dtype)
            kl = state.new_zeros(())
        z_tok = self.z_proj(z).unsqueeze(1)
        enc = torch.cat([z_tok, state_tok, img_tokens], dim=1)
        enc = self._add_pos(enc, self.enc_pos)
        memory = self.encoder(enc)
        queries = self.dec_queries.expand(b, -1, -1)
        dec = self.decoder(queries, memory)
        pred = self.action_head(dec)
        if actions is not None:
            if action_mask is None:
                action_mask = torch.ones(b, self.cfg.chunk_size, device=state.device, dtype=pred.dtype)
            mask = action_mask.unsqueeze(-1)
            l1 = ((pred - actions).abs() * mask).sum() / mask.sum().clamp_min(1.0)
            loss = l1 + self.cfg.kl_weight * kl
        else:
            l1 = state.new_zeros(())
            loss = l1
        return {"pred_actions": pred, "loss": loss, "l1": l1, "kl": kl}


def preprocess_depth(depth_u16: np.ndarray, image_size: int, depth_max_mm: float) -> torch.Tensor:
    import cv2

    z = np.asarray(depth_u16, dtype=np.float32)
    if z.shape[0] != image_size or z.shape[1] != image_size:
        z = cv2.resize(z, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    z[z <= 0] = 0.0
    z = np.clip(z, 0.0, depth_max_mm) / max(depth_max_mm, 1.0)
    return torch.from_numpy(z.astype(np.float32))


def preprocess_image(image_rgb: np.ndarray, image_size: int, backbone: str) -> torch.Tensor:
    import cv2

    if image_rgb.shape[0] != image_size or image_rgb.shape[1] != image_size:
        image_rgb = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    x = image_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    t = torch.from_numpy(x)
    if backbone != "tiny":
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        t = (t - mean) / std
    return t


class ACTPolicy:
    def __init__(self, model: ACT, stats: dict[str, Any], device: str = "cpu") -> None:
        self.model = model
        self.cfg = model.cfg
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.state_mean = np.asarray(stats["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(stats["state_std"], dtype=np.float32)
        self.action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(stats["action_std"], dtype=np.float32)
        self._action_queue: deque[np.ndarray] = deque()
        self._ensembler: ACTTemporalEnsembler | None = None
        if self.cfg.uses_temporal_ensemble():
            self._ensembler = ACTTemporalEnsembler(float(self.cfg.ensemble_temp), self.cfg.chunk_size)

    def reset(self) -> None:
        self._action_queue = deque()
        if self._ensembler is not None:
            self._ensembler.reset()

    @torch.no_grad()
    def _predict_chunk(
        self,
        image_rgb: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
        state: np.ndarray,
        depth_u16: np.ndarray | None = None,
    ) -> np.ndarray:
        # Decorated for AsyncReplanPolicy, which calls this outside select_action.
        if isinstance(image_rgb, (list, tuple)):
            imgs = [preprocess_image(im, self.cfg.image_size, self.cfg.backbone) for im in image_rgb]
            img = torch.stack(imgs, dim=0).unsqueeze(0).to(self.device)
        else:
            img = preprocess_image(image_rgb, self.cfg.image_size, self.cfg.backbone).unsqueeze(0).to(self.device)
        st = (np.asarray(state, dtype=np.float32) - self.state_mean) / self.state_std
        st_t = torch.from_numpy(st).unsqueeze(0).to(self.device)
        depth_t = None
        if self.cfg.use_depth:
            if depth_u16 is None:
                raise ValueError("depth_u16 required when ACTConfig.use_depth is true")
            depth_t = preprocess_depth(depth_u16, self.cfg.image_size, self.cfg.depth_max_mm).unsqueeze(0).to(
                self.device
            )
        pred = self.model(images=img, state=st_t, depth=depth_t)["pred_actions"][0].detach().cpu().numpy()
        return pred * self.action_std + self.action_mean

    @torch.no_grad()
    def select_action(
        self,
        image_rgb: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
        state: np.ndarray,
        depth_u16: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._ensembler is not None:
            # Temporal ensembling requires a fresh chunk every env step (LeRobot).
            chunk = self._predict_chunk(image_rgb, state, depth_u16=depth_u16)
            return self._ensembler.update(chunk)
        # Queue mode: fill n_action_steps, then pop. Do not replan every frame.
        if not self._action_queue:
            chunk = self._predict_chunk(image_rgb, state, depth_u16=depth_u16)
            n = min(int(self.cfg.n_action_steps), int(self.cfg.chunk_size), chunk.shape[0])
            self._action_queue.extend(np.asarray(row, dtype=np.float32) for row in chunk[:n])
        return np.asarray(self._action_queue.popleft(), dtype=np.float32)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps(self.cfg.to_dict(), indent=2) + "\n")
        stats = {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }
        (directory / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        torch.save(self.model.state_dict(), directory / "model.pt")

    @classmethod
    def load(cls, directory: Path, device: str = "cpu") -> ACTPolicy:
        directory = Path(directory)
        cfg_dir = directory if (directory / "config.json").is_file() else directory.parent
        cfg = ACTConfig.from_dict(json.loads((cfg_dir / "config.json").read_text()))
        if cfg.backend == "jax":
            from so101.models.act_jax import ACTPolicyJAX

            return ACTPolicyJAX.load(directory, device=device)
        weights = find_policy_weights(directory)
        stats = json.loads((cfg_dir / "stats.json").read_text())
        model = ACT(cfg)
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        return cls(model, stats, device=device)


def find_policy_weights(directory: Path) -> Path:
    """Prefer best_val, then last, then legacy run_dir/model.pt (or model.msgpack for JAX)."""
    directory = Path(directory)
    for name in ("model.pt", "model.msgpack"):
        for sub in ("best_val", "last", ""):
            path = directory / sub / name if sub else directory / name
            if path.is_file():
                return path
    raise FileNotFoundError(f"No ACT weights under {directory}")
