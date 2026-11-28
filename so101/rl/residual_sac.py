"""Small residual Soft Actor-Critic (joint-space deltas on top of a base policy)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ResidualSACConfig:
    obs_dim: int = 9
    act_dim: int = 6
    hidden: int = 128
    max_delta_deg: float = 4.0
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    alpha: float = 0.05
    batch_size: int = 64
    buffer_size: int = 50_000
    update_every: int = 1
    updates_per_step: int = 1
    start_learning_after: int = 256
    device: str = "cpu"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResidualSACConfig:
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in fields})


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int, max_delta: float) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), -5.0, 2.0)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z) * self.max_delta
        # tanh correction
        log_prob = dist.log_prob(z) - torch.log(1.0 - torch.tanh(z).pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _ = self.forward(obs)
        return torch.tanh(mu) * self.max_delta


class _Replay:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.i = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.i] = obs
        self.act[self.i] = act
        self.rew[self.i, 0] = rew
        self.next_obs[self.i] = next_obs
        self.done[self.i, 0] = float(done)
        self.i = (self.i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int) -> tuple[np.ndarray, ...]:
        idx = np.random.randint(0, self.size, size=batch)
        return (
            self.obs[idx],
            self.act[idx],
            self.rew[idx],
            self.next_obs[idx],
            self.done[idx],
        )


class ResidualSAC:
    """Learn joint residuals δa; final command = clip(base + δa)."""

    def __init__(self, cfg: ResidualSACConfig | None = None) -> None:
        self.cfg = cfg or ResidualSACConfig()
        device = torch.device(self.cfg.device)
        self.device = device
        o, a, h = self.cfg.obs_dim, self.cfg.act_dim, self.cfg.hidden
        self.actor = _GaussianActor(o, a, h, self.cfg.max_delta_deg).to(device)
        self.q1 = _MLP(o + a, 1, h).to(device)
        self.q2 = _MLP(o + a, 1, h).to(device)
        self.q1_t = _MLP(o + a, 1, h).to(device)
        self.q2_t = _MLP(o + a, 1, h).to(device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.opt_q = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.cfg.lr
        )
        self.buffer = _Replay(self.cfg.buffer_size, o, a)
        self._steps = 0

    def act(self, obs: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                a = self.actor.mean_action(o)
            else:
                a, _ = self.actor.sample(o)
        return a.squeeze(0).cpu().numpy().astype(np.float32)

    def push(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.add(obs, act, rew, next_obs, done)
        self._steps += 1

    def update(self) -> dict[str, float]:
        if self.buffer.size < max(self.cfg.start_learning_after, self.cfg.batch_size):
            return {}
        stats: dict[str, float] = {}
        for _ in range(int(self.cfg.updates_per_step)):
            obs, act, rew, next_obs, done = self.buffer.sample(self.cfg.batch_size)
            obs_t = torch.as_tensor(obs, device=self.device)
            act_t = torch.as_tensor(act, device=self.device)
            rew_t = torch.as_tensor(rew, device=self.device)
            next_t = torch.as_tensor(next_obs, device=self.device)
            done_t = torch.as_tensor(done, device=self.device)

            with torch.no_grad():
                next_a, next_logp = self.actor.sample(next_t)
                q1_n = self.q1_t(torch.cat([next_t, next_a], dim=-1))
                q2_n = self.q2_t(torch.cat([next_t, next_a], dim=-1))
                q_n = torch.min(q1_n, q2_n) - self.cfg.alpha * next_logp
                target = rew_t + (1.0 - done_t) * self.cfg.gamma * q_n

            q1 = self.q1(torch.cat([obs_t, act_t], dim=-1))
            q2 = self.q2(torch.cat([obs_t, act_t], dim=-1))
            q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            self.opt_q.zero_grad()
            q_loss.backward()
            self.opt_q.step()

            new_a, logp = self.actor.sample(obs_t)
            q1_pi = self.q1(torch.cat([obs_t, new_a], dim=-1))
            q2_pi = self.q2(torch.cat([obs_t, new_a], dim=-1))
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.cfg.alpha * logp - q_pi).mean()
            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()

            with torch.no_grad():
                for p, pt in zip(self.q1.parameters(), self.q1_t.parameters(), strict=True):
                    pt.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)
                for p, pt in zip(self.q2.parameters(), self.q2_t.parameters(), strict=True):
                    pt.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

            stats = {
                "q_loss": float(q_loss.item()),
                "actor_loss": float(actor_loss.item()),
                "buffer": float(self.buffer.size),
            }
        return stats

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "cfg": asdict(self.cfg),
            },
            path / "residual_sac.pt",
        )
        (path / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path | str, device: str | None = None) -> ResidualSAC:
        path = Path(path)
        ckpt_file = path / "residual_sac.pt" if path.is_dir() else path
        blob = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        cfg = ResidualSACConfig.from_dict(blob.get("cfg") or {})
        if device:
            cfg.device = device
        agent = cls(cfg)
        agent.actor.load_state_dict(blob["actor"])
        agent.q1.load_state_dict(blob["q1"])
        agent.q2.load_state_dict(blob["q2"])
        agent.q1_t.load_state_dict(blob["q1"])
        agent.q2_t.load_state_dict(blob["q2"])
        return agent
