"""Residual RL helpers: dense depth rewards + SAC residual on top of ACT."""

from so101.rl.dense_reward import DenseDepthReward, DenseDepthRewardConfig, RewardStep
from so101.rl.phase_mix_policy import PhaseMixPolicy
from so101.rl.residual_sac import ResidualSAC, ResidualSACConfig

__all__ = [
    "DenseDepthReward",
    "DenseDepthRewardConfig",
    "PhaseMixPolicy",
    "RewardStep",
    "ResidualSAC",
    "ResidualSACConfig",
]
