"""Shared two-layer stochastic actor from Section 4.1."""
from __future__ import annotations
import torch
from torch import nn
from torch.distributions import Normal


class SharedSquashedGaussianActor(nn.Module):
    def __init__(self, observation_dim: int = 54, action_dim: int = 3, hidden_dim: int = 256, log_std_min: float = -5.0, log_std_max: float = 2.0, activation: str = "relu") -> None:
        super().__init__()
        self.observation_dim, self.action_dim = observation_dim, action_dim
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        activation_cls = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}.get(activation)
        if activation_cls is None: raise ValueError(f"unsupported actor activation: {activation}")
        self.backbone = nn.Sequential(nn.Linear(observation_dim, hidden_dim), activation_cls(), nn.Linear(hidden_dim, hidden_dim), activation_cls())
        self.mean, self.log_std = nn.Linear(hidden_dim, action_dim), nn.Linear(hidden_dim, action_dim)

    def distribution(self, observations: torch.Tensor) -> Normal:
        h = self.backbone(observations)
        return Normal(self.mean(h), self.log_std(h).clamp(self.log_std_min, self.log_std_max).exp())

    def sample(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(observations)
        raw = dist.rsample(); action = torch.tanh(raw)
        log_prob = (dist.log_prob(raw) - torch.log(1.0 - action.square() + 1e-6)).sum(dim=-1)
        return action, log_prob

    def deterministic(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.distribution(observations).mean)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.deterministic(observations)
