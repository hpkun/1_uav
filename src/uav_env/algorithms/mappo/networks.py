"""Orthogonally initialized shared Actor and identity-conditioned Critic."""

from __future__ import annotations

from math import sqrt
from typing import Sequence

import torch
from torch import Tensor, nn


def _activation(name: str) -> type[nn.Module]:
    if name.lower() == "relu": return nn.ReLU
    if name.lower() == "tanh": return nn.Tanh
    raise ValueError("activation must be relu or tanh")


def _mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int, activation: str, output_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
    previous = input_dim
    act = _activation(activation)
    for size in hidden_sizes:
        linear = nn.Linear(previous, int(size)); nn.init.orthogonal_(linear.weight, gain=sqrt(2)); nn.init.zeros_(linear.bias)
        layers.extend([linear, act()]); previous = int(size)
    output = nn.Linear(previous, output_dim); nn.init.orthogonal_(output.weight, gain=output_gain); nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


class SharedActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 15, hidden_sizes: Sequence[int] = (128, 128), activation: str = "relu") -> None:
        super().__init__(); self.net = _mlp(obs_dim, hidden_sizes, action_dim, activation, 0.01)

    def forward(self, observations: Tensor, available_actions: Tensor | None = None) -> Tensor:
        logits = self.net(observations)
        if available_actions is not None:
            mask = available_actions.bool()
            if torch.any(mask.sum(dim=-1) == 0): raise ValueError("Every actor row must have an available action")
            logits = logits.masked_fill(~mask, -1.0e9)
        return logits


class CentralizedCritic(nn.Module):
    def __init__(self, state_dim: int, num_agents: int, hidden_sizes: Sequence[int] = (256, 256), activation: str = "relu") -> None:
        super().__init__(); self.num_agents = num_agents; self.net = _mlp(state_dim + num_agents, hidden_sizes, 1, activation, 1.0)

    def forward(self, global_states: Tensor) -> Tensor:
        flat = global_states.reshape(-1, global_states.shape[-1])
        identities = torch.eye(self.num_agents, device=flat.device, dtype=flat.dtype)
        states = flat[:, None, :].expand(-1, self.num_agents, -1)
        ids = identities[None, :, :].expand(flat.shape[0], -1, -1)
        values = self.net(torch.cat([states, ids], dim=-1)).squeeze(-1)
        return values.reshape(*global_states.shape[:-1], self.num_agents)
