"""Finite-logit masked categorical helpers."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Categorical


def masked_categorical(logits: Tensor, available_actions: Tensor) -> Categorical:
    mask = available_actions.bool()
    if torch.any(mask.sum(-1) == 0): raise ValueError("All actions are masked")
    return Categorical(logits=logits.masked_fill(~mask, -1.0e9))


def sample_actions(logits: Tensor, available_actions: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor]:
    distribution = masked_categorical(logits, available_actions)
    actions = torch.argmax(distribution.logits, dim=-1) if deterministic else distribution.sample()
    return actions, distribution.log_prob(actions), distribution.entropy()
