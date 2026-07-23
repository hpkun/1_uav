"""Serializable running return normalization."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class ValueNormalizer:
    def __init__(self, epsilon: float = 1.0e-5) -> None:
        self.mean = torch.tensor(0.0, dtype=torch.float64); self.var = torch.tensor(1.0, dtype=torch.float64)
        self.count = torch.tensor(epsilon, dtype=torch.float64); self.epsilon = epsilon

    def update(self, values: Tensor) -> None:
        data = values.detach().double().reshape(-1)
        if data.numel() == 0: return
        batch_mean, batch_var, batch_count = data.mean(), data.var(unbiased=False), data.numel()
        delta = batch_mean - self.mean; total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta.square() * self.count * batch_count / total
        self.var, self.count = m2 / total, total

    def normalize(self, values: Tensor) -> Tensor:
        return (values - self.mean.to(values.device, values.dtype)) / torch.sqrt(self.var.to(values.device, values.dtype) + self.epsilon)

    def denormalize(self, values: Tensor) -> Tensor:
        return values * torch.sqrt(self.var.to(values.device, values.dtype) + self.epsilon) + self.mean.to(values.device, values.dtype)

    def state_dict(self) -> dict[str, Any]: return {"mean": self.mean, "var": self.var, "count": self.count, "epsilon": self.epsilon}
    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean, self.var, self.count = state["mean"].double(), state["var"].double(), state["count"].double()
        self.epsilon = float(state.get("epsilon", 1.0e-5))
