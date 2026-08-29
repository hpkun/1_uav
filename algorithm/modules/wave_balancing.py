"""Inverse-frequency wave weights for policy/value losses."""
from __future__ import annotations

from typing import Any
import numpy as np
import torch

from .base import CapabilityModule


class WaveBalancingModule(CapabilityModule):
    name = "wave_balancing"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.target = str(self.config.get("loss_target", "actor_critic"))
        if self.target not in {"actor_only", "critic_only", "actor_critic"}:
            raise ValueError(f"invalid wave-balance target: {self.target}")
        self.max_weight = float(self.config.get("max_weight", 3.0)); self.epsilon = float(self.config.get("epsilon", 1e-6))

    def compute_numpy(self, wave_indices: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        wave = np.asarray(wave_indices, dtype=np.int64)
        counts=np.asarray([(wave==k).sum() for k in (1,2,3)],dtype=np.float64); total=max(float(counts.sum()),1.0); fractions=counts/total
        if not self.enabled or np.count_nonzero(counts) <= 1:
            weights=np.ones_like(wave,dtype=np.float32); per=np.ones(3,dtype=float)
        else:
            raw=np.zeros(3,dtype=float); present=counts>0; raw[present]=1.0/np.maximum(fractions[present],self.epsilon)
            norm=sum(raw[k-1]*counts[k-1] for k in (1,2,3))/total; per=np.minimum(raw/max(norm,self.epsilon),self.max_weight)
            weights=np.asarray([per[int(k)-1] if 1<=int(k)<=3 else 1.0 for k in wave.reshape(-1)],dtype=np.float32).reshape(wave.shape)
        metrics={**{f"samples_wave_{k}":float(counts[k-1]) for k in (1,2,3)},**{f"fraction_wave_{k}":float(fractions[k-1]) for k in (1,2,3)},**{f"weight_wave_{k}":float(per[k-1] if counts[k-1]>0 else 0.0) for k in (1,2,3)}}
        return weights,metrics

    def compute_tensor(self, wave_indices: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        weights,metrics=self.compute_numpy(wave_indices.detach().cpu().numpy())
        return torch.as_tensor(weights,dtype=torch.float32,device=wave_indices.device),metrics

    @property
    def actor_enabled(self): return self.enabled and self.target in {"actor_only","actor_critic"}
    @property
    def critic_enabled(self): return self.enabled and self.target in {"critic_only","actor_critic"}


__all__=["WaveBalancingModule"]
