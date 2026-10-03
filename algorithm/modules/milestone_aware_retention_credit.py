"""MARC-MAPPO V1: milestone credit, tempered balance, and retention.

MARC differs from IWSC (no quality critic or replay target), BRSC (no outcome
classifier or heuristic redistribution), HTA (no manager/options), policy
anchor (no frozen global policy), and the legacy wave-balancing module (MARC
uses an actor-only tempered, capped, mean-preserving weight inside one joint
objective).  The deployed actor remains the unchanged 52D feed-forward policy.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from .base import CapabilityModule


MARC_MAPPO_VERSION = 1


def compute_local_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    alive_masks: torch.Tensor,
    next_alive_masks: torch.Tensor,
    wave_transition_flags: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE with normal TD bootstrap but no lambda trace across a wave boundary."""
    if rewards.shape != values.shape or rewards.shape != next_values.shape:
        raise ValueError("MARC rewards and values must share [time, env, agent]")
    if alive_masks.shape != rewards.shape or next_alive_masks.shape != rewards.shape:
        raise ValueError("MARC alive masks must match rewards")
    if dones.shape != rewards.shape[:2] or wave_transition_flags.shape != dones.shape:
        raise ValueError("MARC dones/transition flags must be [time, env]")
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        continuation = (1.0 - dones[step].unsqueeze(-1)) * next_alive_masks[step]
        delta = rewards[step] + gamma * continuation * next_values[step] - values[step]
        local_trace = continuation * (1.0 - wave_transition_flags[step].unsqueeze(-1))
        gae = delta + gamma * gae_lambda * local_trace * gae
        advantages[step] = gae * alive_masks[step]
    returns = (advantages + values) * alive_masks
    return advantages, returns


def continuation_coefficients(
    wave_indices: torch.Tensor, total_waves: int = 3, alpha: float = 1.0
) -> torch.Tensor:
    wave = wave_indices.to(dtype=torch.float32)
    if total_waves < 1 or alpha < 0:
        raise ValueError("invalid MARC continuation coefficient parameters")
    if torch.any((wave < 1) | (wave > total_waves)):
        raise ValueError("MARC wave index outside configured range")
    return 1.0 / (1.0 + float(alpha) * (float(total_waves) - wave))


def tempered_wave_weights(
    wave_indices: torch.Tensor,
    alive_masks: torch.Tensor,
    *,
    max_waves: int = 3,
    temperature: float = 0.5,
    weight_min: float = 0.5,
    weight_max: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Actor-only alive-frequency weights with hard bounds and E_alive[w]=1."""
    if alive_masks.shape[:-1] != wave_indices.shape:
        raise ValueError("MARC alive mask must be wave shape + [agents]")
    if not (0.0 <= temperature <= 1.0 and 0 < weight_min <= 1 <= weight_max):
        raise ValueError("invalid MARC wave weight parameters")
    counts = torch.stack([
        (alive_masks * (wave_indices == wave).to(alive_masks.dtype).unsqueeze(-1)).sum()
        for wave in range(1, max_waves + 1)
    ]).to(dtype=torch.float64)
    total = counts.sum()
    present = counts > 0
    present_count = int(present.sum().item())
    per_wave = torch.zeros(max_waves, dtype=torch.float64, device=wave_indices.device)
    if total <= 0:
        weights = torch.ones_like(wave_indices, dtype=torch.float32)
    elif present_count <= 1:
        per_wave[present] = 1.0
        weights = torch.ones_like(wave_indices, dtype=torch.float32)
    else:
        raw = torch.zeros_like(per_wave)
        raw[present] = (total / (float(present_count) * counts[present])).pow(temperature)

        def weighted_mean(scale: float) -> float:
            clipped = (raw * scale).clamp(weight_min, weight_max)
            return float((counts * clipped).sum().item() / total.item())

        low, high = 0.0, 1.0
        while weighted_mean(high) < 1.0:
            high *= 2.0
        for _ in range(80):
            middle = (low + high) / 2.0
            if weighted_mean(middle) < 1.0:
                low = middle
            else:
                high = middle
        per_wave[present] = (raw[present] * high).clamp(weight_min, weight_max)
        weights = torch.ones_like(wave_indices, dtype=torch.float32)
        for wave in range(1, max_waves + 1):
            if bool(present[wave - 1]):
                weights = torch.where(
                    wave_indices == wave,
                    per_wave[wave - 1].to(dtype=torch.float32),
                    weights,
                )
    fractions = counts / total.clamp_min(1.0)
    effective = float((counts * per_wave).sum().item() / total.item()) if total > 0 else 1.0
    metrics: dict[str, float] = {}
    for wave in range(1, max_waves + 1):
        metrics[f"marc_samples_wave{wave}"] = float(counts[wave - 1])
        metrics[f"marc_fraction_wave{wave}"] = float(fractions[wave - 1])
        metrics[f"marc_weight_wave{wave}"] = (
            float(per_wave[wave - 1]) if bool(present[wave - 1]) else 0.0
        )
    metrics["marc_effective_weight_mean"] = effective
    return weights, metrics


def successful_wave_from_transition(
    source_wave: int, *, spawned_next_wave: bool, episode_done: bool, red_success: bool
) -> int | None:
    wave = int(source_wave)
    if wave in (1, 2) and bool(spawned_next_wave) and not bool(episode_done):
        return wave
    if wave == 3 and bool(episode_done) and bool(red_success):
        return 3
    return None


class MilestoneAwareRetentionCreditModule(CapabilityModule):
    name = "milestone_aware_retention_credit"

    def __init__(self, config: dict[str, Any] | None = None, seed: int = 0) -> None:
        super().__init__(config)
        self.version = MARC_MAPPO_VERSION
        self.max_waves = int(self.config.get("max_waves", 3))
        self.continuation_alpha = float(self.config.get("continuation_alpha", 1.0))
        self.wave_balance_temperature = float(self.config.get("wave_balance_temperature", 0.5))
        self.wave_weight_min = float(self.config.get("wave_weight_min", 0.5))
        self.wave_weight_max = float(self.config.get("wave_weight_max", 2.0))
        self.retention_coefficient = float(self.config.get("retention_coefficient", 0.01))
        self.retention_bank_size_per_wave = int(self.config.get("retention_bank_size_per_wave", 512))
        self.retention_samples_per_wave = int(self.config.get("retention_samples_per_wave", 32))
        self.retention_min_samples_per_wave = int(self.config.get("retention_min_samples_per_wave", 32))
        self.retention_stride = int(self.config.get("retention_stride", 8))
        if self.max_waves != 3:
            raise ValueError("MARC V1 requires max_waves=3")
        if self.continuation_alpha < 0:
            raise ValueError("MARC continuation_alpha must be nonnegative")
        if not (0 <= self.wave_balance_temperature <= 1):
            raise ValueError("MARC wave_balance_temperature must be in [0,1]")
        if not (0 < self.wave_weight_min <= 1 <= self.wave_weight_max):
            raise ValueError("MARC wave weights must straddle one")
        if self.retention_coefficient < 0:
            raise ValueError("MARC retention coefficient must be nonnegative")
        if min(
            self.retention_bank_size_per_wave,
            self.retention_samples_per_wave,
            self.retention_min_samples_per_wave,
            self.retention_stride,
        ) <= 0:
            raise ValueError("MARC retention sizes and stride must be positive")
        if self.retention_samples_per_wave > self.retention_bank_size_per_wave:
            raise ValueError("MARC sample count exceeds bank capacity")
        self.banks = {
            wave: deque(maxlen=self.retention_bank_size_per_wave)
            for wave in range(1, self.max_waves + 1)
        }
        self.ingested_success_count = {wave: 0 for wave in self.banks}
        self.rng = np.random.default_rng(int(seed) ^ 0x4D415243)

    def coefficients(self, wave_indices: torch.Tensor) -> torch.Tensor:
        return continuation_coefficients(
            wave_indices, self.max_waves, self.continuation_alpha
        )

    def wave_weights(
        self, wave_indices: torch.Tensor, alive_masks: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return tempered_wave_weights(
            wave_indices,
            alive_masks,
            max_waves=self.max_waves,
            temperature=self.wave_balance_temperature,
            weight_min=self.wave_weight_min,
            weight_max=self.wave_weight_max,
        )

    @torch.no_grad()
    def ingest_success_segments(self, segments, actor, device: torch.device) -> None:
        if not self.enabled:
            return
        for segment in segments or []:
            wave = int(segment["wave"])
            if wave not in self.banks:
                raise ValueError(f"invalid MARC success wave: {wave}")
            observations = np.asarray(segment["observations"], dtype=np.float32)
            if observations.ndim != 2 or observations.shape[1] != 52:
                raise ValueError("MARC retention observations must be [N,52]")
            if not len(observations):
                continue
            tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
            distribution = actor.distribution(tensor)
            means = distribution.loc.detach().cpu().numpy().astype(np.float16)
            log_stds = distribution.scale.log().detach().cpu().numpy().astype(np.float16)
            for observation, mean, log_std in zip(observations, means, log_stds):
                self.banks[wave].append(
                    {
                        "observation": observation.astype(np.float16),
                        "reference_mean": mean,
                        "reference_log_std": log_std,
                    }
                )
            self.ingested_success_count[wave] += 1

    def ready_waves(self) -> list[int]:
        return [
            wave
            for wave, bank in self.banks.items()
            if len(bank) >= self.retention_min_samples_per_wave
        ]

    def sample_balanced(self) -> dict[int, list[dict[str, np.ndarray]]]:
        sampled: dict[int, list[dict[str, np.ndarray]]] = {}
        for wave in self.ready_waves():
            bank = self.banks[wave]
            indices = self.rng.choice(
                len(bank), self.retention_samples_per_wave, replace=False
            )
            sampled[wave] = [bank[int(index)] for index in indices]
        return sampled

    def retention_loss(self, actor, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
        zero = next(actor.parameters()).sum() * 0.0
        metrics = {
            **{f"marc_bank_size_wave{wave}": float(len(self.banks[wave])) for wave in self.banks},
            **{f"marc_ready_wave{wave}": float(wave in self.ready_waves()) for wave in self.banks},
            **{f"marc_retention_kl_wave{wave}": 0.0 for wave in self.banks},
            "marc_retention_loss": 0.0,
        }
        sampled = self.sample_balanced()
        if not sampled:
            return zero, metrics
        wave_losses = []
        for wave, rows in sampled.items():
            observations = torch.as_tensor(
                np.stack([row["observation"] for row in rows]),
                dtype=torch.float32,
                device=device,
            )
            reference_mean = torch.as_tensor(
                np.stack([row["reference_mean"] for row in rows]),
                dtype=torch.float32,
                device=device,
            ).detach()
            reference_log_std = torch.as_tensor(
                np.stack([row["reference_log_std"] for row in rows]),
                dtype=torch.float32,
                device=device,
            ).detach()
            current = actor.distribution(observations)
            current_mean = current.loc
            current_log_std = current.scale.log()
            reference_variance = torch.exp(2.0 * reference_log_std)
            current_variance = torch.exp(2.0 * current_log_std)
            kl = (
                current_log_std
                - reference_log_std
                + (reference_variance + (reference_mean - current_mean).square())
                / (2.0 * current_variance)
                - 0.5
            ).sum(dim=-1).mean()
            wave_losses.append(kl)
            metrics[f"marc_retention_kl_wave{wave}"] = float(kl.detach())
        mean_kl = torch.stack(wave_losses).mean()
        loss = self.retention_coefficient * mean_kl
        metrics["marc_retention_loss"] = float(loss.detach())
        return loss, metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "banks": {
                str(wave): [
                    {key: np.asarray(value).copy() for key, value in row.items()}
                    for row in bank
                ]
                for wave, bank in self.banks.items()
            },
            "rng_state": deepcopy(self.rng.bit_generator.state),
            "ingested_success_count": deepcopy(self.ingested_success_count),
        }

    def load_state_dict(self, state: dict[str, Any] | None, *, strict: bool = True) -> None:
        if not self.enabled:
            return
        if state is None:
            if strict:
                raise RuntimeError("MARC checkpoint state is missing")
            return
        if int(state.get("version", -1)) != self.version:
            raise RuntimeError("MARC checkpoint version mismatch")
        restored = state.get("banks", {})
        for wave in self.banks:
            rows = restored.get(str(wave))
            if not isinstance(rows, list):
                raise RuntimeError(f"MARC checkpoint bank B{wave} is missing")
            self.banks[wave].clear()
            for row in rows:
                self.banks[wave].append(
                    {
                        "observation": np.asarray(row["observation"], dtype=np.float16),
                        "reference_mean": np.asarray(row["reference_mean"], dtype=np.float16),
                        "reference_log_std": np.asarray(row["reference_log_std"], dtype=np.float16),
                    }
                )
        self.rng.bit_generator.state = deepcopy(state["rng_state"])
        counts = state.get("ingested_success_count", {})
        self.ingested_success_count = {
            wave: int(counts.get(wave, counts.get(str(wave), 0))) for wave in self.banks
        }


__all__ = [
    "MARC_MAPPO_VERSION",
    "MilestoneAwareRetentionCreditModule",
    "compute_local_gae",
    "continuation_coefficients",
    "successful_wave_from_transition",
    "tempered_wave_weights",
]
