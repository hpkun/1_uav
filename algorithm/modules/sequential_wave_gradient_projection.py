"""Parameter-free ordered upstream Actor-gradient projection for SWGP-MAPPO."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import torch

from .base import CapabilityModule


SWGP_MAPPO_VERSION = 1


def gradient_dot(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> torch.Tensor:
    return sum((a * b).sum() for a, b in zip(left, right))


def gradient_norm(gradient: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(sum(value.square().sum() for value in gradient))


def gradient_cosine(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor],
                    epsilon: float = 1e-12) -> torch.Tensor:
    return gradient_dot(left, right) / (gradient_norm(left) * gradient_norm(right) + epsilon)


def project_nonconflicting(gradient: Sequence[torch.Tensor], reference: Sequence[torch.Tensor],
                           epsilon: float = 1e-12) -> tuple[list[torch.Tensor], torch.Tensor, bool]:
    """Remove only a component opposed to ``reference``; never alter the reference."""
    dot = gradient_dot(gradient, reference)
    reference_norm_sq = sum(value.square().sum() for value in reference)
    applied = bool((dot < 0).detach()) and bool((reference_norm_sq > epsilon).detach())
    if not applied:
        return list(gradient), dot, False
    return [value - dot / (reference_norm_sq + epsilon) * ref
            for value, ref in zip(gradient, reference)], dot, True


def ordered_upstream_pairwise(
    gradients: dict[int, Sequence[torch.Tensor]], epsilon: float = 1e-12,
) -> tuple[dict[int, list[torch.Tensor]], dict[str, Any]]:
    """Protect W1, then protected W2, using ordered pairwise projections for W3."""
    projected = {wave: list(values) for wave, values in gradients.items()}
    diagnostics: dict[str, Any] = {
        "w2_applied": False, "w3_w1_applied": False, "w3_w2_applied": False,
        "w2_dot_before": None, "w2_dot_after": None,
        "w3_w1_dot_before": None, "w3_w1_dot_after": None,
        "w3_w2_dot_before": None, "w3_w2_dot_after": None,
    }
    if 2 in projected and 1 in projected:
        projected[2], dot, applied = project_nonconflicting(projected[2], projected[1], epsilon)
        diagnostics.update(w2_applied=applied, w2_dot_before=dot,
                           w2_dot_after=gradient_dot(projected[2], projected[1]))
    if 3 in projected and 1 in projected:
        projected[3], dot, applied = project_nonconflicting(projected[3], projected[1], epsilon)
        diagnostics.update(w3_w1_applied=applied, w3_w1_dot_before=dot,
                           w3_w1_dot_after=gradient_dot(projected[3], projected[1]))
    if 3 in projected and 2 in projected:
        projected[3], dot, applied = project_nonconflicting(projected[3], projected[2], epsilon)
        diagnostics.update(w3_w2_applied=applied, w3_w2_dot_before=dot,
                           w3_w2_dot_after=gradient_dot(projected[3], projected[2]))
        if 1 in projected:
            diagnostics["w3_w1_dot_after"] = gradient_dot(projected[3], projected[1])
    return projected, diagnostics


def natural_wave_fractions(counts: dict[int, int]) -> dict[int, float]:
    total = sum(max(0, int(counts.get(wave, 0))) for wave in (1, 2, 3))
    return {wave: (max(0, int(counts.get(wave, 0))) / total if total else 0.0)
            for wave in (1, 2, 3)}


def weighted_gradient_sum(gradients: dict[int, Sequence[torch.Tensor]],
                          weights: dict[int, float], template: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    return [sum(weights.get(wave, 0.0) * gradients[wave][index] for wave in gradients)
            if gradients else torch.zeros_like(parameter)
            for index, parameter in enumerate(template)]


class SequentialWaveGradientProjectionModule(CapabilityModule):
    name = "sequential_wave_gradient_projection"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.mode = str(self.config.get("mode", "ordered_upstream_pairwise"))
        self.epsilon = float(self.config.get("epsilon", 1e-12))
        if self.mode != "ordered_upstream_pairwise":
            raise ValueError("SWGP V1 requires mode=ordered_upstream_pairwise")
        if self.epsilon <= 0:
            raise ValueError("SWGP epsilon must be positive")
        self.reset_counters()

    def reset_counters(self) -> None:
        self.total_swgp_minibatches = 0
        self.w2_projection_count = 0
        self.w3_w1_projection_count = 0
        self.w3_w2_projection_count = 0
        self.no_projection_count = 0
        self.single_wave_minibatch_count = 0

    def record(self, available_waves: int, diagnostics: dict[str, Any]) -> None:
        self.total_swgp_minibatches += 1
        self.w2_projection_count += int(diagnostics["w2_applied"])
        self.w3_w1_projection_count += int(diagnostics["w3_w1_applied"])
        self.w3_w2_projection_count += int(diagnostics["w3_w2_applied"])
        projected = any(diagnostics[key] for key in ("w2_applied", "w3_w1_applied", "w3_w2_applied"))
        self.no_projection_count += int(not projected)
        self.single_wave_minibatch_count += int(available_waves == 1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": SWGP_MAPPO_VERSION, "config": deepcopy(self.config),
            "total_swgp_minibatches": self.total_swgp_minibatches,
            "w2_projection_count": self.w2_projection_count,
            "w3_w1_projection_count": self.w3_w1_projection_count,
            "w3_w2_projection_count": self.w3_w2_projection_count,
            "no_projection_count": self.no_projection_count,
            "single_wave_minibatch_count": self.single_wave_minibatch_count,
        }

    def load_state_dict(self, state: dict[str, Any] | None, *, branch_from_plain: bool = False) -> None:
        if state is None:
            if branch_from_plain:
                self.reset_counters(); return
            raise RuntimeError("SWGP checkpoint is missing projection counters")
        if int(state.get("version", -1)) != SWGP_MAPPO_VERSION:
            raise RuntimeError("SWGP checkpoint version mismatch")
        if state.get("config") != self.config:
            raise RuntimeError("SWGP checkpoint config mismatch")
        for name in ("total_swgp_minibatches", "w2_projection_count", "w3_w1_projection_count",
                     "w3_w2_projection_count", "no_projection_count", "single_wave_minibatch_count"):
            setattr(self, name, int(state.get(name, 0)))


__all__ = [
    "SWGP_MAPPO_VERSION", "SequentialWaveGradientProjectionModule", "gradient_dot",
    "gradient_norm", "gradient_cosine", "project_nonconflicting",
    "ordered_upstream_pairwise", "natural_wave_fractions", "weighted_gradient_sum",
]
