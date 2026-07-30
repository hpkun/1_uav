"""Compact progress-log formatting shared by algorithm runners."""

from __future__ import annotations

import math
import re
from numbers import Real
from typing import Any


def safe_metric(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Return a finite numeric metric or a finite fallback for display."""

    fallback = float(default) if isinstance(default, Real) and not isinstance(default, bool) and math.isfinite(float(default)) else 0.0
    parsed = _finite_real(row.get(key))
    return parsed if parsed is not None else fallback


def _finite_real(value: Any) -> float | None:
    """Return a finite float for real numeric values, otherwise None."""

    if isinstance(value, Real) and not isinstance(value, bool):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    return None


def safe_int_metric(row: dict[str, Any], key: str, default: int = 0) -> int:
    """Return a finite integer display metric or a safe fallback."""

    value = safe_metric(row, key, float(default))
    return int(value)


def actor_entropy_mean(row: dict[str, Any]) -> float:
    """Average available independent-actor entropy fields."""

    pattern = re.compile(r"actor_(\d+)_policy_entropy_collect\Z")
    indexed: list[tuple[int, float]] = []
    for key, value in row.items():
        match = pattern.fullmatch(str(key))
        if match is None:
            continue
        parsed = _finite_real(value)
        if parsed is not None:
            indexed.append((int(match.group(1)), parsed))
    values = [value for _, value in sorted(indexed)]
    return sum(values) / len(values) if values else 0.0


def training_entropy(row: dict[str, Any]) -> float:
    """Use MAPPO shared entropy first, otherwise average HAPPO actor entropy."""

    entropy = _finite_real(row.get("rollout_action_entropy"))
    if entropy is not None:
        return entropy
    return actor_entropy_mean(row)


def format_training_log(algorithm: str, row: dict[str, Any]) -> str:
    """Build one compact stdout line from a real training metrics row."""

    return (
        f"[{algorithm} update {safe_int_metric(row, 'update_index'):04d}] "
        f"steps={safe_int_metric(row, 'environment_steps')} "
        f"episodes={safe_int_metric(row, 'episodes')} "
        f"team_return={safe_metric(row, 'rollout_team_episode_return_mean'):.3f} "
        f"per_agent_return={safe_metric(row, 'rollout_mean_per_agent_episode_return'):.3f} "
        f"team_reward={safe_metric(row, 'team_reward_mean'):.4f} "
        f"red_hits={safe_metric(row, 'rollout_red_hits_mean'):.2f} "
        f"blue_hits={safe_metric(row, 'rollout_blue_hits_mean'):.2f} "
        f"red_damage={safe_metric(row, 'rollout_red_effective_damage_mean'):.1f} "
        f"blue_damage={safe_metric(row, 'rollout_blue_effective_damage_mean'):.1f} "
        f"timeout={safe_metric(row, 'timeout_rate'):.2f} "
        f"entropy={training_entropy(row):.3f} "
        f"sps={safe_metric(row, 'samples_per_second'):.1f}"
    )


def format_evaluation_log(algorithm: str, evaluation: dict[str, Any]) -> str:
    """Build one compact stdout line from a real evaluation result row."""

    red_win = safe_metric(evaluation, "overall_red_win_rate", safe_metric(evaluation, "red_win_rate"))
    red_hits = safe_metric(evaluation, "mean_red_hits", safe_metric(evaluation, "mean_hits"))
    red_damage = safe_metric(evaluation, "mean_red_effective_damage", safe_metric(evaluation, "mean_effective_damage"))
    return (
        f"[{algorithm} eval:{evaluation.get('evaluation_split', 'validation')}] "
        f"steps={safe_int_metric(evaluation, 'environment_steps')} "
        f"red_win={red_win:.3f} "
        f"elim_win={safe_metric(evaluation, 'elimination_win_rate'):.3f} "
        f"timeout_win={safe_metric(evaluation, 'timeout_survival_win_rate'):.3f} "
        f"draw={safe_metric(evaluation, 'draw_rate'):.3f} "
        f"timeout={safe_metric(evaluation, 'timeout_rate'):.3f} "
        f"return={safe_metric(evaluation, 'mean_team_episode_return'):.3f} "
        f"red_survivors={safe_metric(evaluation, 'mean_red_survivors'):.2f} "
        f"blue_survivors={safe_metric(evaluation, 'mean_blue_survivors'):.2f} "
        f"red_hits={red_hits:.2f} "
        f"blue_hits={safe_metric(evaluation, 'mean_blue_hits'):.2f} "
        f"red_damage={red_damage:.1f} "
        f"blue_damage={safe_metric(evaluation, 'mean_blue_effective_damage'):.1f}"
    )
