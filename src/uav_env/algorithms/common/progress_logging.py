"""Compact progress-log formatting shared by algorithm runners."""

from __future__ import annotations

from typing import Any


def safe_metric(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Return a finite-ish numeric metric or a safe fallback for display."""

    value = row.get(key, default)
    return float(value) if isinstance(value, (int, float)) else default


def actor_entropy_mean(row: dict[str, Any]) -> float:
    """Average available independent-actor entropy fields."""

    values = [
        safe_metric(row, f"actor_{index}_policy_entropy_collect")
        for index in range(3)
        if f"actor_{index}_policy_entropy_collect" in row
    ]
    return sum(values) / len(values) if values else 0.0


def training_entropy(row: dict[str, Any]) -> float:
    """Use MAPPO shared entropy first, otherwise average HAPPO actor entropy."""

    if "rollout_action_entropy" in row:
        return safe_metric(row, "rollout_action_entropy")
    return actor_entropy_mean(row)


def format_training_log(algorithm: str, row: dict[str, Any]) -> str:
    """Build one compact stdout line from a real training metrics row."""

    return (
        f"[{algorithm} update {int(row['update_index']):04d}] "
        f"steps={int(row['environment_steps'])} "
        f"episodes={int(row['episodes'])} "
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
        f"steps={int(evaluation['environment_steps'])} "
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
