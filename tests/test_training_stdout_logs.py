from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from uav_env.algorithms.common.progress_logging import (
    actor_entropy_mean,
    format_evaluation_log,
    format_training_log,
)
from uav_env.algorithms.mappo.config import load_mappo_config
from uav_env.algorithms.mappo.metrics import combat_outcome_rates
from uav_env.algorithms.mappo.runner import MAPPORunner


def test_evaluation_log_uses_real_combat_outcome_rate_keys() -> None:
    rates = combat_outcome_rates(
        [
            SimpleNamespace(winner="red", termination_reason="blue_eliminated"),
            SimpleNamespace(winner="red", termination_reason="timeout"),
            SimpleNamespace(winner="draw", termination_reason="timeout"),
            SimpleNamespace(winner="blue", termination_reason="red_eliminated"),
        ]
    )
    line = format_evaluation_log(
        "MAPPO",
        {
            "environment_steps": 50000,
            "evaluation_split": "validation",
            **rates,
            "mean_team_episode_return": -2.0,
            "mean_red_hits": 2.0,
            "mean_blue_hits": 1.0,
            "mean_red_effective_damage": 150.0,
            "mean_blue_effective_damage": 60.0,
            "mean_red_survivors": 2.0,
            "mean_blue_survivors": 1.0,
        },
    )
    assert "red_win=0.500" in line
    assert "elim_win=0.250" in line
    assert "timeout_win=0.250" in line
    assert "draw=0.250" in line
    assert "timeout=0.500" in line


def test_mappo_short_collect_produces_real_step_reward_means(tmp_path: Path) -> None:
    config = load_mappo_config("configs/mappo_smoke_3v3_v2.yaml")
    config.update(
        {
            "num_envs": 1,
            "vector_env": "sync",
            "rollout_length": 1,
            "run_id": "collect_metrics",
            "device": "cpu",
            "log_interval": 0,
        }
    )
    runner = MAPPORunner(config, "pytest", output_root=tmp_path)
    try:
        _, diagnostics = runner.collect()
    finally:
        runner.close()
    assert "team_reward_mean" in diagnostics
    assert "agent_reward_sum_mean" in diagnostics
    assert np.isfinite(diagnostics["team_reward_mean"])
    assert np.isfinite(diagnostics["agent_reward_sum_mean"])
    line = format_training_log(
        "MAPPO",
        {
            "update_index": 1,
            "environment_steps": 1,
            "episodes": 0,
            "samples_per_second": 1.0,
            **diagnostics,
        },
    )
    assert "team_reward=0.0000" not in line or np.isclose(diagnostics["team_reward_mean"], 0.0)
    assert f"team_reward={diagnostics['team_reward_mean']:.4f}" in line


def test_happo_actor_entropy_uses_average_of_all_available_actors() -> None:
    row = {
        "actor_0_policy_entropy_collect": 1.0,
        "actor_1_policy_entropy_collect": 2.0,
        "actor_2_policy_entropy_collect": 3.0,
    }
    assert actor_entropy_mean(row) == 2.0
    line = format_training_log(
        "HAPPO",
        {
            "update_index": 1,
            "environment_steps": 128,
            "episodes": 0,
            "rollout_team_episode_return_mean": 0.0,
            "rollout_mean_per_agent_episode_return": 0.0,
            "team_reward_mean": 0.0,
            "samples_per_second": 100.0,
            **row,
        },
    )
    assert "entropy=2.000" in line
