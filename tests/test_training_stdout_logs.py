from uav_env.algorithms.mappo.runner import format_evaluation_log, format_training_log


def test_training_log_contains_reward_and_combat_metrics() -> None:
    line = format_training_log(
        "MAPPO",
        {
            "update_index": 2,
            "environment_steps": 4096,
            "episodes": 3,
            "rollout_team_episode_return_mean": -1.25,
            "rollout_mean_per_agent_episode_return": -0.42,
            "team_reward_mean": -0.01,
            "rollout_red_hits_mean": 1.0,
            "rollout_blue_hits_mean": 0.5,
            "rollout_red_effective_damage_mean": 80.0,
            "rollout_blue_effective_damage_mean": 40.0,
            "timeout_rate": 0.25,
            "rollout_action_entropy": 1.7,
            "samples_per_second": 512.0,
        },
    )
    assert "[MAPPO update 0002]" in line
    assert "team_return=-1.250" in line
    assert "red_hits=1.00" in line
    assert "blue_damage=40.0" in line


def test_evaluation_log_contains_outcome_metrics() -> None:
    line = format_evaluation_log(
        "HAPPO",
        {
            "environment_steps": 50000,
            "evaluation_split": "validation",
            "red_win_rate": 0.4,
            "elimination_red_win_rate": 0.3,
            "timeout_survival_red_win_rate": 0.1,
            "mean_team_episode_return": -2.0,
            "mean_red_hits": 2.0,
            "mean_blue_hits": 1.0,
            "mean_red_effective_damage": 150.0,
            "mean_blue_effective_damage": 60.0,
        },
    )
    assert "[HAPPO eval:validation]" in line
    assert "red_win=0.400" in line
    assert "timeout_win=0.100" in line
    assert "red_damage=150.0" in line
