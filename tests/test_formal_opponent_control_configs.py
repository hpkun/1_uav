from __future__ import annotations

from uav_env.algorithms.mappo.config import load_mappo_config


KEY_FIELDS = (
    "seed",
    "num_envs",
    "vector_env",
    "rollout_length",
    "total_env_steps",
    "evaluation_interval",
    "validation_seed_start",
    "validation_episodes",
    "test_seed_start",
    "test_episodes",
    "checkpoint_interval",
    "checkpoint_selection",
    "run_symmetric_stress_test",
)

ALGORITHM_HYPERPARAMETERS = (
    "gamma",
    "gae_lambda",
    "clip_param",
    "value_clip_param",
    "ppo_epochs",
    "num_mini_batches",
    "actor_lr",
    "critic_lr",
    "entropy_coef",
    "value_loss_coef",
    "max_grad_norm",
    "use_huber_loss",
    "huber_delta",
    "normalize_advantages",
    "use_clipped_value_loss",
    "linear_lr_decay",
    "use_value_normalization",
    "activation",
    "actor_hidden_sizes",
    "critic_hidden_sizes",
    "deterministic_evaluation",
)


EXPECTED = {
    "seed": 1,
    "num_envs": 16,
    "vector_env": "parallel",
    "rollout_length": 128,
    "total_env_steps": 300000,
    "evaluation_interval": 50000,
    "validation_seed_start": 10000,
    "validation_episodes": 20,
    "test_seed_start": 20000,
    "test_episodes": 20,
    "checkpoint_interval": 50000,
    "checkpoint_selection": "combat",
    "run_symmetric_stress_test": False,
}


def _assert_formal_control_config(config: dict, opponent: str) -> None:
    for key, value in EXPECTED.items():
        assert config[key] == value
    assert config["environment"]["kind"] == "3v3"
    assert config["environment"]["scenario"] == "head_on_mirrored_jitter_v2"
    assert config["environment"]["opponent"] == opponent
    assert config["environment"]["multi_terminal_reward_profile"] == "paper_2024_exact"


def test_formal_straight_and_pursuit_control_configs_resolve_expected_fields() -> None:
    straight = load_mappo_config("configs/mappo_formal_straight_3v3_diagnostic.yaml")
    pursuit = load_mappo_config("configs/mappo_formal_pursuit_3v3_diagnostic.yaml")

    _assert_formal_control_config(straight, "straight")
    _assert_formal_control_config(pursuit, "pursuit")


def test_existing_formal_and_learnability_configs_remain_unchanged() -> None:
    formal = load_mappo_config("configs/mappo_3v3_v2.yaml")
    learnability = load_mappo_config("configs/mappo_learnability_3v3.yaml")

    assert formal["environment"]["scenario"] == "head_on_mirrored_jitter_v2"
    assert formal["environment"]["opponent"] == "greedy_combat"
    assert formal["environment"]["multi_terminal_reward_profile"] == "paper_2024_exact"

    assert learnability["environment"]["scenario"] == "head_on_learnability_v1"
    assert learnability["environment"]["opponent"] == "straight"
    assert learnability["environment"]["multi_terminal_reward_profile"] == "paper_2024_exact"


def test_formal_control_configs_differ_only_by_opponent_for_key_fields() -> None:
    straight = load_mappo_config("configs/mappo_formal_straight_3v3_diagnostic.yaml")
    pursuit = load_mappo_config("configs/mappo_formal_pursuit_3v3_diagnostic.yaml")

    for key in KEY_FIELDS:
        assert straight[key] == pursuit[key]

    straight_env = dict(straight["environment"])
    pursuit_env = dict(pursuit["environment"])
    straight_opponent = straight_env.pop("opponent")
    pursuit_opponent = pursuit_env.pop("opponent")
    assert straight_opponent == "straight"
    assert pursuit_opponent == "pursuit"
    assert straight_env == pursuit_env


def test_formal_control_configs_have_no_misleading_recursive_inherits() -> None:
    assert "inherits:" not in open("configs/mappo_formal_straight_3v3_diagnostic.yaml", encoding="utf-8").read()
    assert "inherits:" not in open("configs/mappo_formal_pursuit_3v3_diagnostic.yaml", encoding="utf-8").read()


def test_formal_control_configs_keep_formal_algorithm_hyperparameters() -> None:
    formal = load_mappo_config("configs/mappo_3v3_v2.yaml")
    straight = load_mappo_config("configs/mappo_formal_straight_3v3_diagnostic.yaml")
    pursuit = load_mappo_config("configs/mappo_formal_pursuit_3v3_diagnostic.yaml")

    for key in ALGORITHM_HYPERPARAMETERS:
        assert straight[key] == formal[key]
        assert pursuit[key] == formal[key]
