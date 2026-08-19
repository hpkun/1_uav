from pathlib import Path
import copy
import numpy as np
import pytest
import yaml

from uav_combat.training.evaluator import episode_return_metrics, evaluate
from uav_combat.training.runner import MADSACTrainingRunner
from uav_combat.training.vector_env import SyncVectorEnv


ROOT = Path(__file__).resolve().parents[1]


def configs():
    environment = yaml.safe_load((ROOT / "configs/combat_environment.yaml").read_text(encoding="utf-8"))
    algorithm = yaml.safe_load((ROOT / "configs/madsac.yaml").read_text(encoding="utf-8"))
    return environment, algorithm


def test_protocol_and_interface_dimensions():
    _, algorithm = configs()
    assert algorithm["network"] == {
        "observation_dim": 52,
        "action_dim": 3,
        "num_agents": 4,
        "actor_hidden_layers": [256, 256],
        "critic_hidden_layers": [256, 256],
        "attention_heads": 2,
    }
    assert algorithm["training"]["num_train_envs"] == 24
    assert algorithm["training"]["evaluation_episodes"] == 20


def test_vector_seed_formula_no_reuse_and_52d_observations():
    environment, _ = configs()
    vector = SyncVectorEnv(3, environment, base_seed=10, forbidden_seeds=range(10_000_000, 10_000_020))
    observations = vector.reset()
    assert observations.shape == (3, 4, 52)
    assert vector.last_reset_seeds.tolist() == [10, 11, 12]
    for env in vector.envs:
        env.max_steps = 1
    vector.step_batch(np.zeros((3, 4, 3), dtype=np.float32))
    assert vector.last_reset_seeds.tolist() == [13, 14, 15]
    assert len(vector.used_training_seeds) == 6


def test_24_environment_step_adds_exactly_24_transitions(tmp_path):
    environment, algorithm = configs()
    runner = MADSACTrainingRunner(
        environment, algorithm, num_envs=24, total_sampled_steps=24,
        output_dir=tmp_path, smoke=True,
    )
    assert runner.observations.shape == (24, 4, 52)
    result = runner.vector_step()
    assert result["new_transitions"] == 24
    assert runner.trainer.sampled_steps == 24
    assert runner.trainer.replay.size == 24


def test_algorithm1_T_n_d_and_target_schedule_are_unchanged(tmp_path):
    environment, algorithm = configs()
    runner = MADSACTrainingRunner(
        environment, algorithm, num_envs=24, total_sampled_steps=96,
        output_dir=tmp_path, smoke=True,
    )
    runner.trainer.act = lambda observations, masks: np.zeros((24, 4, 3), dtype=np.float32)
    runner.trainer.batch_size = 1
    calls = {"critic": 0, "actor": 0, "target": 0}
    runner.trainer.update_critics = lambda: calls.__setitem__("critic", calls["critic"] + 1) or {
        "critic1_loss": 1.0, "critic2_loss": 1.0, "q_value": 0.0,
    }
    runner.trainer.update_actor = lambda: calls.__setitem__("actor", calls["actor"] + 1) or {
        "actor_loss": 1.0, "entropy": 1.0,
    }
    runner.trainer.update_targets = lambda: calls.__setitem__("target", calls["target"] + 1)
    expected = [
        {"critic": 1, "actor": 0, "target": 0},
        {"critic": 2, "actor": 1, "target": 1},
        {"critic": 3, "actor": 1, "target": 1},
        {"critic": 4, "actor": 2, "target": 2},
    ]
    for counts in expected:
        result = runner.vector_step()
        assert calls == counts
        assert result["scheduler_T"] == 0


def test_n_updates_still_perform_one_target_update_per_actor_branch(tmp_path):
    environment, algorithm = configs()
    algorithm = copy.deepcopy(algorithm)
    algorithm["implementation"]["update_steps_n"] = 3
    algorithm["implementation"]["policy_delay_d"] = 1
    runner = MADSACTrainingRunner(
        environment, algorithm, num_envs=24, total_sampled_steps=24,
        output_dir=tmp_path, smoke=True,
    )
    runner.trainer.batch_size = 1
    runner.trainer.act = lambda observations, masks: np.zeros((24, 4, 3), dtype=np.float32)
    calls = {"critic": 0, "actor": 0, "target": 0}
    runner.trainer.update_critics = lambda: calls.__setitem__("critic", calls["critic"] + 1) or {"critic1_loss": 1.0}
    runner.trainer.update_actor = lambda: calls.__setitem__("actor", calls["actor"] + 1) or {"actor_loss": 1.0}
    runner.trainer.update_targets = lambda: calls.__setitem__("target", calls["target"] + 1)
    runner.vector_step()
    assert calls == {"critic": 3, "actor": 3, "target": 1}


def test_console_logging_is_read_only_and_compact(tmp_path, monkeypatch):
    environment, algorithm = configs()
    runner = MADSACTrainingRunner(
        environment, algorithm, num_envs=24, total_sampled_steps=24_000,
        output_dir=tmp_path, smoke=True,
    )
    assert runner.start_log_line().startswith(
        "[START] mode=smoke | device=cpu | obs=52 | act=3 | agents=4 | hidden=64"
    )
    assert "return=NA | red_loss=NA | atk=NA | lock=NA | kill=NA" in runner.train_log_line()
    for optimizer in (
        runner.trainer.actor_optimizer, runner.trainer.critic1_optimizer,
        runner.trainer.critic2_optimizer,
    ):
        monkeypatch.setattr(optimizer, "step", lambda: pytest.fail("logging must not optimize"))
    runner.train_log_line()


def test_formal_startup_summary_reports_effective_network_and_replay(tmp_path):
    environment, algorithm = configs()
    runner = MADSACTrainingRunner(
        environment, algorithm, num_envs=24, total_sampled_steps=24,
        output_dir=tmp_path,
    )
    summary = runner.startup_summary()
    assert summary["mode"] == "formal"
    assert (summary["observation_dim"], summary["action_dim"], summary["num_agents"]) == (52, 3, 4)
    assert summary["effective_hidden_dim"] == 256
    assert summary["batch_size"] == 1024
    assert summary["replay_capacity"] == 1_000_000


@pytest.mark.parametrize("field,value", [
    ("observation_dim", 54), ("action_dim", 4), ("num_agents", 3),
])
def test_runner_fails_fast_on_network_environment_dimension_mismatch(
    tmp_path, field, value
):
    environment, algorithm = configs()
    algorithm = copy.deepcopy(algorithm)
    algorithm["network"][field] = value
    with pytest.raises(ValueError, match="network/environment dimension mismatch"):
        MADSACTrainingRunner(
            environment, algorithm, total_sampled_steps=24, output_dir=tmp_path
        )


def test_evaluator_exposes_split_combat_and_required_episode_metrics():
    environment, _ = configs()
    environment = copy.deepcopy(environment)
    environment["simulation"]["max_steps"] = 1

    class ZeroActor:
        @staticmethod
        def act(observation, alive_mask, deterministic=True):
            assert deterministic
            return np.zeros((4, 3), dtype=np.float32)

    result = evaluate(ZeroActor(), environment, seeds=[10_000_000])
    expected = {
        "average_return", "average_agent_return", "win_rate", "loss_rate",
        "draw_rate", "timeout_rate", "average_red_loss", "average_blue_loss",
        "average_red_attack_kills", "average_blue_attack_kills",
        "average_red_low_altitude_loss", "average_blue_low_altitude_loss",
        "average_red_high_altitude_loss", "average_blue_high_altitude_loss",
        "average_episode_length", "average_max_horizontal_pair_separation",
    } | {
        f"{side}_{event}_episode_rate"
        for side in ("red", "blue") for event in ("attackable", "lock", "kill")
    }
    assert expected <= result.keys()


def test_episode_return_metric_is_team_sum_and_agent_mean():
    team, agent = episode_return_metrics(np.array([1.0, 2.0, 3.0, 4.0]))
    assert team == 10.0
    assert agent == 2.5
