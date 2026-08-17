from pathlib import Path
import copy
import numpy as np
import pytest
import yaml

from scripts.aggregate_training_runs import mean_ci, read_history, write_rows
from scripts.run_reconstruction_sensitivity import apply_profile, validate_sampled_steps
from uav_combat.training.evaluator import episode_return_metrics, evaluate
from uav_combat.training.runner import PaperTrainingRunner
from uav_combat.training.vector_env import SyncVectorEnv


ROOT = Path(__file__).resolve().parents[1]


def configs():
    env = yaml.safe_load((ROOT / "configs/paper_environment.yaml").read_text(encoding="utf-8"))
    alg = yaml.safe_load((ROOT / "configs/madsac.yaml").read_text(encoding="utf-8"))
    return env, alg


def test_paper_protocol_configures_24_train_20_test_5_runs_95ci():
    _, algorithm = configs()
    training = algorithm["training"]
    assert training["num_train_envs"] == 24
    assert training["evaluation_episodes"] == 20
    assert training["independent_training_runs"] == 5
    assert training["confidence_interval"] == .95
    assert "updates_per_transition" not in algorithm["reproduction_assumptions"]
    assert algorithm["reproduction_assumptions"]["assumed_sampled_steps_per_training_cycle"] == 50_000
    assert algorithm["runtime_logging"] == {
        "console_interval_sampled_steps": 20_000,
        "recent_episode_window": 100,
    }


def test_console_threshold_crossing_and_recent_episode_window(tmp_path):
    environment, algorithm = configs()
    runner = PaperTrainingRunner(
        environment, algorithm, num_envs=24, total_sampled_steps=24_000,
        output_dir=tmp_path, smoke=True,
    )
    runner.trainer.sampled_steps = 19_992
    assert not runner._console_log_due()
    runner.trainer.sampled_steps = 20_016
    assert runner._console_log_due()
    assert runner.next_console_log == 40_000
    assert not runner._console_log_due()

    runner.completed_records = [
        {"team_episode_return": float(i), "red_success": i % 2, "red_losses": float(i + 1)}
        for i in range(105)
    ]
    recent = runner.recent_episode_metrics()
    assert recent["return"] == pytest.approx(np.mean(range(5, 105)))
    assert recent["win"] == pytest.approx(np.mean([i % 2 for i in range(5, 105)]))
    assert recent["red_loss"] == pytest.approx(np.mean(range(6, 106)))


def test_recent_episode_window_handles_short_and_empty_history(tmp_path):
    environment, algorithm = configs()
    runner = PaperTrainingRunner(environment, algorithm, num_envs=4, total_sampled_steps=4, output_dir=tmp_path, smoke=True)
    assert runner.recent_episode_metrics() == {"return": None, "win": None, "red_loss": None}
    assert "return=NA | win=NA | red_loss=NA" in runner.train_log_line()
    runner.completed_records = [
        {"team_episode_return": 2.0, "red_success": 1, "red_losses": 3.0},
        {"team_episode_return": 4.0, "red_success": 0, "red_losses": 1.0},
    ]
    assert runner.recent_episode_metrics() == {"return": 3.0, "win": 0.5, "red_loss": 2.0}


def test_console_log_formats_and_checkpoint_basename(tmp_path, monkeypatch):
    environment, algorithm = configs()
    runner = PaperTrainingRunner(environment, algorithm, num_envs=24, total_sampled_steps=24_000, output_dir=tmp_path, smoke=True)
    evaluation = {
        "sampled_steps": 200_016,
        "average_return": 8.73,
        "average_agent_return": 2.18,
        "win_rate": 0.70,
        "average_red_loss": 2.65,
        "average_episode_length": 112.4,
    }
    assert runner.evaluation_log_line(evaluation) == (
        "[EVAL] steps=200016 | return=8.73 | agent_return=2.18 | win=0.70 "
        "| red_loss=2.65 | ep_len=112.4"
    )
    assert runner.checkpoint_log_line(500_016, tmp_path / "nested" / "checkpoint_500016.pt") == (
        "[CKPT] steps=500016 | saved=checkpoint_500016.pt"
    )
    assert runner.start_log_line() == (
        "[START] device=cpu | envs=24 | seed=0 | total=24000 | batch=64 "
        "| replay=50000 | T=24 | n=1 | d=2"
    )
    done = {
        "sampled_steps": 1_000_008, "completed_episodes": 9458,
        "average_return": 3.25, "win_rate": 0.60, "average_red_loss": 2.50,
    }
    assert runner.done_log_line(done) == (
        "[DONE] steps=1000008 | episodes=9458 | return=3.25 | win=0.60 | red_loss=2.50"
    )

    def unexpected_optimizer_step(*args, **kwargs):
        pytest.fail("console logging must not step an optimizer")

    for optimizer in (
        runner.trainer.actor_optimizer,
        runner.trainer.critic1_optimizer,
        runner.trainer.critic2_optimizer,
    ):
        monkeypatch.setattr(optimizer, "step", unexpected_optimizer_step)
    training_state = (
        runner.trainer.sampled_steps,
        runner.trainer.vector_steps,
        runner.scheduler_T,
        runner.scheduler_update_blocks,
        runner.trainer.critic_update_count,
        runner.trainer.actor_update_count,
        runner.trainer.target_update_count,
    )
    runner.train_log_line()
    runner.evaluation_log_line(evaluation)
    runner.checkpoint_log_line(500_016, "checkpoint_500016.pt")
    runner.done_log_line(done)
    assert training_state == (
        runner.trainer.sampled_steps,
        runner.trainer.vector_steps,
        runner.scheduler_T,
        runner.scheduler_update_blocks,
        runner.trainer.critic_update_count,
        runner.trainer.actor_update_count,
        runner.trainer.target_update_count,
    )


def test_simple_training_seed_formula_and_no_reuse():
    environment, _ = configs()
    vector = SyncVectorEnv(3, environment, base_seed=10, forbidden_seeds=range(10_000_000, 10_000_020))
    vector.reset()
    assert vector.last_reset_seeds.tolist() == [10, 11, 12]
    for env in vector.envs:
        env.max_steps = 1
    vector.step_batch(np.zeros((3, 4, 3), dtype=np.float32))
    assert vector.last_reset_seeds.tolist() == [13, 14, 15]
    assert len(vector.used_training_seeds) == 6


def test_24_environment_sampling_adds_M_transitions(tmp_path):
    environment, algorithm = configs()
    runner = PaperTrainingRunner(environment, algorithm, num_envs=24, total_sampled_steps=24, output_dir=tmp_path, smoke=True)
    assert runner.observations.shape == (24, 4, 45)
    assert len(set(runner.vector.last_reset_seeds)) == 24
    result = runner.vector_step()
    assert result["new_transitions"] == 24
    assert runner.trainer.sampled_steps == 24
    assert runner.trainer.replay.size == 24


def test_algorithm1_T_n_d_and_target_schedule(tmp_path):
    environment, algorithm = configs()
    runner = PaperTrainingRunner(environment, algorithm, num_envs=24, total_sampled_steps=96, output_dir=tmp_path, smoke=True)
    runner.trainer.act = lambda observations, masks: np.zeros((24, 4, 3), dtype=np.float32)
    runner.trainer.batch_size = 1  # remove replay warmup from scheduler semantics
    calls = {"critic": 0, "actor": 0, "target": 0}

    def critic():
        calls["critic"] += 1
        return {"critic1_loss": 1.0, "critic2_loss": 1.0, "q_value": 0.0}
    def actor():
        calls["actor"] += 1
        return {"actor_loss": 1.0, "entropy": 1.0}
    def targets():
        calls["target"] += 1

    runner.trainer.update_critics = critic
    runner.trainer.update_actor = actor
    runner.trainer.update_targets = targets
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
    assert calls["target"] == calls["actor"]
    assert runner.last_actor_metrics == {"actor_loss": 1.0, "entropy": 1.0}


def test_critic_only_block_preserves_last_actor_metrics_and_scheduler_semantics(tmp_path):
    environment, algorithm = configs()
    runner = PaperTrainingRunner(environment, algorithm, num_envs=24, total_sampled_steps=48, output_dir=tmp_path, smoke=True)
    runner.trainer.batch_size = 1
    runner.trainer.act = lambda observations, masks: np.zeros((24, 4, 3), dtype=np.float32)
    calls = {"critic": 0, "actor": 0, "target": 0}
    runner.trainer.update_critics = lambda: calls.__setitem__("critic", calls["critic"] + 1) or {
        "critic1_loss": 2.0, "critic2_loss": 4.0, "q_value": 0.5,
    }
    runner.trainer.update_actor = lambda: calls.__setitem__("actor", calls["actor"] + 1) or {
        "actor_loss": -1.25, "entropy": 2.0,
    }
    runner.trainer.update_targets = lambda: calls.__setitem__("target", calls["target"] + 1)
    runner.trainer.vector_steps = 1
    runner.vector_step()  # vector step 2: actor branch
    actor_metrics = runner.last_actor_metrics.copy()
    sampled_steps = runner.trainer.sampled_steps
    runner.vector_step()  # vector step 3: critic-only branch
    assert runner.last_actor_metrics == actor_metrics
    assert calls == {"critic": 2, "actor": 1, "target": 1}
    assert runner.trainer.sampled_steps == sampled_steps + 24
    assert runner.scheduler_update_blocks == 2


def test_algorithm1_runs_n_updates_but_one_target_update_per_actor_branch(tmp_path):
    environment, algorithm = configs()
    algorithm["reproduction_assumptions"]["update_steps_n"] = 3
    algorithm["reproduction_assumptions"]["policy_delay_d"] = 1
    runner = PaperTrainingRunner(environment, algorithm, num_envs=24, total_sampled_steps=24, output_dir=tmp_path, smoke=True)
    runner.trainer.batch_size = 1
    runner.trainer.act = lambda observations, masks: np.zeros((24, 4, 3), dtype=np.float32)
    calls = {"critic": 0, "actor": 0, "target": 0}
    runner.trainer.update_critics = lambda: calls.__setitem__("critic", calls["critic"] + 1) or {"critic1_loss": 1.0}
    runner.trainer.update_actor = lambda: calls.__setitem__("actor", calls["actor"] + 1) or {"actor_loss": 1.0}
    runner.trainer.update_targets = lambda: calls.__setitem__("target", calls["target"] + 1)
    runner.vector_step()
    assert calls == {"critic": 3, "actor": 3, "target": 1}


def test_scheduler_counter_and_one_million_output_schedule(tmp_path):
    environment, algorithm = configs()
    runner = PaperTrainingRunner(
        environment, algorithm, num_envs=24, total_sampled_steps=1_000_000,
        output_dir=tmp_path, smoke=True,
    )
    assert runner.evaluation_interval == 100_000
    assert runner.checkpoint_interval == 500_000
    assert runner.next_evaluation == 100_000
    assert runner.next_checkpoint == 500_000
    assert runner.scheduler_update_blocks == 0
    runner.scheduler_update_blocks = 7
    checkpoint = tmp_path / "counter.pt"
    runner.save_checkpoint(checkpoint)
    import torch
    extra = torch.load(checkpoint, map_location="cpu", weights_only=False)["extra"]
    assert extra["scheduler_update_blocks"] == 7
    assert "training_cycles" not in extra
    assert runner.summary()["scheduler_update_blocks"] == 7


def test_cuda_request_never_silently_falls_back(monkeypatch):
    import torch
    from uav_combat.madsac import MADSACTrainer
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        MADSACTrainer(hidden_dim=32, attention_heads=2, device="cuda")


def test_evaluation_seed_set_is_twenty_and_disjoint(tmp_path):
    environment, algorithm = configs()
    runner = PaperTrainingRunner(environment, algorithm, num_envs=4, total_sampled_steps=4, output_dir=tmp_path, smoke=True)
    assert runner.evaluation_seeds == list(range(10_000_000, 10_000_020))
    assert not set(runner.vector.used_training_seeds).intersection(runner.evaluation_seeds)


def test_five_run_student_t_95ci_calculation():
    mean, half_width = mean_ci([1, 2, 3, 4, 5])
    assert mean == 3
    assert half_width > 0


def test_figure8_9_aggregation_io_for_five_runs(tmp_path):
    histories = []
    for run in range(5):
        run_dir = tmp_path / f"run_{run}"
        run_dir.mkdir()
        write_rows(run_dir / "evaluation_history.csv", [{
            "sampled_steps": 100000,
            "average_return": float(run),
            "average_agent_return": float(run) / 4,
            "win_rate": run / 10,
            "average_red_loss": 4 - run / 2,
        }])
        histories.append(read_history(run_dir))
    assert set.intersection(*(set(history) for history in histories)) == {100000}
    assert mean_ci([history[100000]["average_return"] for history in histories])[0] == 2
    assert histories[4][100000]["average_agent_return"] == 1


def test_team_and_agent_episode_return_are_not_reward_slot_zero():
    team_return, average_agent_return = episode_return_metrics(np.array([1.0, 2.0, 3.0, 4.0]))
    assert team_return == 10.0
    assert average_agent_return == 2.5


def test_evaluator_sums_local_agent_rewards(monkeypatch):
    import uav_combat.training.evaluator as evaluator_module

    class OneStepEnvironment:
        red_alive_mask = np.ones(4, dtype=np.float32)
        def __init__(self, config): pass
        def reset(self, seed): return np.zeros((4, 45), dtype=np.float32), {}
        def step(self, actions):
            return np.zeros((4, 45), dtype=np.float32), np.array([1, 2, 3, 4], dtype=np.float32), True, False, {
                "red_success": True, "red_losses": 1, "episode_length": 1,
            }

    class ZeroActor:
        def act(self, observation, alive_mask, deterministic=False):
            assert deterministic
            return np.zeros((4, 3), dtype=np.float32)

    monkeypatch.setattr(evaluator_module, "PaperUAVCombatEnv", OneStepEnvironment)
    result = evaluate(ZeroActor(), {}, seeds=[1, 2])
    assert result["average_return"] == 10.0
    assert result["average_agent_return"] == 2.5


def test_sensitivity_overlay_is_one_group_and_canonical_is_unchanged():
    environment, algorithm = configs()
    original_environment, original_algorithm = copy.deepcopy(environment), copy.deepcopy(algorithm)
    candidates = yaml.safe_load((ROOT / "configs/sensitivity_candidates.yaml").read_text(encoding="utf-8"))
    modified_environment, modified_algorithm = apply_profile(
        environment, algorithm, candidates, "weapon", "weapon_weak"
    )
    assert environment == original_environment and algorithm == original_algorithm
    assert modified_algorithm == algorithm
    assert modified_environment["weapon"]["distance_min"] == 1000.0
    assert modified_environment["reproduction_assumptions"]["weapon"]["d_hit"] == 1000.0
    assert modified_environment["reproduction_assumptions"]["sensor"] == environment["reproduction_assumptions"]["sensor"]
    assert candidates["status"] == "CANDIDATE ONLY - NOT PAPER VALUE"


def test_sensitivity_rejects_more_than_200k_steps():
    assert validate_sampled_steps(200_000) == 200_000
    with pytest.raises(ValueError, match="must be executed manually"):
        validate_sampled_steps(200_001)
