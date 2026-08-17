from pathlib import Path
import numpy as np
import yaml

from scripts.aggregate_training_runs import mean_ci, read_history, write_rows
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
    runner.vector_step()
    runner.vector_step()
    assert calls == {"critic": 0, "actor": 0, "target": 0}  # replay < smoke batch 64
    assert runner.scheduler_T == 48
    runner.vector_step()
    assert calls == {"critic": 1, "actor": 0, "target": 1}
    assert runner.scheduler_T == 0
    runner.vector_step()
    assert calls == {"critic": 2, "actor": 1, "target": 2}  # vector t=4 satisfies t mod d


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
            "win_rate": run / 10,
            "average_red_loss": 4 - run / 2,
        }])
        histories.append(read_history(run_dir))
    assert set.intersection(*(set(history) for history in histories)) == {100000}
    assert mean_ci([history[100000]["average_return"] for history in histories])[0] == 2
