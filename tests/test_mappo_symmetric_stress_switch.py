from pathlib import Path
from types import SimpleNamespace

import yaml

from uav_env.algorithms.mappo.runner import MAPPORunner


class DummyModule:
    def parameters(self):
        return []


class DummyWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def run_skeleton(tmp_path: Path, enabled: bool):
    runner = MAPPORunner.__new__(MAPPORunner)
    runner.output_dir = tmp_path
    runner.environment_steps = 0
    runner.update_index = 0
    runner.episodes = 0
    runner.num_agents = 3
    runner.last_evaluation_step = -1
    runner.best_evaluation = None
    runner.device = "cpu"
    runner.schema_metadata = {"environment_schema_version": "homogeneous_3v3_v2_timeaware"}
    runner.actor = DummyModule()
    runner.critic = DummyModule()
    runner.writer = DummyWriter()
    runner.config = {
        "total_env_steps": 4,
        "rollout_length": 2,
        "num_envs": 2,
        "linear_lr_decay": False,
        "evaluation_interval": 4,
        "checkpoint_interval": 100,
        "validation_episodes": 1,
        "validation_seed_start": 10,
        "test_episodes": 1,
        "test_seed_start": 20,
        "checkpoint_selection": "combat",
        "run_symmetric_stress_test": enabled,
    }
    runner.trainer = SimpleNamespace(update=lambda buffer: {})
    runner.collect = lambda: (object(), {"rollout_episode_count": 0.0})
    runner._save = lambda name: (tmp_path / "checkpoints" / name).parent.mkdir(parents=True, exist_ok=True)
    runner.resume = lambda *args, **kwargs: None
    calls = []

    def evaluate(episodes, seed_start, deterministic=None, scenario=None):
        calls.append(scenario)
        return {
            "elimination_win_rate": 0.0,
            "overall_red_win_rate": 0.0,
            "mean_effective_damage": 0.0,
            "mean_survivor_difference": 0.0,
            "mean_hits": 0.0,
            "mean_attack_area_steps": 0.0,
            "mean_team_episode_return": 0.0,
            "red_crash_rate": 0.0,
            "timeout_rate": 0.0,
        }

    runner.evaluate = evaluate
    runner._run_impl()
    summary = yaml.safe_load((tmp_path / "final_summary.yaml").read_text(encoding="utf-8"))
    return calls, summary


def test_symmetric_stress_is_opt_in_for_final_summary(tmp_path: Path):
    calls, summary = run_skeleton(tmp_path / "off", False)
    assert "symmetric_stress_test_v2" not in calls
    assert summary["symmetric_stress_test"] == {}

    calls, summary = run_skeleton(tmp_path / "on", True)
    assert calls.count("symmetric_stress_test_v2") == 2
    assert set(summary["symmetric_stress_test"]) == {"last", "best"}
