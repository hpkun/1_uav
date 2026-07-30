from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest

from uav_env.actions.discrete_15 import DiscreteAction15
from uav_env.algorithms.common.output_safety import prepare_output_dir, validate_run_id
from uav_env.algorithms.happo.config import load_happo_config
import uav_env.algorithms.happo.runner as happo_runner_module
from uav_env.algorithms.happo.runner import HAPPORunner
from uav_env.algorithms.mappo.config import load_mappo_config
import uav_env.algorithms.mappo.runner as mappo_runner_module
from uav_env.algorithms.mappo.runner import MAPPORunner


def _load_run_env_once():
    path = Path("scripts/run_env_once.py").resolve()
    spec = importlib.util.spec_from_file_location("run_env_once", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_id_rejects_unsafe_values_and_nonempty_directories(tmp_path: Path) -> None:
    for value in ("", ".", "..", "a/b", r"a\b", "bad:name"):
        with pytest.raises(ValueError):
            validate_run_id(value)
    used = tmp_path / "runs" / "case" / "used"
    used.mkdir(parents=True)
    (used / "metrics.csv").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_output_dir(tmp_path / "runs", "case", "used")


def test_mappo_and_happo_reject_nonempty_run_id_before_start(tmp_path: Path) -> None:
    for cls, loader, config_path, run_name in (
        (MAPPORunner, load_mappo_config, "configs/mappo_smoke_3v3_v2.yaml", "mappo_case"),
        (HAPPORunner, load_happo_config, "configs/happo_learnability_3v3.yaml", "happo_case"),
    ):
        config = loader(config_path)
        config.update({"run_id": "used", "device": "cpu", "num_envs": 1, "vector_env": "sync", "log_interval": 0})
        used = tmp_path / run_name / "used"
        used.mkdir(parents=True)
        (used / "metrics.csv").write_text("old", encoding="utf-8")
        with pytest.raises(FileExistsError):
            cls(config, run_name, output_root=tmp_path)


def test_run_env_once_hold_uses_discrete_level_hold() -> None:
    module = _load_run_env_once()
    available = np.ones((3, 15), dtype=bool)
    actions = module.sample_actions(np.random.default_rng(1), available, "hold")
    assert actions.tolist() == [int(DiscreteAction15.LEVEL_HOLD)] * 3


def test_run_env_once_rejects_invalid_intervals() -> None:
    module = _load_run_env_once()
    with pytest.raises(Exception):
        module.positive_int("0")
    with pytest.raises(Exception):
        module.positive_int("-1")
    with pytest.raises(Exception):
        module.nonnegative_int("-1")


def test_run_env_once_one_step_outputs_finite_json() -> None:
    module = _load_run_env_once()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        summary = module.run_once(
            "configs/mappo_formal_straight_3v3_diagnostic.yaml",
            "mappo",
            seed=1,
            policy="hold",
            max_steps=1,
            step_log_interval=1,
        )
    assert np.isfinite(summary["team_return"])
    assert np.isfinite(summary["agent_sum_return"])
    assert summary["decision_steps"] == 1
    assert "reward_component_sums" in summary
    assert "side_statistics" in summary
    json.loads(json.dumps(summary))
    assert "[step 0001]" in buffer.getvalue()


def test_run_env_once_preserves_selected_algorithm_config_errors(tmp_path: Path) -> None:
    module = _load_run_env_once()
    bad = tmp_path / "bad_mappo.yaml"
    bad.write_text("vector_env: bogus\n", encoding="utf-8")
    with pytest.raises(ValueError, match="vector_env"):
        module.load_algorithm_config(str(bad), "mappo")


class _Writer:
    def add_scalar(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass

    def flush(self) -> None:
        pass


class _Trainer:
    actor_optimizer = type("Optimizer", (), {"param_groups": []})()
    critic_optimizer = type("Optimizer", (), {"param_groups": []})()

    def update(self, buffer):
        return {}


def _fake_runner(cls, tmp_path: Path, log_interval: int):
    runner = object.__new__(cls)
    runner.config = {
        "total_env_steps": 102,
        "rollout_length": 1,
        "num_envs": 2,
        "evaluation_interval": 2,
        "checkpoint_interval": 1000,
        "validation_episodes": 1,
        "validation_seed_start": 10,
        "test_episodes": 1,
        "test_seed_start": 20,
        "checkpoint_selection": "combat",
        "log_interval": log_interval,
        "linear_lr_decay": False,
        "run_symmetric_stress_test": False,
    }
    runner.environment_steps = 100
    runner.update_index = 0
    runner.episodes = 0
    runner.num_agents = 3
    runner.output_dir = tmp_path
    runner.device = "cpu"
    runner.writer = _Writer()
    runner.trainer = _Trainer()
    runner.best_evaluation = None
    runner.last_evaluation_step = None
    runner.schema_metadata = {"environment_schema_version": "homogeneous_3v3_v2_timeaware"}
    runner.actor = type("Net", (), {"parameters": lambda self: []})()
    runner.critic = type("Net", (), {"parameters": lambda self: []})()
    runner.collect = lambda: (
        object(),
        {
            "rollout_team_episode_return_mean": 1.0,
            "rollout_mean_per_agent_episode_return": 0.5,
            "team_reward_mean": 0.25,
            "agent_reward_sum_mean": 0.75,
            "rollout_red_hits_mean": 1.0,
            "rollout_blue_hits_mean": 0.0,
            "rollout_red_effective_damage_mean": 10.0,
            "rollout_blue_effective_damage_mean": 0.0,
            "timeout_rate": 0.0,
            "rollout_action_entropy": 1.2,
            "actor_0_policy_entropy_collect": 1.0,
            "actor_1_policy_entropy_collect": 2.0,
            "actor_2_policy_entropy_collect": 3.0,
        },
    )
    runner.evaluate = lambda *args, **kwargs: {
        "overall_red_win_rate": 1.0,
        "elimination_win_rate": 1.0,
        "timeout_survival_win_rate": 0.0,
        "draw_rate": 0.0,
        "timeout_rate": 0.0,
        "mean_team_episode_return": 2.0,
        "mean_red_hits": 1.0,
        "mean_blue_hits": 0.0,
        "mean_red_effective_damage": 10.0,
        "mean_blue_effective_damage": 0.0,
        "mean_red_survivors": 3.0,
        "mean_blue_survivors": 0.0,
    }
    runner.resume = lambda *args, **kwargs: None
    runner._save = lambda *args, **kwargs: None
    return runner


def _freeze_time(monkeypatch, start: float, row_time: float) -> None:
    values = iter([start, row_time, row_time + 1.0])

    def fake_time() -> float:
        try:
            return next(values)
        except StopIteration:
            return row_time + 1.0

    monkeypatch.setattr(mappo_runner_module.time, "time", fake_time)


@pytest.mark.parametrize("cls,label", [(MAPPORunner, "MAPPO"), (HAPPORunner, "HAPPO")])
def test_log_interval_one_prints_update_validation_and_test_progress(cls, label: str, tmp_path: Path, capsys, monkeypatch) -> None:
    _freeze_time(monkeypatch, 100.0, 110.0)
    runner = _fake_runner(cls, tmp_path, log_interval=1)
    cls._run_impl(runner)
    output = capsys.readouterr().out
    assert f"[{label} update 0001]" in output
    assert f"[{label} eval:validation]" in output
    assert f"[{label} eval:test_initial]" in output
    assert f"[{label} eval:test_last]" in output
    assert f"[{label} eval:test_best]" in output


@pytest.mark.parametrize("cls", [MAPPORunner, HAPPORunner])
def test_log_interval_zero_suppresses_progress_logs(cls, tmp_path: Path, capsys, monkeypatch) -> None:
    _freeze_time(monkeypatch, 100.0, 110.0)
    runner = _fake_runner(cls, tmp_path, log_interval=0)
    cls._run_impl(runner)
    assert "[" not in capsys.readouterr().out


@pytest.mark.parametrize("cls", [MAPPORunner, HAPPORunner])
def test_samples_per_second_uses_only_new_steps_after_resume(cls, tmp_path: Path, monkeypatch) -> None:
    _freeze_time(monkeypatch, 100.0, 110.0)
    runner = _fake_runner(cls, tmp_path, log_interval=0)
    cls._run_impl(runner)
    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["samples_per_second"]) == pytest.approx(0.2)


def test_happo_uses_shared_logging_module_not_mappo_runner_logging_functions() -> None:
    source = Path("src/uav_env/algorithms/happo/runner.py").read_text(encoding="utf-8")
    assert "from uav_env.algorithms.common.progress_logging import format_evaluation_log, format_training_log" in source
    assert "from uav_env.algorithms.mappo.runner import format_evaluation_log" not in source
    assert "from uav_env.algorithms.mappo.runner import format_training_log" not in source
