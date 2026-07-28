from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from uav_env.algorithms.happo.checkpoint import load_happo_checkpoint, save_happo_checkpoint
from uav_env.algorithms.happo.config import load_happo_config, validate_happo_config
from uav_env.algorithms.happo.networks import IndependentActorSet, JointCentralizedCritic
from uav_env.algorithms.happo.runner import HAPPORunner, REWARD_COMPONENT_NAMES
from uav_env.algorithms.happo.trainer import HAPPOTrainer
from uav_env.algorithms.mappo.metrics import evaluation_key
from uav_env.algorithms.mappo.value_normalizer import ValueNormalizer
from uav_env.combat.events import EpisodeOutcome


def _small_cfg() -> dict:
    cfg = load_happo_config("configs/happo_base.yaml")
    cfg.update(
        {
            "seed": 11,
            "device": "cpu",
            "num_envs": 1,
            "vector_env": "sync",
            "rollout_length": 2,
            "total_env_steps": 2,
            "evaluation_interval": 999999,
            "validation_episodes": 1,
            "test_episodes": 1,
            "ppo_epochs": 1,
            "actor_num_mini_batches": 1,
            "critic_epochs": 1,
            "critic_num_mini_batches": 1,
        }
    )
    return cfg


def _metadata(obs_dim: int = 3, state_dim: int = 4, num_agents: int = 3) -> dict:
    return {
        "environment_schema_version": "homogeneous_3v3_v2_timeaware",
        "observation_schema": "fixed_slots_timeaware_v2",
        "global_state_schema": "fixed_slots_timeaware_v2",
        "reward_profile": "paper_2024_exact",
        "scenario_profile": "head_on_learnability_v1",
        "obs_dim": obs_dim,
        "state_dim": state_dim,
        "num_agents": num_agents,
    }


def _assert_state_dict_close(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        av, bv = a[key], b[key]
        if isinstance(av, torch.Tensor):
            assert torch.allclose(av, bv)
        elif isinstance(av, dict):
            _assert_state_dict_close(av, bv)
        elif isinstance(av, list):
            assert len(av) == len(bv)
            for item_a, item_b in zip(av, bv):
                if isinstance(item_a, dict):
                    _assert_state_dict_close(item_a, item_b)
                elif isinstance(item_a, torch.Tensor):
                    assert torch.allclose(item_a, item_b)
                else:
                    assert item_a == item_b
        else:
            assert av == bv


def test_happo_checkpoint_roundtrip(tmp_path: Path) -> None:
    cfg = _small_cfg()
    actors = IndependentActorSet([3, 3, 3], [2, 2, 2], [8], seed=5)
    critic = JointCentralizedCritic(4, [8])
    normalizer = ValueNormalizer()
    trainer = HAPPOTrainer(actors, critic, cfg, normalizer, torch.device("cpu"))
    for agent_id, optimizer in enumerate(trainer.actor_optimizers):
        loss = actors[agent_id](torch.randn(4, 3), torch.ones(4, 2, dtype=torch.bool)).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    critic_loss = critic(torch.randn(4, 4)).square().mean()
    trainer.critic_optimizer.zero_grad()
    critic_loss.backward()
    trainer.critic_optimizer.step()
    normalizer.update(torch.tensor([1.0, 2.0, 3.0]))
    metadata = _metadata()
    path = tmp_path / "happo.pt"
    rng_state = trainer.order_rng.bit_generator.state
    expected_next_order = trainer.next_update_order()
    trainer.order_rng.bit_generator.state = rng_state
    runner_state = {
        "agent_order_rng_state": trainer.order_rng.bit_generator.state,
        "actor_minibatch_rng_states": [rng.bit_generator.state for rng in trainer.actor_minibatch_rngs],
        "critic_minibatch_rng_state": trainer.critic_minibatch_rng.bit_generator.state,
        "vector_env_state": [],
        "current": {},
        "episodes": 0,
        "episode_team_return_accumulators": np.zeros(1),
        "episode_agent_sum_return_accumulators": np.zeros(1),
        "reward_component_episode_accumulators": {name: np.asarray([float(index)]) for index, name in enumerate(REWARD_COMPONENT_NAMES)},
    }
    save_happo_checkpoint(path, actors, critic, trainer.actor_optimizers, trainer.critic_optimizer, normalizer, cfg, 7, 2, None, runner_state, metadata)
    restored = IndependentActorSet([3, 3, 3], [2, 2, 2], [8], seed=6)
    restored_critic = JointCentralizedCritic(4, [8])
    restored_normalizer = ValueNormalizer()
    restored_trainer = HAPPOTrainer(restored, restored_critic, cfg, restored_normalizer, torch.device("cpu"))
    data = load_happo_checkpoint(path, restored, restored_critic, restored_trainer.actor_optimizers, restored_trainer.critic_optimizer, restored_normalizer, False, "cpu", metadata)
    assert data["algorithm"] == "happo"
    for agent_id in range(3):
        for a, b in zip(actors[agent_id].parameters(), restored[agent_id].parameters()):
            assert torch.allclose(a, b)
        _assert_state_dict_close(trainer.actor_optimizers[agent_id].state_dict(), restored_trainer.actor_optimizers[agent_id].state_dict())
    _assert_state_dict_close(critic.state_dict(), restored_critic.state_dict())
    _assert_state_dict_close(trainer.critic_optimizer.state_dict(), restored_trainer.critic_optimizer.state_dict())
    _assert_state_dict_close(normalizer.state_dict(), restored_normalizer.state_dict())
    assert data["runner_state"]["reward_component_episode_accumulators"]["terminal_reward"].shape == (1,)
    restored_trainer.order_rng.bit_generator.state = data["runner_state"]["agent_order_rng_state"]
    assert restored_trainer.next_update_order() == expected_next_order


def test_happo_checkpoint_rejects_wrong_algorithm_schema_and_shapes(tmp_path: Path) -> None:
    cfg = _small_cfg()
    actors = IndependentActorSet([3, 3, 3], [2, 2, 2], [8], seed=5)
    critic = JointCentralizedCritic(4, [8])
    normalizer = ValueNormalizer()
    trainer = HAPPOTrainer(actors, critic, cfg, normalizer, torch.device("cpu"))
    path = tmp_path / "happo.pt"
    save_happo_checkpoint(path, actors, critic, trainer.actor_optimizers, trainer.critic_optimizer, normalizer, cfg, 0, 0, None, {}, _metadata())

    bad_algo = tmp_path / "mappo.pt"
    torch.save({"algorithm": "mappo"}, bad_algo)
    with pytest.raises(ValueError, match="MAPPO conversion is unsupported"):
        load_happo_checkpoint(bad_algo, actors)

    with pytest.raises(ValueError, match="actor count mismatch"):
        load_happo_checkpoint(path, IndependentActorSet([3, 3], [2, 2], [8], seed=1))

    changed_scenario = {**_metadata(), "scenario_profile": "different"}
    with pytest.raises(ValueError, match="scenario_profile"):
        load_happo_checkpoint(path, actors, critic, trainer.actor_optimizers, trainer.critic_optimizer, normalizer, False, "cpu", changed_scenario)

    changed_actor_only_allowed = {**_metadata(), "scenario_profile": "different", "reward_profile": "different", "global_state_schema": "different"}
    load_happo_checkpoint(path, actors, actor_only=True, expected_metadata=changed_actor_only_allowed)

    changed_obs_schema = {**_metadata(), "observation_schema": "different"}
    with pytest.raises(ValueError, match="observation_schema"):
        load_happo_checkpoint(path, actors, actor_only=True, expected_metadata=changed_obs_schema)

    incompatible = IndependentActorSet([4, 3, 3], [2, 2, 2], [8], seed=2)
    with pytest.raises(ValueError, match="actor_0 dimensions"):
        load_happo_checkpoint(path, incompatible, actor_only=True, expected_metadata=_metadata(obs_dim=3))


def test_happo_short_collect_update_on_v2_environment(tmp_path: Path) -> None:
    cfg = _small_cfg()
    runner = HAPPORunner(cfg, "pytest_happo", output_root=tmp_path)
    try:
        buffer, rollout = runner.collect()
        assert buffer.team_rewards.shape == (2, 1)
        assert buffer.advantages.shape == (2, 1)
        assert np.isfinite(buffer.advantages).all()
        metrics = runner.trainer.update(buffer)
        assert metrics["factor_update_count"] == 3.0
        assert "actor_0_policy_loss" in metrics
        assert np.isfinite([v for v in metrics.values() if isinstance(v, float)]).all()
        assert "team_reward_mean" in rollout
        assert "agent_reward_sum_mean" in rollout
        for name in REWARD_COMPONENT_NAMES:
            assert f"{name}_mean" in rollout
            assert f"{name}_per_step" in rollout
            assert f"{name}_abs_mean" in rollout
            assert f"{name}_per_episode" in rollout
    finally:
        runner.close()


def test_happo_final_validation_test_loop_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _small_cfg()
    cfg.update({"total_env_steps": 0, "validation_seed_start": 222, "test_seed_start": 333, "run_id": "final_loop"})
    runner = HAPPORunner(cfg, "pytest_happo", output_root=tmp_path)
    calls = []

    def fake_evaluate(episodes: int | None = None, seed_start: int = 100000, deterministic: bool | None = None) -> dict[str, float]:
        calls.append((episodes, seed_start, deterministic))
        return {
            "overall_red_win_rate": 0.0,
            "elimination_win_rate": 0.0,
            "timeout_rate": 1.0,
            "red_crash_rate": 0.0,
            "blue_crash_rate": 0.0,
            "mean_episode_return": -1.0,
            "mean_team_episode_return": -1.0,
            "mean_effective_damage": 0.0,
            "mean_survivor_difference": 0.0,
            "mean_hits": 0.0,
            "mean_attack_area_steps": 0.0,
        }

    monkeypatch.setattr(runner, "evaluate", fake_evaluate)
    out = runner.run()
    summary = yaml.safe_load((out / "final_summary.yaml").read_text(encoding="utf-8"))
    assert (out / "checkpoints" / "initial.pt").is_file()
    assert (out / "checkpoints" / "last.pt").is_file()
    assert (out / "checkpoints" / "best.pt").is_file()
    assert set(summary["test_evaluations"]) == {"initial", "last", "best"}
    assert summary["wall_time"] >= 0.0
    assert calls.count((1, 222, None)) == 1
    assert calls.count((1, 333, True)) == 3


def test_happo_evaluation_aliases_and_ground_crash_rate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import uav_env.algorithms.happo.runner as runner_module

    cfg = _small_cfg()
    cfg["run_id"] = "fake_eval"
    runner = HAPPORunner(cfg, "pytest_happo", output_root=tmp_path)

    class FakeAdapter:
        num_agents = 3
        obs_dim = runner.obs_dim
        state_dim = runner.state_dim

        def __init__(self) -> None:
            self.env = SimpleNamespace(close=lambda: None)
            self.episode = -1

        def reset(self, seed: int) -> SimpleNamespace:
            self.episode = seed - 50
            return SimpleNamespace(
                local_obs=np.zeros((3, runner.obs_dim), dtype=np.float32),
                available_action_mask=np.ones((3, 15), dtype=bool),
                agent_alive_mask=np.ones(3, dtype=bool),
            )

        def step(self, action: np.ndarray) -> SimpleNamespace:
            red_ground = 1 if self.episode == 0 else 0
            aircraft = {
                "red_0": {"ground_crashes": red_ground, "ceiling_violations": 1, "hits": 2, "effective_damage": 3.0, "attack_area_steps": 4},
                "red_1": {"ground_crashes": 0, "ceiling_violations": 0, "hits": 0, "effective_damage": 0.0, "attack_area_steps": 0},
                "red_2": {"ground_crashes": 0, "ceiling_violations": 0, "hits": 0, "effective_damage": 0.0, "attack_area_steps": 0},
                "blue_0": {"ground_crashes": 0, "ceiling_violations": 1, "hits": 0, "effective_damage": 0.0, "attack_area_steps": 0},
                "blue_1": {"ground_crashes": 0, "ceiling_violations": 0, "hits": 0, "effective_damage": 0.0, "attack_area_steps": 0},
                "blue_2": {"ground_crashes": 0, "ceiling_violations": 0, "hits": 0, "effective_damage": 0.0, "attack_area_steps": 0},
            }
            outcome = EpisodeOutcome("red", True, True, "timeout", 1, 0.5, 2, 1)
            return SimpleNamespace(
                team_reward=1.5,
                agent_reward_sum=4.5,
                terminated=False,
                truncated=True,
                info={"outcome": outcome, "statistics": {"aircraft": aircraft}},
            )

    monkeypatch.setattr(runner_module, "make_adapter_from_description", lambda description: FakeAdapter())
    try:
        result = runner.evaluate(episodes=2, seed_start=50, deterministic=True)
    finally:
        runner.close()
    assert result["mean_episode_return"] == pytest.approx(result["mean_team_episode_return"])
    assert result["mean_effective_damage"] == pytest.approx(result["mean_red_effective_damage"])
    assert result["mean_hits"] == pytest.approx(result["mean_red_hits"])
    assert result["mean_attack_area_steps"] == pytest.approx(result["mean_red_attack_area_steps"])
    assert result["mean_survivor_difference"] == pytest.approx(result["mean_red_survivors"] - result["mean_blue_survivors"])
    assert result["red_crash_rate"] == pytest.approx(0.5)
    assert result["blue_crash_rate"] == pytest.approx(0.0)
    assert result["mean_blue_ceiling_violations"] == pytest.approx(1.0)


def test_happo_evaluation_key_accepts_smoke_and_combat_aliases() -> None:
    result = {
        "overall_red_win_rate": 0.25,
        "elimination_win_rate": 0.5,
        "red_crash_rate": 0.0,
        "blue_crash_rate": 0.25,
        "mean_episode_return": 1.0,
        "mean_team_episode_return": 1.0,
        "mean_effective_damage": 3.0,
        "mean_hits": 2.0,
        "mean_attack_area_steps": 4.0,
        "mean_red_survivors": 2.0,
        "mean_blue_survivors": 1.0,
        "timeout_rate": 0.1,
    }
    assert evaluation_key(result, "smoke") == pytest.approx((0.25, -0.25, 1.0))
    assert evaluation_key(result, "combat")[2] == pytest.approx(3.0)
    assert evaluation_key(result, "combat")[3] == pytest.approx(1.0)


def test_happo_reward_component_accumulators_restore_shape(tmp_path: Path) -> None:
    cfg = _small_cfg()
    first = HAPPORunner(cfg, "pytest_happo", output_root=tmp_path)
    try:
        first.reward_component_episode_accumulators["situation_reward"][0] = 12.5
        first._save("accumulators.pt")
        restored = HAPPORunner(cfg, "pytest_happo_restore", output_root=tmp_path)
        try:
            restored.resume(str(first.output_dir / "checkpoints" / "accumulators.pt"))
            assert restored.reward_component_episode_accumulators["situation_reward"][0] == pytest.approx(12.5)
            bad = {name: np.zeros(2) for name in REWARD_COMPONENT_NAMES}
            with pytest.raises(ValueError, match="shape mismatch"):
                restored._restore_reward_component_accumulators(bad)
        finally:
            restored.close()
    finally:
        first.close()


def test_happo_keyboard_interrupt_saves_interrupted_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _small_cfg()
    cfg["run_id"] = "interrupt"
    runner = HAPPORunner(cfg, "pytest_happo", output_root=tmp_path)
    monkeypatch.setattr(runner, "_run_impl", lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        runner.run()
    path = runner.output_dir / "checkpoints" / "interrupted.pt"
    assert path.is_file()
    data = torch.load(path, map_location="cpu", weights_only=False)
    assert data["algorithm"] == "happo"
    assert runner._closed


def test_happo_config_optimizer_eps_and_weight_decay_validation() -> None:
    cfg = _small_cfg()
    assert cfg["optimizer_eps"] == pytest.approx(1.0e-5)
    assert cfg["weight_decay"] == pytest.approx(0.0)
    validate_happo_config(cfg)
    bad_eps = {**cfg, "optimizer_eps": 0.0}
    with pytest.raises(ValueError, match="optimizer_eps"):
        validate_happo_config(bad_eps)
    bad_decay = {**cfg, "weight_decay": -1.0}
    with pytest.raises(ValueError, match="weight_decay"):
        validate_happo_config(bad_decay)
