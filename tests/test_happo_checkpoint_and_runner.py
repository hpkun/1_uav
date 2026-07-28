from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from uav_env.algorithms.happo.checkpoint import load_happo_checkpoint, save_happo_checkpoint
from uav_env.algorithms.happo.config import load_happo_config
from uav_env.algorithms.happo.networks import IndependentActorSet, JointCentralizedCritic
from uav_env.algorithms.happo.runner import HAPPORunner
from uav_env.algorithms.happo.trainer import HAPPOTrainer
from uav_env.algorithms.mappo.value_normalizer import ValueNormalizer


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


def test_happo_checkpoint_roundtrip(tmp_path: Path) -> None:
    cfg = _small_cfg()
    actors = IndependentActorSet([3, 3, 3], [2, 2, 2], [8], seed=5)
    critic = JointCentralizedCritic(4, [8])
    normalizer = ValueNormalizer()
    trainer = HAPPOTrainer(actors, critic, cfg, normalizer, torch.device("cpu"))
    metadata = {"environment_schema_version": "x", "observation_schema": "o", "global_state_schema": "g", "reward_profile": "r", "scenario_profile": "s", "obs_dim": 3, "state_dim": 4, "num_agents": 3}
    path = tmp_path / "happo.pt"
    runner_state = {
        "agent_order_rng_state": trainer.order_rng.bit_generator.state,
        "actor_minibatch_rng_states": [rng.bit_generator.state for rng in trainer.actor_minibatch_rngs],
        "critic_minibatch_rng_state": trainer.critic_minibatch_rng.bit_generator.state,
        "vector_env_state": [],
        "current": {},
        "episodes": 0,
        "episode_team_return_accumulators": np.zeros(1),
        "episode_agent_sum_return_accumulators": np.zeros(1),
    }
    save_happo_checkpoint(path, actors, critic, trainer.actor_optimizers, trainer.critic_optimizer, normalizer, cfg, 7, 2, None, runner_state, metadata)
    restored = IndependentActorSet([3, 3, 3], [2, 2, 2], [8], seed=6)
    restored_critic = JointCentralizedCritic(4, [8])
    restored_trainer = HAPPOTrainer(restored, restored_critic, cfg, ValueNormalizer(), torch.device("cpu"))
    data = load_happo_checkpoint(path, restored, restored_critic, restored_trainer.actor_optimizers, restored_trainer.critic_optimizer, ValueNormalizer(), False, "cpu", metadata)
    assert data["algorithm"] == "happo"
    for a, b in zip(actors[0].parameters(), restored[0].parameters()):
        assert torch.allclose(a, b)


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
    finally:
        runner.close()
