import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

from uav_combat.mappo import (
    CentralizedValueCritic, MAPPOTrainer, RolloutBatch, SharedMAPPOActor,
    compute_gae,
)
from uav_combat.training.mappo_runner import MAPPOTrainingRunner


ROOT = Path(__file__).resolve().parents[1]


def configs():
    environment = yaml.safe_load(
        (ROOT / "configs/combat_environment.yaml").read_text(encoding="utf-8")
    )
    algorithm = yaml.safe_load(
        (ROOT / "configs/mappo.yaml").read_text(encoding="utf-8")
    )
    return environment, algorithm


def test_mappo_actor_matches_madsac_network_size_and_log_prob_round_trip():
    actor = SharedMAPPOActor()
    linear = [layer for layer in actor.backbone if isinstance(layer, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear] == [
        (52, 256), (256, 256)
    ]
    observations = torch.randn(3, 4, 52)
    actions, sampled_log_prob, entropy = actor.sample(observations)
    evaluated_log_prob, evaluated_entropy = actor.evaluate_actions(
        observations, actions
    )
    assert actions.shape == (3, 4, 3)
    assert sampled_log_prob.shape == entropy.shape == (3, 4)
    assert torch.allclose(sampled_log_prob, evaluated_log_prob, atol=2e-4)
    assert torch.allclose(entropy, evaluated_entropy)


def test_centralized_value_critic_uses_two_head_attention_and_masks_dead_agents():
    critic = CentralizedValueCritic(hidden_dim=32, attention_heads=2)
    observations = torch.randn(2, 4, 52, requires_grad=True)
    masks = torch.tensor([[1, 1, 1, 1], [1, 0, 1, 0]], dtype=torch.float32)
    values, attention = critic(observations, masks, return_attention=True)
    assert values.shape == (2, 4)
    assert attention.shape == (2, 2, 4, 4)
    assert torch.count_nonzero(values[1, [1, 3]]) == 0
    assert torch.allclose(
        torch.diagonal(attention, dim1=-2, dim2=-1), torch.zeros(2, 2, 4)
    )
    values.sum().backward()
    assert critic.wq.weight.grad is not None


def test_gae_stops_at_agent_death_and_environment_done():
    rewards = torch.tensor([[[1.0]], [[2.0]], [[4.0]]])
    values = torch.zeros_like(rewards)
    next_values = torch.full_like(rewards, 10.0)
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    alive = torch.ones_like(rewards)
    next_alive = torch.tensor([[[1.0]], [[1.0]], [[0.0]]])
    advantages, returns = compute_gae(
        rewards, values, next_values, dones, alive, next_alive,
        gamma=1.0, gae_lambda=1.0,
    )
    assert torch.allclose(advantages[:, 0, 0], torch.tensor([13.0, 2.0, 4.0]))
    assert torch.equal(advantages, returns)


def test_mappo_update_is_finite_and_changes_both_networks():
    trainer = MAPPOTrainer(
        hidden_dim=32, attention_heads=2, ppo_epochs=2, minibatch_size=4,
    )
    rng = np.random.default_rng(4)
    observations = rng.normal(size=(3, 2, 4, 52)).astype(np.float32)
    masks = np.ones((3, 2, 4), dtype=np.float32)
    actions, log_probs = [], []
    for step in range(3):
        action, log_prob = trainer.act(
            observations[step], masks[step], return_log_prob=True
        )
        actions.append(action); log_probs.append(log_prob)
    rollout = RolloutBatch(
        observations=observations,
        actions=np.stack(actions),
        old_log_probs=np.stack(log_probs),
        rewards=rng.normal(size=(3, 2, 4)).astype(np.float32),
        dones=np.zeros((3, 2), dtype=np.float32),
        alive_masks=masks,
        next_observations=observations + 0.01,
        next_alive_masks=masks,
    )
    actor_before = copy.deepcopy(next(trainer.actor.parameters()).detach())
    critic_before = copy.deepcopy(next(trainer.critic.parameters()).detach())
    metrics = trainer.update(rollout)
    assert not torch.equal(actor_before, next(trainer.actor.parameters()))
    assert not torch.equal(critic_before, next(trainer.critic.parameters()))
    assert all(np.isfinite(value) for value in metrics.values())
    assert trainer.ppo_update_count == 1
    assert trainer.actor_update_count == trainer.critic_update_count == 4


def test_mappo_runner_collects_true_parallel_on_policy_rollout(tmp_path):
    environment, algorithm = configs()
    environment = copy.deepcopy(environment)
    environment["simulation"]["max_steps"] = 2
    runner = MAPPOTrainingRunner(
        environment, algorithm, num_envs=2, total_sampled_steps=8,
        output_dir=tmp_path, smoke=True,
    )
    try:
        rollout = runner.collect_rollout(2)
        assert rollout.observations.shape == (2, 2, 4, 52)
        assert rollout.actions.shape == (2, 2, 4, 3)
        assert runner.trainer.sampled_steps == 4
        assert runner.vector.num_workers == 2
        assert len(set(runner.vector.worker_pids)) == 2
    finally:
        runner.vector.close()


def test_mappo_checkpoint_restores_algorithm_and_counters(tmp_path):
    trainer = MAPPOTrainer(hidden_dim=32, attention_heads=2)
    trainer.sampled_steps = 123
    path = tmp_path / "mappo.pt"
    trainer.save(path, {"environment_version": "2.1"})
    restored = MAPPOTrainer(hidden_dim=32, attention_heads=2)
    extra = restored.load(path)
    assert restored.sampled_steps == 123
    assert extra["environment_version"] == "2.1"
    assert all(torch.equal(a, b) for a, b in zip(
        trainer.actor.parameters(), restored.actor.parameters()
    ))
