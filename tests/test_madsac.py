import copy
import numpy as np
import pytest
import torch
from torch import nn

from uav_combat.madsac import AttentionCritic, MADSACTrainer, ReplayBuffer, SharedSquashedGaussianActor
from uav_combat.madsac.trainer import (
    batch_mean_agent_sum, joint_actions_with_own_gradient, masked_slot_mean,
)


def transition(value=0.0, done=False):
    observation = np.full((4, 52), value, dtype=np.float32)
    action = np.zeros((4, 3), dtype=np.float32)
    reward = np.ones(4, dtype=np.float32)
    mask = np.ones(4, dtype=np.float32)
    return observation, action, reward, observation + .1, done, mask, mask


def test_section4_actor_has_two_256_hidden_layers_and_is_shared():
    actor = SharedSquashedGaussianActor()
    linear_layers = [layer for layer in actor.backbone if isinstance(layer, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [(52, 256), (256, 256)]
    observations = torch.randn(3, 4, 52)
    actions, log_probability = actor.sample(observations)
    assert actions.shape == (3, 4, 3) and log_probability.shape == (3, 4)
    assert torch.isfinite(actions).all() and torch.isfinite(log_probability).all()


def test_deterministic_action_is_exactly_tanh_distribution_mean():
    actor = SharedSquashedGaussianActor()
    observations = torch.randn(7, 4, 52)
    assert torch.equal(
        actor.deterministic(observations),
        torch.tanh(actor.distribution(observations).mean),
    )


def test_madsac_policy_statistics_are_finite_and_distinguish_action_moments():
    trainer = MADSACTrainer(hidden_dim=32, attention_heads=2)
    observations = np.random.default_rng(3).normal(size=(5, 4, 52)).astype(np.float32)
    masks = np.ones((5, 4), dtype=np.float32)
    metrics = trainer.policy_statistics(observations, masks)
    expected = {
        f"{prefix}_{name}"
        for prefix in (
            "deterministic_action_mean", "deterministic_action_abs_mean",
            "policy_log_std_mean",
        )
        for name in ("psi", "theta", "v")
    }
    assert metrics.keys() == expected
    assert all(np.isfinite(value) for value in metrics.values())


def test_equation19_20_batch_mean_agent_sum_reduction():
    values = torch.ones(2, 4)
    mixed = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.float32)
    assert batch_mean_agent_sum(values, mixed).item() == 2.5
    assert batch_mean_agent_sum(values, torch.ones(2, 4)).item() == 4.0
    assert batch_mean_agent_sum(values, torch.tensor([[1, 0, 0, 0]] * 2, dtype=torch.float32)).item() == 1.0
    assert batch_mean_agent_sum(values, torch.zeros(2, 4)).item() == 0.0
    assert masked_slot_mean(values, torch.ones(2, 4)).item() == 1.0


class ScalarCritic(nn.Module):
    def __init__(self, value=1.0):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, observations, actions, alive_mask=None):
        return self.value.expand(observations.shape[:2])


@pytest.mark.parametrize("alive_count,expected_loss", [(4, 4.0), (2, 2.0)])
def test_actual_equation19_critic_loss_sums_agents(alive_count, expected_loss):
    trainer = MADSACTrainer(hidden_dim=32, attention_heads=2, batch_size=1, replay_capacity=4)
    trainer.critic1, trainer.critic2 = ScalarCritic(), ScalarCritic()
    trainer.critic1_optimizer = torch.optim.SGD(trainer.critic1.parameters(), lr=0.01)
    trainer.critic2_optimizer = torch.optim.SGD(trainer.critic2.parameters(), lr=0.01)
    trainer.compute_target = lambda batch: torch.zeros_like(batch["rewards"])
    observation, action, reward, next_observation, done, _, _ = transition()
    alive = np.array([1] * alive_count + [0] * (4 - alive_count), dtype=np.float32)
    trainer.replay.push(observation, action, reward, next_observation, done, alive, alive)
    metrics = trainer.update_critics()
    assert metrics["critic1_loss"] == expected_loss
    assert metrics["critic2_loss"] == expected_loss


class ControlledActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.0))

    def sample(self, observations):
        actions = self.value * torch.ones((*observations.shape[:-1], 3), device=observations.device)
        log_prob = self.value * torch.ones(observations.shape[:-1], device=observations.device)
        return actions, log_prob


class ZeroActionCritic(nn.Module):
    def forward(self, observations, actions, alive_mask=None):
        return actions[..., 0] * 0.0


def test_actual_equation20_actor_objective_sums_four_agent_terms():
    trainer = MADSACTrainer(hidden_dim=32, attention_heads=2, batch_size=1, replay_capacity=4, alpha=1.0)
    trainer.actor = ControlledActor()
    trainer.actor_optimizer = torch.optim.SGD(trainer.actor.parameters(), lr=0.01)
    trainer.critic1 = ZeroActionCritic()
    trainer.critic2 = ZeroActionCritic()
    trainer.replay.push(*transition())
    metrics = trainer.update_actor()
    assert metrics["actor_loss"] == 4.0


def test_equations16_17_two_head_attention_and_independent_critics():
    critic1, critic2 = AttentionCritic(), AttentionCritic()
    assert critic1.attention_heads == critic2.attention_heads == 2
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(critic1.parameters(), critic2.parameters()))
    observations = torch.randn(2, 4, 52, requires_grad=True)
    actions = torch.randn(2, 4, 3, requires_grad=True)
    q_values, attention = critic1(observations, actions, return_attention=True)
    assert q_values.shape == (2, 4) and attention.shape == (2, 2, 4, 4)
    assert torch.allclose(torch.diagonal(attention, dim1=-2, dim2=-1), torch.zeros(2, 2, 4))
    q_values.sum().backward()
    assert critic1.wq.weight.grad is not None


class ConstantActor(nn.Module):
    def sample(self, observations):
        return torch.zeros((*observations.shape[:-1], 3), device=observations.device), torch.full(observations.shape[:-1], .2, device=observations.device)


class ConstantCritic(nn.Module):
    def __init__(self, value):
        super().__init__(); self.value = value
    def forward(self, observations, actions, alive_mask=None):
        return torch.full(observations.shape[:2], self.value, device=observations.device)


def test_equation18_uses_target_actor_min_double_q_and_entropy():
    trainer = MADSACTrainer(hidden_dim=32, attention_heads=2, batch_size=1, replay_capacity=4)
    trainer.target_actor = ConstantActor()
    trainer.target_critic1 = ConstantCritic(2.0)
    trainer.target_critic2 = ConstantCritic(5.0)
    batch = {
        "next_observations": torch.zeros(1, 4, 52),
        "next_alive_masks": torch.ones(1, 4),
        "rewards": torch.ones(1, 4),
        "dones": torch.zeros(1, 1),
    }
    expected = 1.0 + .99 * (2.0 - .1 * .2)
    assert torch.allclose(trainer.compute_target(batch), torch.full((1, 4), expected))


def test_equations19_to21_critic_and_actor_updates_are_finite_and_separate():
    trainer = MADSACTrainer(hidden_dim=32, attention_heads=2, batch_size=4, replay_capacity=32)
    for i in range(8):
        trainer.replay.push(*transition(i / 10))
    actor_before = copy.deepcopy(next(trainer.actor.parameters()).detach())
    critic_before = copy.deepcopy(next(trainer.critic1.parameters()).detach())
    critic_metrics = trainer.update_critics()
    assert not torch.equal(critic_before, next(trainer.critic1.parameters()))
    assert torch.equal(actor_before, next(trainer.actor.parameters()))
    actor_metrics = trainer.update_actor()
    assert not torch.equal(actor_before, next(trainer.actor.parameters()))
    trainer.update_targets()
    assert trainer.critic_update_count == trainer.actor_update_count == trainer.target_update_count == 1
    assert all(np.isfinite(value) for value in critic_metrics.values())
    assert all(np.isfinite(value) for value in actor_metrics.values())


def test_equation21_agent_term_detaches_every_other_action_path():
    actions = torch.randn(2, 4, 3, requires_grad=True)
    agent_index, other_index = 1, 3
    joint = joint_actions_with_own_gradient(actions, agent_index)
    controlled_q_i = joint[:, agent_index].sum() + 17.0 * joint[:, other_index].sum()
    gradient = torch.autograd.grad(controlled_q_i, actions)[0]
    assert torch.count_nonzero(gradient[:, agent_index]) > 0
    assert torch.count_nonzero(gradient[:, other_index]) == 0
    assert torch.count_nonzero(gradient[:, 0]) == 0
    assert torch.count_nonzero(gradient[:, 2]) == 0


def test_shared_actor_gradient_sums_all_and_only_alive_own_action_terms():
    actions = torch.randn(1, 4, 3, requires_grad=True)
    alive = torch.tensor([1.0, 0.0, 1.0, 1.0])
    terms = []
    for agent_index in range(4):
        joint = joint_actions_with_own_gradient(actions, agent_index)
        terms.append(alive[agent_index] * joint[:, agent_index].sum())
    gradient = torch.autograd.grad(torch.stack(terms).sum(), actions)[0]
    assert torch.count_nonzero(gradient[:, 0]) == 3
    assert torch.count_nonzero(gradient[:, 1]) == 0
    assert torch.count_nonzero(gradient[:, 2]) == 3
    assert torch.count_nonzero(gradient[:, 3]) == 3


def test_actor_update_with_one_live_agent_is_finite():
    trainer = MADSACTrainer(hidden_dim=32, attention_heads=2, batch_size=2, replay_capacity=8)
    one_alive = np.array([1, 0, 0, 0], dtype=np.float32)
    for value in (0.0, 0.1):
        observation, action, reward, next_observation, done, _, _ = transition(value)
        trainer.replay.push(
            observation, action, reward, next_observation, done,
            alive_masks=one_alive, next_alive_masks=one_alive,
        )
    metrics = trainer.update_actor()
    assert all(np.isfinite(value) for value in metrics.values())


def test_replay_stores_done_and_dead_masks_for_equation18():
    replay = ReplayBuffer(capacity=4, chunk_size=2)
    data = transition(done=True)
    replay.push(*data[:5], alive_masks=data[5], next_alive_masks=np.zeros(4, dtype=np.float32))
    batch = replay.sample(1, np.random.default_rng(1))
    assert batch["dones"].item() == 1
    assert batch["next_alive_masks"].sum().item() == 0


def test_replay_and_equation18_preserve_distinct_local_rewards():
    trainer = MADSACTrainer(hidden_dim=32, attention_heads=2, batch_size=1, replay_capacity=4)
    observation, action, _, next_observation, done, mask, _ = transition()
    local_rewards = np.array([10.0, 0.5, -10.0, -0.25], dtype=np.float32)
    trainer.replay.push(observation, action, local_rewards, next_observation, done, mask, np.zeros(4, dtype=np.float32))
    batch = trainer.replay.sample(1, np.random.default_rng(0))
    assert torch.allclose(batch["rewards"][0], torch.tensor(local_rewards))
    # No surviving next agent means Eq. (18) is exactly the per-agent r_i vector.
    assert torch.allclose(trainer.compute_target(batch)[0], torch.tensor(local_rewards))


def test_checkpoint_is_small_and_does_not_copy_replay(tmp_path):
    trainer = MADSACTrainer(hidden_dim=32, attention_heads=2, batch_size=1, replay_capacity=8)
    trainer.replay.push(*transition())
    state = trainer.checkpoint_state({"scheduler_T": 24})
    assert "replay" not in state and state["extra"]["scheduler_T"] == 24
    path = tmp_path / "checkpoint.pt"
    trainer.save(path)
    restored = MADSACTrainer(hidden_dim=32, attention_heads=2, batch_size=1, replay_capacity=8)
    restored.load(path)
    assert restored.replay.size == 0
    assert all(torch.equal(a, b) for a, b in zip(trainer.actor.parameters(), restored.actor.parameters()))
