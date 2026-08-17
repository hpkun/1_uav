import copy
import numpy as np
import torch
from torch import nn

from uav_combat.madsac import AttentionCritic, MADSACTrainer, ReplayBuffer, SharedSquashedGaussianActor


def transition(value=0.0, done=False):
    observation = np.full((4, 45), value, dtype=np.float32)
    action = np.zeros((4, 3), dtype=np.float32)
    reward = np.ones(4, dtype=np.float32)
    mask = np.ones(4, dtype=np.float32)
    return observation, action, reward, observation + .1, done, mask, mask


def test_section4_actor_has_two_256_hidden_layers_and_is_shared():
    actor = SharedSquashedGaussianActor()
    linear_layers = [layer for layer in actor.backbone if isinstance(layer, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [(45, 256), (256, 256)]
    observations = torch.randn(3, 4, 45)
    actions, log_probability = actor.sample(observations)
    assert actions.shape == (3, 4, 3) and log_probability.shape == (3, 4)
    assert torch.isfinite(actions).all() and torch.isfinite(log_probability).all()


def test_equations16_17_two_head_attention_and_independent_critics():
    critic1, critic2 = AttentionCritic(), AttentionCritic()
    assert critic1.attention_heads == critic2.attention_heads == 2
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(critic1.parameters(), critic2.parameters()))
    observations = torch.randn(2, 4, 45, requires_grad=True)
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
        "next_observations": torch.zeros(1, 4, 45),
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


def test_replay_stores_done_and_dead_masks_for_equation18():
    replay = ReplayBuffer(capacity=4, chunk_size=2)
    data = transition(done=True)
    replay.push(*data[:5], alive_masks=data[5], next_alive_masks=np.zeros(4, dtype=np.float32))
    batch = replay.sample(1, np.random.default_rng(1))
    assert batch["dones"].item() == 1
    assert batch["next_alive_masks"].sum().item() == 0


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
