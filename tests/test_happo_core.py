from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from uav_env.algorithms.happo.networks import IndependentActorSet, JointCentralizedCritic
from uav_env.algorithms.happo.rollout_buffer import HAPPORolloutBuffer, compute_scalar_gae
from uav_env.algorithms.happo.trainer import HAPPOTrainer, happo_policy_loss, update_happo_factor
from uav_env.algorithms.mappo.value_normalizer import ValueNormalizer


def test_independent_actor_parameters_do_not_share_storage() -> None:
    actors = IndependentActorSet([4, 4, 4], [3, 3, 3], [8], seed=7)
    assert actors[0] is not actors[1] and actors[1] is not actors[2]
    ptrs = [{p.data_ptr() for p in actor.parameters()} for actor in actors.actors]
    assert ptrs[0].isdisjoint(ptrs[1])
    assert ptrs[0].isdisjoint(ptrs[2])
    before_1 = [p.detach().clone() for p in actors[1].parameters()]
    before_2 = [p.detach().clone() for p in actors[2].parameters()]
    opt = torch.optim.Adam(actors[0].parameters(), lr=0.01)
    logits = actors[0](torch.randn(5, 4), torch.ones(5, 3, dtype=torch.bool))
    loss = logits.square().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert all(torch.equal(a, b) for a, b in zip(before_1, actors[1].parameters()))
    assert all(torch.equal(a, b) for a, b in zip(before_2, actors[2].parameters()))


def test_order_rng_fixed_random_and_checkpoint_state() -> None:
    cfg = {
        "seed": 3,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "fixed_agent_order": True,
    }
    actors = IndependentActorSet([4, 4, 4], [3, 3, 3], [8], seed=3)
    trainer = HAPPOTrainer(actors, JointCentralizedCritic(5, [8]), cfg, ValueNormalizer(), torch.device("cpu"))
    assert trainer.next_update_order() == [0, 1, 2]
    cfg["fixed_agent_order"] = False
    a = HAPPOTrainer(actors, JointCentralizedCritic(5, [8]), cfg, ValueNormalizer(), torch.device("cpu"))
    b = HAPPOTrainer(actors, JointCentralizedCritic(5, [8]), cfg, ValueNormalizer(), torch.device("cpu"))
    assert a.next_update_order() == b.next_update_order()
    state = copy.deepcopy(a.order_rng.bit_generator.state)
    next_a = a.next_update_order()
    a.order_rng.bit_generator.state = state
    assert a.next_update_order() == next_a


def test_happo_factor_math_oracle_and_inactive_ratio_one() -> None:
    factor = torch.ones(2, 2)
    old0 = torch.zeros(2, 2)
    new0 = torch.log(torch.tensor([[2.0, 0.5], [1.5, 1.0]]))
    active0 = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    after0 = update_happo_factor(factor, new0, old0, active0)
    expected0 = torch.tensor([[2.0, 0.5], [1.0, 1.0]])
    assert torch.allclose(after0, expected0)
    new1 = torch.log(torch.tensor([[3.0, 2.0], [0.25, 4.0]]))
    active1 = torch.ones(2, 2)
    after1 = update_happo_factor(after0, new1, old0, active1)
    assert torch.allclose(after1, expected0 * torch.exp(new1))


def test_happo_policy_loss_matches_equation_11_positive_and_negative_advantage() -> None:
    old = torch.zeros(4)
    new = torch.log(torch.tensor([1.3, 0.7, 1.5, 0.5]))
    advantage = torch.tensor([2.0, 2.0, -2.0, -2.0])
    factor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    active = torch.ones(4)
    loss, ratio, _, _ = happo_policy_loss(new, old, advantage, active, factor, 0.2)
    clipped = torch.clamp(ratio, 0.8, 1.2)
    expected = -(factor * torch.minimum(ratio * advantage, clipped * advantage)).mean()
    assert torch.allclose(loss, expected)


def test_inactive_samples_do_not_create_nan_or_actor_update() -> None:
    cfg = {
        "seed": 1,
        "actor_lr": 0.01,
        "critic_lr": 0.01,
        "clip_param": 0.2,
        "value_clip_param": 0.2,
        "ppo_epochs": 1,
        "actor_num_mini_batches": 1,
        "critic_epochs": 1,
        "critic_num_mini_batches": 1,
        "entropy_coef": 0.01,
        "value_loss_coef": 1.0,
        "max_grad_norm": 10.0,
        "normalize_advantages": True,
        "use_value_normalization": False,
        "use_clipped_value_loss": True,
        "use_huber_loss": True,
        "huber_delta": 10.0,
        "fixed_agent_order": True,
    }
    actors = IndependentActorSet([3, 3, 3], [2, 2, 2], [8], seed=1)
    trainer = HAPPOTrainer(actors, JointCentralizedCritic(4, [8]), cfg, ValueNormalizer(), torch.device("cpu"))
    buffer = HAPPORolloutBuffer(2, 1, 3, 3, 4, action_dim=2)
    buffer.available_action_masks[:] = True
    buffer.alive_masks[:-1, :, 0] = 0.0
    buffer.advantages[:] = 1.0
    buffer.returns[:] = 1.0
    before = [p.detach().clone() for p in actors[0].parameters()]
    metrics = trainer.update(buffer)
    assert np.isfinite([v for v in metrics.values() if isinstance(v, float)]).all()
    assert all(torch.equal(a, b) for a, b in zip(before, actors[0].parameters()))
    assert metrics["factor_update_count"] == 3.0


def test_factor_detach_prevents_previous_actor_gradient() -> None:
    factor_source = torch.tensor([2.0, 3.0], requires_grad=True)
    old = torch.zeros(2)
    new = torch.zeros(2, requires_grad=True)
    loss, _, _, _ = happo_policy_loss(new, old, torch.ones(2), torch.ones(2), factor_source, 0.2)
    loss.backward()
    assert factor_source.grad is None
    assert new.grad is not None


def test_scalar_gae_uses_team_reward_not_per_agent_rewards() -> None:
    rewards = np.asarray([[1.0]], dtype=np.float32)
    values = np.asarray([[2.0], [0.0]], dtype=np.float32)
    terminated = np.asarray([[False]])
    truncated = np.asarray([[True]])
    terminal = np.asarray([[10.0]], dtype=np.float32)
    no_bootstrap = np.asarray([[0.0]], dtype=np.float32)
    advantages, returns = compute_scalar_gae(rewards, values, terminated, truncated, terminal, no_bootstrap, 0.99, 0.95)
    assert advantages.shape == (1, 1)
    assert returns.shape == (1, 1)
    assert advantages[0, 0] == pytest.approx(-1.0)
    critic = JointCentralizedCritic(61)
    assert critic(torch.zeros(2, 61)).shape == (2,)
