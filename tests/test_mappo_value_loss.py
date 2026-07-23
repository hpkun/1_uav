import pytest
import torch

from uav_env.algorithms.mappo.trainer import ppo_value_loss


def test_value_clipping_switch_changes_hand_value():
    new, old, target, mask = torch.tensor([2.0]), torch.tensor([0.0]), torch.tensor([3.0]), torch.ones(1)
    unclipped = ppo_value_loss(new, old, target, mask, .2, False, False, 1.0)
    clipped = ppo_value_loss(new, old, target, mask, .2, True, False, 1.0)
    assert unclipped == pytest.approx(.5)
    assert clipped == pytest.approx(.5 * 2.8**2)


def test_huber_mse_and_critic_mask():
    new = torch.tensor([3.0, 1000.0]); zero = torch.zeros(2); mask = torch.tensor([1.0, 0.0])
    huber = ppo_value_loss(new, zero, zero, mask, .2, False, True, 2.0)
    mse = ppo_value_loss(new, zero, zero, mask, .2, False, False, 2.0)
    assert huber == pytest.approx(4.0)
    assert mse == pytest.approx(4.5)
