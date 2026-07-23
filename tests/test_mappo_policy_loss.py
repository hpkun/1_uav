import pytest
import torch

from uav_env.algorithms.mappo.trainer import masked_mean, ppo_policy_loss


def test_ratio_one_and_masked_sample():
    new = old = torch.zeros(3)
    advantages = torch.tensor([2.0, 4.0, 1000.0])
    mask = torch.tensor([1.0, 1.0, 0.0])
    loss, ratio, clipped, fraction, kl = ppo_policy_loss(new, old, advantages, mask, .2)
    assert loss == pytest.approx(-3.0)
    assert torch.equal(ratio, clipped)
    assert fraction == 0.0 and kl == 0.0


def test_positive_upper_and_negative_lower_clipping():
    old = torch.zeros(2)
    new = torch.log(torch.tensor([2.0, .2]))
    advantage = torch.tensor([1.0, -1.0])
    loss, *_ = ppo_policy_loss(new, old, advantage, torch.ones(2), .2)
    assert loss == pytest.approx(-(.2))  # mean of min(2,1.2)=1.2 and min(-.2,-.8)=-.8


def test_all_zero_actor_mask_is_finite_zero():
    loss, _, _, fraction, kl = ppo_policy_loss(torch.ones(2), torch.zeros(2), torch.ones(2), torch.zeros(2), .2)
    assert loss == 0 and fraction == 0 and kl == 0
    assert masked_mean(torch.tensor([float("nan")]), torch.zeros(1)).isnan()  # NaN inputs remain diagnosable
