from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch

from algorithm.common.interception_checkpoint import read, restore, save
from algorithm.common.interception_vector import InterceptionVectorEnv
from algorithm.mappo.gae import gae
from algorithm.mappo.interception import MAPPO, TrainConfig, clean_central
from env.interception import InterceptionConfig


def test_gae_timeout_bootstraps_but_does_not_link_episodes():
    values = torch.tensor([[[1., float("nan")]]] * 3)
    rewards = torch.tensor([[[1., float("nan")]], [[2., float("nan")]], [[3., float("nan")]]])
    next_values = torch.tensor([[[float("nan"), 0]], [[2., 0]], [[4., 0]]])
    mask = torch.tensor([[[True, False]]] * 3)
    advantage, returns = gae(rewards, values, next_values, mask, mask,
                             torch.tensor([[True],[False],[False]]),
                             torch.tensor([[False],[True],[False]]), .5, 1.)
    torch.testing.assert_close(advantage[:, 0], torch.tensor([[0.,0.],[2.,0.],[4.,0.]]))
    assert torch.isfinite(returns).all()


def test_central_padding_cleanup():
    state = torch.zeros(54)
    state[:6] = float("nan")
    state[7:13] = 2.
    state[13] = 1.
    clean = clean_central(state)
    assert torch.isfinite(clean).all()
    assert (clean[:7] == 0).all() and (clean[7:13] == 2.).all()


def test_cuda_rollout_update_and_exact_resume(tmp_path):
    config = InterceptionConfig()
    tc = TrainConfig(workers=2, horizon=4, epochs=1)
    model = MAPPO(tc)
    with InterceptionVectorEnv(config, tc.workers, tc.seed) as vector:
        assert len(set(vector.worker_pids)) == 2
        assert not np.array_equal(vector.observations[0], vector.observations[1])
        assert vector._seed(0).entropy[0] == 0
        assert not np.array_equal(vector._seed(0).generate_state(4),
                                  np.random.SeedSequence([1, tc.seed]).generate_state(4))
        rollout, _ = model.collect(vector)
        assert rollout.observations.shape == (4, 2, 5, 52)
        assert rollout.values.shape == (4, 2, 5)
        assert ((rollout.actions > 0) & (rollout.actions < 1)).all()
        # Pad invalid slots with NaNs. Only real actors can affect the update.
        rollout.observations[~rollout.current_masks] = float("nan")
        result = model.update(rollout)
        assert result["valid_samples"] == 24
        assert all(np.isfinite(list(result.values())))
        assert (model.critic[-1].weight.grad[3:] == 0).all()
        save(tmp_path/"state.pt", model, config, vector)
        payload = read(tmp_path/"state.pt")
        expected, _ = model.collect(vector)
        model.update(expected)
        with pytest.raises(ValueError, match="configuration mismatch"):
            restore(payload, model, replace(config, pursuer_tau=1.6), vector)
        restore(payload, model, config, vector)
        actual, _ = model.collect(vector)
        for key in vars(expected):
            torch.testing.assert_close(getattr(actual, key), getattr(expected, key), rtol=0, atol=0)
        before = {key: value.clone() for key,value in model.actor.state_dict().items()}
        model.update(actual)
        # Run the same restored step again and verify both optimizers' effects.
        after = {key: value.clone() for key,value in model.actor.state_dict().items()}
        critic = {key: value.clone() for key,value in model.critic.state_dict().items()}
        assert any(not torch.equal(value, after[key]) for key, value in before.items())
        restore(payload, model, config, vector)
        repeat, _ = model.collect(vector)
        model.update(repeat)
        for network, weights in ((model.actor, after), (model.critic, critic)):
            for key,value in network.state_dict().items():
                torch.testing.assert_close(value, weights[key], rtol=0, atol=0)


def test_legacy_checkpoint_is_rejected():
    path = Path("archive/legacy/tests/fixtures/formal_5v5_before_8v8.pt")
    with pytest.raises(ValueError, match="incompatible checkpoint"):
        read(path)


def test_vector_uses_actual_terminal_state_before_reset():
    config = InterceptionConfig(scale="1v1", dt=10., max_steps=1,
                                evader_mode="constant_velocity", constant_direction=(0,0,1),
                                evader_speed_fraction=.1)
    with InterceptionVectorEnv(config, 2, 17) as vector:
        result = vector.step(np.full((2,5,3), .5))
        assert result.truncated.all() and not result.terminated.any()
        np.testing.assert_array_equal(result.transition_states[:, -1], 0.)
        np.testing.assert_array_equal(result.states[:, -1], 1.)
        assert all(e["outcome"] == "timeout" for e in result.episodes)
