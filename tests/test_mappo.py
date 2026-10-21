import dataclasses
import json
import numpy as np
import pytest
import torch
from env import EnvConfig
from algorithm.mappo.config import TrainConfig
from algorithm.mappo.networks import Actor, Critic
from algorithm.mappo.trainer import MAPPOTrainer, Rollout, gae, require_cuda
from algorithm.common.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from algorithm.mappo.evaluation import evaluate
from algorithm.common.vector_env import ParallelVectorEnv
from tools.render_replay import render_replay


def test_network_dimensions_and_beta_probability():
    actor, critic = Actor(), Critic()
    obs = torch.randn(7, 12)
    action, logp = actor.act(obs)
    assert action.shape == (7, 3) and torch.all((action > 0) & (action < 1))
    assert critic(torch.randn(7, 90)).shape == (7, 5)
    dist = actor.distribution(obs)
    manual = ((dist.concentration1 - 1) * action.log() + (dist.concentration0 - 1) * torch.log1p(-action)
              + torch.lgamma(dist.concentration1 + dist.concentration0) - torch.lgamma(dist.concentration1) - torch.lgamma(dist.concentration0)).sum(-1)
    torch.testing.assert_close(logp, manual)
    logp2, entropy = actor.evaluate(obs, action)
    torch.testing.assert_close((logp2 - logp).exp(), torch.ones(7))
    torch.testing.assert_close((logp2 + np.log(1.1) - logp).exp(), torch.full((7,), 1.1))
    assert torch.isfinite(entropy).all()
    torch.testing.assert_close(actor.act(obs, deterministic=True)[0], dist.mean)


@pytest.mark.parametrize('terminated,truncated,expected', [(False, False, 2.5), (True, False, -2.), (False, True, 2.5)])
def test_gae_bootstrap(terminated, truncated, expected):
    values = torch.tensor([[[3.]]])
    adv, returns = gae(torch.tensor([[[1.]]]), values, torch.tensor([[[5.]]]), torch.ones_like(values), torch.ones_like(values),
                       torch.tensor([[terminated]]), torch.tensor([[truncated]]), .9, .95)
    assert adv.item() == pytest.approx(expected)
    assert returns.item() == pytest.approx(expected + 3)


def test_gae_failure_masks_and_episode_boundary():
    r = torch.tensor([[[2., 2.]], [[999., 4.]]])
    v, nv = torch.ones_like(r), torch.full_like(r, 100.)
    cur = torch.tensor([[[1., 1.]], [[0., 1.]]])
    nxt = torch.tensor([[[0., 1.]], [[0., 1.]]])
    done = torch.tensor([[False], [True]])
    trunc = torch.zeros_like(done)
    a, ret = gae(r, v, nv, cur, nxt, done, trunc, .9, .95)
    assert a[0, 0, 0] == 1 and a[1, 0, 0] == 0 and ret[1, 0, 0] == 0
    assert a[1, 0, 1] == 3
    assert a[0, 0, 1] == pytest.approx(91 + .9 * .95 * 3)
    trunc[0] = True
    b, _ = gae(r, v, nv, cur, nxt, done, trunc, .9, .95)
    assert b[0, 0, 1] == 91  # no recursion into reset episode


@pytest.mark.parametrize('scale', ['3U3P', '5U5P', '3U5P'])
def test_parallel_shapes_seed_independence_and_snapshot(scale):
    c = EnvConfig(scale=scale, max_steps=2)
    with ParallelVectorEnv(c, 2, seed=42) as vector:
        assert len(set(vector.worker_pids)) == 2
        assert vector.observations.shape == (2, c.num_uavs, 12)
        assert not np.array_equal(vector.observations[0], vector.observations[1])
        initial = vector.observations.copy()
        snapshot = vector.snapshot()
        action = np.tile([.5, 1/3, 1/8], (2, c.num_uavs, 1))
        first = vector.step(action)
        vector.restore(snapshot)
        np.testing.assert_array_equal(vector.observations, initial)
        again = vector.step(action)
        np.testing.assert_array_equal(first.observations, again.observations)
        last = vector.step(action)
        assert last.truncated.all()
        assert not np.array_equal(last.observations, last.transition_observations)
        assert last.current_masks.shape == last.next_masks.shape == (2, c.num_uavs)


def test_single_worker_and_failure_masks():
    with ParallelVectorEnv(EnvConfig(failure_events={1:[1]}), 1, seed=42) as vector:
        x = vector.step(np.tile([.5, 1/3, 1/8], (1, 3, 1)))
        assert x.current_masks[0, 1] and not x.next_masks[0, 1]
        assert not x.observations[0, 1].any()


def test_cuda_mandatory(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA is mandatory'):
        require_cuda()


def test_real_ppo_checkpoint_resume_evaluate_and_replay(tmp_path):
    require_cuda()
    cfg = TrainConfig(num_envs=1, horizon=5, ppo_epochs=2, minibatch_size=8)
    env_cfg = EnvConfig(max_steps=5, failure_events={2:[1]})
    trainer = MAPPOTrainer(cfg)
    with ParallelVectorEnv(env_cfg, 1, cfg.seed) as vector:
        rollout, summaries = trainer.collect(vector)
        assert len(summaries) == 1
        assert rollout.current_masks[1, 0, 1] == 1
        assert rollout.next_masks[1, 0, 1] == 0
        assert not rollout.current_masks[2:, 0, 1].any()
        before = [p.detach().clone() for p in trainer.actor.parameters()]
        metrics = trainer.update(rollout)
        assert metrics['optimizer_steps'] > 0
        assert metrics['valid_samples'] == 12
        assert any(not torch.equal(a, b) for a, b in zip(before, trainer.actor.parameters()))
        path = tmp_path / 'model.pt'
        save_checkpoint(path, trainer, env_cfg, vector, (0, 0, 0))
        payload = read_checkpoint(path)
        expected, _ = trainer.collect(vector)
        trainer.update(expected)
        clone = MAPPOTrainer(cfg)
        restore_checkpoint(payload, clone, env_cfg, vector)
        actual, _ = clone.collect(vector)
        for key in ('actions', 'observations', 'rewards', 'current_masks', 'next_masks'):
            torch.testing.assert_close(getattr(expected, key), getattr(actual, key), rtol=0, atol=0)
        clone.update(actual)
        for x, y in zip(trainer.actor.parameters(), clone.actor.parameters()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        for x, y in zip(trainer.critic.parameters(), clone.critic.parameters()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        for opt_a,opt_b in ((trainer.actor_optimizer,clone.actor_optimizer),(trainer.critic_optimizer,clone.critic_optimizer)):
            for slot_a,slot_b in zip(opt_a.state.values(),opt_b.state.values()):
                for key in slot_a:
                    torch.testing.assert_close(slot_a[key],slot_b[key],rtol=0,atol=0)
        report = evaluate(clone, env_cfg, [900000], tmp_path / 'trajectory.json')
        assert len(report['episodes']) == 1
        data = json.loads((tmp_path / 'trajectory.json').read_text())
        assert len(data['frames']) == 6
        output = render_replay(tmp_path / 'trajectory.json', tmp_path / 'replay.html')
        html = output.read_text(encoding='utf-8')
        assert 'Plotly.react' in html and 'UAV 状态查询' in html and 'cdn.plot.ly' not in html[:1000]
        with pytest.raises(ValueError, match='configuration mismatch'):
            restore_checkpoint(payload, clone, EnvConfig(scale='5U5P'))


def test_inactive_samples_cannot_affect_updates():
    require_cuda()
    c = TrainConfig(num_envs=1, horizon=4, ppo_epochs=1, minibatch_size=32)
    with ParallelVectorEnv(EnvConfig(failure_events={1:[1]}), 1, c.seed) as vector:
        trainer = MAPPOTrainer(c)
        batch, _ = trainer.collect(vector)
    valid = batch.current_masks.bool()
    fields = {f.name: getattr(batch, f.name).clone() for f in dataclasses.fields(batch)}
    fields['rewards'][~valid] = 1e20
    fields['log_probs'][~valid] = float('nan')
    fields['actions'][~valid] = float('nan')
    fields['observations'][~valid] = float('nan')
    modified = Rollout(**fields)
    a, b = MAPPOTrainer(c), MAPPOTrainer(c)
    torch.manual_seed(10)
    ma = a.update(batch)
    torch.manual_seed(10)
    mb = b.update(modified)
    assert ma == mb
    for pa, pb in zip(a.actor.parameters(), b.actor.parameters()):
        torch.testing.assert_close(pa, pb, rtol=0, atol=0)


def test_empty_valid_batch_skips_optimizer():
    require_cuda()
    c = TrainConfig(num_envs=1, horizon=1, ppo_epochs=1)
    trainer = MAPPOTrainer(c)
    with ParallelVectorEnv(EnvConfig(), 1, c.seed) as vector:
        rollout, _ = trainer.collect(vector)
    rollout.current_masks.zero_()
    before = [p.detach().clone() for p in trainer.actor.parameters()]
    metrics = trainer.update(rollout)
    assert metrics['optimizer_steps'] == 0
    assert trainer.updates == 0
    for a, b in zip(before, trainer.actor.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
