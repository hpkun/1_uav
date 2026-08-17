import copy
import numpy as np
import torch
import pytest

from uav_combat.madsac import AttentionCritic,MADSACTrainer,ReplayBuffer
from uav_combat.madsac.trainer import masked_mean


def fields(n=1):
    o=np.random.default_rng(1).normal(size=(n,4,45)).astype(np.float32); a=np.ones((n,4,3),np.float32); r=np.ones((n,4),np.float32); m=np.array([[1,0,1,0]],np.float32).repeat(n,0); d=np.zeros((n,1),bool)
    return o,a,r,o+.1,d,m,m.copy()


def test_dead_executed_action_and_replay_action_zero():
    trainer=MADSACTrainer(hidden_dim=32,replay_capacity=8,batch_size=1); o,a,r,no,d,m,nm=fields()
    executed=trainer.act(o[0],m[0]); assert np.all(executed[m[0]==0]==0)
    replay=ReplayBuffer(8); replay.push(o[0],a[0],r[0],no[0],False,m[0],nm[0]); batch=replay.sample(1,np.random.default_rng(2))
    assert torch.all(batch["actions"][0,m[0]==0]==0)


def test_attention_dead_keys_zero_and_all_other_dead_finite():
    critic=AttentionCritic(hidden_dim=32,attention_heads=2); o=torch.randn(2,4,45); a=torch.randn(2,4,3); mask=torch.tensor([[1,0,1,0],[1,0,0,0]],dtype=torch.float32)
    q,w=critic(o,a,mask,return_attention=True); assert torch.isfinite(q).all() and torch.isfinite(w).all()
    assert torch.all(w[0,:,:,1]==0) and torch.all(w[0,:,:,3]==0) and torch.all(w[1]==0)


def test_masked_actor_and_critic_mean_excludes_dead():
    values=torch.tensor([[1.,100.,3.,100.]],requires_grad=True); mask=torch.tensor([[1.,0.,1.,0.]])
    loss=masked_mean(values,mask); assert loss.item()==2; loss.backward(); assert torch.equal(values.grad,torch.tensor([[.5,0,.5,0]]))


def test_one_live_and_random_death_mask_updates_finite():
    trainer=MADSACTrainer(hidden_dim=32,replay_capacity=32,batch_size=4,policy_delay=1)
    rng=np.random.default_rng(3)
    for i in range(8):
        o,a,r,no,d,m,nm=fields(); m=np.zeros(4,np.float32); m[i%4]=1; nm=rng.integers(0,2,size=4).astype(np.float32); trainer.replay.push(o[0],a[0],r[0],no[0],False,m,nm)
    metrics=trainer.update(); assert all(np.isfinite(v) for v in metrics.values() if isinstance(v,float))


@pytest.mark.parametrize("mask",[
    [1,1,1,1],[1,1,0,0],[1,0,1,0],[0,0,0,1],[0,0,0,0],
])
def test_random_mask_patterns_never_nan(mask):
    trainer=MADSACTrainer(hidden_dim=32,replay_capacity=8,batch_size=2,policy_delay=1)
    o,a,r,no,d,_,_=fields(2); m=np.asarray(mask,np.float32)
    for i in range(2): trainer.replay.push(o[i],a[i],r[i],no[i],False,m,m)
    metrics=trainer.update(); assert all(np.isfinite(v) for v in metrics.values() if isinstance(v,float))


def test_dead_next_agent_has_no_bootstrap():
    trainer=MADSACTrainer(hidden_dim=32,replay_capacity=4,batch_size=1); o,a,r,no,d,m,nm=fields(); nm[:]=0; trainer.replay.push(o[0],a[0],r[0],no[0],False,m[0],nm[0])
    batch=trainer.replay.sample(1,np.random.default_rng(1)); assert torch.allclose(trainer.compute_target(batch),batch["rewards"])


def test_all_live_mask_matches_unmasked_attention():
    critic=AttentionCritic(hidden_dim=32); o=torch.randn(2,4,45); a=torch.randn(2,4,3)
    assert torch.allclose(critic(o,a),critic(o,a,torch.ones(2,4)))


def test_replay_batch_wrap_roundtrip_and_rng_determinism():
    replay=ReplayBuffer(5,chunk_size=2); data=fields(7); replay.push_batch(*data)
    assert replay.size==5 and replay.position==2
    restored=ReplayBuffer(5,chunk_size=2); restored.load_state_dict(replay.state_dict())
    one=replay.sample(3,np.random.default_rng(9)); two=restored.sample(3,np.random.default_rng(9))
    assert all(torch.equal(one[k],two[k]) for k in one) and one["alive_masks"].shape==(3,4)


def test_checkpoint_replay_rng_steps_and_resumed_update(tmp_path):
    trainer=MADSACTrainer(hidden_dim=32,replay_capacity=16,batch_size=2,policy_delay=1,seed=7)
    data=fields(4); trainer.replay.push_batch(*data); trainer.sampled_env_steps=4; trainer.vector_steps=2; trainer.update(); path=tmp_path/"full.pt"; trainer.save(path,True)
    expected=trainer.rng.random(); restored=MADSACTrainer(hidden_dim=32,replay_capacity=16,batch_size=2,policy_delay=1,seed=99); restored.load(path,True)
    assert restored.sampled_env_steps==4 and restored.vector_steps==2 and restored.replay.size==4 and restored.rng.random()==expected
    assert all(np.isfinite(v) for v in restored.update().values() if isinstance(v,float))
