import numpy as np
import torch
from algorithm.mappo.trainer import MAPPOTrainer,RolloutBatch
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer
from algorithm.modular_mappo.buffer import ModularRolloutBatch,contiguous_chunks

def data(seed=4):
 r=np.random.default_rng(seed);T,E,A,F=4,2,4,52;obs=r.normal(size=(T,E,A,F)).astype("f");actions=np.tanh(r.normal(size=(T,E,A,3))).astype("f");raw=np.arctanh(np.clip(actions,-.999,.999));alive=np.ones((T,E,A),"f");dones=np.zeros((T,E),"f");rew=r.normal(size=(T,E,A)).astype("f")
 base=dict(observations=obs,actions=actions,raw_actions=raw,old_log_probs=np.zeros((T,E,A),"f"),rewards=rew,dones=dones,alive_masks=alive,next_observations=obs.copy(),next_alive_masks=alive.copy())
 return base
def test_all_off_network_initialization_and_actions_equal_baseline():
 b=MAPPOTrainer(hidden_dim=32,ppo_epochs=1,minibatch_size=8,seed=7);m=ModularMAPPOTrainer(hidden_dim=32,ppo_epochs=1,minibatch_size=8,seed=7,modules_config={});assert all(torch.equal(v,m.actor.state_dict()[k]) for k,v in b.actor.state_dict().items());assert all(torch.equal(v,m.critic.state_dict()[k]) for k,v in b.critic.state_dict().items());x=np.random.default_rng(1).normal(size=(2,4,52)).astype("f");assert np.allclose(b.act(x,deterministic=True),m.act(x,deterministic=True)[0])
def test_context_critic_only_actor_invariant():
 m=ModularMAPPOTrainer(hidden_dim=16,modules_config={"wave_context":{"enabled":True,"context_target":"critic_only"}});x=np.ones((1,4,52),"f");a1=m.act(x,np.ones((1,4)),True,context=m.context_numpy([1],[3]))[0];a3=m.act(x,np.ones((1,4)),True,context=m.context_numpy([3],[3]))[0];assert np.array_equal(a1,a3)
def test_recurrent_history_changes_output_and_chunks_contiguous():
 m=ModularMAPPOTrainer(hidden_dim=16,modules_config={"recurrent_memory":{"enabled":True,"hidden_dim":8,"sequence_length":2}});x=np.ones((1,4,52),"f");alive=np.ones((1,4),"f");h,_=m.initial_hidden(1);a1,h1=m.act(x,alive,True,hidden=h,episode_mask=np.ones(1));a2,_=m.act(x,alive,True,hidden=h1,episode_mask=np.ones(1));assert not np.allclose(a1,a2);assert contiguous_chunks(5,1,2)==[(0,0,2),(0,2,4),(0,4,5)]
def test_modular_update_checkpoint_protocol(tmp_path):
 d=data();m=ModularMAPPOTrainer(hidden_dim=16,ppo_epochs=1,minibatch_size=8,modules_config={});ctx=np.zeros((4,2,0),"f");r=ModularRolloutBatch(**d,raw_environment_rewards=d["rewards"].copy(),wave_indices=np.ones((4,2),int),total_waves=np.ones((4,2),int),contexts=ctx,next_contexts=ctx,episode_masks=np.ones((4,2),"f"));metrics=m.update(r);assert all(np.isfinite(list(metrics.values())));p=tmp_path/"x.pt";m.save(p);m.load(p);bad=ModularMAPPOTrainer(hidden_dim=16,modules_config={"popart":{"enabled":True}});import pytest
 with pytest.raises(RuntimeError):bad.load(p)

def test_all_off_update_is_numerically_baseline_equivalent():
 d=data(9);b=MAPPOTrainer(hidden_dim=16,ppo_epochs=1,minibatch_size=8,seed=3);m=ModularMAPPOTrainer(hidden_dim=16,ppo_epochs=1,minibatch_size=8,seed=3,modules_config={});obs=torch.as_tensor(d["observations"])
 with torch.no_grad():
  dist=b.actor.distribution(obs);d["old_log_probs"]=b.actor._squashed_log_prob(dist,torch.as_tensor(d["raw_actions"]),torch.as_tensor(d["actions"])).numpy()
 rb=RolloutBatch(**d);ctx=np.zeros((4,2,0),"f");rm=ModularRolloutBatch(**d,raw_environment_rewards=d["rewards"].copy(),wave_indices=np.ones((4,2),int),total_waves=np.ones((4,2),int),contexts=ctx,next_contexts=ctx,episode_masks=np.ones((4,2),"f"));state=torch.random.get_rng_state();b.update(rb);torch.random.set_rng_state(state);m.update(rm)
 for p,q in zip(b.actor.parameters(),m.actor.parameters()):assert torch.allclose(p,q,atol=2e-6,rtol=2e-6)
 for p,q in zip(b.critic.parameters(),m.critic.parameters()):assert torch.allclose(p,q,atol=2e-6,rtol=2e-6)
