import copy
import numpy as np
import torch

from uav_combat.madsac import AttentionCritic, MADSACTrainer, ReplayBuffer, SharedSquashedGaussianActor


def transition(value=0):
    o=np.full((4,45),value,np.float32); a=np.zeros((4,3),np.float32); r=np.ones(4,np.float32)
    return o,a,r,o+1,False


def test_replay_push_sample_wrap_shapes_dtype():
    replay=ReplayBuffer(capacity=3,chunk_size=2)
    for i in range(5): replay.push(*transition(i))
    assert replay.size==3 and replay.position==2
    batch=replay.sample(2,np.random.default_rng(1))
    assert batch["observations"].shape==(2,4,45) and batch["actions"].dtype==torch.float32


def test_shared_actor_and_finite_squashed_log_probability():
    actor=SharedSquashedGaussianActor(); obs=torch.randn(8,4,45); actions, logp=actor.sample(obs)
    assert actions.shape==(8,4,3) and logp.shape==(8,4) and torch.isfinite(logp).all()
    assert actor(obs[:,0]).shape == (8,3)


def test_double_critics_independent_attention_shape_and_gradient():
    c1,c2=AttentionCritic(),AttentionCritic()
    assert all(p1.data_ptr()!=p2.data_ptr() for p1,p2 in zip(c1.parameters(),c2.parameters()))
    o,a=torch.randn(3,4,45,requires_grad=True),torch.randn(3,4,3,requires_grad=True)
    q,w=c1(o,a,return_attention=True); assert q.shape==(3,4) and w.shape==(3,2,4,4)
    q.sum().backward(); assert c1.wq.weight.grad is not None and torch.isfinite(c1.wq.weight.grad).all()


def test_target_minimum_q_and_one_update_finite_gradients():
    trainer=MADSACTrainer(hidden_dim=32,attention_heads=2,replay_capacity=64,batch_size=4,policy_delay=1)
    for i in range(8): trainer.replay.push(*transition(i/10))
    actor_before=copy.deepcopy(next(trainer.actor.parameters()).detach()); critic_before=copy.deepcopy(next(trainer.critic1.parameters()).detach()); target_before=copy.deepcopy(next(trainer.target_actor.parameters()).detach())
    metrics=trainer.update()
    assert all(np.isfinite(v) for v in metrics.values() if isinstance(v,float))
    assert not torch.equal(actor_before,next(trainer.actor.parameters())) and not torch.equal(critic_before,next(trainer.critic1.parameters())) and not torch.equal(target_before,next(trainer.target_actor.parameters()))
    batch=trainer.replay.sample(4,np.random.default_rng(2)); target=trainer.compute_target(batch)
    assert target.shape==(4,4) and torch.isfinite(target).all()


def test_short_end_to_end_smoke():
    from uav_combat.environment import PaperUAVCombatEnv
    env=PaperUAVCombatEnv(sensor_noise=False); obs,_=env.reset(1)
    trainer=MADSACTrainer(hidden_dim=32,replay_capacity=64,batch_size=4,policy_delay=1)
    for _ in range(6):
        actions=trainer.act(obs); next_obs,rewards,term,trunc,_=env.step(actions); trainer.replay.push(obs,actions,rewards,next_obs,term or trunc); obs=next_obs
    metrics=trainer.update(); assert metrics["actor_updated"] and metrics["target_updated"]


def test_checkpoint_roundtrip(tmp_path):
    trainer=MADSACTrainer(hidden_dim=32,replay_capacity=16,batch_size=2)
    path=tmp_path/"checkpoint.pt"; trainer.save(path)
    restored=MADSACTrainer(hidden_dim=32,replay_capacity=16,batch_size=2); restored.load(path)
    assert all(torch.equal(a,b) for a,b in zip(trainer.actor.parameters(),restored.actor.parameters()))


def test_vector_environment_contract():
    from uav_combat.training import SyncVectorEnv
    vector=SyncVectorEnv(2,base_seed=5); obs=vector.reset()
    result=vector.step(np.zeros((2,4,3),np.float32))
    assert obs.shape==(2,4,45) and result[0].shape==(2,4,45) and result[1].shape==(2,4)
