"""Small, actual CUDA combat acceptance run; never launches formal training."""
import argparse
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from env.aircombat import AirCombatConfig,AirCombatEnv
from env.aircombat.config import ENVIRONMENT_VERSION
from algorithm.aircombat import train,evaluate
from algorithm.mappo.config import ProtocolTrainConfig
from algorithm.mappo.trainer import ProtocolMAPPOTrainer
from algorithm.common.runtime import require_cuda
from algorithm.common.aircombat_vector import AirCombatVectorEnv,CombatEpisodeAccumulator
from algorithm.common.checkpoint import save_protocol_checkpoint,read_protocol_checkpoint,restore_protocol_checkpoint


def equal_tree(a,b):
    if isinstance(a,torch.Tensor): torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray): np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal_tree(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal_tree(x,y)
    elif dataclasses.is_dataclass(a):
        for f in dataclasses.fields(a):equal_tree(getattr(a,f.name),getattr(b,f.name))
    elif hasattr(a,'__dict__'): equal_tree(vars(a),vars(b))
    else: assert a==b


def episode(config, seed, scenario=None, *, random_actions=False):
    env=AirCombatEnv(config)
    obs,info=env.reset(seed,{'scenario':scenario} if scenario else None)
    tracker=CombatEpisodeAccumulator();tracker.reset(info)
    h=hashlib.sha256()
    rng=np.random.default_rng(seed+100000)
    while True:
        # Fixed Table-I steady-forward command is a validation input, never a
        # controller used by the environment/training policy.
        actions=(rng.uniform(.3,.7,(config.num_uavs,3)) if random_actions else
                 np.tile([.5,1/3,1/8],(config.num_uavs,1)))
        obs,r,term,trunc,info=env.step(actions)
        tracker.add(info,r)
        assert np.isfinite(obs).all() and np.isfinite(r).all()
        np.testing.assert_array_equal(r,sum(info['reward_components'].values()))
        for array in (obs,r,env.state.friendly_states,env.state.adversarial_states,
                      env.state.friendly_alive,env.state.adversarial_alive,env.state.assignments,
                      env.state.friendly_dwell,env.state.adversarial_dwell):
            h.update(array.tobytes())
        if term or trunc: break
    return dict(**tracker.summary(info),terminated=term,truncated=trunc,
                numerical_counters=info['numerical_counters'],trajectory_sha256=h.hexdigest(),
                scenario=scenario or 'random_training',seed=seed,
                action_source='uniform(.3,.7) numerical stress' if random_actions else 'fixed Table I steady-forward test input')


def validate(output):
    require_cuda('cuda')
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('use a new acceptance output directory')
    output.mkdir(parents=True,exist_ok=True)
    result=dict(device=torch.cuda.get_device_name(0),torch_version=torch.__version__,environment_version=ENVIRONMENT_VERSION)
    result['complete_episodes']={scale:episode(AirCombatConfig(scale=scale),771) for scale in ('3v3','5v5','3v5')}
    assert all(not r['numerical_failures'] for r in result['complete_episodes'].values())
    result['numerical_stress']=episode(AirCombatConfig(),771,random_actions=True)
    result['deterministic_3v3']=episode(AirCombatConfig(),772,'head_on_3v3')
    assert not result['deterministic_3v3']['numerical_failures']
    repeated=episode(AirCombatConfig(),771)
    assert repeated==result['complete_episodes']['3v3']
    result['fixed_seed_repeat_exact']=True
    # Every scale gets real sample/update checks on the paper's normal 250-step env.
    result['real_ppo']={}
    for scale in ('3v3','5v5','3v5'):
        ec=AirCombatConfig(scale=scale)
        c=ProtocolTrainConfig(num_envs=2,horizon=4,ppo_epochs=1,minibatch_size=32)
        t=ProtocolMAPPOTrainer(c)
        with AirCombatVectorEnv(ec,2) as v:
            r,_=t.collect(v)
            before=copy.deepcopy(t.actor.state_dict())
            metric=t.update(r)
            assert any(not torch.equal(before[k],p) for k,p in t.actor.state_dict().items())
            result['real_ppo'][scale]=dict(rollout=t.rollout_metrics,ppo=metric)
            path=output/f'{scale}_resume.pt'
            save_protocol_checkpoint(path,t,env_config=ec,vector=v)
            payload=read_protocol_checkpoint(path,environment_version=ENVIRONMENT_VERSION)
            r1,e1=t.collect(v);m1=t.update(r1)
            expected=dict(actor=copy.deepcopy(t.actor.state_dict()),critic=copy.deepcopy(t.critic.state_dict()),
                          actor_optimizer=copy.deepcopy(t.actor_optimizer.state_dict()),
                          critic_optimizer=copy.deepcopy(t.critic_optimizer.state_dict()),vector=v.snapshot())
        # Fresh spawned workers, not just in-place restoration.
        t2=ProtocolMAPPOTrainer(c)
        with AirCombatVectorEnv(ec,2) as v2:
            restore_protocol_checkpoint(payload,t2,env_config=ec,vector=v2)
            r2,e2=t2.collect(v2);m2=t2.update(r2)
            equal_tree(r1,r2);equal_tree(e1,e2);equal_tree(m1,m2)
            actual=dict(actor=t2.actor.state_dict(),critic=t2.critic.state_dict(),
                        actor_optimizer=t2.actor_optimizer.state_dict(),
                        critic_optimizer=t2.critic_optimizer.state_dict(),vector=v2.snapshot())
            equal_tree(expected,actual)
        result['real_ppo'][scale]['fresh_worker_resume_exact']=True
    c=ProtocolTrainConfig(num_envs=2,horizon=32,ppo_epochs=1,minibatch_size=128,
                          training_episodes=2,checkpoint_interval_episodes=1)
    t=train(AirCombatConfig(),c,output/'short_train')
    result['short_train']=dict(completed_episodes=t.training_episode_count,sampled_steps=t.sampled_steps,updates=t.updates)
    payload=read_protocol_checkpoint(output/'short_train'/'latest.pt',environment_version=ENVIRONMENT_VERSION)
    restored=ProtocolMAPPOTrainer(c)
    restore_protocol_checkpoint(payload,restored,env_config=AirCombatConfig())
    result['checkpoint_evaluation']=evaluate(restored,AirCombatConfig(),[10000000,10000001])
    result['performance_reproduction_claim']=False
    target=output/'acceptance.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(report=str(target.resolve()),short_train=result['short_train'],
        episodes=result['complete_episodes'],resume_all_scales=True),ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='outputs/xu_aircombat_acceptance')
    args=p.parse_args()
    validate(args.output)


if __name__=='__main__':main()
