"""Focused fixed-scale input experiment checks, including one bounded CUDA smoke."""
import copy
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from algorithm.experiments import configurations,manifest,export_plan
from algorithm.mappo.config import ProtocolTrainConfig
from algorithm.mappo.networks import normalize_inputs
from algorithm.mappo.trainer import ProtocolMAPPOTrainer
from algorithm.mappo.diagnostics import diagnose_actor,diagnose_critic
from algorithm.common.checkpoint import save_protocol_checkpoint,read_protocol_checkpoint,restore_protocol_checkpoint
from algorithm.common.aircombat_vector import AirCombatVectorEnv
from algorithm.aircombat import train,evaluate
from env.aircombat import AirCombatEnv
from env.aircombat.config import ENVIRONMENT_VERSION
from env.dynamics import physical_actions
from tests.test_aircombat_training import tensor_equal


def test_fixed_12_and_60_scales_zero_slots_no_clipping_and_legacy_identity():
    row=torch.tensor([1,400,np.pi,-np.pi,np.pi,10000,200,-np.pi,np.pi,0,20000,30000],dtype=torch.float64)
    expected=torch.tensor([1,1,1,-1,1,1,.5,-1,1,0,2,3],dtype=torch.float64)
    torch.testing.assert_close(normalize_inputs(row,'fixed_scale_v1'),expected,rtol=0,atol=0)
    slots=torch.stack((row,torch.zeros_like(row),row,torch.zeros_like(row),row))
    torch.testing.assert_close(normalize_inputs(slots.flatten(),'fixed_scale_v1'),
        torch.stack((expected,torch.zeros_like(row),expected,torch.zeros_like(row),expected)).flatten(),rtol=0,atol=0)
    assert normalize_inputs(row) is row
    with pytest.raises(ValueError):ProtocolTrainConfig(observation_normalization='running')


def test_configuration_single_variable_and_paper_isolation(tmp_path):
    e,b=configurations('headon_xu_3v3',episodes=2500)
    en,n=configurations('headon_xu_norm_3v3',episodes=2500)
    assert e==en and replace(n,observation_normalization='none')==b
    assert n.num_envs==10 and n.horizon==1500 and n.evaluation_episodes==200
    assert not manifest('headon_xu_norm_3v3')['paper_experiment']
    assert len(export_plan(tmp_path/'plan.json')['training_runs'])==37


def test_cuda_input_response_diagnostics_and_training_checkpoint_smoke(tmp_path):
    ec,formal=configurations('headon_xu_norm_3v3',episodes=2500)
    raw_cfg=replace(formal,observation_normalization='none')
    raw=ProtocolMAPPOTrainer(raw_cfg);norm=ProtocolMAPPOTrainer(formal)
    # Same seed produces exactly the same initialized weights; only inputs differ.
    tensor_equal(raw.actor.state_dict(),norm.actor.state_dict())
    tensor_equal(raw.critic.state_dict(),norm.critic.state_dict())
    env=AirCombatEnv(ec);obs,info=env.reset(1000)
    before=env.snapshot();state=info['centralized_state'][None]
    d={}
    for name,t in (('raw',raw),('fixed',norm)):
        a=diagnose_actor(t.actor,obs,source='synthetic_physical_inputs')
        c=diagnose_critic(t.critic,state,source='synthetic_physical_inputs')
        d[name]=dict(actor=a['layers'][0]['saturation_fraction'],critic=c['layers'][0]['saturation_fraction'])
    tensor_equal(before,env.snapshot())
    # Default legacy path remains precisely the former unscaled forward.
    with torch.no_grad():
        x=raw.tensor(obs)
        p=torch.nn.functional.softplus(raw.actor.network(x))+1
        torch.testing.assert_close(raw.actor.distribution(x).mean,p[:,:3]/(p[:,:3]+p[:,3:]),rtol=0,atol=0)
        manual=torch.cat([h(raw.critic.network(raw.tensor(state))) for h in raw.critic.value_heads],-1)
        torch.testing.assert_close(raw.critic(raw.tensor(state)),manual,rtol=0,atol=0)
    variants=np.repeat(obs[:1],3,axis=0)
    variants[1,11]+=1000
    variants[2,[2,3,4,7,8,9]]+=.2
    actions,_=norm.sample_actions(variants,np.ones(3),deterministic=True)
    assert torch.isfinite(actions).all() and not torch.equal(actions[0],actions[1]) and not torch.equal(actions[0],actions[2])
    # Stochastic Beta controls as well as deterministic controls stay in range.
    sampled,_=norm.sample_actions(np.repeat(obs,50,axis=0),np.ones(150))
    controls=physical_actions(torch.cat((actions,sampled)).cpu().numpy())
    assert np.isfinite(controls).all()
    assert np.all(controls>=np.array([-np.pi,-1,0])) and np.all(controls<=np.array([np.pi,2,8]))
    # Explicit diagnostic budget; unchanged formal catalogue remains 2500/10/1500.
    cfg=replace(formal,training_episodes=2,num_envs=10,horizon=2,ppo_epochs=1,
                evaluation_episodes=2,checkpoint_interval_episodes=1)
    short_env=replace(ec,max_steps=2)
    t=train(short_env,cfg,tmp_path/'run',console=False)
    report=evaluate(t,short_env,[10010000,10010001])
    assert report['completed_episodes']==2 and report['numerical_failures']==0
    payload=read_protocol_checkpoint(tmp_path/'run/latest.pt',environment_version=ENVIRONMENT_VERSION)
    clone=ProtocolMAPPOTrainer(cfg)
    restore_protocol_checkpoint(payload,clone,env_config=short_env)
    torch.testing.assert_close(t.sample_actions(obs,info['active_mask'],deterministic=True)[0],
        clone.sample_actions(obs,info['active_mask'],deterministic=True)[0],rtol=0,atol=0)
    # Reject both restore directions, including a historical config missing field.
    for source_mode,target_mode in (('fixed_scale_v1','none'),('none','fixed_scale_v1')):
        p=copy.deepcopy(payload);p['train_config']['observation_normalization']=source_mode
        target=ProtocolMAPPOTrainer(replace(cfg,observation_normalization=target_mode))
        with pytest.raises(ValueError,match='configuration mismatch'):restore_protocol_checkpoint(p,target,env_config=short_env)
    old=copy.deepcopy(payload);old['train_config'].pop('observation_normalization')
    with pytest.raises(ValueError,match='configuration mismatch'):restore_protocol_checkpoint(old,clone,env_config=short_env)
    # Exact continuation of a real half-rollout checkpoint, including PPO/RNG.
    pending=read_protocol_checkpoint(tmp_path/'run/episodes_000001.pt',environment_version=ENVIRONMENT_VERSION)
    with AirCombatVectorEnv(short_env,10) as vector:
        resume=ProtocolMAPPOTrainer(cfg);restore_protocol_checkpoint(pending,resume,env_config=short_env,vector=vector)
        r,_=resume.collect(vector);metrics=resume.update(r)
    tensor_equal(t.actor.state_dict(),resume.actor.state_dict())
    tensor_equal(t.critic.state_dict(),resume.critic.state_dict())
    assert metrics['valid_samples']>0 and all(np.isfinite(metrics[k]) for k in ('policy_loss','critic_loss','entropy'))
    rows=[json.loads(s) for s in (tmp_path/'run/train.jsonl').read_text().splitlines()]
    assert rows and rows[-1]['ppo']['numerical_diagnostics']['tanh']['actor_tanh1']['finite']
    result=dict(seed=1000,threshold='abs(tanh)>=0.99',saturation=d,
        action_distance_delta=(actions[1]-actions[0]).cpu().tolist(),action_angle_delta=(actions[2]-actions[0]).cpu().tolist(),
        physical_min=controls.min(0).tolist(),physical_max=controls.max(0).tolist(),
        smoke=dict(workers=10,episodes=t.training_episode_count,joint_steps=t.sampled_steps,updates=t.updates,
                   evaluation_episodes=2,numerical_failures=0,exact_resume=True),
        ppo_tanh=rows[-1]['ppo']['numerical_diagnostics']['tanh'])
    (tmp_path/'acceptance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('FIXED_SCALE_ACCEPTANCE '+json.dumps(result))
