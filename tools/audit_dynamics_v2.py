"""Reproducible bounded v2 audit; never launches a formal training budget."""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from env.aircombat import AirCombatConfig,AirCombatEnv
from env.aircombat.config import ENVIRONMENT_VERSION
from env.aircombat.framed_dynamics import initial_frame,step,SimulationFault,frame_error
from env.paper_dynamics import rk4_step
from env.dynamics import velocity_vector
from algorithm.common.runtime import require_cuda
from algorithm.common.checkpoint import read_protocol_checkpoint,save_protocol_checkpoint,restore_protocol_checkpoint
from algorithm.common.aircombat_vector import AirCombatVectorEnv,CombatEpisodeAccumulator
from algorithm.mappo.config import ProtocolTrainConfig
from algorithm.mappo.trainer import ProtocolMAPPOTrainer
from algorithm.aircombat import evaluate,train


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def deterministic():
    config=AirCombatConfig(); old=np.array([0.,0.,6000.,220.,.1,.3]);new=old.copy();frame=initial_frame(new)
    errors=np.zeros(4)
    for i in range(80):
        control=[.35*math.sin(i/20),.3,1.3]
        old=rk4_step(old,control,dt=config.dt,gravity=config.gravity).state
        new,frame,_=step(new,control,frame,config)
        errors=np.maximum(errors,[np.linalg.norm(old[:3]-new[:3]),abs(old[3]-new[3]),abs(old[4]-new[4]),abs(old[5]-new[5])])
    stress=[]
    for kind in ('random','sustained'):
        rng=np.random.default_rng(447);y=np.array([0.,0.,6000.,200.,1.45,.3]);frame=initial_frame(y)
        old=y.copy();old_failure=None;count=0;minimum=1.;max_error=0.;max_projection=0.;max_velocity_jump=0.;max_spin=0.;max_gauge=0.
        for i in range(5000):
            control=rng.uniform([-math.pi,-1,0],[math.pi,2,8]) if kind=='random' else np.array([0.,1.,8.])
            if old_failure is None:
                reference=rk4_step(old,control,dt=config.dt,gravity=config.gravity)
                if reference.accepted:
                    old=reference.state;old[3]=np.clip(old[3],100,400)
                else:old_failure=dict(step=i+1,reason=reference.reason)
            previous=velocity_vector(y);y,frame,d=step(y,control,frame,config)
            count+=d['continuation_substeps'];minimum=min(minimum,d['min_horizontal_ratio'])
            max_error=max(max_error,frame_error(frame));max_projection=max(max_projection,d['frame_projection_error'])
            max_velocity_jump=max(max_velocity_jump,float(np.linalg.norm(velocity_vector(y)-previous)))
            max_spin=max(max_spin,d['max_frame_spin_rate']);max_gauge=max(max_gauge,d['max_reference_axis_error'])
        stress.append(dict(kind=kind,steps=5000,old_first_failure=old_failure,new_faults=0,
            continuation_substeps=count,min_horizontal_ratio=minimum,max_orthogonality_error=max_error,
            max_projection_error=max_projection,max_velocity_change_per_dt=max_velocity_jump,
            max_frame_spin_rad_per_s=max_spin,max_reference_axis_error_rad=max_gauge))
    # Assumption sensitivity, explicitly a deterministic trajectory comparison,
    # not performance tuning or a network parameter search.
    sensitivity=[]
    for cap in (.05,.1,.2):
        c=replace(config,pole_cap_ratio=cap);y=np.array([0.,0.,6000.,200.,1.5,.3]);f=initial_frame(y)
        for _ in range(100):y,f,_=step(y,[.7,1,8],f,c)
        sensitivity.append(dict(pole_cap_ratio=cap,final_state=y.tolist(),final_direction=f[:,0].tolist()))
    base=np.array(sensitivity[1]['final_state'])
    for item in sensitivity:
        item['position_difference_from_cap_0_1_m']=float(np.linalg.norm(np.array(item['final_state'])[:3]-base[:3]))
    convergence=[]
    for substeps in (8,16,32,64):
        c=replace(config,frame_polar_substeps=substeps,frame_normal_substeps=substeps)
        y=np.array([0.,0.,6000.,200.,math.pi/2,.3]);f=initial_frame(y)
        for _ in range(12):y,f,_=step(y,[.7,1,8],f,c)
        convergence.append(dict(substeps=substeps,final_state=y.tolist(),direction=f[:,0].tolist()))
    finest=np.array(convergence[-1]['final_state'])
    for item in convergence:
        value=np.array(item['final_state'])
        item['difference_from_64_substeps']=dict(position_m=float(np.linalg.norm(value[:3]-finest[:3])),
            speed_m_per_s=float(abs(value[3]-finest[3])),gamma_rad=float(abs(value[4]-finest[4])),
            psi_rad=float(abs(value[5]-finest[5])))
    return dict(environment_version=ENVIRONMENT_VERSION,normal_80_step_max_errors=dict(zip(
        ('position_m','speed_m_per_s','gamma_rad','psi_rad'),errors.tolist())),stress=stress,
        engineering_assumption_sensitivity=sensitivity,polar_resolution_convergence=convergence)


def diagnostic_policies(output,historical):
    require_cuda('cuda');trainer=ProtocolMAPPOTrainer(ProtocolTrainConfig())
    policies=[('fixed_trim',None),('untrained_actor',trainer)]
    source=None
    if historical:
        path=Path(historical);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        # Explicit diagnostic-only transfer of Actor WEIGHTS. Never restore a
        # v1 optimizer, vector, RNG, training count, or protocol into v2.
        payload=read_protocol_checkpoint(path,environment_version='xu-aircombat-v1.0.0')
        historical_trainer=ProtocolMAPPOTrainer(ProtocolTrainConfig(**payload['train_config']))
        historical_trainer.actor.load_state_dict(payload['actor'])
        policies.append(('historical_actor',historical_trainer))
        source=dict(path=str(path.resolve()),sha256=digest,environment_version=payload['environment_version'],
            training_episodes=payload['training_episode_count'],updates=payload['updates'],usage='Actor weights only; diagnostic, not resumed/new training')
    rows=[]
    with (output/'policy_trajectories.jsonl').open('x',encoding='utf-8') as stream:
        for name,actor in policies:
            for mode in ('angle_rk4','frame_rk4'):
                episodes=[];faults=[]
                for seed in (10000000,10000001,10000002):
                    env=AirCombatEnv(AirCombatConfig(integrator=mode));obs,info=env.reset(seed)
                    tracker=CombatEpisodeAccumulator();tracker.reset(info)
                    while True:
                        if actor is None:actions=np.tile([.5,1/3,1/8],(3,1))
                        else:actions=actor.sample_actions(obs,info['active_mask'],deterministic=True)[0].cpu().numpy()
                        try:obs,reward,terminated,truncated,info=env.step(actions)
                        except SimulationFault as exc:
                            faults.append(dict(seed=seed,last_valid_step=env.state.step_count,diagnostic=exc.diagnostic));break
                        tracker.add(info,reward)
                        record=dict(policy=name,integrator=mode,seed=seed,step=env.state.step_count,
                            red=env.state.friendly_states.tolist(),blue=env.state.adversarial_states.tolist(),
                            events=info['events'],reward=reward.tolist(),assignments=info['assignments'].tolist(),
                            numerical=info['numerical_diagnostics'],red_controls=info['red_controls'].tolist(),
                            blue_controls=info['blue_controls'].tolist())
                        stream.write(json.dumps(record,allow_nan=False)+'\n')
                        if terminated or truncated:episodes.append(tracker.summary(info));break
                rows.append(dict(policy=name,integrator=mode,attempted=3,completed=len(episodes),fault_count=len(faults),
                    faults=faults,episodes=episodes,win_rate_on_completed_episodes=float(np.mean([e['win'] for e in episodes])) if episodes else None))
    if source:
        assert hashlib.sha256(Path(historical).read_bytes()).hexdigest()==source['sha256']
    return dict(cuda=torch.cuda.get_device_name(),historical_actor=source,results=rows)


def cuda_acceptance(output):
    require_cuda('cuda')
    cfg=ProtocolTrainConfig(num_envs=10,horizon=8,ppo_epochs=1,training_episodes=1,
        checkpoint_interval_episodes=1,minibatch_size=128)
    ec=AirCombatConfig()
    # A real 250-step episode with the unchanged environment; short rollout
    # and one PPO epoch are labelled audit-only, not the formal protocol.
    trainer=train(ec,cfg,output/'short_train',log_every_updates=5)
    evaluation=evaluate(trainer,ec,[11000000,11000001,11000002])
    write(output/'short_evaluation.json',evaluation)
    # Polar internal state and full ten-worker continuation, optimizer/RNG and
    # half-rollout semantics are additionally asserted in pytest.
    from tests.test_aircombat_training import tensor_equal
    trainer=ProtocolMAPPOTrainer(replace(cfg,training_episodes=2))
    with AirCombatVectorEnv(ec,10) as vector:
        for i in range(2):r,_=trainer.collect(vector);trainer.update(r)
        checkpoint=output/'ten_worker_resume.pt'
        save_protocol_checkpoint(checkpoint,trainer,env_config=ec,vector=vector)
        payload=read_protocol_checkpoint(checkpoint,environment_version=ENVIRONMENT_VERSION)
        r1,e1=trainer.collect(vector);m1=trainer.update(r1)
        state=copy.deepcopy((trainer.actor.state_dict(),trainer.critic.state_dict(),trainer.actor_optimizer.state_dict(),
            trainer.critic_optimizer.state_dict(),vector.snapshot(),torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state()))
        clone=ProtocolMAPPOTrainer(trainer.config);restore_protocol_checkpoint(payload,clone,env_config=ec,vector=vector)
        r2,e2=clone.collect(vector);m2=clone.update(r2)
        tensor_equal((r1,e1,m1),(r2,e2,m2))
        tensor_equal(state,(clone.actor.state_dict(),clone.critic.state_dict(),clone.actor_optimizer.state_dict(),
            clone.critic_optimizer.state_dict(),vector.snapshot(),torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state()))
        workers=len(vector.worker_pids)
    return dict(cuda=torch.cuda.get_device_name(),workers=workers,exact_resume=True,
        completed_training_episodes=1,evaluation={k:v for k,v in evaluation.items() if k!='episodes'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('deterministic','policies','cuda'),required=True)
    p.add_argument('--output',required=True);p.add_argument('--historical-checkpoint');args=p.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    target=output/(args.phase+'.json')
    if target.exists():raise FileExistsError('use a fresh audit output directory')
    result=deterministic() if args.phase=='deterministic' else diagnostic_policies(output,args.historical_checkpoint) if args.phase=='policies' else cuda_acceptance(output)
    write(target,result);print(json.dumps(result if args.phase!='policies' else dict(output=str(target)),indent=2))


if __name__=='__main__':main()
