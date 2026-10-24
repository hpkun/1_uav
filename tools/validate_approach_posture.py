"""Bounded CUDA acceptance only; no formal reward-ablation training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from env.aircombat import AirCombatConfig,AirCombatEnv
from env.aircombat.posture_reward import add_statistics,summarize_statistics
from env.aircombat.config import ENVIRONMENT_VERSION
from algorithm.common.runtime import require_cuda
from algorithm.experiments import load_model,write_json,configurations
from algorithm.common.aircombat_vector import AirCombatVectorEnv
from algorithm.common.checkpoint import save_protocol_checkpoint,read_protocol_checkpoint,restore_protocol_checkpoint
from algorithm.mappo.trainer import ProtocolMAPPOTrainer
from algorithm.mappo.config import ProtocolTrainConfig


def equal(a,b):
    from tests.test_aircombat_training import tensor_equal
    tensor_equal(a,b)


def historical_paper_replay(checkpoint,trace_path):
    """Read-only paper evaluation plus fixed-action counterfactual; no resume."""
    checkpoint=Path(checkpoint);before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    trainer,cfg,payload=load_model(checkpoint)
    assert cfg.posture_reward_mode=='paper'
    trace=json.loads(Path(trace_path).read_text(encoding='utf-8'))
    env=AirCombatEnv(cfg);obs,info=env.reset(trace['seed'],{'scenario':trace['scenario']})
    gated=AirCombatEnv(AirCombatConfig(**{**cfg.to_dict(),'posture_reward_mode':'approach_gated'}))
    gated.reset(trace['seed'],{'scenario':trace['scenario']})
    events=[];original_posture=0.;new_posture=0.;totals={}
    for frame in trace['frames'][1:]:
        actions=trainer.sample_actions(obs,info['active_mask'],deterministic=True)[0].cpu().numpy()
        np.testing.assert_array_equal(actions,frame['red_actions'])
        obs,r,term,trunc,info=env.step(actions)
        other=gated.step(actions)
        np.testing.assert_array_equal(r,frame['reward'])
        for name in ('friendly_states','adversarial_states','assignments','friendly_alive','adversarial_alive','blue_controls'):
            np.testing.assert_array_equal(info[name],frame[name]);np.testing.assert_array_equal(info[name],other[4][name])
        for name,value in frame['reward_components'].items():np.testing.assert_array_equal(info['reward_components'][name],value)
        for name in ('attack','allied','shootdown','terminal'):
            np.testing.assert_array_equal(info['reward_components'][name],other[4]['reward_components'][name])
        assert (term,trunc)==other[2:4] and info['events']==other[4]['events']
        events.extend(info['events']);original_posture+=float(info['reward_components']['posture'].sum())
        new_posture+=float(other[4]['reward_components']['posture'].sum())
        add_statistics(totals,other[4]['posture_diagnostics'])
    assert events==trace['events'] and (term or trunc)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==before
    return dict(checkpoint=str(checkpoint),sha256=before,training_episodes=payload['training_episode_count'],
        replay_steps=len(trace['frames'])-1,paper_actions_states_rewards_events_exact=True,
        gated_fixed_action_physics_other_four_rewards_events_exact=True,
        paper_posture_sum_all_red=original_posture,gated_posture_sum_all_red=new_posture,
        posture_diagnostics=summarize_statistics(totals),
        usage='paper checkpoint read-only evaluation; recorded actions fed to another environment, never resumed gated training')


def cuda_parallel(output):
    env_config,formal=configurations('posture_approach_gated_3v3')
    # Explicit audit budget only. Formal catalogue and all its hyperparameters
    # remain unchanged. max_steps=250 and 10 spawn workers are retained here.
    cfg=ProtocolTrainConfig(num_envs=10,horizon=8,ppo_epochs=1,minibatch_size=128)
    trainer=ProtocolMAPPOTrainer(cfg);metrics=[]
    with AirCombatVectorEnv(env_config,10) as vector:
        r,e=trainer.collect(vector);metrics.append(trainer.update(r))
        assert r.current_masks.bool().all() and torch.isfinite(r.rewards).all()
        path=output/'gated_resume.pt';save_protocol_checkpoint(path,trainer,env_config=env_config,vector=vector)
        payload=read_protocol_checkpoint(path,environment_version=ENVIRONMENT_VERSION)
        r1,e1=trainer.collect(vector);m1=trainer.update(r1)
        state=copy.deepcopy((trainer.actor.state_dict(),trainer.critic.state_dict(),trainer.actor_optimizer.state_dict(),
            trainer.critic_optimizer.state_dict(),trainer.rollout_metrics,vector.snapshot(),random.getstate(),
            np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all()))
        clone=ProtocolMAPPOTrainer(cfg);restore_protocol_checkpoint(payload,clone,env_config=env_config,vector=vector)
        r2,e2=clone.collect(vector);m2=clone.update(r2)
        equal((r1,e1,m1),(r2,e2,m2))
        equal(state,(clone.actor.state_dict(),clone.critic.state_dict(),clone.actor_optimizer.state_dict(),
            clone.critic_optimizer.state_dict(),clone.rollout_metrics,vector.snapshot(),random.getstate(),
            np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all()))
        assert r2.current_masks.bool().all() and torch.isfinite(r2.rewards).all()
        assert trainer.rollout_metrics['numerical_failures']==0
        for m in (metrics[0],m1):
            assert m['valid_samples']==240
            for name in ('actor_total_loss','policy_loss','critic_loss','entropy','approx_kl'):
                assert np.isfinite(m[name])
        return dict(workers=len(vector.worker_pids),seed_rule=cfg.worker_seeds(),max_episode_steps=env_config.max_steps,
            audit_horizon=8,audit_ppo_epochs=1,sampled_joint_steps=trainer.sampled_steps,updates=trainer.updates,
            all_480_uav_samples_valid=True,numerical_failures=0,exact_resume=True,ppo=[metrics[0],m1],
            posture_diagnostics=trainer.rollout_metrics['posture_diagnostics'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--historical-checkpoint');p.add_argument('--historical-trace');a=p.parse_args()
    require_cuda('cuda');output=Path(a.output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('use a fresh audit output directory')
    output.mkdir(parents=True)
    result=dict(cuda=torch.cuda.get_device_name(),parallel=cuda_parallel(output))
    if a.historical_checkpoint and a.historical_trace:
        result['historical']=historical_paper_replay(a.historical_checkpoint,a.historical_trace)
    write_json(output/'acceptance.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
