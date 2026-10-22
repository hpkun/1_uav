"""Real 12/60 transitions, exact episode checkpoint targets and resumable buffer."""
import numpy as np
import torch
from env.aircombat.rewards import COMPONENTS
from .trainer import Rollout,clean_protocol_state


def blank_row(trainer,e,n):
    shape=(e,n)
    row=[torch.zeros((*shape,12),device=trainer.device),torch.zeros((e,60),device=trainer.device),
         torch.zeros((*shape,3),device=trainer.device)]
    row.extend(torch.zeros(shape,device=trainer.device) for _ in range(6))
    row.extend((torch.ones(e,device=trainer.device,dtype=torch.bool),torch.zeros(e,device=trainer.device,dtype=torch.bool)))
    return row


def finish_row(trainer,b,n):
    # Evaluate the successor critic once per original vector wave, regardless
    # of checkpoint splits: even GEMM batch-shape roundoff must not feed back
    # from observational checkpoint logging into PPO targets.
    count=b['cursor']
    with torch.no_grad():
        nv=trainer.critic(b['final_states'][:count])[...,:n]
        masks=b['row'][8][:count].bool()
        b['row'][5][:count]=torch.where(masks,nv,torch.zeros_like(nv))
    b['rows'].append(b['row']);b['row']=None;b['cache']=None;b['final_states']=None;b['cursor']=0


def collect(trainer,vector,joint_steps=None,*,episode_limit=None,checkpoint_callback=None):
    c,e,n=trainer.config,vector.num_envs,vector.config.num_uavs
    budget=c.horizon*e if joint_steps is None else joint_steps
    if type(budget) is not int or not 1<=budget<=c.horizon*e:
        raise ValueError('invalid joint rollout budget')
    if episode_limit is not None and (type(episode_limit) is not int or episode_limit<1):
        raise ValueError('episode_limit must be positive')
    b=getattr(trainer,'pending_collection',None)
    if b is None:
        b=dict(rows=[],row=None,cache=None,cursor=0,episodes=[],parts=dict.fromkeys(COMPONENTS,0.),
               valid_count=0,sampled=0,failures=0,speed_clips=0,numerical_events=[],budget=budget,
               episode_limit=episode_limit,horizon=c.horizon,num_envs=e,num_uavs=n)
    elif (b['num_envs'],b['num_uavs'],b['horizon'])!=(e,n,c.horizon):
        raise ValueError('pending rollout/vector dimensions mismatch')
    trainer.pending_collection=b
    # A split vector wave resumes its remaining worker IDs before the next wave.
    # Exact episode targets do not shorten the normal per-worker horizon.
    while len(b['rows'])<c.horizon and b['sampled']<b['budget'] and (b['episode_limit'] is None or len(b['episodes'])<b['episode_limit']):
        if b['row'] is None:
            b['row']=blank_row(trainer,e,n);b['cursor']=0
            # Sample the whole vector wave once. Checkpoint bookkeeping may
            # split its execution, but never changes Beta sampling batch/RNG.
            wave=min(e,b['budget']-b['sampled'])
            obs=trainer.tensor(vector.observations[:wave]);masks=trainer.tensor(vector.masks[:wave])
            states=clean_protocol_state(trainer.tensor(vector.states[:wave]),masks)
            with torch.no_grad():
                actions,logp=trainer.sample_actions(obs,masks)
                if not c.ppo_masking:
                    actions=torch.where(masks.bool().unsqueeze(-1),actions,torch.full_like(actions,.5))
                    logp,_=trainer.actor.evaluate(obs,actions)
                values=trainer.critic(states)[...,:n]
            b['cache']=[obs,states,actions,logp,values,masks]
            b['final_states']=torch.zeros((wave,60),device=trainer.device)
        count=min(e-b['cursor'],b['budget']-b['sampled'])
        if b['episode_limit'] is not None:count=min(count,b['episode_limit']-len(b['episodes']))
        if checkpoint_callback is not None:
            count=min(count,c.checkpoint_interval_episodes-trainer.training_episode_count%c.checkpoint_interval_episodes)
        ids=np.arange(b['cursor'],b['cursor']+count)
        obs,states,actions,logp,values,masks=[x[ids] for x in b['cache']]
        tr=vector.step(actions.cpu().numpy(),indices=ids)
        next_masks=trainer.tensor(tr.next_masks)
        final=clean_protocol_state(trainer.tensor(tr.transition_states),next_masks)
        b['final_states'][ids]=final
        selected=[obs,states,actions,logp,values,torch.zeros_like(values),trainer.tensor(tr.rewards),masks,next_masks,
                  trainer.tensor(tr.terminated,torch.bool),trainer.tensor(tr.truncated,torch.bool)]
        for dst,src in zip(b['row'],selected):dst[ids]=src
        b['episodes'].extend(tr.episodes)
        for worker,info in zip(ids,tr.infos):
            b['valid_count']+=int(info['current_active_mask'].sum())
            b['failures']+=int(info['numerical_failure'])
            b['speed_clips']+=info['numerical_delta']['speed_clips']
            if info['numerical_failure']:
                b['numerical_events'].append(dict(worker=int(worker),step=info['episode_length'],failures=info['failures']))
            for k in b['parts']:b['parts'][k]+=float(info['reward_components'][k].sum())
        b['sampled']+=count;b['cursor']+=count
        trainer.sampled_steps+=count;trainer.training_episode_count+=len(tr.episodes)
        if b['cursor']==len(b['cache'][0]):finish_row(trainer,b,n)
        if checkpoint_callback is not None and tr.episodes and trainer.training_episode_count%c.checkpoint_interval_episodes==0:
            checkpoint_callback(trainer,vector)
    if b['row'] is not None:finish_row(trainer,b,n)
    trainer.rollout_metrics=dict(sampled_steps=b['sampled'],valid_uav_steps=b['valid_count'],
        numerical_failures=b['failures'],numerical_events=b['numerical_events'],speed_clips=b['speed_clips'],
        reward_components_mean={k:v/max(1,b['valid_count']) for k,v in b['parts'].items()})
    rollout=Rollout(*(torch.stack([r[i] for r in b['rows']]) for i in range(11)))
    trainer.pending_collection=None
    return rollout,b['episodes']
