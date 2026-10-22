"""Real 12D/60D transitions; final-state bootstrap precedes auto-reset."""
import numpy as np
import torch
from env.aircombat.rewards import COMPONENTS
from .trainer import Rollout, clean_protocol_state


def collect(trainer, vector, joint_steps=None, *, episode_limit=None):
    c,e,n = trainer.config,vector.num_envs,vector.config.num_uavs
    budget = c.horizon*e if joint_steps is None else joint_steps
    if type(budget) is not int or not 1<=budget<=c.horizon*e:
        raise ValueError('invalid joint rollout budget')
    if episode_limit is not None and (type(episode_limit) is not int or episode_limit<1):
        raise ValueError('episode_limit must be positive')
    rows,episodes = [],[]
    parts = dict.fromkeys(COMPONENTS,0.)
    valid_count = sampled = failures = speed_clips = 0
    numerical_events = []
    while len(rows)<c.horizon and sampled<budget and (episode_limit is None or len(episodes)<episode_limit):
        count = min(e,budget-sampled)
        # At most one completion per selected worker step, so the budget is exact.
        if episode_limit is not None:
            count = min(count,episode_limit-len(episodes))
        ids = np.arange(count)
        obs = trainer.tensor(vector.observations[ids])
        masks = trainer.tensor(vector.masks[ids])
        states = clean_protocol_state(trainer.tensor(vector.states[ids]),masks)
        with torch.no_grad():
            actions,logp = trainer.sample_actions(obs,masks)
            values = trainer.critic(states)[...,:n]
        transition = vector.step(actions.cpu().numpy(),indices=ids)
        next_masks = trainer.tensor(transition.next_masks)
        with torch.no_grad():
            final_states = clean_protocol_state(trainer.tensor(transition.transition_states),next_masks)
            next_values = trainer.critic(final_states)[...,:n]
            next_values = torch.where(next_masks.bool(),next_values,torch.zeros_like(next_values))
        shape = (e,n)
        row = [torch.zeros((*shape,12),device=trainer.device),torch.zeros((e,60),device=trainer.device),
               torch.zeros((*shape,3),device=trainer.device)]
        row.extend(torch.zeros(shape,device=trainer.device) for _ in range(6))
        row.extend((torch.ones(e,device=trainer.device,dtype=torch.bool),torch.zeros(e,device=trainer.device,dtype=torch.bool)))
        selected = [obs,states,actions,logp,values,next_values,trainer.tensor(transition.rewards),masks,next_masks,
                    trainer.tensor(transition.terminated,torch.bool),trainer.tensor(transition.truncated,torch.bool)]
        for dst,src in zip(row,selected):
            dst[:count] = src
        rows.append(row)
        episodes.extend(transition.episodes)
        for worker,info in zip(ids,transition.infos):
            valid_count += int(info['current_active_mask'].sum())
            failures += int(info['numerical_failure'])
            speed_clips += info['numerical_delta']['speed_clips']
            if info['numerical_failure']:
                numerical_events.append(dict(worker=int(worker),step=info['episode_length'],failures=info['failures']))
            for k in parts:
                parts[k] += float(info['reward_components'][k].sum())
        sampled += count
    trainer.sampled_steps += sampled
    trainer.training_episode_count += len(episodes)
    trainer.rollout_metrics = dict(sampled_steps=sampled,valid_uav_steps=valid_count,numerical_failures=failures,
                                   numerical_events=numerical_events,speed_clips=speed_clips,
                                   reward_components_mean={k:v/max(1,valid_count) for k,v in parts.items()})
    return Rollout(*(torch.stack([r[i] for r in rows]) for i in range(11))),episodes
