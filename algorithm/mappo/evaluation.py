"""Independent deterministic evaluation, with training seeds explicitly excluded."""
import numpy as np
import torch
from env.factory import make_environment
from algorithm.common.runtime import require_cuda
from algorithm.common.metrics import EpisodeAccumulator, episode_metrics


def evaluate(trainer, env_config, seeds, trace_path=None, *, provenance=None):
    require_cuda(str(trainer.device))
    summaries=[]
    for episode,seed in enumerate(seeds):
        env=make_environment(env_config)
        obs,info=env.reset(seed=int(seed))
        tracker=EpisodeAccumulator(); tracker.reset(info)
        recorder=None
        if trace_path and episode==0:
            from tools.recording import TrajectoryRecorder
            recorder=TrajectoryRecorder(env_config,seed,info)
            recorder.data['provenance']=provenance or {'policy':'in_memory_mappo','update':trainer.updates,'sampled_steps':trainer.sampled_steps}
        while True:
            with torch.no_grad():
                clean=np.where(info['active_mask'][:,None],obs,0)
                actions,_=trainer.actor.act(trainer.tensor(clean),deterministic=True)
            obs,reward,term,trunc,info=env.step(actions.cpu().numpy())
            tracker.add(info,reward)
            if recorder: recorder.append(info)
            if term or trunc: break
        summary=tracker.summary(info)
        summary.update(seed=int(seed),incomplete=info['incomplete'],all_inactive=info['all_inactive'],numerical_counters=info['numerical_counters'])
        summaries.append(summary)
        if recorder: recorder.save(trace_path)
    if not summaries: raise ValueError('at least one evaluation seed required')
    result=episode_metrics(summaries)
    result.update(episodes=summaries,mean_completion_rate=result['completion_rate'],mean_reward=result['mean_episode_return'])
    return result


def evaluation_seeds(config):
    values=list(config.evaluation_seeds)
    while len(values)<config.evaluation_episodes: values.append(max(values)+1)
    return tuple(values[:config.evaluation_episodes])


def ensure_independent(seeds, training_seed, num_envs, episodes):
    for seed in seeds:
        offset=int(seed)-training_seed
        if offset>=0 and offset//num_envs<=int(episodes[offset%num_envs]):
            raise ValueError(f'evaluation seed {seed} was used for training')
