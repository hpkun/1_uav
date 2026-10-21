"""Deterministic CUDA evaluation on explicit, independent scenario seeds."""
import numpy as np
import torch
from ..environment import Cooperative3DEnv
from ..recording import TrajectoryRecorder
from .trainer import require_cuda


def evaluate(trainer, env_config, seeds, trace_path=None):
    require_cuda()
    if trainer.device.type != 'cuda':
        raise RuntimeError('evaluation requires a CUDA policy')
    summaries = []
    for episode, seed in enumerate(seeds):
        env = Cooperative3DEnv(env_config)
        obs, info = env.reset(seed=int(seed))
        recorder = TrajectoryRecorder(env_config, seed, info) if trace_path and episode == 0 else None
        total = np.zeros(env_config.num_uavs)
        components = dict.fromkeys(('progress', 'completion', 'separation', 'efficiency', 'terminal'), 0.0)
        while True:
            with torch.no_grad():
                actions, _ = trainer.actor.act(trainer.tensor(obs), deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(actions.cpu().numpy())
            total += rewards
            for key, value in info['reward_components'].items():
                components[key] += float(value.sum())
            if recorder:
                recorder.append(info)
            if terminated or truncated:
                break
        if recorder:
            recorder.save(trace_path)
        summaries.append(dict(seed=int(seed), reward=float(total.mean()), reward_components=components,
                              success=info['success'], incomplete=info['incomplete'], all_inactive=info['all_inactive'],
                              episode_length=info['episode_length'], completed_waypoints=info['completed_waypoints'],
                              total_waypoints=info['total_waypoints'], completion_rate=info['completion_rate'],
                              numerical_counters=info['numerical_counters']))
    if not summaries:
        raise ValueError('at least one evaluation seed required')
    return dict(episodes=summaries, success_rate=float(np.mean([r['success'] for r in summaries])),
                mean_completion_rate=float(np.mean([r['completion_rate'] for r in summaries])),
                mean_reward=float(np.mean([r['reward'] for r in summaries])))
