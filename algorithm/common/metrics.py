"""Exact per-step deltas, active-UAV-step denominators and episode metrics."""
import numpy as np

COMPONENTS = ('progress', 'completion', 'separation', 'efficiency', 'terminal')


def target_distances(info, *, previous=False):
    states = info['previous_states'] if previous else info['states']
    assignments = info.get('transition_assignments', info['assignments'])
    mask = info.get('current_active_mask', info['active_mask'])
    ids = np.flatnonzero(mask & (assignments >= 0))
    return np.linalg.norm(info['waypoints'][assignments[ids]] - states[ids, :3], axis=1)


class EpisodeAccumulator:
    def reset(self, info):
        self.reward = np.zeros(len(info['states']))
        self.parts = dict.fromkeys(COMPONENTS, 0.0)
        self.steps, self.valid_steps = 0, 0
        self.switches = 0

    def add(self, info, reward):
        self.reward += reward
        self.steps += 1
        self.valid_steps += int(info['current_active_mask'].sum())
        for key in COMPONENTS:
            self.parts[key] += float(info['reward_components'][key].sum())
        self.switches += int(((info['transition_assignments'] != info['assignments']) & info['next_active_mask']).sum())

    def summary(self, info):
        return dict(episode_return=float(self.reward.mean()), success=info['success'], completion_rate=info['completion_rate'],
                    completed_waypoints=info['completed_waypoints'], episode_length=self.steps,
                    reward_components_mean={k:v/max(1,len(self.reward)) for k,v in self.parts.items()},
                    assignment_switches=self.switches)


def episode_metrics(episodes):
    keys = {'mean_episode_return':'episode_return', 'success_rate':'success', 'completion_rate':'completion_rate',
            'mean_completed_waypoints':'completed_waypoints', 'mean_episode_length':'episode_length'}
    result = {out:float(np.mean([e[key] for e in episodes])) if episodes else None for out,key in keys.items()}
    result['joint_episode_count'] = len(episodes)
    result['reward_components_mean'] = {k:float(np.mean([e['reward_components_mean'][k] for e in episodes])) if episodes else None for k in COMPONENTS}
    return result
