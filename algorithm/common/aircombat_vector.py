"""Spawn vector reusing transport/reset semantics without inspection state."""
import numpy as np
from env.aircombat import AirCombatConfig, AirCombatEnv
from env.aircombat.rewards import COMPONENTS
from env.aircombat.posture_reward import add_statistics,summarize_statistics
from .vector_env import ParallelVectorEnv


class CombatEpisodeAccumulator:
    def reset(self, info):
        self.reward = np.zeros(len(info['active_mask']))
        self.parts = dict.fromkeys(COMPONENTS,0.)
        self.steps = self.valid_steps = self.numerical_failures = 0
        self.numerical_events = []
        self.posture_totals = {}
        self.risks = {k:np.zeros(len(self.reward)) for k in ('proximity','shared_exposure')}

    def add(self, info, reward):
        if 'posture_diagnostics' in info:
            if not hasattr(self,'posture_totals'):self.posture_totals={}
            add_statistics(self.posture_totals,info['posture_diagnostics'])
        self.reward += reward
        self.steps += 1
        self.valid_steps += int(info['current_active_mask'].sum())
        self.numerical_failures += int(info['numerical_failure'])
        if info['numerical_failure']:
            self.numerical_events.append(dict(step=info['episode_length'],failures=info['failures']))
        for k in COMPONENTS:
            self.parts[k] += float(info['reward_components'][k].sum())
        for k in self.risks:
            self.risks[k] += info['risk_terms'][k]

    def summary(self, info):
        result=dict(episode_return=float(self.reward.mean()), episode_length=self.steps,
                    win=info['win'], outcome=info['outcome'],blue_shot_down=info['blue_shot_down'],
                    red_survivors=info['red_survivors'],numerical_failures=self.numerical_failures,
                    numerical_events=self.numerical_events.copy(),
                    numerical_diagnostics=info.get('numerical_diagnostics',{}).copy(),
                    proximity_risk=float(self.risks['proximity'].mean()),
                    shared_exposure_risk=float(self.risks['shared_exposure'].mean()),
                    risk_totals_by_uav={k:v.tolist() for k,v in self.risks.items()},
                    reward_components_total={k:v/len(self.reward) for k,v in self.parts.items()},
                    reward_components_mean={k:v/max(1,self.valid_steps) for k,v in self.parts.items()})
        if getattr(self,'posture_totals',None):result['posture_diagnostics']=summarize_statistics(self.posture_totals)
        return result


def combat_metrics(episodes):
    fields = dict(mean_episode_return='episode_return',mean_episode_length='episode_length',
                  win_rate='win',mean_blue_shot_down='blue_shot_down',mean_red_survivors='red_survivors')
    result = {k:float(np.mean([e[v] for e in episodes])) if episodes else None for k,v in fields.items()}
    result['completed_episodes'] = len(episodes)
    result['numerical_failures'] = sum(e['numerical_failures'] for e in episodes)
    result['termination_counts'] = dict(normal=sum(e.get('outcome') in ('win','red_eliminated','mutual_elimination') for e in episodes),
                                       timeout=sum(e.get('outcome')=='timeout' for e in episodes),
                                       numerical_failure=sum(e.get('outcome')=='numerical_failure' for e in episodes))
    for key in ('proximity_risk','shared_exposure_risk'):
        result[key] = float(np.mean([e[key] for e in episodes])) if episodes and all(key in e for e in episodes) else None
    result['reward_components_mean'] = {k:float(np.mean([e['reward_components_mean'][k] for e in episodes]))
                                        if episodes else None for k in COMPONENTS}
    posture_totals={}
    for e in episodes:
        if 'posture_diagnostics' in e:add_statistics(posture_totals,e['posture_diagnostics'])
    if posture_totals:result['posture_diagnostics']=summarize_statistics(posture_totals)
    return result


class AirCombatVectorEnv(ParallelVectorEnv):
    def __init__(self, config=None, num_envs=10, training_seed_index=1, timeout=60):
        self.training_seed_index = training_seed_index
        super().__init__(config or AirCombatConfig(),num_envs,seed=1000*training_seed_index,timeout=timeout,
                         environment_factory=AirCombatEnv,accumulator_factory=CombatEpisodeAccumulator)

    def _seed(self, i):
        # Initial seed rule is published; subsequent continuation is an assumption.
        return int(1000*self.training_seed_index+i) if self.episodes[i]==0 else None
