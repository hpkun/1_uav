import copy
import numpy as np
from env.dynamics import physical_actions
from env.integrator import rk4
from .config import RegionConfig, ENVIRONMENT_VERSION
from .scenario import initialize
from .regions import centers_at, score_transition
from .policies import rule_actions
from .observation import observe, centralized
from .reward import compute_reward

TEAMS=('red','blue')
NUMERICAL=('speed_clips','gamma_clips','nonfinite_states')


class RegionCompetitionEnv:
    version=ENVIRONMENT_VERSION
    observation_dim,state_dim,action_dim=12,60,3

    def __init__(self,config=None):
        self.config=config if isinstance(config,RegionConfig) else RegionConfig(**(config or {}))
        self.rng=np.random.default_rng()
        self.done=True

    def reset(self,seed=None,options=None):
        if seed is not None: self.rng=np.random.default_rng(seed)
        initial=initialize(self.config,self.rng,options)
        self.states={t:initial[t+'_states'] for t in TEAMS}
        self.active={t:initial[t+'_active'] for t in TEAMS}
        self.anchors,self.phase=initial['anchors'],initial['phase']
        self.step_count=0; self.done=False
        self.episode_seed=seed
        self.region_ids=np.arange(self.config.num_uavs)%self.config.num_regions
        self.centers=centers_at(self.anchors,self.phase,0,self.config)
        self.scores=dict.fromkeys(TEAMS,0.)
        self.counters={t:dict.fromkeys(NUMERICAL,0) for t in TEAMS}
        self.info=self._base_info()
        return self.info['red']['observations'].copy(),copy.deepcopy(self.info)

    def _base_info(self):
        result=dict(environment_version=self.version,step=self.step_count,time=self.step_count*self.config.dt,
                    episode_seed=self.episode_seed,blue_policy=self.config.blue_policy,safe_distance=self.config.safe_distance,
                    regions=self.centers.copy(),region_ids=self.region_ids.copy(),scores=self.scores.copy(),
                    remaining_steps=max(0,self.config.max_steps-self.step_count),events=[],finished=self.done)
        for t in TEAMS:
            other='blue' if t=='red' else 'red'
            obs=observe(self.states[t],self.active[t],self.states[other],self.active[other],self.centers,self.region_ids,self.step_count,self.config)
            result[t]=dict(states=self.states[t].copy(),active_mask=self.active[t].copy(),observations=obs,
                           numerical_counters=self.counters[t].copy())
        result.update(states=self.states['red'].copy(),active_mask=self.active['red'].copy(),centralized_state=centralized(result['red']['observations']))
        return result

    def policy_actions(self,team,mode='region_safe'):
        if team not in TEAMS: raise ValueError('unknown team')
        return rule_actions(self.states[team],self.active[team],self.centers,self.region_ids,self.config,mode,
                            orbit_sign=1 if team=='red' else -1)

    def step(self,red_actions):
        if self.done: raise RuntimeError('reset required')
        return self.step_teams(red_actions,self.policy_actions('blue',self.config.blue_policy))

    def step_teams(self,red_actions,blue_actions):
        """Explicit independent control interface for deterministic diagnostics/controllers."""
        if self.done: raise RuntimeError('reset required')
        c=self.config
        before={t:self.states[t].copy() for t in TEAMS}; masks={t:self.active[t].copy() for t in TEAMS}
        previous_centers=self.centers.copy()
        actions={t:np.array(a,dtype=float,copy=True) for t,a in zip(TEAMS,(red_actions,blue_actions))}
        controls={}; numerical={}
        # Validate BOTH teams before any mutation.
        for t in TEAMS:
            a=actions[t]
            if a.shape!=(c.num_uavs,3) or not np.isfinite(a[masks[t]]).all() or ((a[masks[t]]<0)|(a[masks[t]]>1)).any():
                raise ValueError('each team needs finite active actions [N,3] in [0,1]')
            a[~masks[t]]=0  # Ignored slots remain finite in lossless JSON/checkpoint records.
        for t in TEAMS:
            controls[t]=np.zeros_like(actions[t]); controls[t][masks[t]]=physical_actions(actions[t][masks[t]])
            numerical[t]={k:np.zeros(c.num_uavs,dtype=int) for k in NUMERICAL}
            for i in np.flatnonzero(masks[t]):
                old=self.counters[t].copy()
                self.states[t][i],finite=rk4(self.states[t][i],controls[t][i],c,self.counters[t])
                if not finite: self.active[t][i]=False
                for k in NUMERICAL: numerical[t][k][i]=self.counters[t][k]-old[k]
        self.step_count+=1
        self.centers=centers_at(self.anchors,self.phase,self.step_count*c.dt,c)
        scoring=score_transition(before,self.states,previous_centers,self.centers,
                                 {t:masks[t]&self.active[t] for t in TEAMS},c,self.step_count)
        for t in TEAMS: self.scores[t]+=float(scoring['exclusive'][t].sum())
        terminated=not self.active['red'].any() or not self.active['blue'].any()
        truncated=self.step_count>=c.max_steps and not terminated
        self.done=bool(terminated or truncated)
        difference=self.scores['red']-self.scores['blue']
        outcome=0 if abs(difference)<1e-10 else (1 if difference>0 else -1)
        self.info=self._base_info()
        self.info.update(previous_regions=previous_centers,events=scoring['events'],terminated=bool(terminated),truncated=bool(truncated),
                         bootstrap_terminal=self.done,winner=('red' if outcome>0 else 'blue' if outcome<0 else 'draw') if self.done else None,
                         end_reason=('no_active_team' if terminated else 'deadline' if truncated else None))
        for t,sign in [('red',1),('blue',-1)]:
            reward,parts=compute_reward(before[t],self.states[t],previous_centers,self.region_ids,masks[t],self.active[t],
                                       scoring['credits'][t],numerical[t],sign*outcome,self.done,c)
            self.info[t].update(previous_states=before[t],current_active_mask=masks[t],actions=actions[t].copy(),
                                physical_actions=controls[t],numerical_events=numerical[t],reward_components=parts,rewards=reward,
                                control_credit=scoring['credits'][t],dwell=scoring['dwell'][t],coverage=scoring['coverage'][t],
                                score_delta=float(scoring['exclusive'][t].sum()))
        self.info.update(current_active_mask=masks['red'],next_active_mask=self.active['red'].copy())
        return self.info['red']['observations'].copy(),self.info['red']['rewards'].copy(),bool(terminated),bool(truncated),copy.deepcopy(self.info)

    def snapshot(self):
        return copy.deepcopy(dict(states=self.states,active=self.active,anchors=self.anchors,phase=self.phase,centers=self.centers,
                                  region_ids=self.region_ids,step_count=self.step_count,done=self.done,scores=self.scores,counters=self.counters,
                                  info=self.info,episode_seed=self.episode_seed,rng=self.rng.bit_generator.state))

    def restore(self,snapshot):
        data=copy.deepcopy(snapshot)
        self.rng.bit_generator.state=data.pop('rng')
        for key,value in data.items(): setattr(self,key,value)
