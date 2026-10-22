"""Simultaneous bilateral 3-DOF combat with stable slots and auditable transitions."""
import copy
import numpy as np
from env.dynamics import physical_actions
from env.integrator import rk4
from .config import AirCombatConfig, ENVIRONMENT_VERSION
from .scenarios import generate
from .assignment import assignment_matrices
from .geometry import pair_advantages
from .opponent import choose_blue_controls
from .observations import observe
from .combat import resolve_attacks
from .rewards import compute_rewards, COMPONENTS, allied_risk_terms


class AirCombatEnv:
    version = ENVIRONMENT_VERSION
    observation_dim, state_dim, action_dim = 12,60,3

    def __init__(self, config=None):
        self.config = config if isinstance(config,AirCombatConfig) else AirCombatConfig(**(config or {}))
        self.rng = np.random.default_rng()
        self.state = None
        self.done = True

    def _assign(self):
        assignments,self.ha,self.hb,self.f = assignment_matrices(self.state,self.config)
        self.state.assignments = assignments

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = generate(self.config,self.rng,options)
        self.done = False
        self.counters = dict(speed_clips=0,nonfinite_states=0,integration_failures=0)
        self._assign()
        obs,central = observe(self.state,self.config)
        return obs,self._info(central,[],False)

    def _info(self, central, events, numerical_failure):
        s = self.state
        win = not s.adversarial_alive.any() and not numerical_failure
        mutual = win and not s.friendly_alive.any()
        return dict(environment_version=self.version, centralized_state=central,
                    active_mask=s.friendly_alive.copy(), friendly_alive=s.friendly_alive.copy(),
                    adversarial_alive=s.adversarial_alive.copy(), assignments=s.assignments.copy(),
                    friendly_states=s.friendly_states.copy(),adversarial_states=s.adversarial_states.copy(),
                    H_a=self.ha.copy(),H_b=self.hb.copy(),F=self.f.copy(), events=copy.deepcopy(events),
                    episode_length=s.step_count, win=bool(win), success=bool(win),
                    outcome='numerical_failure' if numerical_failure else 'mutual_elimination' if mutual else
                            'win' if win else 'red_eliminated' if not s.friendly_alive.any() else
                            'timeout' if s.step_count>=self.config.max_steps else 'ongoing',
                    blue_shot_down=sum(e['attacker_side']=='Red' for e in s.events),red_survivors=int(s.friendly_alive.sum()),
                    numerical_failure=numerical_failure, numerical_counters=self.counters.copy())

    def step(self, actions):
        if self.done:
            raise RuntimeError('reset required before step')
        s,c = self.state,self.config
        current = s.friendly_alive.copy()
        assignment = s.assignments.copy()
        actions = np.asarray(actions,dtype=float)
        if actions.shape != (c.num_uavs,3) or not np.isfinite(actions[current]).all() or np.any((actions[current]<0)|(actions[current]>1)):
            raise ValueError('live Red actions must be finite [N,3] in [0,1]')
        failures,events = [],[]
        before_counters = self.counters.copy()
        try:
            blue_controls,choices = choose_blue_controls(s,c,self.rng)
        except FloatingPointError as exc:
            blue_controls,choices = np.zeros((c.num_adversaries,3)),np.full(c.num_adversaries,-1)
            failures.append(dict(side='Blue',id=None,reason=str(exc)))
        red_controls = np.zeros_like(actions)
        red_controls[current] = physical_actions(actions[current])
        next_red,next_blue = s.friendly_states.copy(),s.adversarial_states.copy()
        if not failures:
            for side,states,alive,controls,result in (
                ('Red',s.friendly_states,current,red_controls,next_red),
                ('Blue',s.adversarial_states,s.adversarial_alive,blue_controls,next_blue)):
                for i in np.flatnonzero(alive):
                    diagnostic = {}
                    result[i],ok = rk4(states[i],controls[i],c,self.counters,diagnostics=diagnostic)
                    if not ok:
                        failures.append(dict(side=side,id=int(i),reason=diagnostic['reason']))
        s.step_count += 1
        if failures:
            # Numerical failure is never a kill, a win, or a shaped reward.
            if self.counters['integration_failures']==before_counters['integration_failures']:
                self.counters['integration_failures'] += 1
            rewards = np.zeros(c.num_uavs)
            parts = {k:rewards.copy() for k in COMPONENTS}
            risks = dict(proximity=rewards.copy(),shared_exposure=rewards.copy())
        else:
            s.friendly_states,s.adversarial_states = next_red,next_blue
            _,ra,ba,posture = pair_advantages(next_red,next_blue,c)
            events = resolve_attacks(s,ra,ba,c.attack_dwell_steps)
            rewards,parts = compute_rewards(s,current,assignment,ra,ba,posture,events,c)
            risks = allied_risk_terms(next_red,s.friendly_alive,s.adversarial_alive,ba,c)
        terminated = bool(failures or not s.friendly_alive.any() or not s.adversarial_alive.any())
        truncated = bool(not terminated and s.step_count>=c.max_steps)
        self.done = terminated or truncated
        self._assign()
        obs,central = observe(s,c)
        info = self._info(central,events,bool(failures))
        info.update(current_active_mask=current,next_active_mask=s.friendly_alive.copy(),
                    transition_assignments=assignment,reward_components=parts,
                    risk_terms=risks,terminated=terminated,truncated=truncated,
                    numerical_delta={k:self.counters[k]-before_counters[k] for k in self.counters},
                    failures=failures,blue_action_indices=choices,
                    red_controls=red_controls,blue_controls=blue_controls)
        return obs,rewards,terminated,truncated,info

    def snapshot(self):
        return copy.deepcopy(dict(config=self.config.to_dict(),state=self.state,done=self.done,
                                  counters=self.counters,rng=self.rng.bit_generator.state))

    def restore(self, payload):
        if AirCombatConfig(**payload['config']).to_dict() != self.config.to_dict():
            raise ValueError('environment snapshot configuration mismatch')
        p = copy.deepcopy(payload)
        self.state,self.done,self.counters = p['state'],p['done'],p['counters']
        self.rng.bit_generator.state = p['rng']
        self._assign()
