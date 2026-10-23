"""Simultaneous bilateral 3-DOF combat with stable slots and auditable transitions."""
import copy
import numpy as np
from env.dynamics import physical_actions
from .framed_dynamics import advance, initial_frame, validate, SimulationFault
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
        self.state.friendly_frames=np.array([initial_frame(y) for y in self.state.friendly_states])
        self.state.adversarial_frames=np.array([initial_frame(y) for y in self.state.adversarial_states])
        self.done = False
        self.fault = None
        self.diagnostics=dict(min_horizontal_ratio=1.,pole_entries=0,continuation_substeps=0,
                              frame_orthogonality_error=0.,frame_norm_error=0.,frame_projection_error=0.)
        if self.config.integrator=='frame_rk4':
            self.diagnostics.update(max_frame_spin_rate=0.,max_reference_axis_error=0.)
        self.diagnostics['min_horizontal_ratio']=min(
            np.linalg.norm(frames[alive,:2,0],axis=1).min()
            for frames,alive in ((self.state.friendly_frames,self.state.friendly_alive),
                                 (self.state.adversarial_frames,self.state.adversarial_alive)))
        active_frames=np.concatenate((self.state.friendly_frames[self.state.friendly_alive],
                                      self.state.adversarial_frames[self.state.adversarial_alive]))
        self.diagnostics['frame_norm_error']=float(np.max(abs(np.linalg.norm(active_frames,axis=1)-1)))
        self.diagnostics['frame_orthogonality_error']=float(np.max(abs(
            np.swapaxes(active_frames,1,2)@active_frames-np.eye(3))))
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
                    numerical_failure=numerical_failure, numerical_counters=self.counters.copy(),
                    numerical_diagnostics=self.diagnostics.copy(),integrator=self.config.integrator)

    def step(self, actions):
        if self.done:
            raise RuntimeError('reset required before step; fault='+str(self.fault))
        s,c = self.state,self.config
        current = s.friendly_alive.copy()
        assignment = s.assignments.copy()
        actions = np.asarray(actions,dtype=float)
        if actions.shape != (c.num_uavs,3) or not np.isfinite(actions[current]).all() or np.any((actions[current]<0)|(actions[current]>1)):
            raise ValueError('live Red actions must be finite [N,3] in [0,1]')
        failures,events = [],[]
        before_counters = self.counters.copy()
        rng_before=copy.deepcopy(self.rng.bit_generator.state)
        red_controls = np.zeros_like(actions)
        red_controls[current] = physical_actions(actions[current])
        next_red,next_blue = s.friendly_states.copy(),s.adversarial_states.copy()
        red_frames,blue_frames = s.friendly_frames.copy(),s.adversarial_frames.copy()
        numerical=[]
        try:
            blue_controls,choices = choose_blue_controls(s,c,self.rng)
            for side,states,alive,controls,result,frames in (
                ('Red',s.friendly_states,current,red_controls,next_red,red_frames),
                ('Blue',s.adversarial_states,s.adversarial_alive,blue_controls,next_blue,blue_frames)):
                for i in np.flatnonzero(alive):
                    previous_frame=frames[i].copy()
                    result[i],frames[i],diagnostic = advance(states[i],controls[i],frames[i],c)
                    item=dict(side=side,id=int(i),control=controls[i].copy(),**diagnostic)
                    if diagnostic['pole_entries']:
                        item.update(frame_before=previous_frame.tolist(),frame_after=frames[i].tolist())
                    numerical.append(item)
        except FloatingPointError as exc:
            # No partially advanced state, combat resolution, reward, or terminal
            # transition is returned. The worker propagates this exception.
            self.rng.bit_generator.state=rng_before
            self.done=True
            self.fault=dict(step=s.step_count,integrator=c.integrator,cause=str(exc),
                red_controls=red_controls.tolist(),friendly_states=s.friendly_states.tolist(),
                adversarial_states=s.adversarial_states.tolist(),
                friendly_frames=s.friendly_frames.tolist(),adversarial_frames=s.adversarial_frames.tolist())
            raise SimulationFault('joint simulation transaction failed',**self.fault) from exc
        s.friendly_frames,s.adversarial_frames=red_frames,blue_frames
        for d in numerical:
            self.counters['speed_clips']+=d['speed_clips']
            self.diagnostics['min_horizontal_ratio']=min(self.diagnostics['min_horizontal_ratio'],d['min_horizontal_ratio'])
            for name in ('pole_entries','continuation_substeps'): self.diagnostics[name]+=d[name]
            for name in ('frame_orthogonality_error','frame_norm_error','frame_projection_error','max_frame_spin_rate','max_reference_axis_error'):
                if name in d:self.diagnostics[name]=max(self.diagnostics[name],d[name])
        s.step_count += 1
        s.friendly_states,s.adversarial_states = next_red,next_blue
        _,ra,ba,posture = pair_advantages(next_red,next_blue,c)
        events = resolve_attacks(s,ra,ba,c.attack_dwell_steps)
        rewards,parts = compute_rewards(s,current,assignment,ra,ba,posture,events,c)
        risks = allied_risk_terms(next_red,s.friendly_alive,s.adversarial_alive,ba,c)
        terminated = bool(not s.friendly_alive.any() or not s.adversarial_alive.any())
        truncated = bool(not terminated and s.step_count>=c.max_steps)
        self.done = terminated or truncated
        self._assign()
        obs,central = observe(s,c)
        if not np.isfinite(obs).all() or not np.isfinite(central).all() or not np.isfinite(rewards).all():
            self.done=True
            self.fault=dict(step=s.step_count,cause='non-finite observation/state/reward after combat resolution')
            raise SimulationFault('non-finite combat transition; no transition returned',**self.fault)
        info = self._info(central,events,bool(failures))
        info.update(current_active_mask=current,next_active_mask=s.friendly_alive.copy(),
                    transition_assignments=assignment,reward_components=parts,
                    risk_terms=risks,terminated=terminated,truncated=truncated,
                    numerical_delta={k:self.counters[k]-before_counters[k] for k in self.counters},
                    failures=failures,blue_action_indices=choices,
                    red_controls=red_controls,blue_controls=blue_controls,numerical_step=numerical)
        return obs,rewards,terminated,truncated,info

    def snapshot(self):
        if self.fault is not None: raise SimulationFault('cannot checkpoint a faulted environment',fault=self.fault)
        return copy.deepcopy(dict(environment_version=self.version,config=self.config.to_dict(),state=self.state,done=self.done,
                                  counters=self.counters,diagnostics=self.diagnostics,rng=self.rng.bit_generator.state))

    def restore(self, payload):
        if payload.get('environment_version')!=self.version or AirCombatConfig(**payload['config']).to_dict() != self.config.to_dict():
            raise ValueError('environment snapshot configuration mismatch')
        p = copy.deepcopy(payload)
        for states,frames,alive in ((p['state'].friendly_states,p['state'].friendly_frames,p['state'].friendly_alive),
                                    (p['state'].adversarial_states,p['state'].adversarial_frames,p['state'].adversarial_alive)):
            for i in np.flatnonzero(alive):validate(states[i],frames[i])
        self.state,self.done,self.counters = p['state'],p['done'],p['counters']
        self.diagnostics,self.fault=p['diagnostics'],None
        self.rng.bit_generator.state = p['rng']
        self._assign()
