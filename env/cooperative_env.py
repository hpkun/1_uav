"""Single environment definition for all three inspection task scales."""
import numpy as np
from .assignment import assign
from .config import ENVIRONMENT_VERSION, EnvConfig
from .dynamics import physical_actions
from .integrator import rk4
from .models import pairwise_distances
from .observation import observe, centralized
from .reward import compute_rewards
from .scenario import generate


class Cooperative3DEnv:
    version = ENVIRONMENT_VERSION
    observation_dim, state_dim, action_dim = 12, 90, 3

    def __init__(self, config=None):
        self.config = config if isinstance(config, EnvConfig) else EnvConfig(**(config or {}))
        self.rng = np.random.default_rng()
        self.task = None
        self.done = True

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.task = generate(self.config, self.rng, options)
        self.step_count, self.done = 0, False
        self.counters = dict(speed_clips=0, gamma_clips=0, nonfinite_states=0)
        self.numerical_delta = dict.fromkeys(self.counters, 0)
        self.numerical_events = {k: np.zeros(self.config.num_uavs, dtype=int) for k in self.counters}
        self.assignments = assign(self.task, self.config.assignment_mode)
        obs = observe(self.task, self.assignments, self.config)
        return obs, self._info(obs, [], False)

    def _info(self, obs, events, success):
        t = self.task
        return dict(environment_version=self.version, success=bool(success), incomplete=not bool(success),
                    all_inactive=not bool(t.active_mask.any()), episode_length=self.step_count,
                    completed_waypoints=int(t.completed.sum()), total_waypoints=len(t.waypoints),
                    completion_rate=float(t.completed.mean()), active_mask=t.active_mask.copy(),
                    assignments=self.assignments.copy(), states=t.states.copy(), waypoints=t.waypoints.copy(),
                    priorities=t.priorities.copy(), completed=t.completed.copy(),
                    pairwise_distances=pairwise_distances(t.states), events=events,
                    centralized_state=centralized(obs, t, self.config), numerical_counters=self.counters.copy(),
                    numerical_delta=self.numerical_delta.copy(), numerical_events={k:v.copy() for k,v in self.numerical_events.items()})

    def step(self, actions):
        if self.done:
            raise RuntimeError('reset required before step')
        t, c = self.task, self.config
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (c.num_uavs, 3):
            raise ValueError('actions must have shape [N,3]')
        active_actions = actions[t.active_mask]
        if not np.isfinite(active_actions).all() or np.any((active_actions < 0) | (active_actions > 1)):
            raise ValueError('active actions must be finite in [0,1]')
        before, current_mask, prior_assignment = t.states.copy(), t.active_mask.copy(), self.assignments.copy()
        controls = np.zeros_like(actions)
        controls[current_mask] = physical_actions(active_actions)
        # Idle UAVs hold current speed, heading and flight-path angle using
        # the same differential equations (trim loads, never state increments).
        idle = current_mask & (prior_assignment < 0)
        controls[idle] = np.column_stack((np.zeros(idle.sum()), np.sin(t.states[idle, 4]), np.cos(t.states[idle, 4])))
        failures = []
        prior_counters = self.counters.copy()
        self.numerical_events = {k: np.zeros(c.num_uavs, dtype=int) for k in self.counters}
        for i in np.flatnonzero(current_mask):
            before_integration = self.counters.copy()
            t.states[i], finite = rk4(t.states[i], controls[i], c, self.counters)
            for key in self.counters:
                self.numerical_events[key][i] = self.counters[key] - before_integration[key]
            if not finite:
                t.active_mask[i] = False
                failures.append(dict(uav_id=int(i), reason='nonfinite_integration'))
        self.step_count += 1
        self.numerical_delta = {k: self.counters[k] - prior_counters[k] for k in self.counters}
        for i in c.failure_events.get(self.step_count, []):
            if t.active_mask[i]:
                t.active_mask[i] = False
                failures.append(dict(uav_id=i, reason='injected_failure'))
        # End-position sphere test. Lowest active UAV ID owns each first visit,
        # regardless of assignment, so physical visits cannot be ignored.
        events = []
        for j in np.flatnonzero(~t.completed):
            eligible = np.flatnonzero(t.active_mask & (np.linalg.norm(t.states[:, :3] - t.waypoints[j], axis=1) <= c.completion_radius))
            if len(eligible):
                owner = int(eligible[0])
                t.completed[j] = True
                events.append(dict(step=self.step_count, waypoint_id=int(j), uav_id=owner))
        success = bool(t.completed.all())
        terminated = success or not t.active_mask.any()
        truncated = self.step_count >= c.max_steps and not terminated
        rewards, parts = compute_rewards(before, t.states, t.waypoints, prior_assignment, current_mask,
                                         t.active_mask, events, self.step_count, success, c)
        self.assignments = assign(t, c.assignment_mode)
        obs = observe(t, self.assignments, c)
        self.done = bool(terminated or truncated)
        info = self._info(obs, events, success)
        info.update(previous_states=before, transition_assignments=prior_assignment,
                    current_active_mask=current_mask, next_active_mask=t.active_mask.copy(),
                    physical_actions=controls, reward_components=parts, rewards=rewards.copy(), failures=failures)
        return obs, rewards.astype(np.float32), bool(terminated), bool(truncated), info
