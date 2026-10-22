"""Reproducible fixed and random scenarios using only an environment RNG."""
import numpy as np
from .models import TaskState
from .dynamics import wrap_angle


def generate(config, rng, options=None):
    n, m = config.num_uavs, config.num_waypoints
    options = options or {}
    unknown = set(options) - {'states', 'waypoints', 'priorities', 'active_mask'}
    if unknown:
        raise ValueError(f'unknown reset options: {unknown}')
    s = config.scenario
    if config.random_scenario:
        points = np.column_stack([rng.uniform(*r, m) for r in (s.waypoint_x, s.waypoint_y, s.waypoint_z)])
        positions = []
        for _ in range(s.max_sampling_attempts):
            p = np.r_[rng.uniform(-s.formation_half_width, s.formation_half_width, 2), rng.uniform(*s.uav_altitude)]
            if all(np.linalg.norm(p - q) >= s.initial_min_separation for q in positions):
                positions.append(p)
                if len(positions) == n:
                    break
        if len(positions) != n:
            raise ValueError('cannot sample separated UAV positions; revise scenario ranges')
        positions = np.asarray(positions)
        target = points.mean(axis=0) - positions
        heading = np.arctan2(target[:, 1], target[:, 0]) + rng.uniform(-s.heading_jitter, s.heading_jitter, n)
        states = np.column_stack((positions, rng.uniform(*s.initial_speed, n), rng.uniform(*s.initial_gamma, n), heading))
    else:
        states = np.column_stack((np.zeros(n), (np.arange(n) - (n - 1) / 2) * 300, np.full(n, 3000), np.full(n, 225), np.zeros((n, 2))))
        points = np.column_stack((np.full(m, 7000), (np.arange(m) - (m - 1) / 2) * 700, np.linspace(2700, 3300, m)))
    states = np.array(options.get('states', states), dtype=np.float64, copy=True)
    points = np.array(options.get('waypoints', points), dtype=np.float64, copy=True)
    priorities = np.array(options.get('priorities', np.ones(m)), dtype=np.float64, copy=True)
    active = np.array(options.get('active_mask', np.ones(n, dtype=bool)), copy=True)
    if states.shape != (n, 6) or points.shape != (m, 3) or priorities.shape != (m,) or active.shape != (n,):
        raise ValueError('scenario dimensions do not match scale')
    if not all(np.isfinite(x).all() for x in (states, points, priorities, active)):
        raise ValueError('scenario must be finite')
    if np.any((priorities < 0) | (priorities > 1)) or not np.isin(active, (0, 1)).all():
        raise ValueError('priorities and active flags outside valid ranges')
    if np.any((states[:, 3] < config.v_min) | (states[:, 3] > config.v_max)):
        raise ValueError('initial speed outside published range')
    if np.any(np.abs(np.cos(states[active.astype(bool), 4])) <= config.denominator_epsilon):
        raise ValueError('initial flight state at angle-coordinate singularity')
    states[:, 5] = wrap_angle(states[:, 5])
    return TaskState(states, points, priorities, np.zeros(m, dtype=bool), active.astype(bool))
