from collections import Counter
import numpy as np

from .config import SCALES
from .motion import advance, project
from .protected_area import contains_intruder, navigation_point


def unit(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm else np.zeros_like(vector)


def validate(pursuers, evaders, config, *, subset=False, spawn=False):
    counts = SCALES[config.scale]
    if subset and config.scale == "3v1" and 1 <= len(pursuers) <= 3:
        counts = (len(pursuers), counts[1])
    radii = (config.pursuer_collision_radius, config.evader_collision_radius)
    for states, count, speed, radius in zip(
        (pursuers, evaders), counts, (config.pursuer_vmax, config.evader_vmax), radii
    ):
        if states.shape != (count, 6) or not np.isfinite(states).all():
            raise ValueError("invalid initial state shape or values")
        if np.any(states[:, :3] < np.asarray(config.lower)+radius) or np.any(states[:, :3] > np.asarray(config.upper)-radius):
            raise ValueError("initial position outside arena")
        if np.any(np.linalg.norm(states[:, 3:], axis=1) > speed + 1e-10):
            raise ValueError("initial speed exceeds limit")
    points = np.vstack((pursuers, evaders))[:, :3]
    i, j = np.triu_indices(len(points), 1)
    sizes = np.repeat(radii, counts)
    clearance = config.minimum_spawn_clearance if spawn else 0.
    if np.any(np.linalg.norm(points[i] - points[j], axis=1) <= sizes[i]+sizes[j]+clearance):
        raise ValueError("initial overlap")
    distances = np.linalg.norm(pursuers[:, None, :3] - evaders[:, :3], axis=-1)
    speed = np.linalg.norm(pursuers[:, None, 3:]-evaders[:, 3:], axis=-1)
    if np.any((distances >= config.capture_inner_radius) & (distances <= config.capture_outer_radius)
              & (speed <= config.capture_relative_speed_max)):
        raise ValueError("initial pair inside capture radius")
    if any(contains_intruder(position, config) for position in evaders[:, :3]):
        raise ValueError("initial breach")
    for states, cap in ((pursuers, config.pursuer_vertical_vmax),
                        (evaders, config.evader_vertical_vmax)):
        if cap is not None and np.any(np.abs(states[:, 5]) > cap + 1e-10):
            raise ValueError("initial vertical speed exceeds limit")
    horizon = config.dt * config.max_steps
    # A necessary geometric reachability test, not a guarantee against evasion.
    # Each endpoint lies in a ball centered on its uncommanded lag displacement.
    p = advance(pursuers, np.zeros(3), config.pursuer_tau, horizon)
    e = advance(evaders, np.zeros(3), config.evader_tau, horizon)
    reach_p = advance(np.zeros(6), np.array([config.pursuer_vmax, 0., 0.]),
                      config.pursuer_tau, horizon)[0]
    reach_e = advance(np.zeros(6), np.array([config.evader_vmax, 0., 0.]),
                      config.evader_tau, horizon)[0]
    # Bound over the whole horizon, including motion away from the lag centers.
    shift = np.linalg.norm(p[:, :3] - pursuers[:, :3], axis=1)[:, None]
    shift = shift + np.linalg.norm(e[:, :3] - evaders[:, :3], axis=1)
    if np.any(np.all(distances > reach_p + reach_e + shift + config.capture_outer_radius, axis=0)):
        raise ValueError("target outside combined reachable envelope")


def generate(config, rng, options):
    scenario = options.get("scenario", "random")
    if scenario not in ("fixed", "random", "custom"):
        raise ValueError("unknown scenario")
    n, m = SCALES[config.scale]
    if scenario == "custom":
        p, e = (np.array(options[name], dtype=float, copy=True) for name in ("pursuers", "evaders"))
        validate(p, e, config, subset=options.get('paired', False))
        return p, e, {}
    failures = Counter()
    for _ in range(1 if scenario == "fixed" else config.spawn_attempts):
        fixed = scenario == "fixed"
        p, e = np.zeros((n, 6)), np.zeros((m, 6))
        goal_point = navigation_point(config)

        def draw(bounds, fraction=.5):
            return bounds[0]+fraction*(bounds[1]-bounds[0]) if fixed else rng.uniform(*bounds)

        for j in range(m):
            lo, hi = config.evader_azimuth
            heading = draw((lo+(hi-lo)*j/m, lo+(hi-lo)*(j+1)/m))
            distance = options.get("distance")
            if distance is None:
                distance = draw(config.spawn_distance)
            height = draw(config.evader_height, (.35, .65)[j])
            dz = height - goal_point[2]
            if not np.isfinite(distance) or distance <= abs(dz):
                failures["invalid target distance for sampled height"] += 1
                break
            horizontal = np.sqrt(distance**2-dz**2)
            e[j, :3] = [goal_point[0]+horizontal*np.cos(heading),
                        goal_point[1]+horizontal*np.sin(heading), height]
            speed = 0. if fixed else draw(config.evader_initial_speed)
            e[j, 3:] = project(unit(goal_point-e[j, :3]), speed,
                               config.evader_vertical_vmax)
        else:
            for i in range(n):
                lo, hi = config.pursuer_azimuth
                heading = draw((lo+(hi-lo)*i/n, lo+(hi-lo)*(i+1)/n), .63)
                if config.task == "S0" and fixed:
                    heading = np.arctan2(*(e[0, :2]-config.center[:2])[::-1])
                radius = draw(config.defense_distance, .4+.2*(i % 2))
                height = e[0, 2] if config.task == "S0" and fixed else draw(
                    config.pursuer_height, (i+1)/(n+1))
                p[i, :3] = [config.center[0]+radius*np.cos(heading),
                            config.center[1]+radius*np.sin(heading), height]
                if not fixed:
                    direction = unit(rng.normal(size=3))
                    p[i, 3:] = draw(config.pursuer_initial_speed)*direction
            try:
                validate(p, e, config, spawn=True)
            except ValueError as exc:
                failures[str(exc)] += 1
            else:
                return p, e, dict(failures)
            continue
        continue
    raise ValueError(f"initialization rejected: {dict(failures)}")


def evader_commands(evaders, evader_mask, pursuers, config, phases, time):
    commands = np.zeros((len(evaders), 3))
    destination = navigation_point(config)
    for j in np.flatnonzero(evader_mask):
        point = evaders[j, :3]
        goal = unit(destination - point)
        direction = goal
        if config.evader_mode == "constant_velocity":
            direction = unit(np.asarray(config.constant_direction))
        elif config.evader_mode == "bounded_maneuver":
            phase = phases[j] + 2 * np.pi * time / config.maneuver_period
            direction = unit(goal + config.maneuver_amplitude *
                             np.array([np.cos(phase), np.sin(phase), np.sin(2 * phase)]))
        elif config.evader_mode == "reactive_evasion":
            direction = reactive_direction(evaders[j], pursuers, config, goal)
        commands[j] = project(direction, config.evader_speed_fraction * config.evader_vmax,
                              config.evader_vertical_vmax)
    return commands


def reactive_direction(state, pursuers, config, goal):
    nearby = pursuers[np.linalg.norm(pursuers[:, :3]-state[:3], axis=1) <= config.reactive_range]
    if not len(nearby):
        return goal
    axis = np.eye(3)[np.argmin(abs(goal))]
    lateral = unit(np.cross(goal, axis))
    vertical = np.cross(goal, lateral)
    azimuth = np.arange(8)*np.pi/4
    ring = np.cos(azimuth[:, None])*lateral + np.sin(azimuth[:, None])*vertical
    angle = np.radians(config.reactive_max_angle_deg)
    candidates = np.vstack([goal, *(np.cos(a)*goal+np.sin(a)*ring for a in (angle/2, angle))])
    speed = config.evader_speed_fraction*config.evader_vmax
    # Predict the very same vertically capped commands used for actual motion.
    command = project(candidates, speed, config.evader_vertical_vmax)
    closest = np.full(len(candidates), np.inf)
    fits = np.ones(len(candidates), dtype=bool)
    margin = config.evader_collision_radius
    for t in np.linspace(0, config.reactive_horizon, 9)[1:]:
        predicted = advance(np.broadcast_to(state, (len(command), 6)), command,
                            config.evader_tau, t)[:, :3]
        # Constant current Pursuer velocity is an explicit prediction assumption.
        other = nearby[:, :3]+t*nearby[:, 3:]
        closest = np.minimum(closest, np.linalg.norm(predicted[:, None]-other, axis=-1).min(axis=1))
        fits &= ((predicted >= np.asarray(config.lower)+margin) &
                 (predicted <= np.asarray(config.upper)-margin)).all(axis=1)
    risk = np.maximum(0., 1.-closest/config.reactive_danger_distance)
    scores = (candidates @ goal - config.reactive_risk_weight*risk**2
              + config.reactive_turn_weight*(candidates @ unit(state[3:])))
    if fits.any():
        scores[~fits] = -np.inf
    return candidates[np.argmax(scores)]
