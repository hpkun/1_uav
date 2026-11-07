from collections import Counter
import numpy as np

from .config import SCALES
from .motion import advance, project


def unit(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm else np.zeros_like(vector)


def validate(pursuers, evaders, config):
    for states, count, speed in zip(
        (pursuers, evaders), SCALES[config.scale], (config.pursuer_vmax, config.evader_vmax)
    ):
        if states.shape != (count, 6) or not np.isfinite(states).all():
            raise ValueError("invalid initial state shape or values")
        if np.any(states[:, :3] < config.lower) or np.any(states[:, :3] > config.upper):
            raise ValueError("initial position outside arena")
        if np.any(np.linalg.norm(states[:, 3:], axis=1) > speed + 1e-10):
            raise ValueError("initial speed exceeds limit")
    points = np.vstack((pursuers, evaders))[:, :3]
    i, j = np.triu_indices(len(points), 1)
    if np.any(np.linalg.norm(points[i] - points[j], axis=1) <= config.collision_distance):
        raise ValueError("initial overlap")
    distances = np.linalg.norm(pursuers[:, None, :3] - evaders[:, :3], axis=-1)
    if np.any(distances <= config.capture_radius):
        raise ValueError("initial pair inside capture radius")
    if np.any(np.linalg.norm(evaders[:, :3] - config.center, axis=1) <= config.radius):
        raise ValueError("initial breach")
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
    if np.any(np.all(distances > reach_p + reach_e + shift + config.capture_radius, axis=0)):
        raise ValueError("target outside combined reachable envelope")


def generate(config, rng, options):
    scenario = options.get("scenario", "random")
    if scenario not in ("fixed", "random", "custom"):
        raise ValueError("unknown scenario")
    n, m = SCALES[config.scale]
    if scenario == "custom":
        p, e = (np.array(options[name], dtype=float, copy=True) for name in ("pursuers", "evaders"))
        validate(p, e, config)
        return p, e, {}
    failures = Counter()
    for _ in range(1 if scenario == "fixed" else config.spawn_attempts):
        fixed = scenario == "fixed"
        angle = 0. if fixed else rng.uniform(-np.pi, np.pi)
        p, e = np.zeros((n, 6)), np.zeros((m, 6))
        directions = []
        for j in range(m):
            heading = angle + j * np.pi
            slope = 0. if fixed else rng.uniform(-.12, .12)
            direction = unit(np.array([np.cos(heading), np.sin(heading), slope]))
            directions.append(direction)
            distance = options.get("distance")
            if distance is None:
                distance = np.mean(config.spawn_distance) if fixed else rng.uniform(*config.spawn_distance)
            e[j, :3] = config.center + float(distance) * direction
        for i in range(n):
            direction = directions[i % m]
            tangent = np.array([-direction[1], direction[0], 0.])
            lane = (0., 8., -8.)[i // m]
            p[i, :3] = config.center + config.defense_distance * direction + lane * tangent
        try:
            validate(p, e, config)
        except ValueError as exc:
            failures[str(exc)] += 1
        else:
            return p, e, dict(failures)
    raise ValueError(f"initialization rejected: {dict(failures)}")


def evader_commands(evaders, evader_mask, pursuers, config, phases, time):
    commands = np.zeros((len(evaders), 3))
    for j in np.flatnonzero(evader_mask):
        point = evaders[j, :3]
        goal = unit(np.asarray(config.center) - point)
        direction = goal
        if config.evader_mode == "constant_velocity":
            direction = unit(np.asarray(config.constant_direction))
        elif config.evader_mode == "bounded_maneuver":
            phase = phases[j] + 2 * np.pi * time / config.maneuver_period
            direction = unit(goal + config.maneuver_amplitude *
                             np.array([np.cos(phase), np.sin(phase), np.sin(2 * phase)]))
        elif config.evader_mode == "reactive_evasion":
            away = point - pursuers[:, :3]
            distances = np.linalg.norm(away, axis=1)
            weight = np.maximum(0., 1. - distances / config.reactive_range)
            repel = (away / np.maximum(distances[:, None], 1e-12) * weight[:, None]).sum(0)
            # The blended vector can slow the target even in a symmetric
            # approach where lateral repulsion cancels.
            direction = goal + config.reactive_gain * repel
        commands[j] = project(direction, config.evader_speed_fraction * config.evader_vmax)
    return commands
