"""Preset Evader references tracked with the shared flight controller."""
from dataclasses import dataclass

import numpy as np

from ..preset_tracking import tracking_request
from ..preset_trajectory import ReferencePath, TrajectoryConfig, make_bounded_paths
from .config import PRESET_EVADER_MODES
from .motion import project
from .protected_area import navigation_point


@dataclass(frozen=True)
class EvaderReference:
    config: TrajectoryConfig
    paths: tuple[ReferencePath, ...]
    seed: int

    def commands(self, evaders, mask, time):
        active = np.flatnonzero(mask)
        paths = [self.paths[j] for j in active]
        positions, velocities, _, _ = zip(*(path.sample(time) for path in paths))
        c = self.config
        requests = tracking_request(paths, evaders[active], time, c.dt, c,
                                    np.asarray(positions), np.asarray(velocities))
        commands = np.zeros_like(evaders[:, :3])
        commands[active] = project(requests/c.vmax, c.vmax)
        return commands


def build_evader_reference(states, config, seed):
    speed = config.evader_speed_fraction*config.evader_vmax
    tracking = TrajectoryConfig(
        starts=states[:, :3], endpoints=[navigation_point(config)]*len(states),
        initial_velocities=np.zeros_like(states[:, 3:]), start_jitter=(0., 0., 0.),
        lower=config.lower, upper=config.upper, body_radius=config.evader_collision_radius,
        dt=config.dt, tau=config.evader_tau, vmax=speed,
        reference_speed=min(TrajectoryConfig.reference_speed, speed),
        horizontal_acceleration_max=config.horizontal_acceleration_max,
        climb_vmax=config.climb_vmax, descent_vmax=config.descent_vmax,
        event_time_tolerance=config.event_time_tolerance,
    )
    paths = make_bounded_paths(PRESET_EVADER_MODES[config.evader_mode], states, tracking, seed)
    return EvaderReference(tracking, tuple(paths), seed)
