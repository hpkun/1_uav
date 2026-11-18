"""Project-specific delayed-convergence reference geometry for the isolated 20v20 scene."""
import numpy as np

from .intruder_policy import IntruderPlan
from .trajectory import Reference
from .scenario import diagnostics, inside_region


PROFILE = "20v20_wedge_delayed_convergence_v1"
BETA_1 = .90
BETA_2 = .65
MAGIC = 0x20DC2026


def attack_frame(config):
    center = np.mean(np.asarray(config.intruder_starts, float)[:, :2], axis=0)
    vector = np.asarray(config.protected_center[:2], float)-center
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("invalid shared attack axis")
    longitudinal = vector/norm
    lateral = np.array((-longitudinal[1], longitudinal[0]))
    return longitudinal, lateral


def sample_20v20_delayed_convergence_plan(config, index, rng, generation_seed):
    """Preserve historical start draws; use independent per-member control streams."""
    start = np.asarray(config.intruder_starts[index], float)+rng.uniform(
        -config.intruder_start_jitter_m, config.intruder_start_jitter_m, 3)
    rng.uniform(-1., 1., (2, 2))  # Historical Level-1 pairing draws.
    goal = np.asarray(config.protected_center, float).copy()
    if config.intruder_goal_altitude_mode == "preserve_start_altitude":
        goal[2] = np.clip(start[2], goal[2], goal[2]+config.protected_height)
    chord = float(np.linalg.norm(goal-start))
    if not np.isfinite(start).all() or not np.isfinite(goal).all() or chord <= 0 or inside_region(start, config):
        raise ValueError("invalid delayed-convergence endpoints")
    e_long, e_lat = attack_frame(config)
    relative = start[:2]-goal[:2]
    long0 = float(relative@e_long)
    lane0 = float(relative@e_lat)
    if long0 >= 0:
        raise ValueError("20v20 intruder start is beyond the shared target")
    lateral_limit = min(12., .03*chord)
    vertical_limit = min(6., .015*chord)
    seed_words = [(generation_seed>>(32*n))&0xffffffff for n in range(4)]
    for attempt in range(config.init_max_attempts):
        controls_rng = np.random.default_rng(np.random.SeedSequence(
            seed_words+[index, attempt, MAGIC]))
        lateral_noise = controls_rng.uniform(-lateral_limit, lateral_limit, 2)
        vertical_noise = controls_rng.uniform(-vertical_limit, vertical_limit, 2)
        controls = []
        for fraction, beta, side, up in zip((1/3, 2/3), (BETA_1, BETA_2),
                                             lateral_noise, vertical_noise):
            point = np.empty(3)
            point[:2] = goal[:2]+(1.-fraction)*long0*e_long+(beta*lane0+side)*e_lat
            point[2] = start[2]+fraction*(goal[2]-start[2])+up
            controls.append(point)
        if not np.isfinite(controls).all():
            continue
        try:
            reference = Reference(start, goal, config.intruder_speeds[index], "bezier",
                                  control_points=controls)
        except ValueError:
            continue
        geometry = reference.geometry_diagnostics()
        if not (3. <= geometry["maximum_chord_deviation_m"] <= 80. and
                3. <= geometry["maximum_heading_deviation_from_chord_deg"] <= 60.):
            continue
        if (np.any(reference.points <= config.lower) or np.any(reference.points >= config.upper)
                or not diagnostics(config, [reference])["scenario_validity"]["region_crossing"]):
            continue
        point = lambda value: tuple(float(x) for x in value)
        metadata = dict(trajectory_profile=PROFILE,
            construction="project-specific trajectory construction",
            control_point_1=point(controls[0]), control_point_2=point(controls[1]),
            initial_lane_offset_m=lane0, beta1=BETA_1, beta2=BETA_2,
            lateral_noise_limit_m=lateral_limit, vertical_noise_limit_m=vertical_limit,
            lateral_noise_1_m=float(lateral_noise[0]),
            lateral_noise_2_m=float(lateral_noise[1]),
            vertical_noise_1_m=float(vertical_noise[0]),
            vertical_noise_2_m=float(vertical_noise[1]))
        return IntruderPlan("level2_maneuvering", point(start), point(goal),
            point(controls[0]), point(controls[1]), float(config.intruder_speeds[index]),
            int(generation_seed), reference, metadata)
    raise ValueError("20v20 delayed-convergence sampling exhausted geometry attempts")
