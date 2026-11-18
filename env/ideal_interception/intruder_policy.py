"""Pre-programmed, defender-independent intruder reference plans."""
from dataclasses import dataclass

import numpy as np

from .trajectory import Reference


@dataclass(frozen=True)
class IntruderPlan:
    policy: str
    start: tuple[float, float, float]
    goal: tuple[float, float, float]
    control_point_1: tuple[float, float, float]
    control_point_2: tuple[float, float, float]
    speed: float
    generation_seed: int
    reference: Reference
    profile_metadata: dict | None = None

    def metadata(self):
        altitudes=self.reference.points[:,2]
        row=dict(policy=self.policy,start=self.start,goal=self.goal,
            control_points=(self.control_point_1,self.control_point_2),
            speed=self.speed,generation_seed=self.generation_seed,
            polyline_sha256=self.reference.specification()["polyline_sha256"],
            start_altitude_m=self.start[2],goal_altitude_m=self.goal[2],
            altitude_change_m=self.goal[2]-self.start[2],
            minimum_altitude_m=float(altitudes.min()),
            maximum_altitude_m=float(altitudes.max()),
            maximum_absolute_altitude_deviation_from_start_m=float(
                np.max(np.abs(altitudes-self.start[2]))))
        if self.policy=="level2_maneuvering":
            row["intruder_motion_level"]="level2_maneuvering"
        if self.profile_metadata is not None:
            row.update(self.profile_metadata)
        return row


def sample_zuo_bezier_plan(config, index, rng, generation_seed):
    """One reset-time candidate; caller rejects invalid scene geometry.

    The bounded uniform start/control distribution is a project adaptation,
    not a parameter distribution supplied by Zuo et al.
    """
    start=np.asarray(config.intruder_starts[index],float)+rng.uniform(
        -config.intruder_start_jitter_m,config.intruder_start_jitter_m,3)
    goal=np.asarray(config.protected_center,float).copy()
    if config.intruder_goal_altitude_mode=="preserve_start_altitude":
        goal[2]=np.clip(start[2],goal[2],goal[2]+config.protected_height)
    span=goal-start
    tangent=span/np.linalg.norm(span)
    lateral=np.array((-tangent[1],tangent[0],0.))
    lateral/=np.linalg.norm(lateral)
    vertical=np.cross(tangent,lateral)
    amplitude=config.intruder_bezier_jitter_ratio*float(np.linalg.norm(span))
    offsets=rng.uniform(-amplitude,amplitude,(2,2))
    controls=[]
    for fraction,(side,up) in zip((1/3,2/3),offsets):
        controls.append(start+fraction*span+side*lateral+up*vertical)
    reference=Reference(start,goal,config.intruder_speeds[index],"bezier",
                        control_points=controls)
    point=lambda value: tuple(float(x) for x in value)
    return IntruderPlan("zuo_bezier",point(start),point(goal),point(controls[0]),
                        point(controls[1]),float(config.intruder_speeds[index]),
                        int(generation_seed),reference)


def sample_level2_bezier_plan(config,index,rng,generation_seed):
    """Pre-generate a defender-independent cubic S-turn; reject only geometry.

    Consume the Level-1 control draws on the scene stream, so all six seeded
    starts remain paired with the existing Level-1 scenes.
    """
    from .scenario import diagnostics,inside_region
    start=np.asarray(config.intruder_starts[index],float)+rng.uniform(
        -config.intruder_start_jitter_m,config.intruder_start_jitter_m,3)
    rng.uniform(-1.,1.,(2,2))
    goal=np.asarray(config.protected_center,float).copy()
    if config.intruder_goal_altitude_mode=="preserve_start_altitude":
        goal[2]=np.clip(start[2],goal[2],goal[2]+config.protected_height)
    span=goal-start
    length=float(np.linalg.norm(span))
    if not np.isfinite(start).all() or not np.isfinite(goal).all() or length<=0 or inside_region(start,config):
        raise ValueError("invalid Level-2 endpoints")
    parallel=span/length
    lateral=np.array((-parallel[1],parallel[0],0.))
    lateral/=np.linalg.norm(lateral)
    vertical=np.cross(parallel,lateral)
    seed_words=[(generation_seed>>(32*n))&0xffffffff for n in range(4)]
    for attempt in range(config.init_max_attempts):
        controls_rng=np.random.default_rng(np.random.SeedSequence(
            seed_words+[index,attempt,0x53327572]))
        a1,a2=controls_rng.uniform(.08,.14,2)*length
        sign=1. if controls_rng.integers(0,2) else -1.
        vertical_offset=controls_rng.uniform(-.04,.04)*length
        first=start+span/3+sign*a1*lateral+vertical_offset*vertical
        second=start+2*span/3-sign*a2*lateral-vertical_offset*vertical
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            continue
        try:
            reference=Reference(start,goal,config.intruder_speeds[index],"bezier",
                                control_points=(first,second))
        except ValueError:
            continue
        geometry=reference.geometry_diagnostics()
        if (geometry["maximum_chord_deviation_m"]<8.
                or geometry["maximum_heading_deviation_from_chord_deg"]<8.):
            continue
        if not diagnostics(config,[reference])["scenario_validity"]["region_crossing"]:
            continue
        point=lambda value: tuple(float(x) for x in value)
        return IntruderPlan("level2_maneuvering",point(start),point(goal),
            point(first),point(second),float(config.intruder_speeds[index]),
            int(generation_seed),reference)
    raise ValueError("Level-2 control sampling exhausted geometry attempts")
