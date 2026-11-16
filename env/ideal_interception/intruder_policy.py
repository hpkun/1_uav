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

    def metadata(self):
        return dict(policy=self.policy,start=self.start,goal=self.goal,
            control_points=(self.control_point_1,self.control_point_2),
            speed=self.speed,generation_seed=self.generation_seed,
            polyline_sha256=self.reference.specification()["polyline_sha256"])


def sample_zuo_bezier_plan(config, index, rng, generation_seed):
    """One reset-time candidate; caller rejects invalid scene geometry.

    The bounded uniform start/control distribution is a project adaptation,
    not a parameter distribution supplied by Zuo et al.
    """
    start=np.asarray(config.intruder_starts[index],float)+rng.uniform(
        -config.intruder_start_jitter_m,config.intruder_start_jitter_m,3)
    goal=np.asarray(config.protected_center,float)
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
