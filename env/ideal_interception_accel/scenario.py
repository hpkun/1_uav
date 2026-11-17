"""Seeded bilateral scenarios with event-driven nearest-distance association."""
from dataclasses import asdict
import hashlib
import json
import numpy as np

from .trajectory import Reference
from .geometry import region_entry
from .assignment import nearest_distance_assignment,hungarian_fixed_assignment
from .config import DEFENDER_SLOTS,INTRUDER_SLOTS
from .intruder_policy import sample_zuo_bezier_plan,sample_level2_bezier_plan


def inside_region(position, config):
    relative = np.asarray(position)-np.asarray(config.protected_center)
    return (relative[0]**2+relative[1]**2 <= config.protected_radius**2
            and 0 <= relative[2] <= config.protected_height)


def diagnostics(config,paths):
    length=float(max(np.asarray(config.upper)[:2]-np.asarray(config.lower)[:2]))
    budget=config.dt*config.max_steps
    defender_ratio=max(config.defender_speeds[:config.defender_count])*config.dt/length
    intruder_ratio=max(config.intruder_speeds[:config.intruder_count])*config.dt/length
    rows=[]
    for path in paths:
        entry=None
        points=path.points
        center=np.asarray(config.protected_center)
        low=np.minimum(points[:-1],points[1:])
        high=np.maximum(points[:-1],points[1:])
        possible=(np.all(high[:,:2]>=center[:2]-config.protected_radius,axis=1)
                  & np.all(low[:,:2]<=center[:2]+config.protected_radius,axis=1)
                  & (high[:,2]>=center[2]) & (low[:,2]<=center[2]+config.protected_height))
        for index in np.flatnonzero(possible):
            left=path.distance[index]/path.speed
            velocity=(points[index+1]-points[index])/path.lengths[index]*path.speed
            hit=region_entry(points[index],velocity,center,config.protected_radius,
                             config.protected_height,path.lengths[index]/path.speed)
            if hit is not None:
                entry=left+hit
                break
        margin=float(min(np.min(path.points-np.asarray(config.lower)),
                         np.min(np.asarray(config.upper)-path.points)))
        rows.append(dict(duration=path.duration,duration_fraction=path.duration/budget,
            protected_region_entry_time=entry,
            path_class="region_crossing" if entry is not None else "bypass",
            entry_within_episode=bool(entry is not None and entry<=budget),
            path_complete_within_episode=bool(path.duration<=budget),
            minimum_boundary_margin=margin,
            altitude_range=[float(path.points[:,2].min()),float(path.points[:,2].max())],
            within_bounds=bool(margin>=0),**path.geometry_diagnostics()))
    return dict(reference_length=length,step_ratio=intruder_ratio,
                defender_step_ratio=defender_ratio,intruder_step_ratio=intruder_ratio,
                maximum_step_ratio=max(defender_ratio,intruder_ratio),
                protected_radius_ratio=config.protected_radius/length,paths=rows,
                scenario_validity=dict(geometry=True,
                    region_crossing=all(row["within_bounds"] and row["entry_within_episode"] for row in rows)))


def validate_scenario(report,kind="geometry"):
    if kind not in ("geometry","region_crossing"):
        raise ValueError("unknown scenario type")
    if not report["scenario_validity"][kind]:
        raise ValueError(f"scenario does not satisfy declared {kind} geometry")


def initialize(config, seed):
    sequence = seed if isinstance(seed,np.random.SeedSequence) else np.random.SeedSequence(
        config.seed if seed is None else seed)
    seed_id = int.from_bytes(sequence.generate_state(4,dtype="<u4").tobytes(),"little")
    rng = np.random.default_rng(sequence)
    d, intr = config.defender_count, config.intruder_count
    failure = None
    for _ in range(config.init_max_attempts):
        defender_positions = np.asarray(config.defender_starts[:d],float).copy()
        if config.initialization == "random":
            defender_positions += rng.uniform(-config.random_range,config.random_range,defender_positions.shape)
        if config.intruder_policy=="zuo_bezier":
            try:
                sampler=(sample_level2_bezier_plan if config.intruder_motion_level=="level2_maneuvering"
                         else sample_zuo_bezier_plan)
                plans=[sampler(config,j,rng,seed_id) for j in range(intr)]
            except ValueError:
                failure="invalid cubic Bezier candidate"
                continue
            intruder_positions=np.asarray([plan.start for plan in plans],float)
            paths=[plan.reference for plan in plans]
        else:
            plans=[]
            intruder_positions = np.asarray(config.intruder_starts[:intr],float).copy()
            if config.initialization == "random":
                intruder_positions += rng.uniform(-config.random_range,config.random_range,intruder_positions.shape)
            paths=[Reference(intruder_positions[j],config.intruder_ends[j],
                             config.intruder_speeds[j],config.intruder_mode,"stop")
                   for j in range(intr)]
        positions = np.concatenate((defender_positions,intruder_positions))
        if np.any(positions <= config.lower) or np.any(positions >= config.upper):
            failure = "member outside region"
            continue
        if any(np.linalg.norm(positions[a]-positions[b]) <= config.contact_distance+1e-9
               for a in range(len(positions)) for b in range(a+1,len(positions))):
            failure = "initial contact overlap"
            continue
        if any(inside_region(position,config) for position in intruder_positions):
            failure = "intruder starts inside protected region"
            continue
        if plans and not diagnostics(config,paths)["scenario_validity"]["region_crossing"]:
            failure="intruder plan does not cross protected region within episode budget"
            continue
        break
    else:
        raise ValueError(f"initialization failed after {config.init_max_attempts} attempts: {failure}")
    for plan in plans:
        plan.reference.plan_metadata=plan.metadata()
    defenders = np.zeros((DEFENDER_SLOTS,6),float)
    intruders = np.zeros((INTRUDER_SLOTS,6),float)
    directions = np.asarray(config.defender_directions[:d],float)
    directions /= np.linalg.norm(directions,axis=1)[:,None]
    defenders[:d,:3] = defender_positions
    defenders[:d,3:] = directions*np.asarray(config.defender_speeds[:d])[:,None]
    intruders[:intr,:3] = intruder_positions
    for j,path in enumerate(paths):
        intruders[j,3:] = path.sample(0.)[1]
    assignment = (hungarian_fixed_assignment(defenders[:d,:3],intruders[:intr,:3])
                  if config.assignment_policy=="hungarian_fixed" else
                  nearest_distance_assignment(defenders[:,:3],intruders[:,:3],
                      np.arange(DEFENDER_SLOTS)<d,np.arange(INTRUDER_SLOTS)<intr))
    if config.intruder_policy=="zuo_bezier":
        # Only effective geometry and sampling settings belong to formal identity.
        fields=("scale","dt","max_steps","lower","upper","defender_speeds",
                "intruder_speeds","defender_starts","defender_directions",
                "intruder_starts","intruder_policy","intruder_start_jitter_m",
                "intruder_bezier_jitter_ratio","intruder_goal_altitude_mode",
                "protected_center","protected_radius",
                "protected_height","contact_distance")
        identity={key:getattr(config,key) for key in fields}
        if config.intruder_motion_level=="level2_maneuvering":
            identity["intruder_motion_level"]=config.intruder_motion_level
            identity.pop("intruder_bezier_jitter_ratio")
        if config.scenario_profile_version=="avoidance_stress_v1":
            identity["scenario_profile_version"]=config.scenario_profile_version
            identity["defender_circle_radius_m"]=config.defender_circle_radius_m
        if config.scale in ("6v6","15v15"):
            identity["assignment_policy"]=config.assignment_policy
    else:
        identity=asdict(config)
        for key in ("seed","initialization","random_range","init_max_attempts"):
            identity.pop(key)
        identity.pop("intruder_motion_level",None)
        if config.scale not in ("6v6","15v15"):
            identity.pop("assignment_policy")
    # Hash the historical physical slots, excluding newly appended inactive slots.
    identity_d=5 if config.scale in ("1v1","3v1","5v2") else config.defender_count
    identity_i=2 if config.scale in ("1v1","3v1","5v2") else config.intruder_count
    for key,count in (("defender_speeds",identity_d),("defender_starts",identity_d),
                      ("defender_directions",identity_d),("intruder_speeds",identity_i),
                      ("intruder_starts",identity_i),("intruder_ends",identity_i)):
        if key in identity:
            identity[key]=identity[key][:count]
    identity_defenders=defenders[:identity_d]
    identity_intruders=intruders[:identity_i]
    identity = dict(config=identity,defenders=identity_defenders.tolist(),intruders=identity_intruders.tolist(),
                    assignment=assignment,paths=[path.specification() for path in paths],
                    policy=config.intruder_policy,
                    plans=[path.plan_metadata for path in paths] if plans else [])
    scenario_id = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":"),
                                             allow_nan=False).encode()).hexdigest()
    return defenders,intruders,paths,assignment,seed_id,scenario_id
