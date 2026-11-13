"""Seeded bilateral scenario construction and fixed task association."""
from dataclasses import asdict
import hashlib
import json
import numpy as np

from .trajectory import Reference
from .geometry import region_entry


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
            within_bounds=bool(margin>=0)))
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
        intruder_positions = np.asarray(config.intruder_starts[:intr],float).copy()
        if config.initialization == "random":
            defender_positions += rng.uniform(-config.random_range,config.random_range,defender_positions.shape)
            intruder_positions += rng.uniform(-config.random_range,config.random_range,intruder_positions.shape)
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
        break
    else:
        raise ValueError(f"initialization failed after {config.init_max_attempts} attempts: {failure}")
    paths = [Reference(intruder_positions[j],config.intruder_ends[j],
                       config.intruder_speeds[j],config.intruder_mode,"stop")
             for j in range(intr)]
    defenders = np.zeros((5,6),float)
    intruders = np.zeros((2,6),float)
    directions = np.asarray(config.defender_directions[:d],float)
    directions /= np.linalg.norm(directions,axis=1)[:,None]
    defenders[:d,:3] = defender_positions
    defenders[:d,3:] = directions*np.asarray(config.defender_speeds[:d])[:,None]
    intruders[:intr,:3] = intruder_positions
    for j,path in enumerate(paths):
        intruders[j,3:] = path.sample(0.)[1]
    assignment = {i:i % intr for i in range(d)}
    identity = asdict(config)
    for key in ("seed","initialization","random_range","init_max_attempts"):
        identity.pop(key)
    identity = dict(config=identity,defenders=defenders.tolist(),intruders=intruders.tolist(),
                    assignment=assignment,paths=[path.specification() for path in paths])
    scenario_id = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":"),
                                             allow_nan=False).encode()).hexdigest()
    return defenders,intruders,paths,assignment,seed_id,scenario_id
