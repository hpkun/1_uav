"""Versioned, model-free trajectory capture for the ideal interception environment."""
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys

import numpy as np

from algorithm.common.serialization import serializable
from env.ideal_interception import InterceptionConfig, InterceptionEnv
from env.ideal_interception.config import PROTOCOL
from env.ideal_interception.trajectory import Reference

TRACE_VERSION = "uav-ideal-trace-v1"


def references_from_specs(specs):
    references = []
    for spec in specs:
        path = Reference(spec["start"],spec["end"],spec["speed"],spec["mode"])
        if path.specification() != spec:
            raise ValueError("reference path specification or hash mismatch")
        references.append(dict(specification=spec,points=path.points.tolist()))
    return references


def build_trace(config,seed,scenario_id,path_specs,frames,result):
    from env.ideal_interception.scenario import diagnostics
    paths=[Reference(spec["start"],spec["end"],spec["speed"],spec["mode"]) for spec in path_specs]
    first=frames[0]
    entities=[dict(id=i,role="defender",speed=config.defender_speeds[i],
                   initial_state=first["defenders"][i]) for i in range(5) if first["defender_mask"][i]]
    entities += [dict(id=5+j,role="intruder",speed=config.intruder_speeds[j],
                      initial_state=first["intruders"][j]) for j in range(2) if first["intruder_mask"][j]]
    return serializable(dict(trace_version=TRACE_VERSION,environment_protocol=PROTOCOL,
        config=asdict(config),seed=seed,scenario_id=scenario_id,
        scene=dict(lower=config.lower,upper=config.upper,
            protected_center=config.protected_center,protected_radius=config.protected_radius,
            protected_height=config.protected_height),
        entities=entities,scene_diagnostics=diagnostics(config,paths),
        references=references_from_specs(path_specs),frames=frames,result=result))


def from_evaluation(result):
    if result["protocol"]["version"] != PROTOCOL or not result["episodes"]:
        raise ValueError("incompatible ideal evaluation")
    episode = result["episodes"][0]
    return build_trace(InterceptionConfig(**result["environment"]),episode["seed"],
        episode["scenario_id"],episode["intruder_paths"],result["first_trace"],
        dict(outcome=episode["outcome"],time=episode["time"],events=episode["events"],
             episode_return=episode["episode_return"],reward_parts=episode["reward_parts"]))


def record(config,seed=17,actions=None,scenario_type="geometry"):
    """Capture a deterministic geometric fixture without MAPPO or CUDA."""
    env = InterceptionEnv(config)
    _,info = env.reset(seed)
    from env.ideal_interception.scenario import validate_scenario
    validate_scenario(info["scene_diagnostics"],scenario_type)
    path_specs = info["intruder_paths"]
    frames = [serializable(dict(time=0.,step=0,outcome="ongoing",
        defenders=info["defenders"],intruders=info["intruders"],
        defender_mask=info["defender_mask"],intruder_mask=info["intruder_mask"],events=[]))]
    all_events = []
    reward_parts = {}
    episode_return = 0.
    while True:
        action = np.full((5,3),.5) if actions is None else np.asarray(actions(info),float)
        _,_,terminated,truncated,info = env.step(action)
        all_events.extend(info["events"])
        episode_return += info["team_reward"]
        for key,value in info["reward_parts"].items():
            reward_parts[key] = reward_parts.get(key,0.)+value
        frames.append(serializable(dict(time=info["time"],step=info["step"],outcome=info["outcome"],
            defenders=info["defenders"],intruders=info["intruders"],
            defender_mask=info["defender_mask"],intruder_mask=info["intruder_mask"],
            executed_velocities=info["executed_velocities"],events=info["events"])))
        if terminated or truncated:
            break
    trace=build_trace(config,info["seed"],info["scenario_id"],path_specs,frames,
        dict(outcome=info["outcome"],time=info["time"],events=all_events,
             episode_return=episode_return,reward_parts=reward_parts))
    trace["scenario_type"]=scenario_type
    return trace


def write_trace(path,data):
    path = Path(path)
    opener = gzip.open if path.suffix==".gz" else open
    with opener(path,"xt",encoding="utf-8") as stream:
        json.dump(data,stream,separators=(",",":"),allow_nan=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--scale",choices=("1v1","3v1","5v2"),default="1v1")
    parser.add_argument("--mode",choices=("straight","bezier"),default="straight")
    parser.add_argument("--initialization",choices=("fixed","random"),default="fixed")
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument("--scenario-type",choices=("geometry","region_crossing"),default="geometry")
    parser.add_argument("--config",type=Path)
    args=parser.parse_args(argv)
    if args.config:
        if any(flag in (sys.argv[1:] if argv is None else argv)
               for flag in ("--scale","--mode","--initialization")):
            parser.error("--config cannot be combined with scene flags")
        config=InterceptionConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    else:
        config=InterceptionConfig(scale=args.scale,intruder_mode=args.mode,
                                  initialization=args.initialization)
    write_trace(args.output,record(config,args.seed,scenario_type=args.scenario_type))


if __name__=="__main__":
    main()
