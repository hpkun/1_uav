"""Generate complete intruder observations from ideal Reference paths, without env.step."""
import argparse
from dataclasses import asdict
import gzip
import json
import math
from pathlib import Path

from env.ideal_interception import InterceptionConfig, InterceptionEnv
from env.ideal_interception.config import PROTOCOL as ENVIRONMENT_PROTOCOL

PROTOCOL = "uav-ideal-blue-observations-v1"


def generate(config, seed=17):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    env = InterceptionEnv(config)
    _, info = env.reset(seed)
    targets = []
    for j, path in enumerate(env.paths):
        # A non-grid endpoint is metadata, never a fourth irregular observation.
        count = math.floor(path.duration / config.dt + 1e-10) + 1
        while (count - 1) * config.dt > path.duration:
            count -= 1
        observations = []
        for k in range(count):
            t = k * config.dt
            if t > path.duration:
                raise ArithmeticError("regular observation exceeds reference duration")
            position, velocity = path.sample(t)
            observations.append(dict(time=t,position=position.tolist(),
                                     velocity=velocity.tolist(),active=True))
        endpoint_position, endpoint_velocity = path.sample(path.duration)
        targets.append(dict(target_index=j,configured_speed=config.intruder_speeds[j],
            reference_duration=path.duration,reference=path.specification(),
            first_observation_time=0.,last_regular_sample_time=observations[-1]["time"],
            number_of_regular_samples=len(observations),observations=observations,
            endpoint=dict(time=path.duration,position=endpoint_position.tolist(),
                          velocity=endpoint_velocity.tolist(),regular=math.isclose(
                              observations[-1]["time"],path.duration,rel_tol=0,abs_tol=1e-9))))
    return dict(protocol=PROTOCOL,environment_protocol=ENVIRONMENT_PROTOCOL,
        scenario_id=info["scenario_id"],seed=seed,environment_seed=info["seed"],
        scale=config.scale,intruder_mode=config.intruder_mode,
        initialization=config.initialization,dt=config.dt,target_count=len(targets),
        config=json.loads(json.dumps(asdict(config))),targets=targets)


def write_observations(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path,"xt",encoding="utf-8") as stream:
        json.dump(data,stream,separators=(",",":"),allow_nan=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--scale",choices=("1v1","3v1","5v2"))
    parser.add_argument("--mode",choices=("straight","bezier"))
    parser.add_argument("--initialization",choices=("fixed","random"))
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument("--config",type=Path)
    args=parser.parse_args(argv)
    if args.config:
        if any(value is not None for value in (args.scale,args.mode,args.initialization)):
            parser.error("--config cannot be combined with scene flags")
        config=InterceptionConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    else:
        config=InterceptionConfig(scale=args.scale or "1v1",intruder_mode=args.mode or "straight",
                                  initialization=args.initialization or "fixed")
    write_observations(args.out,generate(config,args.seed))


if __name__=="__main__":
    main()
