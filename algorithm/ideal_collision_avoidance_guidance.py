"""Single 6v6 ThreePoint-PIP episode with an isolated friendly avoidance mode."""
import argparse
import json
from pathlib import Path

from .ideal_collision_avoidance import AvoidanceConfig,MODES
from .ideal_multi_predictive_guidance import run_episode
from env.ideal_interception.config import (formal_6v6_level2_config,
    formal_6v6_avoidance_stress_config)
from tools.ideal_interception_trace import write_trace


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method",choices=("three_point_pip",),default="three_point_pip")
    parser.add_argument("--avoidance-mode",choices=MODES,required=True)
    parser.add_argument("--motion-level",choices=("level2_maneuvering",),
                        default="level2_maneuvering")
    parser.add_argument("--scenario-profile",choices=("nominal","stress"),required=True)
    parser.add_argument("--radius-audit",type=Path)
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.seed<0:
        parser.error("seed must be nonnegative")
    profile=None
    if args.scenario_profile=="stress":
        if args.radius_audit is None:
            parser.error("--radius-audit is required for stress")
        profile=json.loads(args.radius_audit.read_text(encoding="utf-8"))["stress_profile"]
        if (not profile["adequate"] or profile["version"]!="avoidance_stress_v1" or
                profile["selected_radius_m"] is None):
            parser.error("radius audit does not establish an adequate stress profile")
        config=formal_6v6_avoidance_stress_config(profile["selected_radius_m"])
    else:
        config=formal_6v6_level2_config()
    summary,trace=run_episode(config,"three_point_pip",args.seed,
        AvoidanceConfig(mode=args.avoidance_mode))
    if profile is not None:
        summary["stress_profile"]=profile
        trace["stress_profile"]=profile
    args.out.mkdir(parents=True,exist_ok=False)
    with (args.out/"summary.json").open("x",encoding="utf-8") as stream:
        json.dump(summary,stream,indent=2,allow_nan=False)
    write_trace(args.out/"episode.json",trace)
    print(f"mode={args.avoidance_mode} seed={args.seed} "
          f"intercepted={summary['intercepted_count']}/6 "
          f"friendly_collision={summary['friendly_collision_count']} "
          f"safety={summary['safety_violation_count']} "
          f"time={summary['episode_resolution_time_s']:.3f}s")


if __name__=="__main__":
    main()
