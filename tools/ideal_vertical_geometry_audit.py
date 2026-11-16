"""Paired strict-Zuo audit of old asset-z and new preserved-altitude goals."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from statistics import mean

from algorithm.ideal_predictive_guidance import run_episode
from env.ideal_interception import InterceptionConfig


def audit(seeds):
    base=InterceptionConfig(scale="1v1")
    configs={"asset_z":replace(base,intruder_goal_altitude_mode="asset_z"),
             "preserve_start_altitude":base}
    rows=[]
    for seed in seeds:
        by_mode={}
        for mode,config in configs.items():
            summary,trace=run_episode(config,"zuo_tp",seed)
            by_mode[mode]=dict(summary=summary,
                scenario_id=trace["scenario_id"],
                path_sha256=trace["references"][0]["specification"]["polyline_sha256"],
                start=trace["references"][0]["specification"]["start"],
                goal=trace["references"][0]["specification"]["end"])
        if by_mode["asset_z"]["start"]!=by_mode["preserve_start_altitude"]["start"]:
            raise AssertionError(f"seed {seed}: initial intruder states differ")
        rows.append(dict(seed=seed,modes=by_mode))
    aggregates={}
    for mode in configs:
        values=[row["modes"][mode]["summary"] for row in rows]
        count=sum(v["predictive_steps_total"] for v in values)
        aggregates[mode]=dict(contact_count=sum(v["interception_count"]>0 for v in values),
            defender_boundary_count=sum(v["outcome"]=="defender_boundary" for v in values),
            breach_count=sum(v["outcome"]=="breach" for v in values),
            mean_minimum_separation_m=mean(v["minimum_separation_m"] for v in values),
            prediction_below_floor_fraction=sum(v["predicted_point_below_floor_count"]
                for v in values)/count,
            prediction_backward_fraction=sum(v["predicted_point_behind_defender_count"]
                for v in values)/count,
            prediction_forward_fraction=sum(v["prediction_forward_fraction"]*
                v["predictive_steps_total"] for v in values)/count,
            predicted_point_in_bounds_fraction=sum(v["predicted_point_in_bounds_fraction"]*
                v["predictive_steps_total"] for v in values)/count)
    return dict(seeds=list(seeds),aggregates=aggregates,trials=rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start",type=int,default=17)
    parser.add_argument("--trials",type=int,default=20)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args(argv)
    if args.seed_start<0 or args.trials<1:
        parser.error("invalid seed range")
    result=audit(range(args.seed_start,args.seed_start+args.trials))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    for mode,summary in result["aggregates"].items():
        print(f"{mode}: contact={summary['contact_count']} "
              f"boundary={summary['defender_boundary_count']} breach={summary['breach_count']}")


if __name__=="__main__":
    main()
