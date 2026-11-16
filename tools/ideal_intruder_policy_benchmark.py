"""Paired, model-free 1v1 guidance trials on randomized intruder plans."""
import argparse
import json
from pathlib import Path
from statistics import mean
import numpy as np

from algorithm.ideal_predictive_guidance import run_episode
from env.ideal_interception import InterceptionConfig


def compare(config, seeds):
    if config.scale!="1v1" or config.intruder_policy!="zuo_bezier":
        raise ValueError("paired benchmark requires formal 1v1 zuo_bezier policy")
    rows=[]
    future_errors={"zuo_tp":[],"three_point_pip":[]}
    for seed in seeds:
        methods={}
        hashes=[]
        for method in ("zuo_tp","three_point_pip"):
            summary,trace=run_episode(config,method,seed)
            future_errors[method].extend(frame["offline_evaluation"]["future_position_error_m"]
                for frame in trace["frames"][1:]
                if "future_position_error_m" in frame["offline_evaluation"])
            path_hash=trace["references"][0]["specification"]["polyline_sha256"]
            hashes.append(path_hash)
            methods[method]=summary
        if hashes[0]!=hashes[1]:
            raise AssertionError(f"seed {seed}: controllers saw different intruder plans")
        rows.append(dict(seed=seed,path_sha256=hashes[0],methods=methods))
    aggregates={}
    for method in ("zuo_tp","three_point_pip"):
        values=[row["methods"][method] for row in rows]
        contact=[v["actual_contact_time_s"] for v in values
                 if v["actual_contact_time_s"] is not None]
        horizon=[v for v in values if v["min_prediction_horizon_s"] is not None]
        predictive_total=sum(v["predictive_steps_total"] for v in values)
        scored_total=sum(v["future_truth_scored_steps"] for v in values)
        beyond_total=sum(v["prediction_beyond_reference_steps"] for v in values)
        errors=future_errors[method]
        if len(errors)!=scored_total:
            raise AssertionError("future truth aggregation differs from trace samples")
        aggregates[method]=dict(
            contact_count=sum(v["interception_count"]>0 for v in values),
            breach_count=sum(v["outcome"]=="breach" for v in values),
            defender_boundary_count=sum(v["outcome"]=="defender_boundary" for v in values),
            deadline_count=sum(v["outcome"]=="deadline_failure" for v in values),
            other_outcomes={outcome:sum(v["outcome"]==outcome for v in values)
                            for outcome in sorted({v["outcome"] for v in values})
                            if outcome not in ("all_intercepted","breach","defender_boundary")},
            mean_contact_time_s=mean(contact) if contact else None,
            mean_minimum_separation_m=mean(v["minimum_separation_m"] for v in values),
            fallback_steps=sum(v["fallback_steps"] for v in values),
            no_positive_pip_solution_steps=sum(v["no_positive_pip_solution_steps"] for v in values),
            horizon_min_s=min(v["min_prediction_horizon_s"] for v in horizon) if horizon else None,
            horizon_mean_s=(sum(v["mean_prediction_horizon_s"]*v["predictive_steps_total"]
                for v in horizon)/predictive_total) if predictive_total else None,
            horizon_max_s=max(v["max_prediction_horizon_s"] for v in horizon) if horizon else None,
            predictive_steps_total=predictive_total,
            future_truth_scored_steps=scored_total,
            future_truth_unscored_steps=predictive_total-scored_total,
            future_truth_coverage=scored_total/predictive_total if predictive_total else None,
            prediction_beyond_reference_steps=beyond_total,
            prediction_beyond_reference_fraction=beyond_total/predictive_total if predictive_total else None,
            future_position_error_mean_m=mean(errors) if errors else None,
            future_position_error_p95_m=float(np.percentile(errors,95)) if errors else None,
            future_position_error_max_m=max(errors) if errors else None)
        if method=="zuo_tp":
            aggregates[method].update(
                prediction_forward_fraction=sum(v["prediction_forward_fraction"]*
                    v["predictive_steps_total"] for v in values)/predictive_total,
                prediction_backward_fraction=sum(v["prediction_backward_fraction"]*
                    v["predictive_steps_total"] for v in values)/predictive_total,
                predicted_point_in_bounds_fraction=sum(v["predicted_point_in_bounds_fraction"]*
                    v["predictive_steps_total"] for v in values)/predictive_total,
                prediction_below_floor_fraction=sum(v["predicted_point_below_floor_count"]
                    for v in values)/predictive_total)
    return dict(policy=config.intruder_policy,seed_count=len(rows),seeds=list(seeds),
                shared_intruder_paths=True,unique_path_hashes=len({row["path_sha256"] for row in rows}),
                aggregates=aggregates,trials=rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start",type=int,default=17)
    parser.add_argument("--trials",type=int,default=20)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.seed_start<0 or args.trials<1:
        parser.error("seed start and trial count must be nonnegative / positive")
    result=compare(InterceptionConfig(scale="1v1"),range(args.seed_start,args.seed_start+args.trials))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    for method,row in result["aggregates"].items():
        print(f"{method}: contact={row['contact_count']}/{result['seed_count']} "
              f"breach={row['breach_count']} boundary={row['defender_boundary_count']} "
              f"fallback={row['fallback_steps']}")


if __name__=="__main__":
    main()
