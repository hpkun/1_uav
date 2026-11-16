"""Strictly paired 6v6 guidance trials with fixed project geometry."""
import argparse
import json
from pathlib import Path

import numpy as np

from algorithm.ideal_multi_predictive_guidance import METHODS,run_episode
from algorithm.ideal_pip_analysis import summarize_samples,acceleration_bins
from env.ideal_interception.config import (PROTOCOL,formal_6v6_config,
    formal_6v6_level2_config,zuo_paper_profile,SIX_DEFENDER_CIRCLE,SIX_INTRUDER_WEDGE)


def benchmark(config,seeds):
    if config.scale!="6v6" or config.assignment_policy!="hungarian_fixed":
        raise ValueError("formal fixed-assignment 6v6 config required")
    trials=[]
    errors={method:[] for method in METHODS}
    horizons={method:[] for method in METHODS}
    offline_samples={method:[] for method in METHODS}
    geometries=[]
    for seed in seeds:
        methods={}
        identity=None
        seed_geometry=None
        for method in METHODS:
            summary,trace=run_episode(config,method,seed)
            current=(summary["scenario_id"],summary["path_hashes"],
                summary["intruder_starts"],summary["defender_starts"],
                summary["original_assignment"],summary["intruder_paths"])
            if identity is None:
                identity=current
                seed_geometry=trace["scene_diagnostics"]["paths"]
                geometries.extend(seed_geometry)
            elif current!=identity:
                raise AssertionError(f"seed {seed}: guidance methods received different scenarios")
            errors[method].extend(row["future_position_error_m"]
                for frame in trace["frames"][1:]
                for row in frame["offline_evaluation"]["pairs"].values()
                if "future_position_error_m" in row)
            horizons[method].extend(pair["prediction_horizon_s"]
                for frame in trace["frames"][1:] for pair in frame["control"]["pairs"].values()
                if pair["prediction_horizon_s"] is not None)
            offline_samples[method].extend(row for frame in trace["frames"][1:]
                for row in frame["offline_evaluation"]["pairs"].values()
                if "velocity_estimate_difference_mps" in row)
            methods[method]=summary
        trials.append(dict(seed=seed,scenario_id=identity[0],path_hashes=identity[1],
            original_assignment=identity[4],geometry_paths=seed_geometry,methods=methods))
    count=len(trials)
    aggregates={}
    for method in METHODS:
        rows=[trial["methods"][method] for trial in trials]
        scored=errors[method]
        if len(scored)!=sum(row["future_truth_scored_steps"] for row in rows):
            raise AssertionError("future truth aggregation mismatch")
        predictive=sum(row["predictive_steps_total"] for row in rows)
        if len(horizons[method])!=predictive:
            raise AssertionError("prediction horizon aggregation mismatch")
        def mean_present(key):
            values=[row[key] for row in rows if row[key] is not None]
            return float(np.mean(values)) if values else None
        aggregates[method]=dict(trials=count,
            mission_success_trials=sum(row["mission_success"] for row in rows),
            mission_success_rate=sum(row["mission_success"] for row in rows)/count,
            total_intercepted=sum(row["intercepted_count"] for row in rows),
            interception_ratio=sum(row["intercepted_count"] for row in rows)/(6*count),
            total_breached=sum(row["breached_count"] for row in rows),
            breach_ratio=sum(row["breached_count"] for row in rows)/(6*count),
            mean_first_interception_time_s=mean_present("first_interception_time_s"),
            mean_last_interception_time_s=mean_present("last_interception_time_s"),
            mean_episode_resolution_time_s=mean_present("episode_resolution_time_s"),
            defender_boundary_count=sum(row["defender_boundary_count"] for row in rows),
            friendly_collision_count=sum(row["friendly_collision_count"] for row in rows),
            intruder_collision_count=sum(row["intruder_collision_count"] for row in rows),
            mean_minimum_defender_defender_separation_m=mean_present(
                "minimum_defender_defender_separation_m"),
            predictive_steps_total=predictive,
            future_truth_scored_steps=len(scored),
            fallback_steps=sum(row["fallback_steps"] for row in rows),
            no_positive_pip_solution_steps=sum(row["no_positive_pip_solution_steps"] for row in rows),
            prediction_horizon_min_s=float(min(horizons[method])) if predictive else None,
            prediction_horizon_mean_s=float(np.mean(horizons[method])) if predictive else None,
            prediction_horizon_p95_s=float(np.percentile(horizons[method],95)) if predictive else None,
            prediction_horizon_max_s=float(max(horizons[method])) if predictive else None,
            future_truth_coverage=len(scored)/predictive if predictive else None,
            prediction_beyond_reference_fraction=(sum(row["prediction_beyond_reference_fraction"]*
                row["predictive_steps_total"] for row in rows if row["predictive_steps_total"])/predictive
                if predictive else None),
            future_position_error_mean_m=float(np.mean(scored)) if scored else None,
            future_position_error_p95_m=float(np.percentile(scored,95)) if scored else None,
            future_position_error_max_m=float(max(scored)) if scored else None)
        aggregates[method]["same_state_pip"]=summarize_samples(offline_samples[method])
        aggregates[method]["acceleration_bins"]=acceleration_bins(offline_samples[method])
    geometry_keys=("path_length_ratio","maximum_chord_deviation_m",
        "maximum_heading_deviation_from_chord_deg","total_absolute_heading_change_deg",
        "curvature_p95_1pm","curvature_max_1pm",
        "normal_acceleration_p95_mps2","normal_acceleration_max_mps2")
    geometry_summary={key:dict(mean=float(np.mean([row[key] for row in geometries])),
        p95=float(np.percentile([row[key] for row in geometries],95)),
        max=float(max(row[key] for row in geometries))) for key in geometry_keys}
    return dict(protocol=PROTOCOL,scale="6v6",
        profile="unified" if config.dt==.05 else "paper_like",
        intruder_motion_level=config.intruder_motion_level,
        geometry=geometry_summary,
        seeds=list(seeds),seed_count=count,paired_scenarios=True,
        aggregates=aggregates,trials=trials)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start",type=int,default=17)
    parser.add_argument("--trials",type=int,default=50)
    parser.add_argument("--profile",choices=("unified","paper_like"),default="unified")
    parser.add_argument("--motion-level",choices=("level1_smooth","level2_maneuvering"),
                        default="level1_smooth")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.trials<1 or args.seed_start<0:
        parser.error("positive trial count and nonnegative seed required")
    helper=(formal_6v6_level2_config if args.motion_level=="level2_maneuvering"
            else formal_6v6_config)
    config=(helper() if args.profile=="unified" else
        zuo_paper_profile(scale="6v6",defender_starts=SIX_DEFENDER_CIRCLE,
            intruder_starts=SIX_INTRUDER_WEDGE,assignment_policy="hungarian_fixed",
            intruder_motion_level=args.motion_level))
    result=benchmark(config,range(args.seed_start,args.seed_start+args.trials))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    for method,row in result["aggregates"].items():
        print(f"{method}: mission={row['mission_success_trials']}/{args.trials} "
              f"intercepted={row['total_intercepted']}/{6*args.trials} "
              f"breached={row['total_breached']}/{6*args.trials}")


if __name__=="__main__":
    main()
