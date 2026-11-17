"""Paired 6v6 ThreePoint-PIP trials differing only in friendly avoidance."""
import argparse
import json
from pathlib import Path

import numpy as np

from algorithm.ideal_collision_avoidance import AvoidanceConfig,MODES
from algorithm.ideal_multi_predictive_guidance import run_episode
from env.ideal_interception.config import (PROTOCOL,formal_6v6_level2_config,
    formal_6v6_avoidance_stress_config)


def benchmark(config,seeds):
    if config.scale!="6v6" or config.intruder_motion_level!="level2_maneuvering":
        raise ValueError("Level-2 6v6 config required")
    seeds=list(seeds)
    trials=[]
    deviations={mode:[] for mode in MODES}
    for seed in seeds:
        methods={}
        identity=None
        for mode in MODES:
            summary,trace=run_episode(config,"three_point_pip",seed,AvoidanceConfig(mode=mode))
            current=(summary["scenario_id"],summary["path_hashes"],
                summary["defender_starts"],summary["intruder_starts"],
                summary["intruder_paths"],summary["original_assignment"])
            if identity is None:
                identity=current
            elif identity!=current:
                raise AssertionError(f"seed {seed}: avoidance modes received different scenarios")
            deviations[mode].extend(pair["avoidance_guidance_deviation_deg"]
                for frame in trace["frames"][1:] for pair in frame["control"]["pairs"].values())
            methods[mode]=summary
        trials.append(dict(seed=seed,scenario_id=identity[0],path_hashes=identity[1],
            defender_starts=identity[2],intruder_starts=identity[3],
            original_assignment=identity[5],methods=methods))
    n=len(trials)
    aggregates={}
    for mode in MODES:
        rows=[trial["methods"][mode] for trial in trials]
        active=sum(row["avoidance_active_steps"] for row in rows)
        eligible=sum(row["eligible_active_defender_steps"] for row in rows)
        if len(deviations[mode])!=eligible:
            raise AssertionError("guidance deviation denominator mismatch")
        def mean_present(key):
            values=[row[key] for row in rows if row[key] is not None]
            return float(np.mean(values)) if values else None
        aggregates[mode]=dict(episodes=n,mission_success_rate=sum(
            row["mission_success"] for row in rows)/n,
            interception_ratio=sum(row["intercepted_count"] for row in rows)/(6*n),
            breach_ratio=sum(row["breached_count"] for row in rows)/(6*n),
            total_intercepted=sum(row["intercepted_count"] for row in rows),
            total_breached=sum(row["breached_count"] for row in rows),
            friendly_collision_count=sum(row["friendly_collision_count"] for row in rows),
            episodes_with_collision_rate=sum(row["friendly_collision_count"]>0 for row in rows)/n,
            safety_violation_count=sum(row["safety_violation_count"] for row in rows),
            episodes_with_safety_violation_rate=sum(row["safety_violation_count"]>0 for row in rows)/n,
            alert_entry_count=sum(row["alert_entry_count"] for row in rows),
            alert_exposure_pair_seconds=sum(row["alert_exposure_pair_seconds"] for row in rows),
            minimum_friendly_distance_m=min(row["minimum_friendly_distance_m"] for row in rows),
            mean_minimum_friendly_distance_m=mean_present("minimum_friendly_distance_m"),
            minimum_predicted_cpa_distance_m=min(row["minimum_predicted_cpa_distance_m"] for row in rows),
            maximum_cpa_risk=max(row["maximum_cpa_risk"] for row in rows),
            mean_first_interception_time_s=mean_present("first_interception_time_s"),
            mean_last_interception_time_s=mean_present("last_interception_time_s"),
            mean_episode_resolution_time_s=mean_present("episode_resolution_time_s"),
            mean_defender_path_length_m=mean_present("mean_defender_path_length_m"),
            total_defender_path_length_m=sum(row["total_defender_path_length_m"] for row in rows),
            avoidance_active_steps=active,eligible_active_defender_steps=eligible,
            avoidance_active_fraction=active/eligible if eligible else None,
            predictive_early_activation_fraction=sum(
                row["predictive_early_activation_count"] for row in rows)/active if active else None,
            guidance_deviation_deg=dict(mean=float(np.mean(deviations[mode])),
                p95=float(np.percentile(deviations[mode],95)),max=float(max(deviations[mode]))))
    return dict(protocol=PROTOCOL,method="three_point_pip",modes=list(MODES),
        intruder_motion_level=config.intruder_motion_level,
        scenario_profile_version=config.scenario_profile_version,
        defender_circle_radius_m=config.defender_circle_radius_m or 50.,
        seeds=seeds,paired_scenarios=True,aggregates=aggregates,trials=trials)


def config_from_audit(path,seeds):
    audit=json.loads(Path(path).read_text(encoding="utf-8"))
    profile=audit["stress_profile"]
    if (audit["seeds"]!=list(seeds) or not profile["adequate"] or
            profile["version"]!="avoidance_stress_v1" or
            profile["selected_radius_m"] is None):
        raise ValueError("radius audit is inadequate or does not match benchmark seeds")
    return formal_6v6_avoidance_stress_config(profile["selected_radius_m"])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-profile",choices=("nominal","stress"),required=True)
    parser.add_argument("--radius-audit",type=Path)
    parser.add_argument("--seed-start",type=int,default=17)
    parser.add_argument("--trials",type=int,default=50)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.seed_start<0 or args.trials<1:
        parser.error("invalid seed range")
    seeds=list(range(args.seed_start,args.seed_start+args.trials))
    if args.scenario_profile=="stress" and args.radius_audit is None:
        parser.error("--radius-audit is required for the stress profile")
    config=(formal_6v6_level2_config() if args.scenario_profile=="nominal"
            else config_from_audit(args.radius_audit,seeds))
    result=benchmark(config,seeds)
    if args.scenario_profile=="stress":
        result["stress_profile"]=json.loads(args.radius_audit.read_text(
            encoding="utf-8"))["stress_profile"]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    for mode,row in result["aggregates"].items():
        print(f"{mode}: MSR={row['mission_success_rate']:.3f} "
              f"ISR={row['interception_ratio']:.3f} "
              f"collisions={row['friendly_collision_count']} "
              f"safety={row['safety_violation_count']} "
              f"active={row['avoidance_active_fraction']:.3f}")


if __name__=="__main__":
    main()
