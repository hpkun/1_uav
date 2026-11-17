"""Paired 6v6 ThreePoint-PIP comparison of ideal and acceleration-limited motion."""
import argparse
import json
from pathlib import Path

from algorithm.ideal_multi_predictive_guidance import run_episode as run_ideal
from algorithm.ideal_multi_predictive_guidance_accel import run_episode as run_accel
from env.ideal_interception.config import formal_6v6_level2_config as ideal_config
from env.ideal_interception_accel.config import formal_6v6_level2_config as accel_config
from tools.ideal_interception_trace import write_trace as write_ideal_trace
from tools.ideal_interception_accel_trace import write_trace as write_accel_trace


def compare(seeds, out=None):
    rows = []
    for seed in seeds:
        baseline, baseline_trace = run_ideal(ideal_config(), "three_point_pip", seed)
        cases = [("ideal", "inf", baseline, baseline_trace, write_ideal_trace)]
        for limit in (12., 8., 5.):
            summary, trace = run_accel(accel_config(defender_acceleration_max=limit),
                                       "three_point_pip", seed)
            if summary["path_hashes"] != baseline["path_hashes"]:
                raise ValueError("paired reference paths differ")
            cases.append((f"accel_{int(limit)}", limit, summary, trace, write_accel_trace))
        for name, limit, summary, trace, writer in cases:
            row = dict(acceleration_limit=limit, seed=seed,
                intercepted_count=summary["intercepted_count"],
                interception_ratio=summary["interception_ratio"],
                breach_count=summary["breached_count"],
                mission_success=summary["mission_success"],
                episode_resolution_time_s=summary["episode_resolution_time_s"],
                mean_interception_time_s=summary["mean_interception_time_s"],
                prediction_horizon_s=summary["prediction_horizon_s"],
                acceleration_limited_fraction=summary.get("acceleration_limited_fraction", 0.),
                scenario_id=summary["scenario_id"],path_hashes=summary["path_hashes"])
            rows.append(row)
            if out is not None:
                writer(out/f"{name}_seed{seed}.json", trace)
    if out is not None:
        with (out/"summary.json").open("x", encoding="utf-8") as stream:
            json.dump(rows, stream, indent=2, allow_nan=False)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--out", type=Path, default=Path("outputs/ideal_accel_sweep"))
    args = parser.parse_args(argv)
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique nonnegative integers")
    args.out.mkdir(parents=True, exist_ok=False)
    for row in compare(args.seeds, args.out):
        print(f"seed={row['seed']} a_max={row['acceleration_limit']} "
              f"interception_ratio={row['interception_ratio']:.3f} "
              f"breach_count={row['breach_count']} "
              f"mission_success={row['mission_success']} "
              f"acceleration_limited_fraction={row['acceleration_limited_fraction']:.3f}")


if __name__ == "__main__":
    main()
