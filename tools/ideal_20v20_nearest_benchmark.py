"""Multi-seed benchmark for the independent 20v20 nearest-target baseline."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from algorithm.ideal_collision_avoidance import AvoidanceConfig, MODES
from algorithm.ideal_20v20_nearest_guidance import run_episode


def benchmark(seed_start, trials, avoidance=None):
    if type(seed_start) is not int or seed_start < 0 or type(trials) is not int or trials < 1:
        raise ValueError("invalid benchmark seed range")
    avoidance = AvoidanceConfig() if avoidance is None else avoidance
    rows = [run_episode(seed, avoidance)[0] for seed in range(seed_start, seed_start+trials)]
    def mean(key):
        return float(np.mean([row[key] for row in rows]))
    separations = [row["minimum_defender_defender_separation_m"] for row in rows
                   if row["minimum_defender_defender_separation_m"] is not None]
    result = dict(experiment="ideal_20v20_nearest_three_point_pip_benchmark",
        seed_start=seed_start, trials=trials, avoidance_mode=avoidance.mode,
        aggregate=dict(mean_ISR=mean("interception_ratio"),
            std_ISR=float(np.std([row["interception_ratio"] for row in rows])),
            mean_BR=mean("breach_ratio"),
            MSR=mean("mission_success"),
            mean_intercepted_count=mean("intercepted_count"),
            mean_breached_count=mean("breached_count"),
            friendly_collision_episode_rate=float(np.mean([
                row["friendly_collision_count"] > 0 for row in rows])),
            mean_friendly_collision_count=mean("friendly_collision_count"),
            unassigned_contact_episode_rate=float(np.mean([
                row["unassigned_contact_count"] > 0 for row in rows])),
            mean_unassigned_contact_count=mean("unassigned_contact_count"),
            mean_initial_unique_target_count=mean("initial_unique_target_count"),
            mean_initial_max_defenders_per_target=mean("initial_max_defenders_per_target"),
            mean_assignment_update_count=mean("assignment_update_count"),
            mean_target_resolution_reassignment_count=mean(
                "target_resolution_reassignment_count"),
            mean_total_reassigned_defenders=mean("total_reassigned_defenders"),
            mean_minimum_defender_defender_separation_m=(
                float(np.mean(separations)) if separations else None)),
        per_seed=rows)
    if not all(math.isfinite(value) for value in result["aggregate"].values()
               if value is not None):
        raise FloatingPointError("nonfinite benchmark aggregate")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=17)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--avoidance-mode", choices=MODES, default="none")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"output file already exists: {args.out}")
    result = benchmark(args.seed_start, args.trials,
                       AvoidanceConfig(mode=args.avoidance_mode))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"trials={args.trials} seed_start={args.seed_start} "
          f"mean_ISR={result['aggregate']['mean_ISR']:.3f} "
          f"mean_BR={result['aggregate']['mean_BR']:.3f} "
          f"MSR={result['aggregate']['MSR']:.3f} out={args.out}")


if __name__ == "__main__":
    main()
