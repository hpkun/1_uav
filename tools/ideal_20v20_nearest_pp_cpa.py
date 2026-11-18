"""Evaluate nearest assignment + pure pursuit + existing predictive CPA-kNN."""
import argparse
import json
from pathlib import Path

import numpy as np

from algorithm.ideal_collision_avoidance import AvoidanceConfig, apply_collision_avoidance
from env.ideal_interception.large_scale_20v20_env import TwentyConfig, TwentyVTwentyEnv


def decide(info, avoidance):
    positions = np.asarray(info["defenders"][:, :3], float)
    velocities = np.asarray(info["defenders"][:, 3:], float)
    targets = np.asarray(info["intruders"][:, :3], float)
    actions = np.full((len(positions), 3), .5)
    active_avoidance = 0
    for i, j in sorted(info["assignment"].items()):
        delta = targets[j]-positions[i]
        distance = float(np.linalg.norm(delta))
        if not np.isfinite(distance) or distance <= 1e-12:
            raise FloatingPointError("pure pursuit has no finite target direction")
        direction, row = apply_collision_avoidance(avoidance, i, positions,
            velocities, info["defender_mask"], delta/distance)
        actions[i] = .5+.5*direction
        active_avoidance += bool(row["avoidance_active"])
    return actions, active_avoidance


def run_episode(seed=17):
    config = TwentyConfig(seed=seed)
    env = TwentyVTwentyEnv(config)
    _, info = env.reset(seed)
    initial = info
    avoidance = AvoidanceConfig(mode="predictive_cpa_knn")
    contacts = []
    active_decisions = 0
    while True:
        actions, count = decide(info, avoidance)
        active_decisions += count
        _, _, terminated, truncated, info = env.step(actions)
        contacts.extend(event["time"] for event in info["events"]
                        if event["kind"] == "interception_contact")
        if terminated or truncated:
            break
    return dict(experiment="ideal_20v20_nearest_pure_pursuit_cpa",
        protocol=info["protocol"], seed=seed, scenario_id=info["scenario_id"],
        assignment_policy=config.assignment_policy,
        guidance_method="pure_pursuit_current_assigned_target",
        avoidance_mode=avoidance.mode, intercepted_count=info["intercepted_count"],
        breached_count=info["breached_count"],
        unassigned_contact_count=info["unassigned_contact_count"],
        interception_ratio=info["interception_ratio"], breach_ratio=info["breach_ratio"],
        mission_success=info["mission_success"], outcome=info["outcome"],
        friendly_collision_count=info["friendly_collision_count"],
        intruder_collision_count=info["intruder_collision_count"],
        defender_boundary_count=info["defender_boundary_count"],
        minimum_defender_defender_separation_m=info["minimum_separation"],
        episode_resolution_time_s=info["time"],
        first_interception_time_s=min(contacts) if contacts else None,
        mean_interception_time_s=float(np.mean(contacts)) if contacts else None,
        last_interception_time_s=max(contacts) if contacts else None,
        initial_unique_target_count=initial["initial_unique_target_count"],
        initial_max_defenders_per_target=initial["initial_max_defenders_per_target"],
        target_resolution_reassignment_count=info["target_resolution_reassignment_count"],
        total_reassigned_defenders=info["total_reassigned_defenders"],
        avoidance_active_decisions=active_decisions,
        path_hashes=[path.specification()["polyline_sha256"] for path in env.paths])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    result = run_episode(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False),
                                          encoding="utf-8")
    print(f"seed={args.seed} intercepted={result['intercepted_count']}/20 "
          f"breached={result['breached_count']}/20 "
          f"interception_ratio={result['interception_ratio']:.3f} "
          f"friendly_collisions={result['friendly_collision_count']} "
          f"time={result['episode_resolution_time_s']:.3f}s")


if __name__ == "__main__":
    main()
