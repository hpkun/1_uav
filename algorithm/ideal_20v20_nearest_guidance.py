"""Independent 20v20 nearest-target, ThreePoint-PIP deterministic episode."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from algorithm.common.serialization import serializable
from algorithm.ideal_collision_avoidance import (AvoidanceConfig, MODES,
    apply_collision_avoidance)
from env.ideal_interception.large_scale_20v20_env import (TwentyConfig,
    TwentyVTwentyEnv, PROTOCOL_20V20)
from prediction.history import IntruderHistory
from prediction.intercept_prediction import estimate_three_point_pip_from_history

TRACE_VERSION = "uav-ideal-20v20-nearest-trace-v1"


def _frame(info, *, control=None):
    frame = dict(time=info["time"], step=info["step"], outcome=info["outcome"],
        defenders=info["defenders"], intruders=info["intruders"],
        defender_mask=info["defender_mask"], intruder_mask=info["intruder_mask"],
        assignment=info["assignment"], events=info["events"],
        assignment_updates=info.get("assignment_updates", []))
    if "executed_velocities" in info:
        frame["executed_velocities"] = info["executed_velocities"]
    if control is not None:
        frame["control"] = control
    return serializable(frame)


def _decision(info, history, config, avoidance):
    history.observe(float(info["time"]), info["intruders"][:, :3],
                    info["intruder_mask"], info["scenario_id"])
    actions = np.full((config.defender_count, 3), .5)
    pairs = {}
    positions = np.asarray(info["defenders"][:, :3])
    velocities = np.asarray(info["defenders"][:, 3:])
    for i, j in sorted(info["assignment"].items()):
        own = positions[i]
        target = np.asarray(info["intruders"][j, :3])
        prediction = estimate_three_point_pip_from_history(
            history, j, own, config.defender_speeds[i])
        if prediction.status == "READY":
            point = np.asarray(prediction.predicted_target_position_m)
            mode = "THREE_POINT_PIP"
        else:
            point = target
            mode = ("WARMUP_PP" if prediction.status == "INSUFFICIENT_HISTORY"
                    else "FALLBACK_PP")
        delta = point-own
        distance = float(np.linalg.norm(delta))
        if not np.isfinite(distance) or distance <= 1e-12:
            delta = target-own
            distance = float(np.linalg.norm(delta))
            mode = "FALLBACK_PP"
        if not np.isfinite(distance) or distance <= 1e-12:
            raise FloatingPointError("no finite 20v20 guidance direction")
        direction = delta/distance
        final, avoidance_row = apply_collision_avoidance(avoidance, i, positions,
            velocities, info["defender_mask"], direction)
        actions[i] = .5+.5*final
        pairs[i] = dict(assigned_target=j, guidance_mode=mode,
            prediction_status=prediction.status,
            prediction_horizon_s=prediction.horizon_s,
            predicted_target_position_m=prediction.predicted_target_position_m,
            estimated_target_velocity_mps=prediction.estimated_target_velocity_mps,
            guidance_direction_unit=final.tolist(),
            pip_equation_degree=prediction.pip_equation_degree,
            pip_discriminant=prediction.pip_discriminant,
            avoidance_adjusted_direction_unit=final.tolist(), **avoidance_row)
    return actions, dict(decision_time_s=float(info["time"]), method="three_point_pip",
        avoidance_mode=avoidance.mode, action=actions.tolist(), pairs=pairs)


def run_episode(seed=17, avoidance=None):
    avoidance = AvoidanceConfig() if avoidance is None else avoidance
    config = TwentyConfig(seed=seed)
    env = TwentyVTwentyEnv(config)
    _, info = env.reset(seed)
    history = IntruderHistory(config.dt, slots=config.intruder_count)
    first = info
    frames = [_frame(info)]
    all_events = []
    predictive = warmup = fallback = no_root = 0
    horizons = []
    while True:
        actions, control = _decision(info, history, config, avoidance)
        for pair in control["pairs"].values():
            warmup += pair["guidance_mode"] == "WARMUP_PP"
            fallback += pair["guidance_mode"] == "FALLBACK_PP"
            no_root += pair["prediction_status"] == "NO_POSITIVE_PIP_SOLUTION"
            if pair["prediction_horizon_s"] is not None:
                predictive += 1
                horizons.append(pair["prediction_horizon_s"])
        _, _, terminated, truncated, info = env.step(actions)
        frames.append(_frame(info, control=control))
        all_events.extend(info["events"])
        if terminated or truncated:
            break
    contacts = [event["time"] for event in all_events
                if event["kind"] == "interception_contact"]
    summary = dict(experiment="ideal_20v20_nearest_three_point_pip",
        protocol=PROTOCOL_20V20, seed=seed, scenario_id=info["scenario_id"],
        assignment_policy="nearest_dynamic", avoidance_mode=avoidance.mode,
        intercepted_count=info["intercepted_count"], breached_count=info["breached_count"],
        remaining_intruder_count=info["remaining_intruder_count"],
        interception_ratio=info["interception_ratio"], breach_ratio=info["breach_ratio"],
        mission_success=info["mission_success"], outcome=info["outcome"],
        friendly_collision_count=info["friendly_collision_count"],
        intruder_collision_count=info["intruder_collision_count"],
        defender_boundary_count=info["defender_boundary_count"],
        episode_resolution_time_s=info["time"],
        first_interception_time_s=min(contacts) if contacts else None,
        mean_interception_time_s=float(np.mean(contacts)) if contacts else None,
        last_interception_time_s=max(contacts) if contacts else None,
        minimum_defender_defender_separation_m=info["minimum_separation"],
        predictive_steps_total=predictive, warmup_steps=warmup,
        fallback_steps=fallback, no_positive_pip_solution_steps=no_root,
        initial_unique_target_count=info["initial_unique_target_count"],
        initial_max_defenders_per_target=info["initial_max_defenders_per_target"],
        assignment_update_count=info["assignment_update_count"],
        mean_unique_targets_assigned=info["mean_unique_targets_assigned"],
        minimum_unique_targets_assigned=info["minimum_unique_targets_assigned"],
        maximum_defenders_per_single_target=info["maximum_defenders_per_single_target"],
        path_hashes=[path.specification()["polyline_sha256"] for path in env.paths],
        original_assignment=first["assignment"])
    trace = dict(trace_version=TRACE_VERSION, environment_protocol=PROTOCOL_20V20,
        config=asdict(config), seed=seed, scenario_id=info["scenario_id"],
        scene_diagnostics=env.scene_diagnostics,
        references=[dict(specification=path.specification(), points=path.points.tolist(),
                         plan_metadata=path.plan_metadata) for path in env.paths],
        initial_assignment=first["assignment"],
        assignment_history=env.assignment_history, frames=frames,
        result=dict(summary=summary, events=all_events,
                    defender_status=info["defender_status"],
                    intruder_status=info["intruder_status"]))
    return serializable(summary), serializable(trace)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--avoidance-mode", choices=MODES, default="none")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"output directory already exists: {args.out}")
    summary, trace = run_episode(args.seed, AvoidanceConfig(mode=args.avoidance_mode))
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out/"summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    with (args.out/"episode.json").open("x", encoding="utf-8") as stream:
        json.dump(trace, stream, separators=(",", ":"), allow_nan=False)
    print(f"seed={args.seed} intercepted={summary['intercepted_count']}/20 "
          f"breached={summary['breached_count']}/20 "
          f"interception_ratio={summary['interception_ratio']:.3f} "
          f"friendly_collisions={summary['friendly_collision_count']} "
          f"initial_unique_targets={summary['initial_unique_target_count']} "
          f"initial_max_defenders_per_target={summary['initial_max_defenders_per_target']} "
          f"assignment_updates={summary['assignment_update_count']} "
          f"episode_time={summary['episode_resolution_time_s']:.3f}s")


if __name__ == "__main__":
    main()
