"""Zuo load-balanced assignment, ThreePoint-PIP and predictive friendly avoidance."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from algorithm.common.serialization import serializable
from algorithm.ideal_20v20_nearest_guidance import _decision, _frame
from algorithm.ideal_collision_avoidance import AvoidanceConfig
from env.ideal_interception.large_scale_20v20_env import TwentyConfig, TwentyVTwentyEnv
from prediction.history import IntruderHistory


PROTOCOL = "uav-ideal-interception-20v20-zuo-allocation-pip-cpa-v3"
TRACE_VERSION = "uav-ideal-trace-v2"
EPS = 1e-12
CONTROLLER_SPEC = dict(
    source="Zuo et al., Drones 2026, 10, 418, Section 2.2, Eqs. 4-9",
    allocation="sequential threat-score minus nonlinear load penalty, Eqs. 4-9",
    guidance="project ThreePoint-PIP, unchanged",
    avoidance="project predictive_cpa_knn: k=4, CPA horizon=3 s, safe=3 m, alert=8 m, max weight=0.40",
    assignment_refresh="initial and affected defenders after target resolution",
    paper_label_note="The paper calls Eqs. 4-9 LB-Hungarian, but specifies a sequential argmax, not a Hungarian subproblem.",
    scene_note="Project 20v20 front-arc Level-2 scenario; not the paper's numerical experiment.")


def load_balanced_assignment(defenders, intruders, defender_mask, intruder_mask, initial_loads=None):
    """Published threat/angle/load utility, sequential by defender index (Eqs. 4-9)."""
    own, target = np.asarray(defenders, float), np.asarray(intruders, float)
    active_d, active_i = np.asarray(defender_mask, bool), np.asarray(intruder_mask, bool)
    if (own.ndim != 2 or target.ndim != 2 or own.shape[1:] != (3,) or
            target.shape[1:] != (3,) or active_d.shape != (len(own),) or
            active_i.shape != (len(target),) or not np.isfinite(own).all() or
            not np.isfinite(target).all()):
        raise ValueError("invalid Zuo allocation state")
    n = len(own)
    alpha, wd, wa, penalty = ((.115, .85, .15, .006) if n <= 20 else
                              (.210, .75, .25, .038) if n <= 35 else
                              (.360, .90, .10, .070))
    targets = np.flatnonzero(active_i)
    loads = Counter(initial_loads or {})
    result = {}
    for i in np.flatnonzero(active_d):
        if not len(targets):
            break
        relative = target[targets]-own[i]
        distances = np.linalg.norm(relative, axis=1)
        angles = np.degrees(np.arccos(np.clip(relative[:, 0]/np.maximum(distances, EPS), -1., 1.)))
        threat = wd/(1.+alpha*distances)+wa/(1.+.5*angles)
        utility = threat-np.array([penalty*loads[int(j)]**1.2 for j in targets])
        j = int(targets[int(np.argmax(utility))])
        result[int(i)] = j
        loads[j] += 1
    return result


class ZuoAssignmentTwentyEnv(TwentyVTwentyEnv):
    """Isolated assignment adapter; the nearest baseline and physical events stay untouched."""

    def reset(self, seed=None):
        _, info = super().reset(seed)
        self.assignment = load_balanced_assignment(self.defenders[:, :3], self.intruders[:, :3],
                                                   self.defender_mask, self.intruder_mask)
        self.original_assignment = self.assignment.copy()
        self.per_pair_initial_distance_m = {i: float(np.linalg.norm(
            self.defenders[i, :3]-self.intruders[j, :3])) for i, j in self.assignment.items()}
        self.assignment_total_cost_m = sum(self.per_pair_initial_distance_m.values())
        self.assignment_history = []
        self.record_assignment("initial_zuo_load_balanced_assignment")
        _, info = self.output(include_paths=True)
        info["assignment_updates"] = self.assignment_history.copy()
        return self.defenders.copy(), info

    def update_assignments_after_member_event(self, before_defenders, before_intruders):
        previous = self.assignment.copy()
        for i in np.flatnonzero(before_defenders & ~self.defender_mask):
            self.assignment.pop(int(i), None)
        for j in np.flatnonzero(before_intruders & ~self.intruder_mask):
            affected = [i for i, target in sorted(self.assignment.items())
                        if target == j and self.defender_mask[i]]
            if not affected:
                continue
            old = self.assignment.copy()
            for i in affected:
                del self.assignment[i]
            mask = np.zeros(len(self.defender_mask), bool)
            mask[affected] = True
            # Existing assignments remain locked. Their loads count in Eq. (7).
            # Re-run the published sequential argmax for affected defenders only.
            self.assignment.update(load_balanced_assignment(
                self.defenders[:, :3], self.intruders[:, :3], mask, self.intruder_mask,
                initial_loads=Counter(self.assignment.values())))
            self.record_assignment("target_resolution_zuo_reassignment", resolved_target=int(j),
                                   affected_defenders=affected, old_assignment=old)
        if self.assignment != previous and not np.any(before_intruders & ~self.intruder_mask):
            self.record_assignment("defender_exit_assignment_removed", old_assignment=previous)

    def output(self, *args, **kwargs):
        observation, info = super().output(*args, **kwargs)
        info["protocol"] = PROTOCOL
        rows = [row for row in self.assignment_history
                if row["reason"] == "target_resolution_zuo_reassignment"]
        counts = [sum(i in row["new_assignment"] for i in row["affected_defenders"])
                  for row in rows]
        info.update(target_resolution_reassignment_count=len(rows),
                    total_reassigned_defenders=sum(counts),
                    mean_reassigned_defenders_per_resolution=(float(np.mean(counts))
                        if counts else 0.),
                    maximum_reassigned_defenders_single_resolution=max(counts, default=0))
        return observation, info


def run_episode(seed=17):
    config = TwentyConfig(seed=seed)
    env = ZuoAssignmentTwentyEnv(config)
    _, info = env.reset(seed)
    initial = info
    history = IntruderHistory(config.dt, slots=config.intruder_count)
    avoidance = AvoidanceConfig(mode="predictive_cpa_knn")
    frames = [_frame(info)]
    events = []
    avoidance_active_decisions = 0
    while True:
        actions, control = _decision(info, history, config, avoidance)
        avoidance_active_decisions += sum(bool(pair["avoidance_active"])
                                          for pair in control["pairs"].values())
        _, _, terminated, truncated, info = env.step(actions)
        frames.append(_frame(info, control=control))
        events.extend(info["events"])
        if terminated or truncated:
            break
    summary = dict(experiment="ideal_20v20_zuo_assignment_three_point_pip",
                   protocol=PROTOCOL, seed=seed,
                   scenario_id=info["scenario_id"], assignment_policy="zuo_load_balanced_target_resolution",
                   guidance_method="three_point_pip", avoidance_mode=avoidance.mode,
                   avoidance_config=asdict(avoidance),
                   avoidance_active_decisions=avoidance_active_decisions,
                   intercepted_count=info["intercepted_count"], breached_count=info["breached_count"],
                   unassigned_contact_count=info["unassigned_contact_count"],
                   interception_ratio=info["interception_ratio"], breach_ratio=info["breach_ratio"],
                   mission_success=info["mission_success"], outcome=info["outcome"],
                   friendly_collision_count=info["friendly_collision_count"],
                   intruder_collision_count=info["intruder_collision_count"],
                   minimum_defender_defender_separation_m=info["minimum_separation"],
                   episode_resolution_time_s=info["time"],
                   initial_unique_target_count=initial["initial_unique_target_count"],
                   initial_max_defenders_per_target=initial["initial_max_defenders_per_target"],
                   target_resolution_reassignment_count=info["target_resolution_reassignment_count"],
                   total_reassigned_defenders=info["total_reassigned_defenders"])
    trace = dict(trace_version=TRACE_VERSION, environment_protocol=PROTOCOL,
                 config=asdict(config), seed=seed, scenario_id=info["scenario_id"],
                 controller_spec=CONTROLLER_SPEC,
                 scene_diagnostics=env.scene_diagnostics,
                 scene=dict(lower=config.lower, upper=config.upper,
                            protected_center=config.protected_center,
                            protected_radius=config.protected_radius,
                            protected_height=config.protected_height),
                 references=[dict(specification=path.specification(), points=path.points.tolist(),
                                  plan_metadata=path.plan_metadata) for path in env.paths],
                 initial_assignment=initial["assignment"], assignment_history=env.assignment_history,
                 frames=frames, result=dict(summary=summary, outcome=info["outcome"],
                                            time=info["time"], events=events,
                                            defender_status=info["defender_status"],
                                            intruder_status=info["intruder_status"]))
    return serializable(summary), serializable(trace)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"output directory already exists: {args.out}")
    summary, trace = run_episode(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out/"summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    with (args.out/"episode.json").open("x", encoding="utf-8") as stream:
        json.dump(trace, stream, separators=(",", ":"), allow_nan=False)
    print(f"seed={args.seed} intercepted={summary['intercepted_count']}/20 "
          f"interception_ratio={summary['interception_ratio']:.3f} "
          f"breached={summary['breached_count']}/20 outcome={summary['outcome']} "
          f"time={summary['episode_resolution_time_s']:.3f}s")


if __name__ == "__main__":
    main()
