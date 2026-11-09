"""Paired speed-tier audits for the unchanged v4 contact task."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import mean

import numpy as np

from algorithm.common.research import atomic_json, summarize
from algorithm.interception import episode
from env.interception import InterceptionConfig, InterceptionEnv
from env.interception.config import TASKS, VERSION
from tools.interception_high_speed_audit import POLICIES, PROFILE


TIERS = {"low": (6, 7), "medium": (12, 14), "upper": (18, 21)}
TASK_NAMES = ("S0", "S1", "S2", "S3")
THRESHOLDS = (10, 5, 3, 2, 1)


def config(tier, task):
    p, e = TIERS[tier]
    options = json.loads(PROFILE.read_text(encoding="utf-8"))
    options.update(task=task, scale=TASKS[task][0], max_steps=900, pursuer_vmax=p, evader_vmax=e,
                   pursuer_initial_speed=(p*2/3, p*5/6), evader_initial_speed=(e*5/7, e*6/7))
    return InterceptionConfig(**options)


def reset(c, seed):
    return InterceptionEnv(c).reset(np.random.SeedSequence([1, seed]))[1]


def verify_pair(low, actual, factor):
    for key, mask in (("pursuers", "active_mask"), ("evaders", "evader_mask")):
        np.testing.assert_array_equal(low[mask], actual[mask])
        np.testing.assert_array_equal(low[key][:, :3], actual[key][:, :3])
        np.testing.assert_allclose(factor*low[key][:, 3:], actual[key][:, 3:], rtol=0, atol=1e-12)


def levels(distance):
    return {str(t): bool(distance <= t+(1e-9 if t == 1 else 0.)) for t in THRESHOLDS}


def add_diagnostics(row, c):
    labels = {"breach": "D_breach", "friendly_collision": "E_friendly_collision",
              "timeout": "F_timeout", "pursuer_boundary_violation": "F_red_boundary",
              "evader_escape": "F_blue_escape"}
    failures = set()
    for pair in row["closest_targets"]:
        j = pair["entity_ids"][1]-5
        distance = pair["center_distance"]
        pair["entered_m"] = levels(distance)
        v = np.asarray(pair["velocities"])
        pair["speed_mps"] = np.linalg.norm(v, axis=1).tolist()
        point = np.asarray(pair["positions"][1])
        goal = np.asarray(c.center)+c.navigation_offset-point
        u = np.asarray(pair["evader_executed_velocity"])
        pair["blue_executed_deviation_deg"] = (float(np.degrees(np.arctan2(
            np.linalg.norm(np.cross(goal, u)), goal@u))) if np.linalg.norm(u)*np.linalg.norm(goal) > 0 else None)
        if j not in row["captured_ids"]:
            failures.add("A_outside_10m" if distance > 10 else "B_10m_to_3m" if distance > 3 else
                         "C_3m_to_1m" if not pair["entered_m"]["1"] else "G_geometric_entry_without_contact")
    if row["outcome"] != "success":
        failures.add(labels.get(row["outcome"], "G_"+row["outcome"]))
    row["failure_labels"] = sorted(failures)
    row["minimum_red_blue_m"] = min((p["center_distance"] for p in row["closest_targets"]), default=None)
    row["entered_m"] = levels(row["minimum_red_blue_m"] if row["minimum_red_blue_m"] is not None else np.inf)
    return row


def statistics(rows):
    pairs = [p for row in rows for p in row["closest_targets"]]
    misses = [p for row in rows for p in row["closest_targets"] if p["entity_ids"][1]-5 not in row["captured_ids"]]
    targets = sum(len(row["target_ids"]) for row in rows)
    distances = [row["minimum_red_blue_m"] for row in rows if row["minimum_red_blue_m"] is not None]
    breach = [row["time"] for row in rows if row["outcome"] == "breach"]
    contacts = [row["first_contact_time"] for row in rows if row["first_contact_time"] is not None]
    reactive = [x for row in rows for x in row["first_evasive_request"].values()]
    return summarize(rows) | dict(
        mean_duration_s=mean(row["time"] for row in rows),
        mean_breach_s=mean(breach) if breach else None,
        mean_first_contact_s=mean(contacts) if contacts else None,
        distance_quantiles_m=dict(zip(("min", "p10", "p25", "p50", "p75", "p90", "max"),
                                      np.quantile(distances, (0,.1,.25,.5,.75,.9,1)).tolist())) if distances else {},
        mean_minimum_red_blue_m=mean(distances) if distances else None,
        any_target_entered_counts={str(t): sum(row["entered_m"][str(t)] for row in rows) for t in THRESHOLDS},
        per_target_entered_counts={str(t): sum(p["entered_m"][str(t)] for p in pairs) for t in THRESHOLDS},
        target_samples=targets,
        approached_10m_without_contact=sum(p["entered_m"]["10"] for p in misses),
        approached_3m_without_contact=sum(p["entered_m"]["3"] for p in misses),
        closest_capped_red_count=sum(p["pursuer_horizontal_acceleration"] >= .95*6 for p in pairs),
        mean_closest_red_acceleration=mean(p["pursuer_horizontal_acceleration"] for p in pairs) if pairs else None,
        mean_closest_relative_speed=mean(p["relative_speed"] for p in pairs) if pairs else None,
        mean_closest_time=mean(p["time"] for p in pairs) if pairs else None,
        mean_closest_blue_deviation_deg=mean(p["blue_executed_deviation_deg"] for p in pairs
                                            if p["blue_executed_deviation_deg"] is not None)
                                         if any(p["blue_executed_deviation_deg"] is not None for p in pairs) else None,
        mean_evasion_start_s=mean(p["time"] for p in reactive) if reactive else None,
        mean_evasion_start_m=mean(p["distance"] for p in reactive) if reactive else None,
        max_horizontal_acceleration=np.max([r["peak_horizontal_acceleration"] for r in rows], axis=0).tolist(),
        mean_horizontal_acceleration=np.mean([r["mean_horizontal_acceleration"] for r in rows], axis=0).tolist(),
        mean_episode_wall_s=mean(row["wall_time_s"] for row in rows),
        controller_us_per_step=1e6*sum(row["controller_s"] for row in rows)/max(1, sum(r["steps"] for r in rows)),
        failure_counts={name: sum(name in row["failure_labels"] for row in rows)
                        for name in sorted({x for row in rows for x in row["failure_labels"]})})


def trial(c, seed, policy, options=None):
    row, _ = episode(c, seed, options or {"scenario": "random"}, policy, audit=True)
    if row["outcome"] == "simulation_failure":
        raise RuntimeError(row["error"])
    return add_diagnostics(row, c)


def run(tier, task, seeds, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seeds = list(seeds)
    c = config(tier, task)
    low = config("low", task)
    initial = {seed: reset(c, seed) for seed in seeds}
    for seed in seeds:
        verify_pair(reset(low, seed), initial[seed], TIERS[tier][0]/TIERS["low"][0])
    designs = [("main", p, c) for p in POLICIES]
    if task == "S2" or tier == "low":
        designs += [(f"horizon_{h}", "predictive", replace(c, predictive_horizon=h)) for h in (20,40)]
    if task == "S2":
        designs.append(("single", "predictive", c))
    groups = []
    for suite, policy, actual in designs:
        rows = []
        for seed in seeds:
            expected = initial[seed]
            options = None
            if suite == "single":
                slot = seed % 3
                options = dict(scenario="custom", paired=True, pursuers=expected["pursuers"][slot:slot+1],
                               evaders=expected["evaders"][expected["evader_mask"]])
            row = trial(actual, seed, policy, options)
            if options:
                np.testing.assert_array_equal(row["initial"]["pursuers"][:1], options["pursuers"])
                row["deployment_ids"] = [slot]
            else:
                np.testing.assert_array_equal(row["initial"]["pursuers"], expected["pursuers"])
                row["deployment_ids"] = np.flatnonzero(expected["active_mask"]).tolist()
            np.testing.assert_array_equal(row["initial"]["evaders"], expected["evaders"])
            rows.append(row)
        record = dict(version=VERSION, tier=tier, task=task, suite=suite, policy=policy, config=asdict(actual),
                      seeds=seeds, statistics=statistics(rows), episodes=rows)
        atomic_json(output/f"{suite}_{policy}.json", record)
        groups.append({k:v for k,v in record.items() if k != "episodes"})
        print(tier, task, suite, policy, record["statistics"]["outcomes"], flush=True)
        if tier == "low" and suite == "main" and any(
                r["outcome"] == "timeout" and not r["entered_m"]["10"] for r in rows):
            designs.append(("timeout_120", policy, replace(c, max_steps=1200)))
    atomic_json(output/"summary.json", dict(groups=groups, seeds=seeds))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tier", choices=TIERS, required=True)
    p.add_argument("--task", choices=TASK_NAMES, required=True)
    p.add_argument("--seed", type=int, default=1200)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.episodes < 1:
        p.error("--episodes must be positive")
    run(args.tier, args.task, range(args.seed,args.seed+args.episodes), args.out)


if __name__ == "__main__":
    main()
