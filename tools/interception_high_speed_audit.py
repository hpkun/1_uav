"""Paired high-speed contact audit with per-target continuous closest approaches."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import mean

import numpy as np

from algorithm.common.research import summarize
from algorithm.common.research import atomic_json
from algorithm.common.serialization import serializable
from algorithm.interception import episode
from env.interception import InterceptionConfig, InterceptionEnv
from env.interception.config import TASKS, VERSION


PROFILE = Path(__file__).resolve().parents[1]/"configs/interception_high_speed.json"
POLICIES = ("stationary", "random", "reference", "predictive")


def summarize_audit(rows):
    def average(key):
        values = [row[key] for row in rows if row[key] is not None]
        return mean(values) if values else None

    misses = [pair for row in rows for pair in row["closest_targets"]
              if pair["entities"][1]-5 not in row["captured_ids"]]
    return summarize(rows) | dict(
        mean_first_contact_s=average("first_contact_time"),
        mean_minimum_red_blue_m=average("minimum_red_blue"),
        mean_minimum_friendly_m=average("minimum_friendly_distance"),
        missed_targets=dict(outside_10m=sum(p["center_distance"] > 10 for p in misses),
                            inside_10m=sum(p["center_distance"] <= 10 for p in misses),
                            acceleration_capped_near_miss=sum(p["center_distance"] <= 10 and
                                p["pursuer_horizontal_acceleration"] >= .95*rows[0]["acceleration_cap"]
                                for p in misses)),
        repeated_close_pass_episodes=sum(any(v > 1 for v in row["close_passes"].values()) for row in rows))


def run(output, seeds, tasks, *, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    overrides = json.loads(PROFILE.read_text(encoding="utf-8"))
    groups = []
    for task in tasks:
        config = InterceptionConfig(**overrides, task=task, scale=TASKS[task][0])
        env = InterceptionEnv(config)
        designs = [(policy, {}, 0) for policy in POLICIES] + [
            ("predictive", {"predictive_horizon": 20.}, 0)]
        if task == "S2":
            designs += [(policy, {"evader_vmax": speed}, 0) for speed in (25., 30.) for policy in POLICIES]
            designs += [("predictive", {"horizontal_acceleration_max": a}, 0) for a in (4., 8.)]
            designs += [("predictive", {}, count) for count in (1, 2)]
        initial = {}
        for initialization, seed in [("fixed", seeds[0]), *(("random", seed) for seed in seeds)]:
            _, info = env.reset(np.random.SeedSequence([1, seed]), {"scenario": initialization})
            initial[initialization, seed] = info
        for policy, changes, count in designs:
            actual = replace(config, **changes)
            destination = output/f"group_{len(groups):03}.json"
            identity = dict(task=task, policy=policy, changes=changes,
                            cohort=count or int(TASKS[task][0].split("v")[0]),
                            config=serializable(asdict(actual)))
            if resume and destination.exists():
                group = json.loads(destination.read_text(encoding="utf-8"))
                if (any(group[k] != v for k, v in identity.items()) or
                        [row["seed"] for row in group["random"]] != list(seeds)):
                    raise ValueError(f"paired audit identity mismatch in {destination}")
                groups.append({k: v for k, v in group.items() if k not in ("fixed", "random")})
                continue
            rows = []
            for (initialization, seed), expected in initial.items():
                p = expected["pursuers"][expected["active_mask"]]
                e = expected["evaders"][expected["evader_mask"]]
                # Balance the single deployed ID without ranking its geometry.
                order = np.arange(len(p))
                if count:
                    order = np.roll(order, seed % len(p))[:count]
                options = (dict(scenario="custom", paired=True, pursuers=p[order], evaders=e) if count
                           else {"scenario": initialization})
                row, _ = episode(actual, seed, options, policy, audit=True)
                np.testing.assert_array_equal(row["initial"]["pursuers"][:len(order)], p[order])
                np.testing.assert_array_equal(row["initial"]["evaders"][:len(e)], e)
                nearest = min((x["center_distance"] for x in row["closest_targets"]), default=None)
                row.update(minimum_red_blue=nearest, acceleration_cap=actual.horizontal_acceleration_max,
                           deployment_ids=order.tolist(), paired_initialization=initialization)
                rows.append(row)
            metrics = summarize_audit(rows[1:])
            group = identity | dict(metrics=metrics, fixed=rows[0], random=rows[1:])
            atomic_json(destination, group)
            groups.append({k: v for k, v in group.items() if k not in ("fixed", "random")})
            print(task, policy, changes, count, metrics["outcomes"], flush=True)
    atomic_json(output/"summary.json", dict(version=VERSION, seeds=list(seeds), groups=groups,
                teams={"pursuers": "red defenders", "evaders": "blue intruders"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument("--tasks", nargs="+", choices=("S0", "S1", "S2", "S3"), default=("S0", "S1", "S2", "S3"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    run(args.out, range(args.seed, args.seed+args.episodes), args.tasks, resume=args.resume)


if __name__ == "__main__":
    main()
