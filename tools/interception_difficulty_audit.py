"""Paired reference-policy audit for the v4 contact task."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from algorithm.common.research import summarize
from algorithm.common.serialization import write_json
from algorithm.interception import episode
from env.interception import InterceptionConfig
from env.interception.config import TASKS, VERSION


def audit(output, episodes, seed):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seeds = list(range(seed, seed+episodes))
    tasks = ("S0", "S1", "S2", "S3")
    policies = ("stationary", "random", "reference", "predictive")
    groups = []
    shared = {}
    for task in tasks:
        scale = TASKS[task][0]
        for difficulty, speed in (("easy", 10.), ("main", 13.), ("hard", 14.)):
            config = InterceptionConfig(task=task, scale=scale, evader_vmax=speed)
            for policy in policies:
                rows = []
                for initialization, trial in [("fixed", seed), *(("random", x) for x in seeds)]:
                    row, _ = episode(config, trial, {"scenario": initialization}, policy)
                    key = scale, initialization, trial
                    expected = shared.setdefault(key, row["initial"])
                    for name in ("pursuers", "evaders"):
                        np.testing.assert_array_equal(row["initial"][name], expected[name])
                    rows.append(row)
                stats = summarize(rows[1:])
                groups.append(dict(task=task, difficulty=difficulty, policy=policy, config=asdict(config),
                                   fixed=rows[0], random=rows[1:], metrics=stats))
                print(task, difficulty, policy, stats["outcomes"], flush=True)
    write_json(output/"audit.json", dict(version=VERSION, seeds=seeds, groups=groups))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    audit(args.out, args.episodes, args.seed)


if __name__ == "__main__":
    main()
