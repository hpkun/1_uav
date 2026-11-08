"""Paired initial-condition audit for S1 and S2 reference controllers."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from algorithm.common.research import summarize
from algorithm.common.serialization import write_json
from algorithm.interception import episode
from env.interception import InterceptionConfig, InterceptionEnv


def paired(config, seed, initialization):
    env = InterceptionEnv(config)
    _, info = env.reset(np.random.SeedSequence([1, seed]), {"scenario": initialization})
    p = info["pursuers"][info["active_mask"]]
    e = info["evaders"][info["evader_mask"]]
    # Rotate the first deployed ID used in smaller teams by seed, without
    # ranking agents by advantageous geometry. Preserve IDs within a cohort.
    order = np.roll(np.arange(len(p)), seed % len(p))
    p = p[order]
    result = []
    for count in (1, 2, 3):
        options = dict(scenario="custom", paired=True, pursuers=p[:count], evaders=e)
        row, _ = episode(config, seed, options, "paired_reference")
        row.update(cohort_size=count, deployment_ids=order[:count].tolist(),
                   paired_initialization=initialization, pair_id=f"{config.task}/{initialization}/{seed}")
        result.append(row)
    return result


def run(output, seeds):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, groups = [], []
    for task in ("S1", "S2"):
        config = InterceptionConfig(task=task)
        for initialization in ("fixed", "random"):
            block = [row for seed in (seeds[:1] if initialization == "fixed" else seeds)
                     for row in paired(config, seed, initialization)]
            rows.extend(block)
            for count in (1, 2, 3):
                selected = [row for row in block if row["cohort_size"] == count]
                stats = summarize(selected)
                groups.append(dict(task=task, initialization=initialization, cohort_size=count,
                                   config=asdict(config), metrics=stats))
                print(task, initialization, count, stats["outcomes"],
                      "mean capture", stats["mean_capture_time_s"], flush=True)
    write_json(output/"paired_audit.json", dict(seeds=seeds, controller="predictive reference",
                                               groups=groups, episodes=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    run(args.out, list(range(args.seed, args.seed+args.episodes)))


if __name__ == "__main__":
    main()
