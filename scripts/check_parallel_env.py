"""Verify that training environments are persistent independent processes."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import numpy as np
import yaml

from uav_combat.training.vector_env import ParallelVectorEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=24)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if min(args.num_envs, args.steps) <= 0:
        raise ValueError("num-envs and steps must be positive")

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/combat_environment.yaml").read_text(encoding="utf-8")
    )
    actions = np.zeros((args.num_envs, 4, 3), dtype=np.float32)
    with ParallelVectorEnv(args.num_envs, config, base_seed=91_000_000) as vector:
        vector.reset()
        started = time.perf_counter()
        for _ in range(args.steps):
            vector.step_batch(actions, auto_reset=False)
        elapsed = time.perf_counter() - started
        print(f"parent_pid={os.getpid()}")
        print(f"backend={vector.backend}")
        print(f"workers={vector.num_workers}")
        print("worker_pids=" + ",".join(map(str, vector.worker_pids)))
        print(f"unique_worker_pids={len(set(vector.worker_pids))}")
        print(f"batch_steps={args.steps}")
        print(f"environment_transitions={args.steps * args.num_envs}")
        print(f"elapsed_seconds={elapsed:.6f}")
        print(
            "environment_transitions_per_second="
            f"{args.steps * args.num_envs / elapsed:.3f}"
        )


if __name__ == "__main__":
    main()
