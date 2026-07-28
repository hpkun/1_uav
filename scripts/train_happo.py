"""Train or resume the independent HAPPO baseline."""

from __future__ import annotations

import argparse

from uav_env.algorithms.happo.config import load_happo_config
from uav_env.algorithms.happo.runner import HAPPORunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--total-env-steps", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--load-actors-only")
    parser.add_argument("--run-name", default="happo")
    args = parser.parse_args()
    config = load_happo_config(args.config)
    for key, value in (
        ("seed", args.seed),
        ("device", args.device),
        ("total_env_steps", args.total_env_steps),
        ("num_envs", args.num_envs),
    ):
        if value is not None:
            config[key] = value
    runner = HAPPORunner(config, args.run_name)
    if args.resume:
        runner.resume(args.resume)
    if args.load_actors_only:
        runner.resume(args.load_actors_only, actor_only=True)
    print(f"Output: {runner.run().resolve()}")


if __name__ == "__main__":
    main()
