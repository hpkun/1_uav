"""Configuration-driven formal and smoke MAPPO runner."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
from uav_combat.training.mappo_runner import MAPPOTrainingRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-sampled-steps", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--env-config", default="configs/persistent_wave_v2_environment.yaml")
    parser.add_argument("--algorithm-config", default="configs/mappo_persistent_wave.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env_path = Path(args.env_config)
    if not env_path.is_absolute():
        env_path = root / env_path
    env_config = yaml.safe_load(env_path.read_text(encoding="utf-8"))
    algorithm_path = Path(args.algorithm_config)
    if not algorithm_path.is_absolute():
        algorithm_path = root / algorithm_path
    algorithm_config = yaml.safe_load(algorithm_path.read_text(encoding="utf-8"))
    runner = MAPPOTrainingRunner(env_config, algorithm_config, args.num_envs,
        args.total_sampled_steps or (192 if args.smoke else None), args.device,
        args.seed, args.output_dir, args.smoke)
    print(runner.start_log_line(), flush=True)
    if args.resume:
        runner.resume(args.resume)
    summary = runner.run()
    (runner.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(runner.done_log_line(summary), flush=True)


if __name__ == "__main__":
    main()
