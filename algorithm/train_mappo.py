"""Configuration-driven formal and smoke MAPPO runner."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from algorithm.mappo.runner import MAPPOTrainingRunner


def resolved(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


class TeeOutput:
    """Write entry-point progress to both the terminal and train.log."""

    def __init__(self, terminal, log_stream) -> None:
        self.terminal = terminal
        self.log_stream = log_stream

    def write(self, value: str) -> int:
        self.terminal.write(value)
        self.log_stream.write(value)
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_stream.flush()


def default_output_dir(seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "outputs" / f"run_{timestamp}_seed_{seed}"


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
    env_path = resolved(args.env_config)
    env_config = yaml.safe_load(env_path.read_text(encoding="utf-8"))
    algorithm_path = resolved(args.algorithm_config)
    algorithm_config = yaml.safe_load(algorithm_path.read_text(encoding="utf-8"))
    configured_seed = int(algorithm_config["training"]["seed"])
    effective_seed = configured_seed if args.seed is None else args.seed
    output_dir = (
        default_output_dir(effective_seed)
        if args.output_dir is None
        else resolved(args.output_dir)
    )
    runner = MAPPOTrainingRunner(env_config, algorithm_config, args.num_envs,
        args.total_sampled_steps or (192 if args.smoke else None), args.device,
        args.seed, output_dir, args.smoke)
    runner.output_dir.mkdir(parents=True, exist_ok=True)
    (runner.output_dir / "env_config.yaml").write_text(
        yaml.safe_dump(env_config, sort_keys=False), encoding="utf-8"
    )
    (runner.output_dir / "algorithm_config.yaml").write_text(
        yaml.safe_dump(algorithm_config, sort_keys=False), encoding="utf-8"
    )
    startup = runner.startup_summary()
    run_config = {
        "device": startup["device"],
        "seed": startup["seed"],
        "num_envs": startup["num_envs_M"],
        "total_sampled_steps": startup["total_sampled_steps"],
        "smoke": runner.smoke,
        "environment_config_path": str(env_path),
        "algorithm_config_path": str(algorithm_path),
        "environment_variant": startup["environment_variant"],
        "environment_version": str(env_config["environment_version"]),
        "algorithm": "MAPPO",
    }
    (runner.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    with (runner.output_dir / "train.log").open("a", encoding="utf-8") as log_stream:
        with redirect_stdout(TeeOutput(sys.stdout, log_stream)):
            print(runner.start_log_line(), flush=True)
            if args.resume:
                runner.resume(resolved(args.resume))
            summary = runner.run()
            (runner.output_dir / "run_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            print(runner.done_log_line(summary), flush=True)


if __name__ == "__main__":
    main()
