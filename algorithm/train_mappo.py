"""Configuration-driven formal and smoke MAPPO runner."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime
import json
from pathlib import Path
import sys
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch

from algorithm.common.checkpoint import validate_checkpoint_for_resume
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
    base = PROJECT_ROOT / "outputs" / f"run_{timestamp}_seed_{seed}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{suffix}")
        suffix += 1
    return candidate


def ensure_fresh_output_directory(output_dir: Path) -> None:
    """Create or accept an empty directory, rejecting any existing contents."""
    if output_dir.exists():
        if not output_dir.is_dir():
            raise RuntimeError(f"fresh output_dir is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise RuntimeError(
                f"fresh training output_dir is non-empty: {output_dir}; "
                "choose a new run directory"
            )
    else:
        output_dir.mkdir(parents=True)


def resolve_run_paths(
    output_arg: str | None,
    resume_arg: str | None,
    seed: int,
) -> tuple[Path, Path | None]:
    """Resolve the final run directory and optional resume checkpoint."""
    if resume_arg is None:
        output_dir = (
            default_output_dir(seed) if output_arg is None else resolved(output_arg)
        )
        ensure_fresh_output_directory(output_dir)
        return output_dir, None
    checkpoint = resolved(resume_arg).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
    checkpoint_parent = checkpoint.parent
    output_dir = checkpoint_parent if output_arg is None else resolved(output_arg).resolve()
    if output_dir != checkpoint_parent:
        raise RuntimeError(
            "resume output_dir must equal checkpoint.parent: "
            f"expected {checkpoint_parent}, got {output_dir}"
        )
    return output_dir, checkpoint


def validate_resume_config_snapshots(
    output_dir: Path,
    env_config: dict,
    algorithm_config: dict,
) -> list[tuple[Path, dict]]:
    """Reject changed snapshots and return legacy snapshots that are missing."""
    missing: list[tuple[Path, dict]] = []
    for name, current in (
        ("env_config.yaml", env_config),
        ("algorithm_config.yaml", algorithm_config),
    ):
        path = output_dir / name
        if not path.exists():
            warnings.warn(
                f"resume run is missing {name}; creating a compatibility snapshot",
                RuntimeWarning,
            )
            missing.append((path, current))
            continue
        stored = yaml.safe_load(path.read_text(encoding="utf-8"))
        if stored != current:
            raise RuntimeError(f"resume {name} mismatch with stored run snapshot")
    return missing


def write_yaml_snapshot(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


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
    output_dir, resume_checkpoint = resolve_run_paths(
        args.output_dir, args.resume, effective_seed
    )
    missing_snapshots: list[tuple[Path, dict]] = []
    resume_state = None
    if resume_checkpoint is not None:
        missing_snapshots = validate_resume_config_snapshots(
            output_dir, env_config, algorithm_config
        )
        resume_state = torch.load(
            resume_checkpoint, map_location="cpu", weights_only=False
        )
        validate_checkpoint_for_resume(resume_state, env_config, algorithm_config)
    runner = MAPPOTrainingRunner(env_config, algorithm_config, args.num_envs,
        args.total_sampled_steps or (192 if args.smoke else None), args.device,
        args.seed, output_dir, args.smoke)
    if resume_checkpoint is None:
        write_yaml_snapshot(runner.output_dir / "env_config.yaml", env_config)
        write_yaml_snapshot(
            runner.output_dir / "algorithm_config.yaml", algorithm_config
        )
    else:
        for path, config in missing_snapshots:
            write_yaml_snapshot(path, config)
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
        "effective_hidden_dim": startup["effective_hidden_dim"],
        "output_dir": str(runner.output_dir.resolve()),
        "resume_checkpoint": (
            None if resume_checkpoint is None else str(resume_checkpoint)
        ),
    }
    if resume_checkpoint is None:
        (runner.output_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2), encoding="utf-8"
        )
    with (runner.output_dir / "train.log").open("a", encoding="utf-8") as log_stream:
        with redirect_stdout(TeeOutput(sys.stdout, log_stream)):
            print(runner.start_log_line(), flush=True)
            if resume_checkpoint is not None:
                runner.resume(resume_checkpoint)
                resume_record = {
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "checkpoint": str(resume_checkpoint),
                    "checkpoint_sampled_steps": int(
                        resume_state.get("sampled_steps", 0)
                    ),
                    "seed": startup["seed"],
                    "device": startup["device"],
                    "num_envs": startup["num_envs_M"],
                    "total_sampled_steps": startup["total_sampled_steps"],
                }
                with (runner.output_dir / "resume_history.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps(resume_record) + "\n")
            summary = runner.run()
            (runner.output_dir / "run_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            print(runner.done_log_line(summary), flush=True)


if __name__ == "__main__":
    main()
