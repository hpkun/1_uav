"""Fresh, CUDA-only comparison for the Jiao 2025 reproduction screening.

This tool never trains. It evaluates All-Off, M5, Jiao-Core and Jiao-Full on
the same 100 development seeds and keeps raw environment performance separate
from the Jiao Eq. (12) training signal.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.protocol import config_sha256
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import is_formal_v2_checkpoint
from algorithm.train_modular_mappo import load_config
from env.factory import make_combat_environment

ENV_CONFIG = ROOT / "configs" / "persistent_wave_v2_environment.yaml"
OUTPUT = ROOT / "outputs" / "jiao2025_reproduction_analysis"
SEED_BASE = 37_000_000
EPISODES = 100
RUNS = {
    "All-Off": ROOT / "outputs" / "pw_alloff_matched_1p5m_seed2023",
    "M5": ROOT / "outputs" / "pw_m5_wave_balance_1p5m_seed2023",
    "Jiao-Core": ROOT / "outputs" / "jiao2025_core_1p5m_seed2023",
    "Jiao-Full": ROOT / "outputs" / "jiao2025_full_1p5m_seed2023",
}
EXPECTED_MODULES = {
    "All-Off": set(),
    "M5": {"wave_balancing"},
    "Jiao-Core": {"wave_context", "recurrent_memory", "popart"},
    "Jiao-Full": {"wave_context", "recurrent_memory", "popart", "multi_wave_reward"},
}


def validate_jiao_evaluation_seeds(seeds) -> list[int]:
    values = [int(seed) for seed in seeds]
    if len(values) != EPISODES or len(set(values)) != EPISODES:
        raise ValueError("Jiao evaluation requires exactly 100 unique seeds")
    if values != list(range(SEED_BASE, SEED_BASE + EPISODES)):
        raise ValueError("Jiao evaluation seeds must be 37,000,000-37,000,099")
    if any(20_000_000 <= seed <= 20_000_199 for seed in values):
        raise ValueError("20M formal-holdout seeds are forbidden")
    if any(35_000_000 <= seed < 37_000_000 for seed in values):
        raise ValueError("old 35M/36M development seeds are forbidden")
    return values


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_jiao_config(name: str, config: dict[str, Any]) -> None:
    enabled = {key for key, value in config["modules"].items() if value.get("enabled", False)}
    if enabled != EXPECTED_MODULES[name]:
        raise RuntimeError(f"{name} module mismatch: {sorted(enabled)}")
    if name.startswith("Jiao"):
        training = config["training"]
        expected = {
            "actor_learning_rate": .0005, "critic_learning_rate": .0005,
            "ppo_epochs": 10, "clip_ratio": .1, "entropy_coefficient": .01,
            "gae_lambda": .95, "gamma": .99, "rollout_steps": 150,
            "minibatch_size": 512, "num_train_envs": 24,
            "total_sampled_steps": 1_500_000,
        }
        for key, value in expected.items():
            if not np.isclose(float(training[key]), float(value)):
                raise RuntimeError(f"{name} {key} mismatch")
        modules = config["modules"]
        if modules["wave_context"].get("encoding") != "scalar_round" or modules["wave_context"].get("context_target") != "actor_critic":
            raise RuntimeError(f"{name} scalar actor/critic F is not enabled")
        if modules["recurrent_memory"].get("mode") != "actor_critic_gru":
            raise RuntimeError(f"{name} actor/critic GRU is not enabled")
        expected_reward = "jiao_r2_replacement" if name == "Jiao-Full" else "none"
        if modules["multi_wave_reward"].get("mode") != expected_reward:
            raise RuntimeError(f"{name} reward mode mismatch")
        for forbidden in ("wave_balancing", "warm_start", "curriculum", "policy_anchor"):
            if modules[forbidden].get("enabled", False):
                raise RuntimeError(f"{name} unexpectedly enables {forbidden}")


def load_checkpoint(name: str, path: Path, device: str):
    if not torch.cuda.is_available() or device != "cuda":
        raise RuntimeError("Jiao reproduction analysis is CUDA-only")
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not is_formal_v2_checkpoint(state):
        raise RuntimeError(f"{path} is not a self-describing Modular MAPPO v2 checkpoint")
    extra = state["extra"]
    config = extra.get("algorithm_config")
    if config is None:
        raise RuntimeError(f"{path} lacks algorithm_config")
    validate_jiao_config(name, config)
    if extra.get("environment_variant") != "persistent_wave_v2":
        raise RuntimeError(f"{path} environment variant mismatch")
    trainer = build_modular_mappo_trainer(
        config, device=device,
        hidden_dim=int(extra["network_architecture"]["hidden_dim"]),
    )
    trainer.load(path)
    metadata = {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": file_sha256(path),
        "sampled_steps": int(state["sampled_steps"]),
        "training_seed": int(extra["training_seed"]),
        "algorithm_config_sha256": config_sha256(config),
        "enabled_modules": sorted(EXPECTED_MODULES[name]),
    }
    return trainer, metadata


def validate_run(name: str, run_dir: Path) -> dict[str, Any]:
    required = ("run_config.json", "run_summary.json", "algorithm_config.yaml", "best_eval.pt", "latest.pt")
    missing = [filename for filename in required if not (run_dir / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"{name} incomplete run: {missing}")
    run = _json(run_dir / "run_config.json")
    summary = _json(run_dir / "run_summary.json")
    config = yaml.safe_load((run_dir / "algorithm_config.yaml").read_text(encoding="utf-8"))
    validate_jiao_config(name, config)
    if run.get("algorithm") != "modular_mappo" or run.get("environment_variant") != "persistent_wave_v2":
        raise RuntimeError(f"{name} run protocol mismatch")
    if int(run.get("seed", -1)) != 2023 or int(summary.get("sampled_steps", -1)) != 1_500_000:
        raise RuntimeError(f"{name} is not the complete seed2023 1.5M run")
    if run.get("algorithm_config_sha256") != config_sha256(config):
        raise RuntimeError(f"{name} algorithm config snapshot hash mismatch")
    return {"complete": True, "sampled_steps": 1_500_000, "training_seed": 2023,
            "run_config": str((run_dir / "run_config.json").resolve())}


def scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value if isinstance(value, (bool, str, int, float)) or value is None else None


def run_episode(trainer, env_config: dict[str, Any], seed: int) -> dict[str, Any]:
    env = make_combat_environment(env_config)
    observation, _ = env.reset(seed)
    alive = env.red_alive_mask.copy()
    actor_hidden, critic_hidden = trainer.initial_hidden(1)
    episode_mask = np.zeros(1, dtype=np.float32)
    raw_return = np.zeros(4, dtype=np.float64)
    wave = 1
    total = int(env_config["persistent_waves"]["total_waves"])
    while True:
        context = trainer.context_numpy(np.asarray([wave]), np.asarray([total]))
        actions, actor_hidden = trainer.act(
            observation[None], alive[None], True, False,
            context, actor_hidden, episode_mask,
        )
        _, critic_hidden = trainer.values_step(
            observation[None], alive[None], context,
            critic_hidden, episode_mask,
        )
        observation, reward, terminated, truncated, info = env.step(actions[0])
        raw_return += reward
        alive = np.asarray(info["red_alive_mask"], dtype=np.float32)
        actor_hidden = trainer.recurrent.apply_alive(actor_hidden, alive[None])
        critic_hidden = trainer.recurrent.apply_alive(critic_hidden, alive[None])
        episode_mask[:] = 1.0
        wave = int(info.get("wave_index", wave))
        total = int(info.get("total_waves", total))
        if terminated or truncated:
            break
    record = {key: scalar(value) for key, value in info.items() if scalar(value) is not None}
    record.update({
        "seed": seed,
        "episode_return": float(raw_return.sum()),
        "timeout": int(info.get("termination_reason") == "red_failure_timeout"),
        "waves_cleared": int(info.get("waves_cleared", 0)),
        "episode_kill_loss_ratio": float(info.get("blue_losses", 0)) / max(float(info.get("red_losses", 0)), 1.0),
    })
    per_wave = [dict(row) for row in info.get("per_wave_metrics", [])]
    for index in (1, 2, 3):
        row = next((value for value in per_wave if int(value.get("wave_index", 0)) == index), None)
        record[f"clear_wave_{index}"] = int(record["waves_cleared"] >= index)
        record[f"reached_wave_{index}"] = int(index == 1 or record["waves_cleared"] >= index - 1)
        record[f"time_spent_wave_{index}"] = None if row is None else int(row["duration_steps"])
    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, float]:
    mean = lambda key: float(np.mean([float(row.get(key, 0) or 0) for row in records]))
    result = {
        "evaluation_episodes": float(len(records)),
        "average_return": mean("episode_return"),
        "clear_wave_1_probability": mean("clear_wave_1"),
        "clear_wave_2_probability": mean("clear_wave_2"),
        "clear_wave_3_probability": mean("clear_wave_3"),
        "average_waves_cleared": mean("waves_cleared"),
        "average_red_loss": mean("red_losses"),
        "average_blue_loss": mean("blue_losses"),
        "average_red_boundary_exits": mean("red_boundary_exits"),
        "average_blue_boundary_exits": mean("blue_boundary_exits"),
        "average_red_ground_losses": mean("red_ground_losses"),
        "average_blue_ground_losses": mean("blue_ground_losses"),
        "timeout_rate": mean("timeout"),
        "average_episode_length": mean("episode_length"),
        "evaluation_boundary_exit_rate": float(np.mean([row.get("red_boundary_exits", 0) > 0 for row in records])),
    }
    result["kill_loss_ratio"] = float(sum(row.get("blue_losses", 0) for row in records)) / max(float(sum(row.get("red_losses", 0) for row in records)), 1.0)
    result["timeout_conditioned_reached_W1"] = result["timeout_rate"]
    for index in (2, 3):
        reached = [row for row in records if row[f"reached_wave_{index}"]]
        result[f"timeout_conditioned_reached_W{index}"] = float(np.mean([row["timeout"] for row in reached])) if reached else 0.0
    return result


def reward_summary(run_dir: Path) -> dict[str, float]:
    path = run_dir / "training_metrics.jsonl"
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = ("raw_environment_reward", "jiao_training_reward", "paper_R2_blue_kill_component",
            "paper_R2_red_loss_component", "paper_R2_wave1", "paper_R2_wave2", "paper_R2_wave3")
    result = {key: float(sum(float(row.get(key, 0.0)) for row in rows)) for key in keys}
    summary_path = run_dir / "run_summary.json"
    if summary_path.is_file():
        totals = _json(summary_path).get("paper_R2_totals", {})
        result.update({
            "paper_R2_blue_kill_component": float(totals.get("blue_kill_component", result["paper_R2_blue_kill_component"])),
            "paper_R2_red_loss_component": float(totals.get("red_loss_component", result["paper_R2_red_loss_component"])),
            **{f"paper_R2_wave{wave}": float(totals.get(f"wave_{wave}", result[f"paper_R2_wave{wave}"])) for wave in (1, 2, 3)},
        })
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU evaluation")
    seeds = validate_jiao_evaluation_seeds(range(SEED_BASE, SEED_BASE + EPISODES))
    env_config = yaml.safe_load(ENV_CONFIG.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "evaluation_cache"
    cache.mkdir(exist_ok=True)
    table: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    integrity = {method: validate_run(method, run_dir) for method, run_dir in RUNS.items()}
    for method, run_dir in RUNS.items():
        for role, filename in (("best", "best_eval.pt"), ("latest", "latest.pt")):
            checkpoint = run_dir / filename
            digest = file_sha256(checkpoint) if checkpoint.is_file() else "missing"
            cache_path = cache / f"{method.lower().replace('-', '_')}_{role}.json"
            cached = _json(cache_path) if cache_path.is_file() else None
            if cached and cached.get("checkpoint_sha256") == digest and cached.get("seeds") == seeds:
                payload = cached
            else:
                trainer, checkpoint_meta = load_checkpoint(method, checkpoint, args.device)
                records = [run_episode(trainer, env_config, seed) for seed in seeds]
                payload = {"checkpoint_sha256": digest, "seeds": seeds, "metadata": checkpoint_meta,
                           "summary": summarize(records), "episodes": records}
                cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            metadata[f"{method}_{role}"] = payload["metadata"]
            table.append({"method": method, "checkpoint_role": role, **payload["summary"]})
            print(f"[EVAL] {method} {role}: W3={payload['summary']['clear_wave_3_probability']:.3f} waves={payload['summary']['average_waves_cleared']:.3f}", flush=True)
    reward_rows = [{"method": method, **reward_summary(run_dir)} for method, run_dir in RUNS.items() if method.startswith("Jiao")]
    write_csv(args.output_dir / "raw_environment_comparison.csv", table)
    write_csv(args.output_dir / "jiao_training_reward_summary.csv", reward_rows)
    result = {"protocol": {"seeds": seeds, "episodes": EPISODES, "device": args.device,
                           "environment": str(ENV_CONFIG), "note": "development comparison; not 20M formal holdout"},
              "run_integrity": integrity, "checkpoints": metadata, "raw_environment_comparison": table,
              "jiao_training_reward_summary": reward_rows}
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    columns = ("method", "checkpoint_role", "clear_wave_1_probability", "clear_wave_2_probability",
               "clear_wave_3_probability", "average_waves_cleared", "average_return",
               "average_red_loss", "average_blue_loss", "kill_loss_ratio",
               "average_red_boundary_exits", "average_red_ground_losses", "timeout_rate",
               "average_episode_length")
    lines = ["# Jiao 2025 reproduction comparison", "",
             "Development evaluation on seeds 37,000,000-37,000,099; raw environment metrics only.", "",
             "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in table:
        lines.append("| " + " | ".join(str(row[key]) for key in columns) + " |")
    lines.extend(["", "Jiao Eq. (12) training signals are stored separately in `jiao_training_reward_summary.csv`.", ""])
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
