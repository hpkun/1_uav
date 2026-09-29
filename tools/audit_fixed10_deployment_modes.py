"""Fixed10 Treatment seed5303 Peak/Final deployment-mode audit.

The deterministic arm is reused from the completed Fixed10 natural-entry audit.
The stochastic arm uses the Actor's original tanh-Gaussian ``rsample`` path.
This file has no training, backward, resume, or optimizer path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.evaluator import episode_return_metrics
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from env.factory import make_combat_environment
from tools.fixed10_wave_state_drift_common import (
    checkpoint_step, json_dump, load_checkpoint, sha256, state_dict_sha256,
)

TOOL_VERSION = 1
TRAINING_SEED = 5303
ENVIRONMENT_SEEDS = tuple(range(88_330_000, 88_330_016))
POLICY_STREAM_IDS = (0, 1, 2)
FORBIDDEN_45M = range(45_000_000, 45_000_200)
RUN_DIR = ROOT / "outputs" / "dev_actor_clip10_fixed10_seed5303_300k"
HISTORICAL_DETERMINISTIC = ROOT / "outputs" / "fixed10_wave_state_drift_audit" / "natural_entry_episodes.csv"
DEFAULT_FULL_OUTPUT = ROOT / "outputs" / "fixed10_deployment_mode_audit"
DEFAULT_SMOKE_OUTPUT = ROOT / "outputs" / "fixed10_deployment_mode_audit_smoke"
CHECKPOINTS = {
    "Peak": {"id": "Treatment10_seed5303_peak", "path": RUN_DIR / "best_eval.pt", "step": 1_701_888},
    "Final": {"id": "Treatment10_seed5303_final", "path": RUN_DIR / "final.pt", "step": 1_805_280},
}
METRICS = ("W1", "W2", "W3", "AverageWaves", "Q2", "Q3", "Return", "RedLoss", "Boundary", "Ground", "EpisodeLength")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed10 Peak/Final deterministic-vs-stochastic audit")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--print-existing", action="store_true")
    return parser


def validate_environment_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(seed) for seed in seeds)
    if any(seed in FORBIDDEN_45M for seed in values):
        raise RuntimeError("45M future-final seeds are forbidden")
    if any(not 88_000_000 <= seed < 89_000_000 for seed in values):
        raise RuntimeError("this audit accepts only independent 88M diagnostic seeds")
    return values


def policy_rng_seed(environment_seed: int, stream_id: int) -> int:
    if int(stream_id) not in POLICY_STREAM_IDS:
        raise ValueError("policy RNG stream id must be 0, 1, or 2")
    validate_environment_seeds([environment_seed])
    payload = f"fixed10-deployment-v1:{int(environment_seed)}:{int(stream_id)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


@contextmanager
def isolated_policy_rng(seed: int, device: str):
    cuda_devices: list[int] = []
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        parsed = torch.device(device)
        cuda_devices = [torch.cuda.current_device() if parsed.index is None else parsed.index]
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(int(seed))
        if cuda_devices:
            torch.cuda.manual_seed_all(int(seed))
        yield


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty result: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_historical_deterministic(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected = []
    expected_seeds = set(ENVIRONMENT_SEEDS)
    for role, spec in CHECKPOINTS.items():
        group = [row for row in rows if row.get("policy_id") == spec["id"]]
        if len(group) != 16:
            raise RuntimeError(f"historical deterministic {role} must have exactly 16 rows")
        keys = [(int(row["diagnostic_episode_seed"]), int(row["checkpoint_step"])) for row in group]
        if len(set(keys)) != 16:
            raise RuntimeError(f"historical deterministic {role} contains duplicate episode keys")
        if {seed for seed, _ in keys} != expected_seeds:
            raise RuntimeError(f"historical deterministic {role} seed coverage mismatch")
        if {step for _, step in keys} != {spec["step"]}:
            raise RuntimeError(f"historical deterministic {role} checkpoint mismatch")
        checkpoint_sha = sha256(spec["path"])
        for row in group:
            waves = int(row["waves_cleared"])
            selected.append({
                "training_seed": TRAINING_SEED, "checkpoint_id": spec["id"], "checkpoint_role": role,
                "checkpoint_step": spec["step"], "checkpoint_sha256": checkpoint_sha,
                "environment_seed": int(row["diagnostic_episode_seed"]),
                "policy_rng_stream_id": None, "policy_rng_seed": None,
                "deployment_mode": "deterministic", "data_source": "HISTORICAL_REUSED",
                "waves_cleared": waves, "episode_length": int(row["episode_length"]),
                "red_losses": int(row["red_losses"]), "red_boundary_exits": int(row["red_boundary_exits"]),
                "red_ground_losses": int(row["red_ground_losses"]),
                "team_episode_return": None,
                "return_unavailable_reason": "HISTORICAL_SOURCE_DID_NOT_RECORD_RETURN",
                "reached_w2": int(row["reached_w2"]), "reached_w3": int(row["reached_w3"]),
                "w3_cleared": int(waves >= 3), "ended_normally": True,
                "termination_reason": "HISTORICAL_NOT_RECORDED",
            })
    if len(selected) != 32:
        raise RuntimeError("historical deterministic reuse must contain exactly 32 episodes")
    return selected


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate empty episode set")
    waves = np.asarray([int(row["waves_cleared"]) for row in records])
    w1, w2, w3 = (float(np.mean(waves >= index)) for index in (1, 2, 3))
    returns = [float(row["team_episode_return"]) for row in records if row.get("team_episode_return") is not None]
    return {
        "W1": w1, "W2": w2, "W3": w3, "AverageWaves": float(np.mean(waves)),
        "Q2": None if w1 == 0 else w2 / w1,
        "Q3": None if w2 == 0 else w3 / w2,
        "Q2_undefined_reason": "W1_ZERO" if w1 == 0 else None,
        "Q3_undefined_reason": "W2_ZERO" if w2 == 0 else None,
        "Return": float(np.mean(returns)) if len(returns) == len(records) else None,
        "Return_unavailable_reason": None if len(returns) == len(records) else "ONE_OR_MORE_EPISODES_LACK_RETURN",
        "RedLoss": float(np.mean([row["red_losses"] for row in records])),
        "Boundary": float(np.mean([row["red_boundary_exits"] for row in records])),
        "Ground": float(np.mean([row["red_ground_losses"] for row in records])),
        "EpisodeLength": float(np.mean([row["episode_length"] for row in records])),
        "episodes": len(records),
    }


def stream_summaries(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for role in CHECKPOINTS:
        for stream_id in POLICY_STREAM_IDS:
            group = [row for row in records if row["checkpoint_role"] == role and row["policy_rng_stream_id"] == stream_id]
            row = {"checkpoint_role": role, "policy_rng_stream_id": stream_id, **aggregate(group)}
            rows.append(row)
    descriptive = {}
    for role in CHECKPOINTS:
        role_rows = [row for row in rows if row["checkpoint_role"] == role]
        descriptive[role] = {}
        for metric in METRICS:
            values = [row[metric] for row in role_rows if row[metric] is not None]
            descriptive[role][metric] = None if not values else {
                "mean": float(statistics.mean(values)),
                "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
                "n_streams_defined": len(values),
            }
    return rows, descriptive


def paired_rows(deterministic: list[dict], stochastic: list[dict]) -> list[dict[str, Any]]:
    output = []
    for mode, rows in (("deterministic", deterministic), ("stochastic", stochastic)):
        keys = sorted({
            (int(row["environment_seed"]), row["policy_rng_stream_id"])
            for row in rows
        }, key=lambda item: (item[0], -1 if item[1] is None else int(item[1])))
        by_key = {(row["checkpoint_role"], int(row["environment_seed"]), row["policy_rng_stream_id"]): row for row in rows}
        for environment_seed, stream_id in keys:
            peak = by_key.get(("Peak", environment_seed, stream_id))
            final = by_key.get(("Final", environment_seed, stream_id))
            if peak is None or final is None:
                raise RuntimeError(f"missing Peak/Final pair for {mode} seed={environment_seed} stream={stream_id}")
            row: dict[str, Any] = {"deployment_mode": mode, "environment_seed": environment_seed,
                                   "policy_rng_stream_id": stream_id}
            for wave in (1, 2, 3):
                p = int(peak["waves_cleared"] >= wave); f = int(final["waves_cleared"] >= wave)
                row[f"peak_w{wave}"] = p; row[f"final_w{wave}"] = f
                row[f"w{wave}_paired_outcome"] = "BOTH_SUCCESS" if p and f else "PEAK_ONLY" if p else "FINAL_ONLY" if f else "BOTH_FAIL"
            row.update({
                "peak_waves": peak["waves_cleared"], "final_waves": final["waves_cleared"],
                "delta_average_waves_episode": final["waves_cleared"] - peak["waves_cleared"],
                "delta_red_loss": final["red_losses"] - peak["red_losses"],
                "delta_boundary": final["red_boundary_exits"] - peak["red_boundary_exits"],
                "delta_ground": final["red_ground_losses"] - peak["red_ground_losses"],
            })
            output.append(row)
    return output


def paired_summary(rows: list[dict], deterministic_summary: dict, stochastic_summary: dict) -> dict:
    result = {}
    for mode in ("deterministic", "stochastic"):
        group = [row for row in rows if row["deployment_mode"] == mode]
        wave_counts = {}
        for wave in (1, 2, 3):
            labels = [row[f"w{wave}_paired_outcome"] for row in group]
            wave_counts[f"W{wave}"] = {name: labels.count(name) for name in ("PEAK_ONLY", "FINAL_ONLY", "BOTH_SUCCESS", "BOTH_FAIL")}
        summary = deterministic_summary if mode == "deterministic" else stochastic_summary
        peak_q3, final_q3 = summary["Peak"]["Q3"], summary["Final"]["Q3"]
        result[mode] = {
            "pair_count": len(group), "independent_environment_scenarios": 16,
            "policy_stream_repeats_per_scenario": 1 if mode == "deterministic" else 3,
            "paired_wave_outcomes": wave_counts,
            "mean_final_minus_peak_AverageWaves": float(np.mean([row["delta_average_waves_episode"] for row in group])),
            "mean_final_minus_peak_RedLoss": float(np.mean([row["delta_red_loss"] for row in group])),
            "mean_final_minus_peak_Boundary": float(np.mean([row["delta_boundary"] for row in group])),
            "mean_final_minus_peak_Ground": float(np.mean([row["delta_ground"] for row in group])),
            "peak_Q3": peak_q3, "final_Q3": final_q3,
            "delta_Q3_final_minus_peak": None if peak_q3 is None or final_q3 is None else final_q3 - peak_q3,
            "reach_W3_delta_final_minus_peak": summary["Final"]["W2"] - summary["Peak"]["W2"],
            "unconditional_W3_delta_final_minus_peak": summary["Final"]["W3"] - summary["Peak"]["W3"],
        }
    return result


def descriptive_label(det: dict, stochastic: dict, full: bool) -> str:
    if not full:
        return "INSUFFICIENT_EVIDENCE"
    det_delta = det["Final"]["Q3"] - det["Peak"]["Q3"] if det["Final"]["Q3"] is not None and det["Peak"]["Q3"] is not None else None
    sto_delta = stochastic["Final"]["Q3"] - stochastic["Peak"]["Q3"] if stochastic["Final"]["Q3"] is not None and stochastic["Peak"]["Q3"] is not None else None
    if sto_delta is None:
        return "INSUFFICIENT_EVIDENCE"
    if sto_delta <= -0.05:
        return "STOCHASTIC_W3_DROP_OBSERVED"
    if det_delta is not None and det_delta <= -0.05 and sto_delta >= 0:
        return "DETERMINISTIC_SPECIFIC_DROP_PLAUSIBLE"
    return "MIXED_OR_SMALL_DIFFERENCE"


def validate_protocol() -> tuple[dict, dict, dict[str, dict[str, Any]]]:
    env_config = yaml.safe_load((RUN_DIR / "runtime_env_config.yaml").read_text(encoding="utf-8"))
    algorithm_config = yaml.safe_load((RUN_DIR / "algorithm_config.yaml").read_text(encoding="utf-8"))
    run_config = json.loads((RUN_DIR / "run_config.json").read_text(encoding="utf-8"))
    network = run_config["network_architecture"]
    if env_config.get("environment_variant") != "persistent_wave_v2": raise RuntimeError("wrong environment variant")
    if int(env_config["persistent_waves"]["total_waves"]) != 3: raise RuntimeError("total_waves must be 3")
    if int(env_config["simulation"]["max_steps"]) != 3000: raise RuntimeError("max_steps must be 3000")
    if (network.get("actor_input_dim"), algorithm_config["network"]["action_dim"], algorithm_config["network"]["num_agents"]) != (52, 3, 4):
        raise RuntimeError("Actor/environment contract must be 52D/3D/4-agent")
    if network.get("actor_gru_hidden_dim") != 0 or network.get("actor_context_dim") != 0 or network.get("entity_attention_enabled"):
        raise RuntimeError("audit requires Plain feed-forward Actor")
    metadata = {}
    for role, spec in CHECKPOINTS.items():
        checkpoint = load_checkpoint(spec["path"])
        if checkpoint_step(checkpoint) != spec["step"]: raise RuntimeError(f"{role} checkpoint step mismatch")
        extra = checkpoint.get("extra", {})
        if extra.get("environment_config_sha256") != run_config["environment_config_sha256"]: raise RuntimeError(f"{role} environment config mismatch")
        if extra.get("algorithm_config_sha256") != run_config["algorithm_config_sha256"]: raise RuntimeError(f"{role} algorithm config mismatch")
        if extra.get("network_architecture") != network: raise RuntimeError(f"{role} network architecture mismatch")
        metadata[role] = {"path": str(spec["path"].relative_to(ROOT)), "sampled_steps": spec["step"],
                          "checkpoint_sha256": sha256(spec["path"]), "actor_state_sha256": state_dict_sha256(checkpoint["actor"])}
    return env_config, algorithm_config, metadata


def run_episode(trainer, env_config: dict, role: str, environment_seed: int, stream_id: int,
                policy_seed: int, checkpoint_meta: dict, device: str) -> dict[str, Any]:
    env = make_combat_environment(deepcopy(env_config))
    observation, _ = env.reset(int(environment_seed))
    alive = env.red_alive_mask.copy()
    agent_returns = np.zeros(4, dtype=np.float64)
    with isolated_policy_rng(policy_seed, device), torch.no_grad():
        while True:
            actions, _ = trainer.act(observation[None], alive[None], deterministic=False)
            observation, reward, terminated, truncated, info = env.step(actions[0])
            agent_returns += reward
            alive = np.asarray(info["red_alive_mask"], dtype=np.float32)
            if terminated or truncated:
                team_return, _ = episode_return_metrics(agent_returns)
                waves = int(info["waves_cleared"])
                return {
                    "training_seed": TRAINING_SEED, "checkpoint_id": CHECKPOINTS[role]["id"],
                    "checkpoint_role": role, "checkpoint_step": checkpoint_meta["sampled_steps"],
                    "checkpoint_sha256": checkpoint_meta["checkpoint_sha256"],
                    "environment_seed": int(environment_seed), "policy_rng_stream_id": int(stream_id),
                    "policy_rng_seed": int(policy_seed), "deployment_mode": "stochastic", "data_source": "NEW_DIAGNOSTIC",
                    "waves_cleared": waves, "episode_length": int(info["episode_length"]),
                    "red_losses": int(info["red_losses"]), "red_boundary_exits": int(info["red_boundary_exits"]),
                    "red_ground_losses": int(info["red_ground_losses"]), "team_episode_return": team_return,
                    "return_unavailable_reason": None, "reached_w2": int(waves >= 1),
                    "reached_w3": int(waves >= 2), "w3_cleared": int(waves >= 3),
                    "ended_normally": True, "termination_reason": str(info["termination_reason"]),
                }


def validate_stochastic_count(rows: list[dict], full: bool) -> None:
    expected = 96 if full else 2
    if len(rows) != expected: raise RuntimeError(f"expected {expected} stochastic episodes, got {len(rows)}")
    keys = [(row["checkpoint_role"], row["environment_seed"], row["policy_rng_stream_id"]) for row in rows]
    if len(set(keys)) != expected: raise RuntimeError("duplicate stochastic episode key")
    if full:
        expected_keys = {(role, seed, stream) for role in CHECKPOINTS for seed in ENVIRONMENT_SEEDS for stream in POLICY_STREAM_IDS}
        if set(keys) != expected_keys: raise RuntimeError("FULL stochastic 2x16x3 coverage mismatch")


def run_audit(device: str, output_dir: Path, smoke: bool) -> dict[str, Any]:
    if device != "cuda": raise RuntimeError("repository policy requires CUDA for checkpoint/environment audit")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    validate_environment_seeds(ENVIRONMENT_SEEDS)
    output_dir = Path(output_dir)
    if output_dir.exists(): raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    if smoke and output_dir.resolve() == DEFAULT_FULL_OUTPUT.resolve():
        raise RuntimeError("SMOKE cannot write to the FULL audit directory")
    output_dir.mkdir(parents=True)
    json_dump(output_dir / "run_status.json", {"status": "IN_PROGRESS", "smoke": smoke})
    checkpoint_before = {role: sha256(spec["path"]) for role, spec in CHECKPOINTS.items()}
    try:
        env_config, algorithm_config, checkpoint_meta = validate_protocol()
        historical = validate_historical_deterministic(read_csv(HISTORICAL_DETERMINISTIC))
        env_seeds = ENVIRONMENT_SEEDS[:1] if smoke else ENVIRONMENT_SEEDS
        stream_ids = POLICY_STREAM_IDS[:1] if smoke else POLICY_STREAM_IDS
        stochastic = []
        actor_memory_before = {}
        actor_memory_after = {}
        for role, spec in CHECKPOINTS.items():
            checkpoint = load_checkpoint(spec["path"])
            trainer = build_modular_mappo_trainer(algorithm_config, device=device, hidden_dim=256, total_sampled_steps=1_805_280)
            trainer.actor.load_state_dict(checkpoint["actor"], strict=True); trainer.actor.eval(); trainer.critic.eval()
            if trainer.actor.training: raise RuntimeError("Actor must be in eval mode")
            actor_memory_before[role] = state_dict_sha256(trainer.actor.state_dict())
            for environment_seed in env_seeds:
                for stream_id in stream_ids:
                    seed = policy_rng_seed(environment_seed, stream_id)
                    stochastic.append(run_episode(trainer, env_config, role, environment_seed, stream_id, seed,
                                                  checkpoint_meta[role], device))
            actor_memory_after[role] = state_dict_sha256(trainer.actor.state_dict())
            if actor_memory_after[role] != actor_memory_before[role]: raise RuntimeError(f"{role} Actor mutated")
            del trainer; torch.cuda.empty_cache()
        validate_stochastic_count(stochastic, full=not smoke)
        deterministic_summary = {role: aggregate([row for row in historical if row["checkpoint_role"] == role]) for role in CHECKPOINTS}
        stochastic_summary = {role: aggregate([row for row in stochastic if row["checkpoint_role"] == role]) for role in CHECKPOINTS}
        stream_rows, stream_descriptive = (stream_summaries(stochastic) if not smoke else ([], {}))
        pairs = paired_rows(historical if not smoke else [row for row in historical if row["environment_seed"] in env_seeds], stochastic)
        pair_summary = paired_summary(pairs, deterministic_summary, stochastic_summary)
        label = descriptive_label(deterministic_summary, stochastic_summary, full=not smoke)
        checkpoint_after = {role: sha256(spec["path"]) for role, spec in CHECKPOINTS.items()}
        if checkpoint_before != checkpoint_after: raise RuntimeError("checkpoint file mutation guard failed")
        inventory = {
            "tool_version": TOOL_VERSION, "tool_source_sha256": sha256(Path(__file__)),
            "mode": "SMOKE" if smoke else "FULL", "device": device,
            "cuda_device": torch.cuda.get_device_name(0), "training_seed": TRAINING_SEED,
            "checkpoints": checkpoint_meta, "environment_config_sha256": sha256(RUN_DIR / "runtime_env_config.yaml"),
            "algorithm_config_sha256": sha256(RUN_DIR / "algorithm_config.yaml"),
            "environment_summary": {"variant": "persistent_wave_v2", "total_waves": 3, "max_steps": 3000,
                                    "observation_dim": 52, "action_dim": 3, "red_agents": 4},
            "environment_seeds": list(env_seeds),
            "policy_rng": [{"environment_seed": seed, "stream_id": stream,
                            "policy_seed": policy_rng_seed(seed, stream)} for seed in env_seeds for stream in stream_ids],
            "common_random_numbers_design": True, "deterministic_data_source": "HISTORICAL_REUSED",
            "expected_stochastic_episodes": 2 if smoke else 96, "actual_stochastic_episodes": len(stochastic),
            "formal_training_performed": False, "45m_accessed": False,
        }
        summary = {"deterministic": deterministic_summary, "stochastic": stochastic_summary,
                   "stochastic_stream_mean_std": stream_descriptive, "descriptive_label": label}
        limitations = {
            "training_seed_replication_count": 1, "training_seed": 5303,
            "independent_environment_scenarios": 16 if not smoke else 1,
            "policy_streams_are_within_scenario_repeats": True,
            "48_stochastic_episodes_are_not_48_independent_scenarios": True,
            "peak_checkpoint_selection_bias": "Peak was selected using historical 44M development evaluation",
            "common_random_numbers_limit": "closed-loop trajectories can diverge, so future environment events are not identical",
            "deterministic_return": "UNAVAILABLE because historical reused CSV did not record return",
            "no_causal_proof": True,
        }
        analysis = {
            "status": "SMOKE_COMPLETE_NOT_FULL" if smoke else "COMPLETE",
            "descriptive_label": label, "full_audit_complete": not smoke,
            "stochastic_episode_count": len(stochastic), "deterministic_episode_count_reused": 32,
            "training_performed": False, "backward_calls": 0, "optimizer_steps": 0,
            "45m_accessed": False, "checkpoint_mutation_guard": {"status": "PASS", "before": checkpoint_before, "after": checkpoint_after},
            "in_memory_actor_mutation_guard": {"status": "PASS", "before": actor_memory_before, "after": actor_memory_after},
        }
        csv_write(output_dir / "deterministic_reused_episodes.csv", historical)
        csv_write(output_dir / "stochastic_episodes.csv", stochastic)
        if stream_rows: csv_write(output_dir / "stochastic_stream_summary.csv", stream_rows)
        csv_write(output_dir / "paired_peak_final.csv", pairs)
        json_dump(output_dir / "inventory.json", inventory)
        json_dump(output_dir / "deployment_mode_summary.json", summary)
        json_dump(output_dir / "paired_peak_final_summary.json", pair_summary)
        json_dump(output_dir / "analysis.json", analysis)
        json_dump(output_dir / "limitations.json", limitations)
        (output_dir / "decision_support.txt").write_text(
            f"STATUS={analysis['status']}\nLABEL={label}\n"
            f"STOCHASTIC_EPISODES={len(stochastic)}\nDETERMINISTIC_SOURCE=HISTORICAL_REUSED\n",
            encoding="utf-8",
        )
        json_dump(output_dir / "run_status.json", {"status": analysis["status"], "smoke": smoke})
        print_report(output_dir)
        return analysis
    except Exception as exc:
        json_dump(output_dir / "run_status.json", {"status": "FAILED", "smoke": smoke,
                                                    "error_type": type(exc).__name__, "error": str(exc)})
        raise


def print_report(output_dir: Path) -> None:
    analysis = json.loads((Path(output_dir) / "analysis.json").read_text(encoding="utf-8"))
    summary = json.loads((Path(output_dir) / "deployment_mode_summary.json").read_text(encoding="utf-8"))
    print("=" * 88)
    print("FIXED10 PEAK/FINAL DEPLOYMENT MODE AUDIT")
    print(f"status={analysis['status']} label={analysis['descriptive_label']}")
    for mode in ("deterministic", "stochastic"):
        for role in ("Peak", "Final"):
            row = summary[mode][role]
            print(f"{mode:13s} {role:5s}: W1={row['W1']:.3f} W2={row['W2']:.3f} W3={row['W3']:.3f} "
                  f"AW={row['AverageWaves']:.3f} Q3={row['Q3']}")
    print(f"output={Path(output_dir)}")
    print("=" * 88)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir or (DEFAULT_SMOKE_OUTPUT if args.smoke else DEFAULT_FULL_OUTPUT)
    if args.print_existing:
        print_report(output_dir); return
    run_audit(args.device, output_dir, args.smoke)


if __name__ == "__main__":
    main()
