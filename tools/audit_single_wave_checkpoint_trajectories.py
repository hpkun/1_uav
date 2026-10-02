"""Optional six-checkpoint boundary-precursor audit (not run by the main audit).

The tool evaluates best/final checkpoints for seeds 5401..5403 on the same
reserved diagnostic seed bank beginning at 89.2M.  It refuses CPU and existing
output directories.  Counterfactual reward values are *offline scores of fixed
trajectories*, never claims about a counterfactual policy distribution.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from env.factory import make_combat_environment
from env.geometry import engagement_geometry

RUN = "outputs/mappo_mlp_single_wave_seed{seed}_1p5m"
TRAINING_SEEDS = (5401, 5402, 5403)
LAGS = (20, 10, 5, 1)


def best_step(run: Path) -> int:
    with (run / "evaluation_history.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    row = max(rows, key=lambda x: (float(x["win_rate"]), float(x["average_return"]), -float(x["average_red_loss"])))
    return int(row["sampled_steps"])


def checkpoint_specs() -> list[dict[str, Any]]:
    specs = []
    for seed in TRAINING_SEEDS:
        run = ROOT / RUN.format(seed=seed)
        specs.extend([
            {"training_seed": seed, "role": "best", "path": run / "best_eval.pt", "expected_step": best_step(run)},
            {"training_seed": seed, "role": "final", "path": run / "checkpoint_1500000.pt", "expected_step": 1_500_000},
        ])
    return specs


def load_trainer(spec: dict[str, Any]):
    state = torch.load(spec["path"], map_location="cuda:0", weights_only=False)
    if int(state.get("sampled_steps", -1)) != spec["expected_step"]:
        raise RuntimeError(f"checkpoint step mismatch: {spec['path']}")
    config = state.get("extra", {}).get("algorithm_config")
    if not isinstance(config, dict):
        config = yaml.safe_load((spec["path"].parent / "algorithm_config.yaml").read_text(encoding="utf-8"))
    trainer = build_modular_mappo_trainer(config, "cuda", total_sampled_steps=1_500_000)
    trainer.load(spec["path"], strict_protocol=True, restore_rng=False)
    trainer.actor.eval(); trainer.critic.eval()
    return trainer


def aircraft_snapshot(env, actions: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for index, own in enumerate(env.red):
        radius = math.hypot(own.x, own.y)
        radial = 0.0 if radius == 0 else (own.x * own.velocity_vector()[0] + own.y * own.velocity_vector()[1]) / radius
        candidates = [(engagement_geometry(own, target).distance, target) for target in env.blue if target.alive]
        geo = engagement_geometry(own, min(candidates, key=lambda x: x[0])[1]) if own.alive and candidates else None
        rows.append({
            "agent": index, "alive": bool(own.alive), "x": own.x, "y": own.y,
            "radius": radius, "boundary_margin": float(env.arena_radius - radius),
            "radial_velocity": radial, "altitude": own.altitude, "speed": own.v,
            "pitch": own.theta, "heading": own.psi,
            "heading_relative_to_outward": float(((own.psi - math.atan2(own.y, own.x) + math.pi) % (2 * math.pi)) - math.pi),
            "nearest_blue_distance": None if geo is None else geo.distance,
            "nearest_blue_off_boresight": None if geo is None else geo.off_boresight,
            "fire_window": False if geo is None else bool(env.weapon.in_fire_window(geo)),
            "fire_ready": bool(env.red_fire_states[index].armed),
            "action_heading": float(actions[index, 0]), "action_pitch": float(actions[index, 1]),
            "action_speed": float(actions[index, 2]),
        })
    return rows


def audit_episode(trainer, env_config: dict[str, Any], episode_seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = make_combat_environment(env_config)
    obs, _ = env.reset(episode_seed)
    alive = env.red_alive_mask.astype(np.float32)
    history: deque[dict[str, Any]] = deque(maxlen=max(LAGS) + 1)
    precursors: list[dict[str, Any]] = []
    total = np.zeros(4, dtype=np.float64)
    while True:
        actions, _ = trainer.act(obs[None], alive[None], deterministic=True)
        action = actions[0]
        before_alive = np.asarray([x.alive for x in env.red], dtype=bool)
        history.append({"step": env.steps, "agents": aircraft_snapshot(env, action)})
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        after_alive = np.asarray([x.alive for x in env.red], dtype=bool)
        newly_dead = np.flatnonzero(before_alive & ~after_alive)
        for agent in newly_dead:
            if math.hypot(env.red[agent].x, env.red[agent].y) <= env.arena_radius:
                continue
            event_rows = []
            for lag in LAGS:
                target = env.steps - lag
                candidates = [x for x in history if x["step"] == target]
                if not candidates:
                    continue
                snap = candidates[0]["agents"][agent]
                event_rows.append({"episode_seed": episode_seed, "boundary_step": env.steps,
                    "lag_steps": lag, **snap})
            tactical = any((row["nearest_blue_distance"] is not None and row["nearest_blue_distance"] <= 4000)
                           or row["fire_window"] for row in event_rows)
            escape_like = bool(event_rows) and all(row["radial_velocity"] > 0 and
                               abs(row["heading_relative_to_outward"]) < math.pi / 2 for row in event_rows)
            candidate = ("TACTICAL_OVERSHOOT_CANDIDATE" if tactical else
                         "INTENTIONAL_ESCAPE_LIKE_CANDIDATE" if escape_like else "UNCLASSIFIED")
            for row in event_rows:
                row["descriptive_candidate"] = candidate
                row["classification_warning"] = "heuristic descriptor, not causal intent inference"
            precursors.extend(event_rows)
        alive = np.asarray(info["red_alive_mask"], dtype=np.float32)
        if terminated or truncated:
            break
    base = float(total.sum())
    outcome = {
        "episode_seed": episode_seed, "episode_return": base,
        "termination_reason": info["termination_reason"], "red_success": bool(info["red_success"]),
        "red_losses": int(info["red_losses"]), "blue_losses": int(info["blue_losses"]),
        "red_boundary_exits": int(info["red_boundary_exits"]),
        "red_ground_losses": int(info["red_ground_losses"]), "episode_length": int(info["episode_length"]),
        **{f"episode_{name}_total": float(info[f"episode_{name}_total"]) for name in ("r1", "r2", "r3", "r4")},
        "red_first_fire_window_step": info.get("red_first_fire_window_step"),
        "red_first_attempt_step": info.get("red_first_attempt_step"),
        "red_first_kill_step": info.get("red_first_kill_step"),
        "first_boundary_exit_step": min((int(r["boundary_step"]) for r in precursors), default=None),
        **{k: info.get(k) for k in ("blue_ground_guard_decision_steps", "blue_ground_guard_override_steps",
            "blue_ground_guard_activations", "blue_ground_guard_activation_ratio", "blue_ground_guard_max_duration_steps")},
    }
    return precursors, outcome


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("status\nNO_ROWS\n", encoding="utf-8"); return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=89_200_000)
    parser.add_argument("--output-dir", default="outputs/single_wave_checkpoint_trajectory_audit")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory; CPU fallback is forbidden")
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    output = ROOT / args.output_dir
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    all_precursors, outcomes, scores = [], [], []
    for spec in checkpoint_specs():
        trainer = load_trainer(spec)
        env_config = yaml.safe_load((spec["path"].parent / "env_config.yaml").read_text(encoding="utf-8"))
        for episode_seed in range(args.seed_base, args.seed_base + args.episodes):
            precursors, outcome = audit_episode(trainer, env_config, episode_seed)
            identity = {"training_seed": spec["training_seed"], "checkpoint_role": spec["role"],
                        "checkpoint_step": spec["expected_step"]}
            all_precursors.extend([{**identity, **row} for row in precursors])
            outcomes.append({**identity, **outcome})
            for boundary_penalty in (-10.0, -20.0, -30.0):
                for timeout_penalty in (0.0, -10.0, -20.0):
                    fixed_score = (outcome["episode_return"] - outcome["episode_r2_total"]
                                   + boundary_penalty * outcome["red_boundary_exits"]
                                   + (timeout_penalty if outcome["termination_reason"] == "red_failure_timeout" else 0.0))
                    scores.append({**identity, "episode_seed": episode_seed,
                        "boundary_penalty": boundary_penalty, "timeout_penalty": timeout_penalty,
                        "fixed_trajectory_counterfactual_score": fixed_score,
                        "warning": "offline fixed-trajectory score; policy behavior is not re-simulated"})
        del trainer; torch.cuda.empty_cache()
    write_csv(output / "boundary_precursors.csv", all_precursors)
    write_csv(output / "episode_outcomes.csv", outcomes)
    write_csv(output / "offline_counterfactual_reward_scores.csv", scores)
    manifest = {"executed": True, "seed_base": args.seed_base, "seed_end": args.seed_base + args.episodes - 1,
                "episodes_per_checkpoint": args.episodes, "checkpoints": [{**s, "path": str(s["path"])} for s in checkpoint_specs()],
                "interpretation_limit": "descriptive trajectory evidence and fixed-trajectory reward scoring only"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "output": str(output), "episodes": len(outcomes),
                      "boundary_precursor_rows": len(all_precursors)}, indent=2))


if __name__ == "__main__":
    main()
