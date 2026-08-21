"""Summarize a V2 smoke run and export one representative combat trajectory."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch

from uav_combat.environment.env import MultiUAVCombatEnv
from uav_combat.environment.geometry import engagement_geometry
from uav_combat.environment.observation import OBSERVATION_DIM
from uav_combat.madsac import MADSACTrainer


def load_trainer(checkpoint: Path) -> MADSACTrainer:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hidden = int(state["actor"]["backbone.0.weight"].shape[0])
    trainer = MADSACTrainer(observation_dim=OBSERVATION_DIM, hidden_dim=hidden)
    trainer.load(checkpoint)
    return trainer


def minimum_enemy_distance(env: MultiUAVCombatEnv) -> float:
    values = [
        engagement_geometry(red, blue).distance
        for red in env.red if red.alive for blue in env.blue if blue.alive
    ]
    return min(values, default=float("nan"))


def rollout(trainer: MADSACTrainer, config: Path, seed: int) -> tuple[dict, list[dict]]:
    env = MultiUAVCombatEnv(config)
    observation, reset_info = env.reset(seed)
    rows = []
    while True:
        actions = trainer.act(observation, env.red_alive_mask, deterministic=True)
        observation, rewards, terminated, truncated, info = env.step(actions)
        rows.append({
            "seed": seed,
            "step": env.steps,
            "time_s": env.steps * env.dt,
            "scenario_mode": reset_info["scenario_mode"],
            "minimum_enemy_distance": minimum_enemy_distance(env),
            "mean_reward": float(np.mean(rewards)),
            "mean_progress_reward": float(np.mean(info["progress_rewards"])),
            "mean_tactical_reward": float(np.mean(info["tactical_rewards"])),
            "mean_fire_reward": float(np.mean(info["fire_opportunity_rewards"])),
            "mean_event_reward": float(np.mean(info["event_rewards"])),
            "mean_abs_heading_action": float(np.mean(np.abs(actions[:, 0]))),
            "mean_abs_pitch_action": float(np.mean(np.abs(actions[:, 1]))),
            "mean_speed_action": float(np.mean(actions[:, 2])),
            "red_fire_window_pairs": info["red_fire_window_pairs"],
            "blue_fire_window_pairs": info["blue_fire_window_pairs"],
            "red_active_locks": info["red_active_locks"],
            "blue_active_locks": info["blue_active_locks"],
            "red_survivors": info["red_survivors"],
            "blue_survivors": info["blue_survivors"],
        })
        if terminated or truncated:
            return {
                "seed": seed,
                "scenario_mode": reset_info["scenario_mode"],
                "episode_length": env.steps,
                "termination_reason": info["termination_reason"],
                "red_attackable": info["red_first_attackable_step"] is not None,
                "red_lock": info["red_first_lock_step"] is not None,
                "red_kill": info["red_first_kill_step"] is not None,
                "red_attack_kills": info["red_attack_kills"],
                "blue_attack_kills": info["blue_attack_kills"],
                "red_altitude_losses": (
                    info["red_low_altitude_losses"] + info["red_high_altitude_losses"]
                ),
            }, rows


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "nonzero_rate": float(np.mean(array != 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = (root / args.checkpoint).resolve()
    metrics_path = (root / args.metrics).resolve()
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = root / "configs/combat_environment.yaml"
    trainer = load_trainer(checkpoint)

    episodes = []
    trajectories = []
    for seed in range(10_000_000, 10_000_000 + args.episodes):
        episode, rows = rollout(trainer, config, seed)
        episodes.append(episode)
        trajectories.append(rows)
    representative_index = next(
        (index for index, episode in enumerate(episodes) if episode["red_kill"]),
        next((index for index, episode in enumerate(episodes) if episode["red_attackable"]), 0),
    )
    representative = trajectories[representative_index]
    with (output / "representative_trajectory.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(representative[0]))
        writer.writeheader()
        writer.writerows(representative)

    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    window = max(len(metrics) // 5, 1)
    channels = [
        "mean_step_reward", "mean_progress_reward", "mean_tactical_reward",
        "mean_fire_opportunity_reward", "mean_event_reward",
        "red_fire_window_pairs", "blue_fire_window_pairs",
        "red_active_locks", "blue_active_locks",
    ]
    report = {
        "sampled_steps": len(metrics) * 24,
        "training_step_channels": {
            key: distribution([row[key] for row in metrics]) for key in channels
        },
        "training_reward_change": {
            key: {
                "first_20_percent_mean": float(np.mean([
                    row[key] for row in metrics[:window]
                ])),
                "last_20_percent_mean": float(np.mean([
                    row[key] for row in metrics[-window:]
                ])),
            }
            for key in channels[:5]
        },
        "deterministic_evaluation": {
            "episodes": len(episodes),
            "red_attackable_rate": float(np.mean([row["red_attackable"] for row in episodes])),
            "red_lock_rate": float(np.mean([row["red_lock"] for row in episodes])),
            "red_kill_rate": float(np.mean([row["red_kill"] for row in episodes])),
            "blue_kills_mean": float(np.mean([row["blue_attack_kills"] for row in episodes])),
            "red_kills_mean": float(np.mean([row["red_attack_kills"] for row in episodes])),
            "red_altitude_losses_mean": float(np.mean([row["red_altitude_losses"] for row in episodes])),
        },
        "representative_episode": episodes[representative_index],
        "representative_trajectory": "representative_trajectory.csv",
        "combat_interaction_gate": bool(
            any(row["red_attackable"] for row in episodes)
            and any(row["red_lock"] for row in episodes)
            and any(row["red_kill"] for row in episodes)
        ),
    }
    (output / "smoke_analysis.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
