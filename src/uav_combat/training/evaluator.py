"""Deterministic evaluation on seeds disjoint from training."""
from __future__ import annotations

import numpy as np
from ..environment.env import MultiUAVCombatEnv


def episode_return_metrics(agent_returns: np.ndarray) -> tuple[float, float]:
    """Team sum and per-agent mean return diagnostics."""
    values = np.asarray(agent_returns, dtype=float)
    if values.shape != (4,):
        raise ValueError("agent_returns must have shape (4,)")
    return float(values.sum()), float(values.mean())


def evaluate(actor, config="configs/combat_environment.yaml", seeds=range(10_000_000, 10_000_020)) -> dict[str, float]:
    records = []
    for seed in seeds:
        env = MultiUAVCombatEnv(config)
        observation, _ = env.reset(int(seed))
        agent_returns = np.zeros(4, dtype=float)
        while True:
            actions = actor.act(observation, env.red_alive_mask, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(actions)
            agent_returns += reward
            if terminated or truncated:
                team_return, mean_agent_return = episode_return_metrics(agent_returns)
                records.append({
                    "episode_return": team_return,
                    "mean_agent_episode_return": mean_agent_return,
                    **info,
                })
                break
    mean = lambda key: float(np.mean([record[key] for record in records]))
    return {
        "average_return": mean("episode_return"),
        "average_agent_return": mean("mean_agent_episode_return"),
        "win_rate": mean("red_success"),
        "loss_rate": mean("blue_win"),
        "draw_rate": mean("draw"),
        "timeout_rate": float(np.mean([
            record["termination_reason"] == "red_failure_timeout" for record in records
        ])),
        "average_red_loss": mean("red_losses"),
        "average_blue_loss": mean("blue_losses"),
        **{
            f"{side}_{event}_episode_rate": float(np.mean([
                record[f"{side}_first_{event}_step"] is not None for record in records
            ]))
            for side in ("red", "blue")
            for event in ("fire_window", "attempt", "hit", "kill")
        },
        "average_red_attack_kills": mean("red_attack_kills"),
        "average_blue_attack_kills": mean("blue_attack_kills"),
        "average_red_boundary_exits": mean("red_boundary_exits"),
        "average_blue_boundary_exits": mean("blue_boundary_exits"),
        "average_red_ground_losses": mean("red_ground_losses"),
        "average_blue_ground_losses": mean("blue_ground_losses"),
        "average_episode_length": mean("episode_length"),
        **{
            f"average_episode_{name}_total": mean(f"episode_{name}_total")
            for name in ("r1", "r2", "r3", "r4")
        },
        "evaluation_episodes": len(records),
    }
