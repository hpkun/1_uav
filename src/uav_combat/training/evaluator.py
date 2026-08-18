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
        "average_red_loss": mean("red_losses"),
        "average_blue_loss": mean("blue_losses"),
        "average_episode_length": mean("episode_length"),
        "evaluation_episodes": float(len(records)),
    }
