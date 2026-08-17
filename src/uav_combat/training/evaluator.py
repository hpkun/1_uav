"""Paper evaluation: twenty fixed seeds disjoint from training."""
from __future__ import annotations

import numpy as np
from ..environment.env import PaperUAVCombatEnv


def evaluate(actor, config="configs/paper_environment.yaml", seeds=range(10_000_000, 10_000_020)) -> dict[str, float]:
    records = []
    for seed in seeds:
        env = PaperUAVCombatEnv(config)
        observation, _ = env.reset(int(seed))
        episode_return = 0.0
        while True:
            actions = actor.act(observation, env.red_alive_mask, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(actions)
            episode_return += float(reward[0])
            if terminated or truncated:
                records.append({"episode_return": episode_return, **info})
                break
    mean = lambda key: float(np.mean([record[key] for record in records]))
    return {
        "average_return": mean("episode_return"),
        "win_rate": mean("red_success"),
        "average_red_loss": mean("red_losses"),
        "average_episode_length": mean("episode_length"),
        "evaluation_episodes": float(len(records)),
    }
