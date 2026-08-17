"""Twenty-seed paper evaluation protocol."""
from __future__ import annotations
import numpy as np
from ..environment.env import PaperUAVCombatEnv


def evaluate(actor, config="configs/paper_environment.yaml", seeds=range(10_000, 10_020)) -> dict[str, float]:
    records = []
    for seed in seeds:
        env = PaperUAVCombatEnv(config); obs, _ = env.reset(int(seed)); total = 0.0
        while True:
            actions = actor(obs, deterministic=True) if hasattr(actor, "__call__") else actor.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(actions); total += float(reward[0])
            if terminated or truncated: break
        records.append({"episode_return": total, **info})
    return {"episode_return": float(np.mean([r["episode_return"] for r in records])), "win_rate": float(np.mean([r["win"] for r in records])), "red_survivors": float(np.mean([r["red_survivors"] for r in records])), "red_losses": float(np.mean([4-r["red_survivors"] for r in records])), "blue_losses": float(np.mean([4-r["blue_survivors"] for r in records])), "attack_kills": float(np.mean([r["attack_kills"] for r in records])), "boundary_losses": float(np.mean([r["boundary_losses"] for r in records])), "episode_length": float(np.mean([r["episode_length"] for r in records]))}
