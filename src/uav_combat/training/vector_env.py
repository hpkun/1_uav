"""Deterministic synchronous vector interface for the paper environments."""
from __future__ import annotations
import numpy as np
from ..environment.env import PaperUAVCombatEnv


class SyncVectorEnv:
    def __init__(self, num_envs: int, config="configs/paper_environment.yaml", base_seed: int = 0) -> None:
        self.envs = [PaperUAVCombatEnv(config) for _ in range(num_envs)]; self.base_seed = base_seed

    def reset(self) -> np.ndarray:
        return np.stack([env.reset(self.base_seed + i)[0] for i, env in enumerate(self.envs)])

    def step(self, actions: np.ndarray):
        results = [env.step(actions[i]) for i, env in enumerate(self.envs)]
        return tuple(np.stack(items) if j < 2 else np.asarray(items) if j < 4 else list(items) for j, items in enumerate(zip(*results)))
