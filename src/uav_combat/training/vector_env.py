"""Synchronous M-environment sampler matching Algorithm 1."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..environment.env import MultiUAVCombatEnv
from ..environment.observation import OBSERVATION_DIM


@dataclass
class VectorStep:
    observations: np.ndarray
    transition_next_observations: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    infos: list[dict]
    alive_masks: np.ndarray
    next_alive_masks: np.ndarray


class SyncVectorEnv:
    """Simple non-overlapping seed allocation: base + episode*M + env_id."""

    def __init__(self, num_envs: int, config="configs/combat_environment.yaml", base_seed: int = 0, forbidden_seeds=()) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.envs = [MultiUAVCombatEnv(config) for _ in range(num_envs)]
        self.num_envs = int(num_envs)
        self.base_seed = int(base_seed)
        self.forbidden_seeds = set(map(int, forbidden_seeds))
        self.used_training_seeds: set[int] = set()
        self.episode_indices = np.zeros(num_envs, dtype=np.int64)
        self.current_observations = np.zeros((num_envs, 4, OBSERVATION_DIM), dtype=np.float32)
        self.current_alive_masks = np.ones((num_envs, 4), dtype=np.float32)
        self.last_reset_seeds = np.zeros(num_envs, dtype=np.int64)

    def seed_for(self, env_id: int, episode_index: int) -> int:
        return self.base_seed + episode_index * self.num_envs + env_id

    def _reset_one(self, env_id: int) -> np.ndarray:
        seed = self.seed_for(env_id, int(self.episode_indices[env_id]))
        if seed in self.forbidden_seeds:
            raise RuntimeError(f"training seed overlaps evaluation set: {seed}")
        if seed in self.used_training_seeds:
            raise RuntimeError(f"training seed reused: {seed}")
        self.used_training_seeds.add(seed)
        self.last_reset_seeds[env_id] = seed
        return self.envs[env_id].reset(seed)[0]

    def reset(self) -> np.ndarray:
        self.current_observations = np.stack([self._reset_one(i) for i in range(self.num_envs)])
        self.current_alive_masks = np.stack([env.red_alive_mask for env in self.envs])
        return self.current_observations.copy()

    def step_batch(self, actions: np.ndarray, auto_reset: bool = True) -> VectorStep:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, 4, 3):
            raise ValueError("vector actions must be [env, 4, 3]")
        alive_before = self.current_alive_masks.copy()
        results = [env.step(actions[i]) for i, env in enumerate(self.envs)]
        transition_next = np.stack([result[0] for result in results])
        rewards = np.stack([result[1] for result in results])
        terminated = np.asarray([result[2] for result in results])
        truncated = np.asarray([result[3] for result in results])
        infos = [result[4] for result in results]
        next_alive = np.stack([info["red_alive_mask"] for info in infos])
        current, current_alive = transition_next.copy(), next_alive.copy()
        if auto_reset:
            for i, done in enumerate(terminated | truncated):
                if done:
                    self.episode_indices[i] += 1
                    current[i] = self._reset_one(i)
                    current_alive[i] = self.envs[i].red_alive_mask
        self.current_observations = current
        self.current_alive_masks = current_alive
        return VectorStep(
            current.copy(), transition_next, rewards, terminated, truncated,
            infos, alive_before, next_alive,
        )

    def step(self, actions: np.ndarray):
        result = self.step_batch(actions)
        return result.observations, result.rewards, result.terminated, result.truncated, result.infos
