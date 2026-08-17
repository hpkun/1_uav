"""Chunked 4-agent CTDE replay buffer with million-transition logical capacity."""
from __future__ import annotations
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int = 1_000_000, num_agents: int = 4, observation_dim: int = 45, action_dim: int = 3, chunk_size: int = 4096) -> None:
        if capacity <= 0: raise ValueError("capacity must be positive")
        self.capacity, self.num_agents = int(capacity), int(num_agents)
        self.observation_dim, self.action_dim, self.chunk_size = observation_dim, action_dim, chunk_size
        self._chunks: dict[int, dict[str, np.ndarray]] = {}; self.position = self.size = 0

    def _chunk(self, index: int) -> tuple[dict[str, np.ndarray], int]:
        ci, offset = divmod(index, self.chunk_size)
        if ci not in self._chunks:
            count = min(self.chunk_size, self.capacity - ci * self.chunk_size)
            self._chunks[ci] = {
                "observations": np.empty((count, self.num_agents, self.observation_dim), np.float32), "actions": np.empty((count, self.num_agents, self.action_dim), np.float32),
                "rewards": np.empty((count, self.num_agents), np.float32), "next_observations": np.empty((count, self.num_agents, self.observation_dim), np.float32), "dones": np.empty((count, 1), np.float32),
            }
        return self._chunks[ci], offset

    def push(self, observations, actions, rewards, next_observations, done: bool) -> None:
        expected = ((self.num_agents, self.observation_dim), (self.num_agents, self.action_dim), (self.num_agents,), (self.num_agents, self.observation_dim))
        arrays = [np.asarray(x, np.float32) for x in (observations, actions, rewards, next_observations)]
        if tuple(a.shape for a in arrays) != expected: raise ValueError(f"invalid replay transition shapes: {[a.shape for a in arrays]}")
        chunk, offset = self._chunk(self.position)
        for key, value in zip(("observations", "actions", "rewards", "next_observations"), arrays): chunk[key][offset] = value
        chunk["dones"][offset, 0] = float(done)
        self.position = (self.position + 1) % self.capacity; self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        if batch_size <= 0 or self.size < batch_size: raise ValueError("not enough transitions")
        indices = (rng or np.random.default_rng()).choice(self.size, size=batch_size, replace=False)
        rows = [self._chunk(int(i)) for i in indices]
        return {key: torch.as_tensor(np.stack([chunk[key][offset] for chunk, offset in rows]), device=device) for key in ("observations", "actions", "rewards", "next_observations", "dones")}
