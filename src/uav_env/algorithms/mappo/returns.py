"""GAE with separate termination, truncation bootstrap, and reset boundaries."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_gae(
    rewards: NDArray[np.float32], values: NDArray[np.float32], terminated: NDArray[np.bool_],
    truncated: NDArray[np.bool_], terminal_values: NDArray[np.float32], gamma: float, gae_lambda: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return advantages/returns for arrays shaped ``[T,E,A]`` and values ``[T+1,E,A]``."""

    if values.shape[0] != rewards.shape[0] + 1 or terminated.shape != rewards.shape[:2] or truncated.shape != rewards.shape[:2]:
        raise ValueError("GAE shape mismatch")
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = np.zeros_like(rewards[0], dtype=np.float32)
    for step in range(rewards.shape[0] - 1, -1, -1):
        term = terminated[step, :, None]
        trunc = truncated[step, :, None]
        bootstrap = np.where(term, 0.0, np.where(trunc, terminal_values[step], values[step + 1]))
        delta = rewards[step] + gamma * bootstrap - values[step]
        continuation = (~(term | trunc)).astype(np.float32)
        gae = delta + gamma * gae_lambda * continuation * gae
        advantages[step] = gae
    return advantages, advantages + values[:-1]
