"""Base Gymnasium-style environment interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

try:
    import gymnasium as gym
except ModuleNotFoundError:  # pragma: no cover - only for minimal source checks
    class _FallbackEnv:
        """Tiny import-time fallback when optional runtime dependencies are absent."""

        @classmethod
        def __class_getitem__(cls, item: object) -> type["_FallbackEnv"]:
            return cls

    class _FallbackGym:
        Env = _FallbackEnv

    gym = _FallbackGym()


class BaseUAVEnv(gym.Env[Any, Any], ABC):
    """Base contract for future UAV environments.

    Subclasses will follow Gymnasium's ``reset()`` and ``step()`` signatures.
    No transition or observation values are fabricated in this project phase.
    """

    @abstractmethod
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """Reset a future environment episode."""

        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Advance a future environment by one decision step."""

        raise NotImplementedError
