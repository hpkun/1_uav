"""Importable multi-UAV environment placeholder."""

from __future__ import annotations

from typing import Any

from uav_env.envs.base_env import BaseUAVEnv


class CombatMultiEnv(BaseUAVEnv):
    """Future heterogeneous multi-UAV Gymnasium environment."""

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """Reset the future multi-UAV environment."""

        raise NotImplementedError("CombatMultiEnv.reset is not implemented")

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Advance the future multi-UAV environment."""

        raise NotImplementedError("CombatMultiEnv.step is not implemented")
