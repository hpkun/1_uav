"""Importable 1v1 environment placeholder."""

from __future__ import annotations

from typing import Any

from uav_env.envs.base_env import BaseUAVEnv


class Combat1v1Env(BaseUAVEnv):
    """Future one-versus-one Gymnasium environment."""

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """Reset the future 1v1 environment."""

        raise NotImplementedError("Combat1v1Env.reset is not implemented")

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Advance the future 1v1 environment."""

        raise NotImplementedError("Combat1v1Env.step is not implemented")
