"""Transparent algorithm-side multi-wave reward transformation."""
from __future__ import annotations

from typing import Any
import numpy as np

from .base import CapabilityModule


class MultiWaveRewardAdapter(CapabilityModule):
    name = "multi_wave_reward"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.mode = str(self.config.get("mode", "none"))
        if self.mode not in {"none", "wave_clear_bonus", "survivor_wave_clear_bonus", "round_scaled"}:
            raise ValueError(f"invalid reward adapter mode: {self.mode}")
        self.bonuses = {int(k.replace("wave", "")): float(v) for k, v in self.config.get("bonuses", {"wave1": 1.0, "wave2": 2.0, "wave3": 3.0}).items()}
        self.scales = {int(k.replace("wave", "")): float(v) for k, v in self.config.get("round_scales", {"wave1": 1.0, "wave2": 1.0, "wave3": 1.0}).items()}
        self.survivor_scale = float(self.config.get("survivor_scale", 0.25))

    def adapt(self, raw_rewards: np.ndarray, infos: list[dict[str, Any]],
              transition_wave_indices: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, float]]:
        raw = np.asarray(raw_rewards, dtype=np.float32); training = raw.copy()
        transition_waves = (
            np.asarray(transition_wave_indices, dtype=np.int64)
            if transition_wave_indices is not None else None
        )
        if transition_waves is not None and transition_waves.shape != (raw.shape[0],):
            raise ValueError("transition_wave_indices must have shape [num_envs]")
        bonus_by_wave = {1: 0.0, 2: 0.0, 3: 0.0}
        if not self.enabled or self.mode == "none":
            return training, self._metrics(raw, training, bonus_by_wave)
        for env_id, info in enumerate(infos):
            # Reward belongs to the state/action transition that started in
            # this wave, never the post-step wave reported after a spawn.
            wave = int(transition_waves[env_id]) if transition_waves is not None else (
                int(info.get("waves_cleared", 0)) if info.get("wave_cleared_this_step", False)
                else int(info.get("wave_index", 1))
            )
            wave = max(1, wave)
            if self.mode == "round_scaled":
                training[env_id] *= self.scales.get(wave, 1.0)
            elif info.get("wave_cleared_this_step", False):
                bonus = self.bonuses.get(wave, 0.0)
                if self.mode == "survivor_wave_clear_bonus":
                    bonus += self.survivor_scale * float(info.get("red_survivors", 0))
                alive = np.asarray(info.get("red_alive_mask", np.ones(raw.shape[1])), dtype=np.float32)
                training[env_id] += bonus * alive
                bonus_by_wave[wave] = bonus_by_wave.get(wave, 0.0) + float(bonus * alive.sum())
        return training, self._metrics(raw, training, bonus_by_wave)

    @staticmethod
    def _metrics(raw: np.ndarray, training: np.ndarray, bonus: dict[int, float]) -> dict[str, float]:
        return {"raw_reward_mean": float(raw.mean()), "training_reward_mean": float(training.mean()),
                "reward_bonus_total": float((training - raw).sum()),
                **{f"reward_bonus_wave{wave}": float(bonus.get(wave, 0.0)) for wave in (1, 2, 3)}}


__all__ = ["MultiWaveRewardAdapter"]
