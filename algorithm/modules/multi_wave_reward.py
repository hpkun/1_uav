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
        if self.mode not in {"none", "wave_clear_bonus", "survivor_wave_clear_bonus", "round_scaled", "jiao_r2_replacement"}:
            raise ValueError(f"invalid reward adapter mode: {self.mode}")
        self.bonuses = {int(k.replace("wave", "")): float(v) for k, v in self.config.get("bonuses", {"wave1": 1.0, "wave2": 2.0, "wave3": 3.0}).items()}
        self.scales = {int(k.replace("wave", "")): float(v) for k, v in self.config.get("round_scales", {"wave1": 1.0, "wave2": 1.0, "wave3": 1.0}).items()}
        self.survivor_scale = float(self.config.get("survivor_scale", 0.25))
        self.last_transition: dict[str, np.ndarray] = {}

    def adapt(self, raw_rewards: np.ndarray, infos: list[dict[str, Any]],
              transition_wave_indices: np.ndarray | None = None,
              red_alive_before: np.ndarray | None = None,
              blue_alive_before: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, float]]:
        raw = np.asarray(raw_rewards, dtype=np.float32); training = raw.copy()
        transition_waves = (
            np.asarray(transition_wave_indices, dtype=np.int64)
            if transition_wave_indices is not None else None
        )
        if transition_waves is not None and transition_waves.shape != (raw.shape[0],):
            raise ValueError("transition_wave_indices must have shape [num_envs]")
        bonus_by_wave = {1: 0.0, 2: 0.0, 3: 0.0}
        blue_component = np.zeros(raw.shape[0], dtype=np.float32)
        red_component = np.zeros(raw.shape[0], dtype=np.float32)
        team_signal = np.zeros(raw.shape[0], dtype=np.float32)
        per_wave = np.zeros((raw.shape[0], 3), dtype=np.float32)
        self.last_transition = {
            "blue_component": blue_component,
            "red_component": red_component,
            "team_signal": team_signal,
            "per_wave": per_wave,
        }
        if not self.enabled or self.mode == "none":
            return training, self._metrics(raw, training, bonus_by_wave, blue_component, red_component, per_wave)
        if self.mode == "jiao_r2_replacement":
            if transition_waves is None or red_alive_before is None or blue_alive_before is None:
                raise ValueError("Jiao R2 requires transition waves and pre-transition red/blue alive masks")
            red_before = np.asarray(red_alive_before, dtype=np.float32)
            blue_before = np.asarray(blue_alive_before, dtype=np.float32)
            if red_before.shape != raw.shape or blue_before.shape != raw.shape:
                raise ValueError("Jiao R2 alive masks must match raw reward shape")
            training.fill(0.0)
            indices = np.arange(1, raw.shape[1] + 1, dtype=np.float32)
            for env_id, info in enumerate(infos):
                red_after = np.asarray(info["red_alive_mask"], dtype=np.float32)
                # persistent_wave_v2 immediately respawns Blue after a clear;
                # the transition itself nevertheless ended with all old Blue dead.
                blue_after = (np.zeros_like(blue_before[env_id]) if info.get("wave_cleared_this_step", False)
                              else np.asarray(info["blue_alive_mask"], dtype=np.float32))
                new_blue_deaths = (blue_before[env_id] > .5) & (blue_after <= .5)
                new_red_deaths = (red_before[env_id] > .5) & (red_after <= .5)
                wave = max(1, min(3, int(transition_waves[env_id])))
                blue_component[env_id] = float((indices[new_blue_deaths] * wave).sum())
                red_component[env_id] = -float(indices[new_red_deaths].sum())
                team_signal[env_id] = blue_component[env_id] + red_component[env_id]
                per_wave[env_id, wave - 1] = team_signal[env_id]
                # Fully cooperative reward goes to agents alive at transition start,
                # including an agent that is lost during this same transition.
                training[env_id] = team_signal[env_id] * red_before[env_id]
                bonus_by_wave[wave] += float(team_signal[env_id])
            return training, self._metrics(raw, training, bonus_by_wave, blue_component, red_component, per_wave)
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
        return training, self._metrics(raw, training, bonus_by_wave, blue_component, red_component, per_wave)

    @staticmethod
    def _metrics(raw: np.ndarray, training: np.ndarray, bonus: dict[int, float],
                 blue: np.ndarray, red: np.ndarray, per_wave: np.ndarray) -> dict[str, float]:
        return {"raw_reward_mean": float(raw.mean()), "training_reward_mean": float(training.mean()),
                "reward_bonus_total": float((training - raw).sum()),
                "raw_environment_reward": float(raw.sum()),
                "jiao_training_reward": float(training.sum()),
                "paper_R2_blue_kill_component": float(blue.sum()),
                "paper_R2_red_loss_component": float(red.sum()),
                **{f"paper_R2_wave{wave}": float(per_wave[:, wave - 1].sum()) for wave in (1, 2, 3)},
                **{f"reward_bonus_wave{wave}": float(bonus.get(wave, 0.0)) for wave in (1, 2, 3)}}


__all__ = ["MultiWaveRewardAdapter"]
