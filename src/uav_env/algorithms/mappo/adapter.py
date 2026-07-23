"""Shape-checked MAPPO adapters without changing Gymnasium environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from uav_env.actions.discrete_15 import DiscreteAction15
from uav_env.envs.combat_1v1_env import Combat1v1Env
from uav_env.envs.combat_multi_env import CombatMultiEnv


@dataclass(frozen=True)
class AdapterStep:
    local_obs: NDArray[np.float32]
    global_state: NDArray[np.float32]
    agent_rewards: NDArray[np.float32]
    agent_alive_mask: NDArray[np.float32]
    available_action_mask: NDArray[np.bool_]
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class MAPPOEnvAdapter:
    """Unify the fixed 1v1 and 2v2 red-agent interfaces."""

    def __init__(self, env: Combat1v1Env | CombatMultiEnv) -> None:
        self.env = env
        self.num_agents = 1 if isinstance(env, Combat1v1Env) else 2
        self.obs_dim = 11 if self.num_agents == 1 else 28
        self.state_dim = 10 if self.num_agents == 1 else 40

    def _pack(self, observation: NDArray[np.float64], info: dict[str, Any], rewards: Sequence[float], terminated: bool, truncated: bool) -> AdapterStep:
        local = np.asarray(observation, dtype=np.float32).reshape(self.num_agents, self.obs_dim)
        if self.num_agents == 1:
            state = np.asarray(info["critic_state"], dtype=np.float32)
            alive = np.asarray([float(info["red_state"].alive)], dtype=np.float32)
            available = np.ones((1, 15), dtype=bool)
            if not bool(alive[0]):
                available[:] = False
                available[0, int(DiscreteAction15.LEVEL_HOLD)] = True
        else:
            state = np.asarray(info["global_state"], dtype=np.float32)
            alive = np.asarray(info["red_agent_alive_mask"], dtype=np.float32)
            available = np.asarray(info["available_action_mask"], dtype=bool)
        if state.shape != (self.state_dim,) or available.shape != (self.num_agents, 15):
            raise ValueError(f"Adapter shape mismatch: state={state.shape}, available={available.shape}")
        return AdapterStep(local, state, np.asarray(rewards, dtype=np.float32), alive, available, terminated, truncated, info)

    def reset(self, seed: int | None = None) -> AdapterStep:
        observation, info = self.env.reset(seed=seed)
        return self._pack(observation, info, np.zeros(self.num_agents), False, False)

    def step(self, actions: NDArray[np.int64]) -> AdapterStep:
        parsed = np.asarray(actions, dtype=np.int64).reshape(self.num_agents)
        env_action: int | NDArray[np.int64] = int(parsed[0]) if self.num_agents == 1 else parsed
        observation, team_reward, terminated, truncated, info = self.env.step(env_action)
        if self.num_agents == 1:
            rewards = [float(team_reward)]
        else:
            rewards = [float(info["agent_rewards"][f"red_{index}"]) for index in range(self.num_agents)]
        return self._pack(observation, info, rewards, terminated, truncated)


class SyncCombatVectorEnv:
    """Small deterministic synchronous vector environment with terminal-state retention."""

    def __init__(self, factories: Sequence[Callable[[], MAPPOEnvAdapter]], base_seed: int) -> None:
        if not factories:
            raise ValueError("At least one environment is required")
        self.envs = [factory() for factory in factories]
        self.base_seed = int(base_seed)
        first = self.envs[0]
        self.num_envs, self.num_agents = len(self.envs), first.num_agents
        self.obs_dim, self.state_dim = first.obs_dim, first.state_dim
        if any((e.num_agents, e.obs_dim, e.state_dim) != (self.num_agents, self.obs_dim, self.state_dim) for e in self.envs):
            raise ValueError("All vector environments must have identical dimensions")

    @staticmethod
    def _stack(steps: Sequence[AdapterStep]) -> dict[str, Any]:
        return {
            "local_obs": np.stack([s.local_obs for s in steps]), "global_state": np.stack([s.global_state for s in steps]),
            "rewards": np.stack([s.agent_rewards for s in steps]), "alive_masks": np.stack([s.agent_alive_mask for s in steps]),
            "available_actions": np.stack([s.available_action_mask for s in steps]),
            "terminated": np.asarray([s.terminated for s in steps], dtype=bool), "truncated": np.asarray([s.truncated for s in steps], dtype=bool),
            "infos": [s.info for s in steps],
        }

    def reset(self) -> dict[str, Any]:
        return self._stack([env.reset(self.base_seed + index) for index, env in enumerate(self.envs)])

    def step(self, actions: NDArray[np.int64]) -> dict[str, Any]:
        if np.asarray(actions).shape != (self.num_envs, self.num_agents):
            raise ValueError("Vector action shape mismatch")
        terminal_steps: list[AdapterStep] = []
        reset_steps: list[AdapterStep | None] = []
        for index, env in enumerate(self.envs):
            try:
                terminal = env.step(np.asarray(actions[index], dtype=np.int64))
                terminal_steps.append(terminal)
                if terminal.terminated or terminal.truncated:
                    reset_steps.append(env.reset())
                else:
                    reset_steps.append(None)
            except Exception as error:
                raise RuntimeError(f"Vector env_index={index}, seed={self.base_seed + index} failed") from error
        result = self._stack(terminal_steps)
        result["terminal_steps"] = terminal_steps
        result["reset_steps"] = reset_steps
        result["next_local_obs"] = np.stack([(reset_steps[i] or terminal_steps[i]).local_obs for i in range(self.num_envs)])
        result["next_global_state"] = np.stack([(reset_steps[i] or terminal_steps[i]).global_state for i in range(self.num_envs)])
        result["next_alive_masks"] = np.stack([(reset_steps[i] or terminal_steps[i]).agent_alive_mask for i in range(self.num_envs)])
        result["next_available_actions"] = np.stack([(reset_steps[i] or terminal_steps[i]).available_action_mask for i in range(self.num_envs)])
        return result
