"""Shape-checked MAPPO adapters without changing Gymnasium environments."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.connection import Connection
import traceback
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
    team_reward: float
    agent_reward_sum: float
    agent_alive_mask: NDArray[np.float32]
    available_action_mask: NDArray[np.bool_]
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class MAPPOEnvAdapter:
    """Unify fixed 1v1, 2v2, and 3v3 red-agent interfaces."""

    def __init__(self, env: Combat1v1Env | CombatMultiEnv) -> None:
        self.env = env
        if isinstance(env, Combat1v1Env):
            self.num_agents, self.obs_dim, self.state_dim = 1, 11, 10
        else:
            self.num_agents = int(env.red_count)
            self.obs_dim = int(env.observation_space.shape[-1])
            self.state_dim = int(env.global_state_dim)

    def _pack(self, observation: NDArray[np.float64], info: dict[str, Any], rewards: Sequence[float], terminated: bool, truncated: bool, team_reward: float | None = None) -> AdapterStep:
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
        agent_rewards = np.asarray(rewards, dtype=np.float32)
        scalar_team_reward = float(np.mean(agent_rewards)) if team_reward is None else float(team_reward)
        if not np.isclose(scalar_team_reward, float(np.mean(agent_rewards)), rtol=1.0e-6, atol=1.0e-6):
            raise ValueError("Environment team reward must equal mean agent rewards")
        return AdapterStep(local, state, agent_rewards, scalar_team_reward, float(np.sum(agent_rewards)), alive, available, terminated, truncated, info)

    def reset(self, seed: int | None = None) -> AdapterStep:
        observation, info = self.env.reset(seed=seed)
        return self._pack(observation, info, np.zeros(self.num_agents), False, False, 0.0)

    def step(self, actions: NDArray[np.int64]) -> AdapterStep:
        parsed = np.asarray(actions, dtype=np.int64).reshape(self.num_agents)
        env_action: int | NDArray[np.int64] = int(parsed[0]) if self.num_agents == 1 else parsed
        observation, team_reward, terminated, truncated, info = self.env.step(env_action)
        if self.num_agents == 1:
            rewards = [float(team_reward)]
        else:
            rewards = [float(info["agent_rewards"][f"red_{index}"]) for index in range(self.num_agents)]
        return self._pack(observation, info, rewards, terminated, truncated, float(team_reward))


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
            "team_rewards": np.asarray([s.team_reward for s in steps], dtype=np.float32),
            "agent_reward_sums": np.asarray([s.agent_reward_sum for s in steps], dtype=np.float32),
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

    def get_state(self) -> list[MAPPOEnvAdapter]:
        """Return serializable adapter/environment state for checkpoint v3."""

        return self.envs

    def set_state(self, state: Sequence[MAPPOEnvAdapter]) -> None:
        """Restore serializable adapter/environment state."""

        restored = list(state)
        if len(restored) != self.num_envs:
            raise ValueError("Vector environment state length mismatch")
        self.envs = restored

    def close(self) -> None:
        """Close synchronous environments when they expose a close method."""

        for adapter in self.envs:
            adapter.env.close()


@dataclass(frozen=True)
class CombatEnvDescription:
    """Serializable fixed environment construction arguments for workers."""

    kind: str
    scenario: str
    opponent: str
    multi_terminal_reward_profile: str | None = None
    functional_mode: str | None = None
    red_roles: tuple[str, ...] | None = None
    relay_enabled: bool | None = None


def make_adapter_from_description(description: CombatEnvDescription, seed: int | None = None) -> MAPPOEnvAdapter:
    """Construct one adapter from a process-safe environment description."""

    from uav_env.envs import make_1v1_env, make_2v2_env, make_3v3_env

    if description.kind == "1v1":
        env = make_1v1_env(description.scenario, description.opponent, seed=seed)
    elif description.kind == "2v2":
        env = make_2v2_env(
            description.scenario, description.opponent, seed=seed,
            multi_terminal_reward_profile=description.multi_terminal_reward_profile,
        )
    elif description.kind == "3v3":
        env = make_3v3_env(
            description.scenario, description.opponent, seed=seed,
            multi_terminal_reward_profile=description.multi_terminal_reward_profile,
            functional_mode=description.functional_mode,
            red_roles=list(description.red_roles) if description.red_roles is not None else None,
            relay_enabled=description.relay_enabled,
        )
    else:
        raise ValueError(f"Unknown environment kind: {description.kind!r}")
    return MAPPOEnvAdapter(env)


def _parallel_worker(connection: Connection, description: CombatEnvDescription, seed: int) -> None:
    """Own one environment and service commands until explicitly closed."""

    adapter: MAPPOEnvAdapter | None = None
    try:
        adapter = make_adapter_from_description(description, seed)
        while True:
            command, payload = connection.recv()
            if command == "reset":
                connection.send(("ok", adapter.reset(payload)))
            elif command == "step":
                terminal = adapter.step(np.asarray(payload, dtype=np.int64))
                reset = adapter.reset() if terminal.terminated or terminal.truncated else None
                connection.send(("ok", (terminal, reset)))
            elif command == "get_state":
                connection.send(("ok", adapter))
            elif command == "set_state":
                adapter = payload
                connection.send(("ok", None))
            elif command == "close":
                adapter.env.close()
                connection.send(("ok", None))
                break
            else:
                raise ValueError(f"Unknown worker command: {command!r}")
    except BaseException as error:
        try:
            connection.send(("error", f"{type(error).__name__}: {error}\n{traceback.format_exc()}"))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class ParallelCombatVectorEnv:
    """Process-parallel fixed vector environment with terminal-state retention."""

    def __init__(self, description: CombatEnvDescription, num_envs: int, base_seed: int, response_timeout: float = 60.0) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.description = description
        self.num_envs = int(num_envs)
        self.base_seed = int(base_seed)
        self.response_timeout = float(response_timeout)
        probe = make_adapter_from_description(description, base_seed)
        self.num_agents, self.obs_dim, self.state_dim = probe.num_agents, probe.obs_dim, probe.state_dim
        probe.env.close()
        context = mp.get_context("spawn")
        self._connections: list[Connection] = []
        self._processes: list[mp.Process] = []
        self._closed = False
        for index in range(self.num_envs):
            parent, child = context.Pipe()
            process = context.Process(
                target=_parallel_worker,
                args=(child, description, self.base_seed + index),
                name=f"uav-env-{index}",
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)

    def _receive(self, index: int) -> Any:
        connection = self._connections[index]
        if not connection.poll(self.response_timeout):
            process = self._processes[index]
            raise RuntimeError(f"Parallel env worker {index} timed out; alive={process.is_alive()}")
        try:
            status, payload = connection.recv()
        except EOFError as error:
            raise RuntimeError(f"Parallel env worker {index} exited unexpectedly") from error
        if status != "ok":
            raise RuntimeError(f"Parallel env worker {index} failed:\n{payload}")
        return payload

    def reset(self) -> dict[str, Any]:
        """Reset all workers with deterministic base-seed offsets."""

        for index, connection in enumerate(self._connections):
            connection.send(("reset", self.base_seed + index))
        return SyncCombatVectorEnv._stack([self._receive(index) for index in range(self.num_envs)])

    def step(self, actions: NDArray[np.int64]) -> dict[str, Any]:
        """Advance all resident workers concurrently and auto-reset completed episodes."""

        action_array = np.asarray(actions)
        if action_array.shape != (self.num_envs, self.num_agents):
            raise ValueError("Vector action shape mismatch")
        for index, connection in enumerate(self._connections):
            connection.send(("step", np.asarray(action_array[index], dtype=np.int64)))
        pairs = [self._receive(index) for index in range(self.num_envs)]
        terminal_steps = [pair[0] for pair in pairs]
        reset_steps = [pair[1] for pair in pairs]
        result = SyncCombatVectorEnv._stack(terminal_steps)
        result["terminal_steps"] = terminal_steps
        result["reset_steps"] = reset_steps
        next_steps = [reset_steps[index] or terminal_steps[index] for index in range(self.num_envs)]
        result["next_local_obs"] = np.stack([step.local_obs for step in next_steps])
        result["next_global_state"] = np.stack([step.global_state for step in next_steps])
        result["next_alive_masks"] = np.stack([step.agent_alive_mask for step in next_steps])
        result["next_available_actions"] = np.stack([step.available_action_mask for step in next_steps])
        return result

    def get_state(self) -> list[MAPPOEnvAdapter]:
        """Fetch each resident environment for exact checkpoint resume."""

        for connection in self._connections:
            connection.send(("get_state", None))
        return [self._receive(index) for index in range(self.num_envs)]

    def set_state(self, state: Sequence[MAPPOEnvAdapter]) -> None:
        """Replace each resident environment from checkpoint state."""

        restored = list(state)
        if len(restored) != self.num_envs:
            raise ValueError("Vector environment state length mismatch")
        for connection, adapter in zip(self._connections, restored):
            connection.send(("set_state", adapter))
        for index in range(self.num_envs):
            self._receive(index)

    @property
    def worker_pids(self) -> tuple[int | None, ...]:
        """Return worker process IDs for lifecycle tests."""

        return tuple(process.pid for process in self._processes)

    @property
    def workers_alive(self) -> tuple[bool, ...]:
        """Return worker liveness flags."""

        return tuple(process.is_alive() for process in self._processes)

    def close(self) -> None:
        """Close all workers, terminating only workers that fail to acknowledge."""

        if self._closed:
            return
        self._closed = True
        for connection, process in zip(self._connections, self._processes):
            if process.is_alive():
                try:
                    connection.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for index, process in enumerate(self._processes):
            if process.is_alive():
                try:
                    self._receive(index)
                except RuntimeError:
                    pass
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        for connection in self._connections:
            connection.close()

    def __enter__(self) -> "ParallelCombatVectorEnv":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback_value: object) -> None:
        self.close()
