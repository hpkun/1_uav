"""Long lived spawn workers with explicit transition and reset observations."""
import multiprocessing as mp
import os
import traceback
import copy
from dataclasses import dataclass
import numpy as np
from env.config import EnvConfig
from env.factory import make_environment
from .metrics import EpisodeAccumulator


def _worker(pipe, config):
    try:
        env = make_environment(config)
        pipe.send(('ok', os.getpid()))
        while True:
            command, payload = pipe.recv()
            if command == 'close':
                break
            if command == 'reset':
                value = env.reset(payload)
            elif command == 'step':
                value = env.step(payload)
            elif command == 'snapshot':
                value = dict(task=env.task, rng=env.rng.bit_generator.state, step=env.step_count,
                             done=env.done, counters=env.counters, assignments=env.assignments,
                             numerical_delta=env.numerical_delta, numerical_events=env.numerical_events)
            elif command == 'restore':
                env.task = payload['task']
                env.rng.bit_generator.state = payload['rng']
                env.step_count, env.done = payload['step'], payload['done']
                env.counters, env.assignments = payload['counters'], payload['assignments']
                env.numerical_delta, env.numerical_events = payload['numerical_delta'], payload['numerical_events']
                value = None
            else:
                raise ValueError(f'unknown worker command {command}')
            pipe.send(('ok', value))
    except BaseException:
        try:
            pipe.send(('error', traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        pipe.close()


@dataclass
class VectorStep:
    observations: np.ndarray
    transition_observations: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    infos: list
    current_masks: np.ndarray
    next_masks: np.ndarray
    states: np.ndarray
    transition_states: np.ndarray
    indices: np.ndarray
    episodes: list


class ParallelVectorEnv:
    def __init__(self, config=None, num_envs=10, seed=1, forbidden_seeds=(), timeout=60):
        if num_envs < 1:
            raise ValueError('num_envs must be positive')
        self.config = config or EnvConfig()
        self.num_envs, self.seed, self.timeout = num_envs, seed, timeout
        self.forbidden_seeds = set(forbidden_seeds)
        self.episodes = np.zeros(num_envs, dtype=np.int64)
        self.connections, self.processes = [], []
        self.closed = False
        self.trackers = [EpisodeAccumulator() for _ in range(num_envs)]
        context = mp.get_context('spawn')
        try:
            for _ in range(num_envs):
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, self.config), daemon=True)
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            self.worker_pids = [self._receive(p) for p in self.connections]
            self.reset()
        except BaseException:
            self.close()
            raise

    def _receive(self, pipe):
        if not pipe.poll(self.timeout):
            raise TimeoutError('environment worker did not respond')
        status, payload = pipe.recv()
        if status != 'ok':
            raise RuntimeError(payload)
        return payload

    def _seed(self, i):
        value = int(self.seed + self.episodes[i] * self.num_envs + i)
        if value in self.forbidden_seeds:
            raise ValueError(f'training seed overlaps evaluation seeds: {value}')
        return value

    def reset(self):
        for i, pipe in enumerate(self.connections):
            pipe.send(('reset', self._seed(i)))
        pairs = [self._receive(pipe) for pipe in self.connections]
        self.observations = np.stack([x[0] for x in pairs])
        self.masks = np.stack([x[1]['active_mask'] for x in pairs])
        self.states = np.stack([x[1]['centralized_state'] for x in pairs])
        for tracker, (_, info) in zip(self.trackers, pairs):
            tracker.reset(info)
        return self.observations.copy()

    def step(self, actions, indices=None):
        indices = np.arange(self.num_envs) if indices is None else np.asarray(indices, dtype=int)
        if indices.ndim != 1 or len(indices) == 0 or len(set(indices.tolist())) != len(indices) or np.any((indices < 0) | (indices >= self.num_envs)):
            raise ValueError('invalid worker indices')
        if np.shape(actions) != (len(indices), self.config.num_uavs, 3):
            raise ValueError('vector action dimensions mismatch')
        for i, action in zip(indices, actions):
            self.connections[i].send(('step', action))
        rows = [self._receive(self.connections[i]) for i in indices]
        transition_obs = np.stack([r[0] for r in rows])
        observations = transition_obs.copy()
        old_masks = self.masks[indices].copy()
        next_masks = np.stack([r[4]['active_mask'] for r in rows])
        self.masks[indices] = next_masks
        transition_states = np.stack([r[4]['centralized_state'] for r in rows])
        states = transition_states.copy()
        resetting = []
        episodes = []
        for k, (i, row) in enumerate(zip(indices, rows)):
            self.trackers[i].add(row[4], row[1])
            if row[2] or row[3]:
                episodes.append(self.trackers[i].summary(row[4]))
                self.episodes[i] += 1
                self.connections[i].send(('reset', self._seed(i)))
                resetting.append((k, i))
        for k, i in resetting:
            observations[k], info = self._receive(self.connections[i])
            self.masks[i] = info['active_mask']
            states[k] = info['centralized_state']
            self.trackers[i].reset(info)
        self.observations[indices] = observations
        self.states[indices] = states
        return VectorStep(observations, transition_obs, np.stack([r[1] for r in rows]),
                          np.array([r[2] for r in rows]), np.array([r[3] for r in rows]),
                          [r[4] for r in rows], old_masks, next_masks, states, transition_states, indices, episodes)

    def snapshot(self):
        for p in self.connections:
            p.send(('snapshot', None))
        return dict(workers=[self._receive(p) for p in self.connections], episodes=self.episodes.copy(),
                    observations=self.observations.copy(), masks=self.masks.copy(), states=self.states.copy(), trackers=copy.deepcopy(self.trackers))

    def restore(self, state):
        if len(state['workers']) != self.num_envs:
            raise ValueError('checkpoint worker count mismatch')
        for p, value in zip(self.connections, state['workers']):
            p.send(('restore', value))
        for p in self.connections:
            self._receive(p)
        self.episodes = state['episodes'].copy()
        self.observations, self.masks = state['observations'].copy(), state['masks'].copy()
        self.states, self.trackers = state['states'].copy(), copy.deepcopy(state['trackers'])

    def close(self):
        if self.closed:
            return
        self.closed = True
        for pipe in self.connections:
            try:
                pipe.send(('close', None))
            except (OSError, BrokenPipeError):
                pass
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for pipe in self.connections:
            pipe.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
