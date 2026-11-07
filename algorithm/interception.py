"""CLI and experiment lifecycle for three-dimensional interception."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

from env.interception import InterceptionConfig, InterceptionEnv, SimulationFailure
from env.interception.config import VERSION, MAX_PURSUERS, SCALES
from env.interception.scenario import unit
from algorithm.common.interception_vector import EpisodeAccumulator, InterceptionVectorEnv
from algorithm.common.interception_checkpoint import save, read, restore, load_weights
from algorithm.common.runtime import require_cuda
from algorithm.common.serialization import serializable, write_json
from algorithm.mappo.interception import MAPPO, TrainConfig


@contextmanager
def run_lock(folder):
    # OS locks release when an interrupted process exits.
    with (folder / ".train.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def write_update(folder, record):
    path = folder / f"update_{record['update']:06}.json"
    with tempfile.NamedTemporaryFile(dir=folder, mode="w", encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(serializable(record), stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train(folder, env_config=None, train_config=None, updates=1, resume=False):
    require_cuda()
    if updates < 1:
        raise ValueError("updates must be positive")
    folder = Path(folder)
    if not resume:
        folder.mkdir(parents=True, exist_ok=False)
    latest = folder / "latest.pt"
    with run_lock(folder):
        payload = read(latest) if resume else None
        if resume:
            env_config = InterceptionConfig(**payload["environment"])
            train_config = TrainConfig(**payload["training"])
        random.seed(train_config.seed)
        np.random.seed(train_config.seed)
        torch.manual_seed(train_config.seed)
        model = MAPPO(train_config)
        with InterceptionVectorEnv(env_config, train_config.workers, train_config.seed) as vector:
            if payload:
                restore(payload, model, env_config, vector)
                if payload["last_log"] is not None:
                    write_update(folder, payload["last_log"])
            else:
                write_json(folder / "config.json", dict(version=VERSION, environment=asdict(env_config),
                                                       training=asdict(train_config)))
                save(latest, model, env_config, vector)
            try:
                for _ in range(updates):
                    rollout, episodes = model.collect(vector)
                    metrics = model.update(rollout)
                    model.episodes += len(episodes)
                    record = dict(update=model.updates, joint_steps=model.steps, episodes=model.episodes,
                                  metrics=metrics, completed_episodes=episodes)
                    save(latest, model, env_config, vector, record)
                    write_update(folder, record)
                    print(json.dumps({k: v for k, v in record.items() if k != "completed_episodes"}), flush=True)
            except (RuntimeError, FloatingPointError) as exc:
                write_json(folder / f"failure_{time.time_ns()}.json",
                           dict(error=str(exc), last_committed=str(latest),
                                simulation_failure="simulation_failure" in str(exc)))
                raise
    return latest


def reference_action(info, config, assignments):
    commands = np.zeros((MAX_PURSUERS, 3))
    for i, j in assignments:
        if not info["evader_mask"][j]:
            continue
        p = info["pursuers"][i]
        e = info["evaders"][j]
        # Relative stopping distance under the same first-order lag. Brake
        # early by two control periods, then hold a non-contact offset.
        outward = unit(e[:3] - np.asarray(config.center))
        delta = e[:3] - p[:3]
        axial = delta @ outward
        gap = axial - 1.4
        relative_velocity = e[3:] - p[3:]
        closing = -relative_velocity @ outward
        target_speed = config.evader_vmax * config.evader_speed_fraction
        brake = config.pursuer_vmax - target_speed
        speed = max(0., closing)
        stopping = config.pursuer_tau * (speed - brake * np.log1p(speed / brake))
        relative_command = (config.pursuer_vmax + target_speed
                            if gap > stopping + 2 * config.dt * speed else -brake)
        if abs(gap) < .55 and abs(closing) < 1.25:
            relative_command = 1.5 * gap - 1.6 * closing
        lateral = delta - axial * outward
        commands[i] = ((relative_command - target_speed) * outward + .75 * lateral +
                       relative_velocity + closing * outward)
    return .5 + .5 * commands / np.maximum(config.pursuer_vmax,
                                           np.linalg.norm(commands, axis=-1, keepdims=True))


def trace_row(info):
    return {key: info[key] for key in ("pursuers", "evaders", "active_mask", "evader_mask",
                                      "events", "step", "time", "outcome", "minimum_separation",
                                      "reward_parts")} | {"commands": info.get("commands")}


def episode(config, seed, options, policy, model=None, trace=False):
    if policy == "reference" and (config.evader_mode != "goal_seeking" or
                                  config.pursuer_tau != config.evader_tau or
                                  config.pursuer_vmax <= config.evader_vmax * config.evader_speed_fraction):
        raise ValueError("reference controller requires goal_seeking, equal lag and faster pursuers")
    env = InterceptionEnv(config)
    rng = np.random.default_rng([2, seed])
    obs, info = env.reset(np.random.SeedSequence([1, seed]), options)
    accumulator = EpisodeAccumulator()
    accumulator.reset(info)
    frames = [trace_row(info)] if trace else []
    pursuers = list(np.flatnonzero(info["active_mask"]))
    assignments = []
    for j in np.flatnonzero(info["evader_mask"]):
        i = min(pursuers, key=lambda i: np.linalg.norm(info["pursuers"][i, :3] - info["evaders"][j, :3]))
        assignments.append((i, j))
        pursuers.remove(i)
    while True:
        if policy == "random":
            action = rng.random((MAX_PURSUERS, 3))
        elif policy == "reference":
            action = reference_action(info, config, assignments)
        else:
            action = model.act(obs, info["active_mask"], deterministic=True)[0].cpu().numpy()
        try:
            obs, rewards, terminated, truncated, info = env.step(action)
        except SimulationFailure as exc:
            result = accumulator.summary(info) | {"outcome": "simulation_failure", "error": str(exc)}
            break
        accumulator.add(info, rewards)
        if trace:
            frames.append(trace_row(info))
        if terminated or truncated:
            result = accumulator.summary(info)
            break
    return result | {"seed": seed}, frames


def summarize(rows):
    total = len(rows)
    successful = [row["time"] for row in rows if row["outcome"] == "success"]
    outcomes = {name: sum(row["outcome"] == name for row in rows)
                for name in ("success", "breach", "collision", "timeout", "simulation_failure")}
    return dict(episodes=total, outcomes=outcomes,
                success_rate=outcomes["success"] / total,
                breach_rate=outcomes["breach"] / total,
                collision_rate=outcomes["collision"] / total,
                timeout_rate=outcomes["timeout"] / total,
                numerical_failure_count=outcomes["simulation_failure"],
                capture_time_samples=len(successful),
                mean_capture_time_s=float(np.mean(successful)) if successful else None,
                minimum_separation_m=min(row["minimum_separation"] for row in rows),
                mean_episode_return=float(np.mean([row["episode_return"] for row in rows])),
                mean_control_effort=float(np.mean([row["control_effort"] for row in rows])))


def evaluate(folder, config, options, policy, seeds, model=None):
    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, seed in enumerate(seeds):
        row, frames = episode(config, seed, options, policy, model, trace=True)
        rows.append(row)
        write_json(folder / f"trajectory_{index:03}.json",
                   dict(version=VERSION, config=asdict(config), seed=seed, policy=policy,
                        scenario=options, result=row, frames=frames))
    summary = summarize(rows)
    write_json(folder / "summary.json", summary | {"runs": rows})
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)

    def environment_flags(command):
        command.add_argument("--env", type=Path, help="JSON overrides of InterceptionConfig")
        command.add_argument("--scale", choices=SCALES)

    init = sub.add_parser("validate")
    environment_flags(init)
    run = sub.add_parser("run")
    environment_flags(run)
    assess = sub.add_parser("evaluate")
    for command in (init, run, assess):
        command.add_argument("--scenario", choices=("fixed", "random", "custom"), default="fixed")
        command.add_argument("--initial", type=Path, help="JSON custom pursuers/evaders or spawn distance")
        command.add_argument("--seed", type=int, default=1000)
    for command in (run, assess):
        command.add_argument("--episodes", type=int, default=1)
        command.add_argument("--out", type=Path, required=True)
    run.add_argument("--policy", choices=("reference", "random"), default="reference")
    assess.add_argument("checkpoint", type=Path)
    fit = sub.add_parser("train")
    environment_flags(fit)
    fit.add_argument("--training", type=Path, help="JSON overrides of TrainConfig")
    fit.add_argument("--updates", type=int, default=1)
    fit.add_argument("--out", type=Path, required=True)
    continuation = sub.add_parser("resume")
    continuation.add_argument("--out", type=Path, required=True)
    continuation.add_argument("--updates", type=int, default=1)
    args = parser.parse_args(argv)
    if args.operation == "resume":
        train(args.out, updates=args.updates, resume=True)
        return
    if args.operation == "evaluate":
        payload = read(args.checkpoint)
        config = InterceptionConfig(**payload["environment"])
        model = MAPPO(TrainConfig(**payload["training"]))
        load_weights(payload, model)
    else:
        overrides = json.loads(args.env.read_text()) if args.env else {}
        if args.scale:
            overrides["scale"] = args.scale
        config = InterceptionConfig(**overrides)
    if args.operation == "train":
        settings = json.loads(args.training.read_text()) if args.training else {}
        train(args.out, config, TrainConfig(**settings), args.updates)
        return
    options = json.loads(args.initial.read_text()) if args.initial else {}
    options["scenario"] = args.scenario
    if args.operation == "validate":
        obs, info = InterceptionEnv(config).reset(args.seed, options)
        print(json.dumps(serializable(dict(version=VERSION, observation_shape=obs.shape,
                                           centralized_shape=info["centralized_state"].shape,
                                           configuration=asdict(config), initial=info,
                                           maximum_acceleration_m_s2=[2 * config.pursuer_vmax / config.pursuer_tau,
                                                                      2 * config.evader_vmax / config.evader_tau])),
                         indent=2, allow_nan=False))
    else:
        seeds = list(range(args.seed, args.seed + args.episodes))
        evaluate(args.out, config, options, args.policy if args.operation == "run" else "checkpoint",
                 seeds, None if args.operation == "run" else model)


if __name__ == "__main__":
    main()
