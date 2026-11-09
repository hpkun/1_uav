"""CLI and experiment lifecycle for three-dimensional interception."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time
from uuid import uuid4

import numpy as np
import torch

from env.interception import InterceptionConfig, InterceptionEnv, SimulationFailure
from env.interception.config import VERSION, MAX_PURSUERS, SCALES, SCENARIO_SCALES, TASKS
from env.interception.scenario import unit
from env.interception.motion import execute_velocity
from env.interception.protected_area import navigation_point
from env.interception.protected_area import navigation_point
from algorithm.common.interception_vector import EpisodeAccumulator, InterceptionVectorEnv
from algorithm.common.interception_checkpoint import PROTOCOL, save, read, restore, load_weights
from algorithm.common.research import (EvaluationProtocol, atomic_json, commit_header, digest,
    evaluation_csv, evaluation_records, file_digest, journal, publish_best, read_json,
    summarize, terminal_line, write_views)
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


def train(folder, env_config=None, train_config=None, updates=1, resume=False, *,
          evaluation=None, log_interval=1):
    if env_config is not None and env_config.scale not in SCALES:
        raise ValueError("3v3 is a reference simulation outside the MAPPO checkpoint protocol")
    require_cuda()
    if updates < 1 or log_interval < 1:
        raise ValueError("updates and log interval must be positive")
    folder = Path(folder)
    if not resume:
        folder.mkdir(parents=True, exist_ok=False)
    latest = folder / "latest.pt"
    with run_lock(folder):
        payload = read(latest) if resume else None
        if resume:
            head = commit_header(latest)
            run = head["run"]
            if (run != payload.get("research_run") or head["update"] != payload["updates"] or
                    run["environment"] != serializable(payload["environment"]) or
                    run["training"] != payload["training"] or
                    payload["steps"] != payload["updates"] * run["training"]["workers"] * run["training"]["horizon"] or
                    head["last_log"] != payload["last_log"] or read_json(folder / "config.json") != run):
                raise ValueError("run configuration or commit metadata differs from checkpoint")
            env_config = InterceptionConfig(**payload["environment"])
            train_config = TrainConfig(**payload["training"])
            evaluation = EvaluationProtocol(**run["evaluation"])
            records = journal(folder, head, repair=True)
            if payload["episodes"] != sum(len(row["completed_episodes"]) for row in records):
                raise ValueError("episode count differs from checkpoint")
        else:
            evaluation = evaluation or EvaluationProtocol()
            run = serializable(dict(id=uuid4().hex, version=VERSION, protocol=PROTOCOL,
                environment=asdict(env_config), training=asdict(train_config), evaluation=asdict(evaluation),
                initial_budget_updates=updates, training_scenario="random",
                seeds=dict(training="[0, seed, worker_id, worker_episode_index]",
                           evaluation_environment="[1, evaluation_seed]",
                           evaluation_policy="[3, evaluation_seed]")))
            records = []
        random.seed(train_config.seed)
        np.random.seed(train_config.seed)
        torch.manual_seed(train_config.seed)
        model = MAPPO(train_config)
        with InterceptionVectorEnv(env_config, train_config.workers, train_config.seed) as vector:
            if payload:
                restore(payload, model, env_config, vector)
            else:
                atomic_json(folder / "config.json", run)
                save(latest, model, env_config, vector, run=run)
            evaluations = evaluation_records(folder, run, model.updates)
            complete_evaluation(folder, run, model, env_config, evaluation, evaluations)
            write_views(folder, records)
            recent = [row for record in records for row in record["completed_episodes"]][-100:]
            previous = digest(records[-1]) if records else None
            planned = model.updates + updates
            try:
                for _ in range(updates):
                    start = time.perf_counter()
                    old_steps = model.steps
                    rollout, episodes = model.collect(vector)
                    sampled = time.perf_counter()
                    metrics = model.update(rollout)
                    optimized = time.perf_counter()
                    model.episodes += len(episodes)
                    record = serializable(dict(update=model.updates, joint_steps=model.steps,
                        completed_episodes_total=model.episodes, completed_episodes=episodes,
                        run_id=run["id"], previous=previous, planned_updates=planned, **metrics,
                        learning_rates=dict(actor=train_config.actor_lr, critic=train_config.critic_lr),
                        gradient_norm_basis="before clipping", sampling_time=sampled-start,
                        optimization_time=optimized-sampled, update_time=optimized-start,
                        sampling_speed=(model.steps-old_steps)/(sampled-start)))
                    save(latest, model, env_config, vector, record, run=run)
                    atomic_json(folder / f"update_{model.updates:06}.json", record)
                    write_views(folder, [record], append=True)
                    previous = digest(record)
                    recent = (recent + episodes)[-100:]
                    if model.updates % log_interval == 0 or model.updates == planned:
                        print(terminal_line(record, recent), flush=True)
                    if evaluation.evaluation_interval_updates and model.updates % evaluation.evaluation_interval_updates == 0:
                        complete_evaluation(folder, run, model, env_config, evaluation, evaluations)
            except (RuntimeError, FloatingPointError) as exc:
                write_json(folder / f"failure_{time.time_ns()}.json",
                           dict(error=str(exc), last_committed=str(latest),
                                simulation_failure="simulation_failure" in str(exc)))
                raise
    return latest


def reference_action(info, config, assignments, *, predictive=False):
    commands = np.zeros((MAX_PURSUERS, 3))
    for i, j in assignments:
        p, e = info["pursuers"][i], info["evaders"][j]
        desired = unit(e[:3]-p[:3])*config.pursuer_vmax
        if predictive:
            # Nominal target navigation from current state only. The simulator
            # still executes the configured target policy independently.
            target = unit(navigation_point(config)-e[:3])*config.evader_vmax*config.evader_speed_fraction
            target = execute_velocity(target, e[3:], config.evader_vmax, config.evader_tau, config)
            times = np.arange(config.predictive_step, config.predictive_horizon+1e-9, config.predictive_step)
            p_lag = -config.pursuer_tau*np.expm1(-times/config.pursuer_tau)
            e_lag = -config.evader_tau*np.expm1(-times/config.evader_tau)
            point = e[:3]+e_lag[:, None]*e[3:]+(times-e_lag)[:, None]*target
            drift = p[:3]+p_lag[:, None]*p[3:]
            gain = (times-p_lag)[:, None]
            request = (point-drift)/gain
            allowed = execute_velocity(request, p[3:], config.pursuer_vmax, config.pursuer_tau, config)
            miss = np.linalg.norm(drift+gain*allowed-point, axis=1)
            ready = ((miss <= config.pursuer_collision_radius+config.evader_collision_radius) |
                     (np.linalg.norm(point[:, :2]-config.center[:2], axis=1) <= config.radius))
            desired = request[np.flatnonzero(ready)[0] if ready.any() else -1]
        commands[i] = desired
    return .5 + .5 * commands / np.maximum(config.pursuer_vmax,
                                           np.linalg.norm(commands, axis=-1, keepdims=True))


def trace_row(info, action=None):
    p, e = info["pursuers"], info["evaders"]
    valid = info["active_mask"][:, None] & info["evader_mask"][None, :]
    return {key: info[key] for key in ("pursuers", "evaders", "active_mask", "evader_mask",
                                      "events", "step", "time", "outcome", "minimum_separation",
                                      "reward_parts", "task", "initialization", "minimum_center_distance",
                                      "minimum_surface_clearance", "reset_seed")} | dict(commands=info.get("commands"),
        physical_time=info["time"], normalized_actions=action,
        requested_velocities=info.get("requested_velocities"), initial_accelerations=info.get("initial_accelerations"),
        command_duration=info.get("command_duration"),
        pursuer_speed=np.where(info["active_mask"], np.linalg.norm(p[:, 3:], axis=1), None),
        evader_speed=np.where(info["evader_mask"], np.linalg.norm(e[:, 3:], axis=1), None),
        distances=np.where(valid, np.linalg.norm(p[:, None, :3]-e[None, :, :3], axis=-1), None),
        relative_speeds=np.where(valid, np.linalg.norm(p[:, None, 3:]-e[None, :, 3:], axis=-1), None))


def episode(config, seed, options, policy, model=None, trace=False, deterministic=True, *, audit=False):
    env = InterceptionEnv(config, diagnostics=audit)
    rng = np.random.default_rng([2, seed])
    obs, info = env.reset(np.random.SeedSequence([1, seed]), options)
    accumulator = EpisodeAccumulator()
    accumulator.reset(info)
    frames = [trace_row(info)] if trace else []
    closest, friendly, end_event, first_contact, inside_10m, close_passes = {}, None, None, None, set(), {}
    if audit:
        started = time.perf_counter()
        controller_s = 0.
        acceleration_integral = np.zeros(2)
        active_seconds = np.zeros(2)
        acceleration_peak = np.zeros(2)
        first_evasive_request = {}
        goal = navigation_point(config)
        taus = np.r_[np.full(MAX_PURSUERS, config.pursuer_tau),
                     np.full(config.evader_slots, config.evader_tau)]
    while True:
        if audit:
            previous = info
            policy_start = time.perf_counter()
        if policy == "random":
            action = rng.random((MAX_PURSUERS, 3))
        elif policy == "stationary":
            action = np.full((MAX_PURSUERS, 3), .5)
        elif policy in ("reference", "predictive", "paired_reference"):
            pursuers = list(np.flatnonzero(info["active_mask"]))
            assignments = []
            for j in np.flatnonzero(info["evader_mask"]):
                i = min(pursuers, key=lambda i: np.linalg.norm(info["pursuers"][i, :3]-info["evaders"][j, :3]))
                assignments.append((i, j))
                pursuers.remove(i)
            action = reference_action(info, config, assignments, predictive=policy in ("predictive", "paired_reference"))
        else:
            action = model.act(obs, info["active_mask"], deterministic=deterministic)[0].cpu().numpy()
        if audit:
            controller_s += time.perf_counter()-policy_start
        try:
            obs, rewards, terminated, truncated, info = env.step(action)
        except SimulationFailure as exc:
            result = accumulator.summary(info) | {"outcome": "simulation_failure", "error": str(exc)}
            break
        accumulator.add(info, rewards)
        if audit:
            durations = info["command_duration"]
            magnitude = np.linalg.norm(info["initial_accelerations"][:, :2], axis=1)
            integral = -taus*np.expm1(-durations/taus)*magnitude
            for side, slots in enumerate((slice(None, MAX_PURSUERS), slice(MAX_PURSUERS, None))):
                acceleration_integral[side] += integral[slots].sum()
                active_seconds[side] += durations[slots].sum()
                acceleration_peak[side] = max(acceleration_peak[side], magnitude[slots].max())
            if config.evader_mode == "reactive_evasion":
                for j in np.flatnonzero(previous["evader_mask"]):
                    if j not in first_evasive_request:
                        request = info["requested_velocities"][MAX_PURSUERS+j]
                        direction = unit(goal-previous["evaders"][j, :3])
                        if np.linalg.norm(unit(request)-direction) > 1e-7:
                            distance = np.linalg.norm(previous["pursuers"][previous["active_mask"], :3]
                                                      -previous["evaders"][j, :3], axis=1).min()
                            first_evasive_request[int(j)] = dict(time=previous["time"], distance=float(distance),
                                                                 request=request.tolist())
            for j, pair in info["closest_targets"].items():
                if j not in closest or pair["center_distance"] < closest[j]["center_distance"]:
                    closest[j] = pair
                if pair["center_distance"] <= 10 and j not in inside_10m:
                    close_passes[j] = close_passes.get(j, 0)+1
            inside_10m = {j for j, pair in info["closest_targets"].items() if pair["center_distance"] <= 10}
            step_friendly = info["closest_friendly"]
            if step_friendly is not None:
                friendly = step_friendly if friendly is None else min(friendly, step_friendly)
            for event in info["events"]:
                if event["kind"] == "interception_contact":
                    if first_contact is None:
                        first_contact = event["time"]
                if event["kind"] == info["outcome"]:
                    end_event = event
        if trace:
            frames.append(trace_row(info, action))
        if terminated or truncated:
            result = accumulator.summary(info)
            break
    if audit:
        result.update(closest_targets=list(closest.values()), minimum_friendly_distance=friendly,
                      first_contact_time=first_contact, terminal_event=end_event, close_passes=close_passes,
                      contact_events=[event for event in env.state.events if event["kind"] == "interception_contact"],
                      wall_time_s=time.perf_counter()-started, controller_s=controller_s,
                      first_evasive_request=first_evasive_request,
                      peak_horizontal_acceleration=acceleration_peak.tolist(),
                      mean_horizontal_acceleration=np.divide(acceleration_integral, active_seconds,
                          out=np.zeros(2), where=active_seconds > 0).tolist())
    return result | {"seed": seed}, frames


def evaluation_episode(config, seed, options, policy, model, trace, deterministic):
    if model is None:
        return episode(config, seed, options, policy, trace=trace)
    # Evaluation only draws from local environment generators and Torch.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(int(np.random.SeedSequence([3, seed]).generate_state(1)[0]))
        return episode(config, seed, options, policy, model, trace, deterministic)


def complete_evaluation(folder, run, model, config, protocol, rows):
    update = model.updates
    interval = protocol.evaluation_interval_updates
    latest = folder / "latest.pt"
    if interval and update and update % interval == 0:
        source_hash = file_digest(latest)
        if rows and rows[-1]["update"] == update:
            if rows[-1]["checkpoint_sha256"] != source_hash:
                raise ValueError("evaluation commit differs from latest checkpoint")
        else:
            options = dict(scenario=protocol.evaluation_scenario)
            runs = [evaluation_episode(config, seed, options, "checkpoint", model, False,
                                       protocol.evaluation_deterministic)[0] for seed in protocol.seeds]
            result = dict(run_id=run["id"], update=update, joint_steps=model.steps, seeds=protocol.seeds,
                          protocol=asdict(protocol), summary=summarize(runs), runs=runs,
                          checkpoint_source=f"latest.pt at update {update}", checkpoint_sha256=source_hash)
            atomic_json(folder / f"evaluation_{update:06}.json", result)
            rows.append(result)
    publish_best(folder, rows, update)
    evaluation_csv(folder, rows)


def evaluate(folder, config, options, policy, seeds, model=None, *, identity=None,
             compressed=False, deterministic=True):
    import gzip
    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, seed in enumerate(seeds):
        row, frames = evaluation_episode(config, seed, options, policy, model, True, deterministic)
        rows.append(row)
        data = dict(version=VERSION, trace_version=2, time_unit="s", config=asdict(config), task=config.task,
                    seed=seed, policy=policy, scenario=options, result=row, frames=frames,
                    deterministic=deterministic if model is not None else None, model=identity,
                    entity_ids=dict(pursuers=list(range(MAX_PURSUERS)),
                                    evaders=list(range(MAX_PURSUERS, MAX_PURSUERS+config.evader_slots))))
        path = folder / f"trajectory_{index:03}.json"
        if compressed:
            with gzip.open(str(path) + ".gz", "xt", encoding="utf-8") as stream:
                json.dump(serializable(data), stream, allow_nan=False, separators=(",", ":"))
        else:
            write_json(path, data)
    summary = summarize(rows)
    write_json(folder / "summary.json", summary | dict(
        version=VERSION, environment=asdict(config), runs=rows, seeds=list(seeds),
        teams={"pursuers": "red defenders", "evaders": "blue intruders"},
        scenario=options, model=identity, policy=policy,
        deterministic=deterministic if model is not None else None))
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)

    def environment_flags(command, scales=SCENARIO_SCALES):
        command.add_argument("--env", type=Path, help="JSON overrides of InterceptionConfig")
        command.add_argument("--scale", choices=scales)
        command.add_argument("--task", choices=[task for task, (scale, _) in TASKS.items() if scale in scales])

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
        command.add_argument("--gzip", action="store_true", help="compress trajectory JSON")
    run.add_argument("--policy", choices=("reference", "predictive", "stationary", "random"), default="reference")
    assess.add_argument("checkpoint", type=Path)
    assess.add_argument("--stochastic", action="store_true")
    fit = sub.add_parser("train")
    environment_flags(fit, SCALES)
    fit.add_argument("--training", type=Path, help="JSON overrides of TrainConfig")
    fit.add_argument("--updates", type=int, default=1)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--evaluation-interval-updates", type=int, default=0)
    fit.add_argument("--evaluation-episodes", type=int, default=5)
    fit.add_argument("--evaluation-seed-base", type=int, default=1000)
    fit.add_argument("--evaluation-scenario", choices=("fixed", "random"), default="random")
    fit.add_argument("--evaluation-deterministic", action=argparse.BooleanOptionalAction, default=True)
    continuation = sub.add_parser("resume")
    continuation.add_argument("--out", type=Path, required=True)
    continuation.add_argument("--updates", type=int, default=1)
    for command in (fit, continuation):
        command.add_argument("--log-interval", type=int, default=1)
    args = parser.parse_args(argv)
    if args.operation == "resume":
        train(args.out, updates=args.updates, resume=True, log_interval=args.log_interval)
        return
    if args.operation == "evaluate":
        source_hash = file_digest(args.checkpoint)
        payload = read(args.checkpoint)
        if file_digest(args.checkpoint) != source_hash:
            raise ValueError("checkpoint changed during load")
        config = InterceptionConfig(**payload["environment"])
        model = MAPPO(TrainConfig(**payload["training"]))
        load_weights(payload, model)
        identity = dict(path=str(args.checkpoint.resolve()), sha256=source_hash,
                        update=payload["updates"], joint_steps=payload["steps"])
    else:
        overrides = json.loads(args.env.read_text()) if args.env else {}
        if args.scale:
            overrides["scale"] = args.scale
        if args.task:
            overrides["task"] = args.task
        if "task" in overrides and "scale" not in overrides:
            overrides["scale"] = TASKS[overrides["task"]][0]
        config = InterceptionConfig(**overrides)
    if args.operation == "train":
        settings = json.loads(args.training.read_text()) if args.training else {}
        protocol = EvaluationProtocol(**{name: getattr(args, name) for name in
                                        EvaluationProtocol.__dataclass_fields__})
        train(args.out, config, TrainConfig(**settings), args.updates,
              evaluation=protocol, log_interval=args.log_interval)
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
                 seeds, None if args.operation == "run" else model,
                 identity=None if args.operation == "run" else identity,
                 compressed=args.gzip, deterministic=not getattr(args, "stochastic", False))


if __name__ == "__main__":
    main()
