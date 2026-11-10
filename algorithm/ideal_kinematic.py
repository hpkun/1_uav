"""Independent CUDA MAPPO entry point for ideal point-mass tasks."""
import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import random
import time
from uuid import uuid4

import numpy as np
import torch

from algorithm.common import terminal
from algorithm.common.interception_checkpoint import load_weights, read, restore, save
from algorithm.common.research import atomic_json
from algorithm.common.runtime import require_cuda
from algorithm.common.serialization import serializable
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.interception import run_lock
from algorithm.mappo.interception import MAPPO, TrainConfig
from env.ideal_kinematic import IdealConfig, IdealEnv
from env.ideal_kinematic.config import PROTOCOL as VERSION, OBS_DIM, STATE_DIM, SLOTS

PROTOCOL = dict(version=VERSION, algorithm="mappo-beta", observation=OBS_DIM,
                state=STATE_DIM, action=3, slots=SLOTS, action_semantics="desired_motion_direction")


def config_from_saved(values):
    restored = dict(values)
    for key in ("lower", "upper", "speeds", "starts", "directions", "waypoints",
                "target_starts", "target_ends"):
        restored[key] = tuple(tuple(item) if isinstance(item, list) else item
                              for item in restored[key])
    return IdealConfig(**restored)


class IdealMAPPO(MAPPO):
    observation_dim = OBS_DIM
    state_dim = STATE_DIM

    @staticmethod
    def clean_states(states):
        slots = states[..., :SLOTS*8].reshape(*states.shape[:-1], SLOTS, 8)
        masks = slots[..., 6:7]
        if not ((masks == 0) | (masks == 1)).all():
            raise ValueError("invalid ideal critic masks")
        return torch.cat((torch.where(masks.bool(), slots, 0.).flatten(-2),
                          states[..., SLOTS*8:]), dim=-1)


class IdealAccumulator:
    def reset(self, info):
        self.seed = info["seed"]
        self.initial_states = info["actual_states"].tolist()
        self.total = 0.
        self.parts = {}
        self.events = []

    def add(self, info, rewards):
        self.total += info["team_reward"]
        self.events.extend(info["events"])
        for key, value in info["reward_parts"].items():
            self.parts[key] = self.parts.get(key, 0.) + value

    def summary(self, info):
        return dict(task=info["task"], outcome=info["outcome"], time=info["time"],
                    steps=info["step"], seed=self.seed, initial_states=self.initial_states,
                    final_states=info["actual_states"].tolist(), status=info["status"],
                    assignment=info["assignment"], episode_return=self.total,
                    reward_parts=self.parts.copy(), events=self.events.copy(),
                    minimum_separation=info["minimum_separation"],
                    warning_count=sum(e["kind"] == "warning" for e in self.events),
                    contact_count=sum(e["kind"] in ("contact", "target_contact") for e in self.events))


def summarize(rows):
    n = len(rows)
    return dict(episodes=n, success_rate=sum(r["outcome"] == "success" for r in rows)/n if n else None,
                boundary_rate=sum(r["outcome"] == "boundary" for r in rows)/n if n else None,
                mean_return=sum(r["episode_return"] for r in rows)/n if n else None,
                mean_warning_count=sum(r["warning_count"] for r in rows)/n if n else None)


def summary_fields(summary):
    return terminal.fields(ep=summary["episodes"], success=terminal.percent(summary["success_rate"]),
                           boundary=terminal.percent(summary["boundary_rate"]),
                           reward=terminal.number(summary["mean_return"]),
                           warnings=terminal.number(summary["mean_warning_count"]))


def vector_env(config, training):
    return ParallelVectorEnv(config, training.workers, training.seed,
                             environment_factory=IdealEnv, accumulator_factory=IdealAccumulator)


def evaluate(model, config, seeds, *, deterministic=True, trace=True):
    require_cuda(str(model.device))
    rows, first_trace = [], []
    for index, seed in enumerate(seeds):
        if not deterministic:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        env = IdealEnv(config)
        obs, info = env.reset(np.random.SeedSequence([1, seed]))
        accumulator = IdealAccumulator()
        accumulator.reset(info)
        frames = [serializable(dict(time=info["time"], states=info["actual_states"],
                                    reference=info["reference"], active_mask=info["active_mask"],
                                    events=[]))] if trace and index == 0 else []
        while True:
            action = model.act(obs, info["active_mask"], deterministic=deterministic)[0].cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            accumulator.add(info, reward)
            if trace and index == 0:
                frames.append(serializable(dict(time=info["time"], states=info["actual_states"],
                    reference=info["reference"], active_mask=info["active_mask"], events=info["events"])))
            if terminated or truncated:
                rows.append(accumulator.summary(info))
                break
        if index == 0:
            first_trace = frames
    return serializable(dict(summary=summarize(rows), episodes=rows, first_trace=first_trace,
                             action_semantics="desired_motion_direction"))


def fit(folder, config=None, training=None, *, total_steps=None, until_steps=None,
        eval_interval=16, evaluation_seeds=None, device=None, resume=False):
    require_cuda(device or "cuda")
    folder = Path(folder)
    latest = folder / "latest.pt"
    if not resume:
        if config is None or training is None or not evaluation_seeds or eval_interval < 1:
            raise ValueError("configuration, training and evaluation protocol required")
        wave = training.workers * training.horizon
        target = total_steps if until_steps is None else until_steps
        if total_steps is None or target is None or not 0 < target <= total_steps or target % wave or total_steps % wave:
            raise ValueError("budget must align with workers*horizon")
        folder.mkdir(parents=True, exist_ok=False)
        run = serializable(dict(id=uuid4().hex, protocol=PROTOCOL, environment=asdict(config),
                    training=asdict(training), total_steps=total_steps,
                    eval_interval=eval_interval, evaluation_seeds=list(evaluation_seeds),
                    device=str(require_cuda(device or "cuda"))))
    with run_lock(folder):
        payload = read(latest, protocol=PROTOCOL) if resume else None
        if resume:
            run = payload["research_run"]
            if run["protocol"] != PROTOCOL or run != json.loads((folder/"config.json").read_text()):
                raise ValueError("ideal run identity mismatch")
            config, training = config_from_saved(run["environment"]), TrainConfig(**run["training"])
            total_steps, eval_interval = run["total_steps"], run["eval_interval"]
            evaluation_seeds = run["evaluation_seeds"]
            if device is not None and str(require_cuda(device)) != run["device"]:
                raise ValueError("resume requires original CUDA device")
        wave = training.workers*training.horizon
        target = total_steps if until_steps is None else until_steps
        if not 0 < target <= total_steps or target % wave or target < (payload["steps"] if payload else 0):
            raise ValueError("invalid resume target")
        random.seed(training.seed)
        np.random.seed(training.seed)
        torch.manual_seed(training.seed)
        model = IdealMAPPO(training, device=run["device"])
        with vector_env(config, training) as vector:
            if payload:
                restore(payload, model, config, vector, protocol=PROTOCOL)
            else:
                atomic_json(folder/"config.json", run)
                save(latest, model, config, vector, run=run, protocol=PROTOCOL, evaluations=[])
            evaluations = payload["evaluations"].copy() if payload else []
            recent = []
            if payload and payload["last_log"]:
                for path in sorted(folder.glob("update_*.json")):
                    row = json.loads(path.read_text())
                    if row["update"] <= model.updates:
                        recent.extend(row["completed_episodes"])
                recent = recent[-100:]
            terminal.training_start("Ideal", terminal.fields(experiment="ideal", task=config.task,
                initialization=config.initialization, target_mode=config.target_mode),
                training, start_update=model.updates, planned_updates=total_steps//wave,
                evaluation=terminal.evaluation_setup(eval_interval, evaluation_seeds,
                    scenario=config.initialization, actions="Beta mean"),
                folder=folder, resume=resume, stop_steps=target)
            while model.steps < target:
                sampled_at = time.perf_counter()
                try:
                    rollout, episodes = model.collect(vector)
                    sampled = time.perf_counter()
                    metrics = model.update(rollout)
                except (RuntimeError, FloatingPointError) as exc:
                    atomic_json(folder/"failure.json", dict(error=str(exc),
                                last_committed=str(latest), update=model.updates))
                    raise
                model.episodes += len(episodes)
                recent = (recent + episodes)[-100:]
                result = None
                if model.updates % eval_interval == 0 or model.steps == total_steps:
                    result = evaluate(model, config, evaluation_seeds)
                    evaluations.append(dict(update=model.updates, summary=result["summary"]))
                record = serializable(dict(update=model.updates, planned_updates=total_steps//wave,
                    joint_steps=model.steps, completed_episodes_total=model.episodes,
                    completed_episodes=episodes, sampling_speed=wave/(sampled-sampled_at), **metrics))
                save(latest, model, config, vector, record, run=run, protocol=PROTOCOL,
                     evaluations=evaluations)
                atomic_json(folder/f"update_{model.updates:06}.json", record)
                with (folder/"training_metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({k:v for k,v in record.items() if k != "completed_episodes"})+"\n")
                terminal.training_progress("Ideal", record, total_steps=total_steps,
                    window=terminal.fields(win=f"{len(recent)}/100",
                        success=terminal.percent(summarize(recent)["success_rate"]),
                        reward=terminal.number(summarize(recent)["mean_return"])))
                if result is not None:
                    atomic_json(folder/f"evaluation_{model.updates:06}.json", result)
                    terminal.emit("Ideal", "EVAL", terminal.fields(upd=model.updates),
                                  summary_fields(result["summary"]))
        terminal.training_end("Ideal", model, latest, paused=target < total_steps)
    return latest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    train, resume, assess = (commands.add_parser(name) for name in ("train", "resume", "evaluate"))
    for command in (train, resume, assess):
        command.add_argument("--out", required=True, type=Path)
        command.add_argument("--device")
    train.add_argument("--task", choices=("IK0", "IK1", "IK2", "IK3"), required=True)
    train.add_argument("--config", type=Path)
    train.add_argument("--total-steps", type=int, required=True)
    train.add_argument("--until-steps", type=int)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--horizon", type=int, default=128)
    train.add_argument("--eval-interval", type=int, default=16)
    train.add_argument("--eval-seed", type=int, default=2200)
    train.add_argument("--eval-episodes", type=int, default=10)
    resume.add_argument("--until-steps", type=int)
    assess.add_argument("checkpoint", type=Path)
    assess.add_argument("--eval-seed", type=int, default=2200)
    assess.add_argument("--eval-episodes", type=int, default=10)
    assess.add_argument("--sample-actions", action="store_true")
    args = parser.parse_args(argv)
    if args.operation == "train":
        options = json.loads(args.config.read_text()) if args.config else {}
        if options.get("task", args.task) != args.task:
            parser.error("task differs from configuration")
        config = IdealConfig(**(options | dict(task=args.task)))
        training = TrainConfig(seed=args.seed, workers=args.workers, horizon=args.horizon)
        fit(args.out, config, training, total_steps=args.total_steps,
            until_steps=args.until_steps, eval_interval=args.eval_interval,
            evaluation_seeds=list(range(args.eval_seed, args.eval_seed+args.eval_episodes)),
            device=args.device)
    elif args.operation == "resume":
        fit(args.out, until_steps=args.until_steps, device=args.device, resume=True)
    else:
        payload = read(args.checkpoint, protocol=PROTOCOL)
        config = config_from_saved(payload["environment"])
        model = IdealMAPPO(TrainConfig(**payload["training"]), device=args.device or "cuda")
        load_weights(payload, model)
        seeds = list(range(args.eval_seed, args.eval_seed+args.eval_episodes))
        args.out.mkdir(parents=True, exist_ok=False)
        result = evaluate(model, config, seeds, deterministic=not args.sample_actions)
        atomic_json(args.out/"evaluation.json", result | dict(protocol=PROTOCOL,
                    checkpoint=str(args.checkpoint), update=payload["updates"], seeds=seeds))
        terminal.emit("Ideal", "EVAL", terminal.fields(experiment="ideal", task=config.task,
            upd=payload["updates"]), summary_fields(result["summary"]), terminal.fields(out=args.out))


if __name__ == "__main__":
    from algorithm.ideal_kinematic import main as canonical_main
    terminal.run_cli("Ideal", canonical_main)
