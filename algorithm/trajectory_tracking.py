"""CUDA training and evaluation for the independent preset tracking protocol."""
import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import random
import time
from uuid import uuid4

import numpy as np
import torch

from algorithm.common.interception_checkpoint import load_weights, read, restore, save
from algorithm.common import terminal
from algorithm.common.research import atomic_json, commit_header, digest, journal, read_json
from algorithm.common.runtime import require_cuda
from algorithm.common.serialization import serializable
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.interception import run_lock
from algorithm.mappo.interception import MAPPO, TrainConfig
from algorithm.trajectory_tracking_metrics import TrackingAccumulator, summarize
from env.interception.config import MAX_PURSUERS
from env.preset_trajectory import MODES
from env.trajectory_rl import OBS_DIM, STATE_DIM, SLOT, VERSION, TrackingConfig, TrackingEnv, TrackingNumericalFailure


PROTOCOL = dict(version=VERSION,algorithm="mappo-beta",observation=OBS_DIM,state=STATE_DIM,
                action=3,slots=MAX_PURSUERS,slot_width=SLOT)
EXPERIMENT = "Preset tracking"


class TrackingMAPPO(MAPPO):
    observation_dim = OBS_DIM
    state_dim = STATE_DIM

    @staticmethod
    def clean_states(states):
        slots = states[...,:-1].reshape(*states.shape[:-1],MAX_PURSUERS,SLOT)
        masks = slots[...,-1:]
        if not ((masks==0)|(masks==1)).all():
            raise ValueError("invalid tracking critic masks")
        return torch.cat((torch.where(masks.bool(),slots,0.).flatten(-2),states[...,-1:]),dim=-1)


def make_vector(config, training):
    return ParallelVectorEnv(config,training.workers,training.seed,environment_factory=TrackingEnv,
                             accumulator_factory=TrackingAccumulator)


def evaluate(model, config, seeds):
    require_cuda(str(model.device))
    rows, trace = [], []
    for index,seed in enumerate(seeds):
        env = TrackingEnv(config)
        obs,info = env.reset(np.random.SeedSequence([1,seed]))
        tracker = TrackingAccumulator()
        tracker.reset(info)
        frames = []
        if index==0:
            frames.append({k:info[k] for k in ("time","actual_states","reference","active_mask")})
        while True:
            action = model.act(obs,info["active_mask"],deterministic=True)[0].cpu().numpy()
            try:
                obs,reward,term,trunc,info = env.step(action)
            except TrackingNumericalFailure as exc:
                info = info | dict(outcome="numerical_failure")
                rows.append(tracker.summary(info) | dict(seed=seed,error=str(exc)))
                break
            tracker.add(info,reward)
            if index==0:
                frames.append({k:info[k] for k in ("time","actual_states","reference","active_mask")})
            if term or trunc:
                rows.append(tracker.summary(info) | dict(seed=seed))
                break
        if index==0:
            trace = frames
    return serializable(dict(summary=summarize(rows),episodes=rows,first_trace=trace))


def _publish_evaluation(folder, update, evaluation):
    path = folder/f"evaluation_{update:06}.json"
    if path.exists():
        if read_json(path)!=evaluation:
            raise ValueError("evaluation differs from the checkpoint commit")
    else:
        atomic_json(path,evaluation)


def _metric_view(folder, records):
    # The journal is authoritative; rebuild a view after an interrupted append.
    with (folder/"training_metrics.jsonl").open("w",encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps({k:v for k,v in row.items() if k not in ("completed_episodes","evaluation")})+"\n")


def fit(folder, config=None, training=None, *, total_steps=None, until_steps=None, interval=16,
        evaluation_seeds=None, device=None, resume=False, expected_mode=None):
    folder = Path(folder)
    if not resume:
        if total_steps is None or interval<1 or not evaluation_seeds or len(set(evaluation_seeds))!=len(evaluation_seeds):
            raise ValueError("positive budget, interval and unique evaluation seeds are required")
        run = serializable(dict(id=uuid4().hex,protocol=PROTOCOL,environment=asdict(config),
            training=asdict(training),total_steps=total_steps,interval=interval,
            evaluation_seeds=list(evaluation_seeds),device=str(require_cuda(device or "cuda"))))
        wave = training.workers*training.horizon
        target = total_steps if until_steps is None else until_steps
        if not 0<target<=total_steps or target%wave or total_steps%wave:
            raise ValueError("budgets must be multiples of workers*horizon")
        folder.mkdir(parents=True,exist_ok=False)
    path = folder/"latest.pt"
    with run_lock(folder):
        payload = read(path,protocol=PROTOCOL) if resume else None
        if resume:
            head = commit_header(path)
            run = payload["research_run"]
            if (head["run"]!=run or run!=read_json(folder/"config.json") or
                    head["update"]!=payload["updates"] or head["last_log"]!=payload["last_log"] or
                    serializable(payload["environment"])!=run["environment"] or payload["training"]!=run["training"]):
                raise ValueError("checkpoint and run identities differ")
            config,training = TrackingConfig(**run["environment"]),TrainConfig(**run["training"])
            if expected_mode is not None and config.trajectory_mode!=expected_mode:
                raise ValueError("checkpoint trajectory mode mismatch")
            if device is not None and str(require_cuda(device))!=run["device"]:
                raise ValueError("exact resume requires the original CUDA device selection")
            total_steps,interval,evaluation_seeds = run["total_steps"],run["interval"],run["evaluation_seeds"]
            records = journal(folder,head,repair=True)
            wave = training.workers*training.horizon
            if payload["steps"]!=len(records)*wave or payload["episodes"]!=sum(
                    len(r["completed_episodes"]) for r in records):
                raise ValueError("checkpoint counters differ from the journal")
        else:
            records = []
        wave = training.workers*training.horizon
        target = total_steps if until_steps is None else until_steps
        if not 0<target<=total_steps or target%wave or total_steps%wave or target<len(records)*wave:
            raise ValueError("budgets must be multiples of workers*horizon and respect the saved run")
        random.seed(training.seed)
        np.random.seed(training.seed)
        torch.manual_seed(training.seed)
        model = TrackingMAPPO(training,device=run["device"])
        with make_vector(config,training) as vector:
            evaluations = []
            if resume:
                restore(payload,model,config,vector,protocol=PROTOCOL)
                for row in records:
                    if row["evaluation"] is not None:
                        _publish_evaluation(folder,row["update"],row["evaluation"])
                        evaluations.append(dict(update=row["update"],summary=row["evaluation"]["summary"]))
                if evaluations!=payload["evaluations"]:
                    raise ValueError("checkpoint evaluation index differs from the journal")
            else:
                atomic_json(folder/"config.json",run)
                save(path,model,config,vector,run=run,protocol=PROTOCOL,evaluations=evaluations)
            _metric_view(folder,records)
            cumulative = sum(r["valid_samples"] for r in records)
            terminal.training_start(EXPERIMENT, terminal.fields(experiment="tracking",
                mode=config.trajectory_mode, uavs=len(config.starts)), training,
                start_update=model.updates, planned_updates=total_steps//wave, folder=folder, resume=resume,
                stop_steps=target, evaluation=terminal.evaluation_setup(interval, evaluation_seeds,
                    scenario="seeded references", actions="Beta mean", final=bool((total_steps//wave) % interval)))
            while model.steps<target:
                try:
                    rollout, episodes = model.collect(vector)
                    metrics = model.update(rollout)
                except (RuntimeError,FloatingPointError) as exc:
                    atomic_json(folder/f"failure_{time.time_ns()}.json",dict(error=str(exc),last_committed=str(path)))
                    raise
                model.episodes += len(episodes)
                cumulative += metrics["valid_samples"]
                result = None
                if model.updates%interval==0 or model.steps==total_steps:
                    result = evaluate(model,config,evaluation_seeds)
                    evaluations.append(dict(update=model.updates,summary=result["summary"]))
                row = serializable(dict(update=model.updates,joint_steps=model.steps,
                    planned_updates=total_steps//wave,run_id=run["id"],
                    previous=digest(records[-1]) if records else None,
                    completed_episodes_total=model.episodes,completed_episodes=episodes,
                    cumulative_valid_samples=cumulative,training=summarize(episodes),evaluation=result,**metrics))
                save(path,model,config,vector,row,run=run,protocol=PROTOCOL,evaluations=evaluations)
                atomic_json(folder/f"update_{model.updates:06}.json",row)
                if result is not None:
                    _publish_evaluation(folder,model.updates,result)
                records.append(row)
                with (folder/"training_metrics.jsonl").open("a",encoding="utf-8") as f:
                    f.write(json.dumps({k:v for k,v in row.items() if k not in ("completed_episodes","evaluation")})+"\n")
                terminal.training_progress(EXPERIMENT, row, total_steps=total_steps)
                if result is not None:
                    terminal.emit(EXPERIMENT, "EVAL", terminal.fields(upd=model.updates),
                                  terminal.tracking_results(result["summary"]))
        terminal.training_end(EXPERIMENT, model, path, paused=target < total_steps)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="operation",required=True)
    train,resume,assess = [sub.add_parser(name) for name in ("train","resume","evaluate")]
    for cmd in (train,resume,assess):
        cmd.add_argument("--out",required=True,type=Path)
        cmd.add_argument("--device")
    for cmd in (train,resume):
        cmd.add_argument("--mode",choices=MODES,required=cmd is train)
        cmd.add_argument("--until-steps",type=int)
    train.add_argument("--total-steps",type=int,required=True)
    train.add_argument("--env",type=Path)
    train.add_argument("--eval-interval",type=int,default=16)
    for field in fields(TrainConfig):
        train.add_argument("--"+field.name.replace("_","-"),type=field.type,default=field.default)
    assess.add_argument("checkpoint",type=Path)
    for cmd in (train,assess):
        cmd.add_argument("--eval-seed",type=int,default=2200)
        cmd.add_argument("--eval-episodes",type=int,default=10)
    a = p.parse_args(argv)
    if a.operation=="resume":
        fit(a.out,until_steps=a.until_steps,device=a.device,resume=True,expected_mode=a.mode)
        return
    if a.eval_seed<0 or a.eval_episodes<1:
        p.error("invalid evaluation seed or episode count")
    seeds = list(range(a.eval_seed,a.eval_seed+a.eval_episodes))
    if a.operation=="train":
        env = json.loads(a.env.read_text(encoding="utf-8")) if a.env else {}
        if env.get("trajectory_mode",a.mode)!=a.mode:
            p.error("mode differs from environment configuration")
        fit(a.out,TrackingConfig(**(env|{"trajectory_mode":a.mode})),
            TrainConfig(**{f.name:getattr(a,f.name) for f in fields(TrainConfig)}),
            total_steps=a.total_steps,until_steps=a.until_steps,interval=a.eval_interval,
            evaluation_seeds=seeds,device=a.device)
    else:
        payload = read(a.checkpoint,protocol=PROTOCOL)
        model = TrackingMAPPO(TrainConfig(**payload["training"]),device=a.device or "cuda")
        load_weights(payload,model)
        config = TrackingConfig(**payload["environment"])
        a.out.mkdir(parents=True,exist_ok=False)
        result = evaluate(model,config,seeds)
        atomic_json(a.out/"evaluation.json",result |
                    dict(mode=config.trajectory_mode,checkpoint=str(a.checkpoint),joint_steps=payload["steps"]))
        terminal.emit(EXPERIMENT, "EVAL", terminal.fields(experiment="tracking",
                      mode=config.trajectory_mode, actions="Beta mean", upd=payload["updates"],
                      seeds=terminal.seed_span(seeds)),
                      terminal.tracking_results(result["summary"]), terminal.fields(out=a.out))


if __name__=="__main__":
    from algorithm.trajectory_tracking import main as canonical_main
    terminal.run_cli(EXPERIMENT, canonical_main)
