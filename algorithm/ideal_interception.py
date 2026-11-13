"""CUDA MAPPO lifecycle for abstract geometric interception."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
from uuid import uuid4
import numpy as np
import torch

from algorithm.common import terminal
from algorithm.common.interception_checkpoint import load_weights, read, restore, save
from algorithm.common.research import atomic_json, digest
from algorithm.common.runtime import require_cuda
from algorithm.common.serialization import serializable
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.interception import run_lock
from algorithm.mappo.interception import MAPPO, TrainConfig
from env.ideal_interception import InterceptionConfig, InterceptionEnv
from env.ideal_interception.config import PROTOCOL as VERSION, OBS_DIM, STATE_DIM

PROTOCOL = dict(version=VERSION,algorithm="mappo-beta",observation=OBS_DIM,
                state=STATE_DIM,action=3,defenders=5,intruders=2,
                action_semantics="defender_desired_motion_direction")


class InterceptionMAPPO(MAPPO):
    observation_dim = OBS_DIM
    state_dim = STATE_DIM

    @staticmethod
    def clean_states(states):
        defenders = states[...,:35].reshape(*states.shape[:-1],5,7)
        intruders = states[...,35:49].reshape(*states.shape[:-1],2,7)
        assignment = states[...,49:59].reshape(*states.shape[:-1],5,2)
        dm,im = defenders[...,-1:],intruders[...,-1:]
        if not (((dm==0)|(dm==1)).all() and ((im==0)|(im==1)).all()):
            raise ValueError("invalid bilateral critic masks")
        return torch.cat((torch.where(dm.bool(),defenders,0.).flatten(-2),
            torch.where(im.bool(),intruders,0.).flatten(-2),
            torch.where(dm.bool(),assignment,0.).flatten(-2),states[...,59:]),dim=-1)


class EpisodeAccumulator:
    def reset(self,info):
        self.initial_defenders = info["defenders"].tolist()
        self.initial_intruders = info["intruders"].tolist()
        self.intruder_paths = info["intruder_paths"]
        self.scene_diagnostics = info["scene_diagnostics"]
        self.scenario_id,self.seed = info["scenario_id"],info["seed"]
        self.total = 0.
        self.parts = {}
        self.events = []

    def add(self,info,rewards):
        self.total += info["team_reward"]
        self.events.extend(info["events"])
        for key,value in info["reward_parts"].items():
            self.parts[key] = self.parts.get(key,0.)+value

    def summary(self,info):
        return dict(scale=info["scale"],outcome=info["outcome"],time=info["time"],
            steps=info["step"],seed=self.seed,scenario_id=self.scenario_id,
            initial_defenders=self.initial_defenders,initial_intruders=self.initial_intruders,
            intruder_paths=self.intruder_paths,
            scene_diagnostics=self.scene_diagnostics,
            final_defenders=info["defenders"].tolist(),final_intruders=info["intruders"].tolist(),
            defender_status=info["defender_status"],intruder_status=info["intruder_status"],
            assignment=info["assignment"],episode_return=self.total,
            reward_parts=self.parts.copy(),events=self.events.copy(),
            interception_count=info["interception_count"],breach_count=info["breach_count"],
            friendly_collision_count=info["friendly_collision_count"],
            intruder_collision_count=info["intruder_collision_count"],
            minimum_separation=info["minimum_separation"],
            defender_exits=sum(s not in ("active","unused") for s in info["defender_status"]),
            intruder_exits=sum(s not in ("active","unused") for s in info["intruder_status"]))


def summarize(rows):
    n = len(rows)
    def average(key):
        values = [row[key] for row in rows if row[key] is not None]
        return sum(values)/len(values) if values else None
    return dict(episodes=n,
        all_intercepted_rate=sum(r["outcome"]=="all_intercepted" for r in rows)/n if n else None,
        breach_rate=sum(r["outcome"]=="breach" for r in rows)/n if n else None,
        friendly_collision_rate=sum(r["outcome"]=="friendly_collision" for r in rows)/n if n else None,
        mean_return=average("episode_return"),mean_interceptions=average("interception_count"),
        mean_time=average("time"),
        mean_minimum_separation=average("minimum_separation"),
        mean_defender_exits=average("defender_exits"),mean_intruder_exits=average("intruder_exits"))


def results(summary):
    return terminal.fields(ep=summary["episodes"],
        intercepted=terminal.percent(summary["all_intercepted_rate"]),
        breach=terminal.percent(summary["breach_rate"]),
        friendly=terminal.percent(summary["friendly_collision_rate"]),
        reward=terminal.number(summary["mean_return"]),
        contacts=terminal.number(summary["mean_interceptions"]))


def model_digest(actor,critic):
    sha = hashlib.sha256()
    for label,state in (("actor",actor),("critic",critic)):
        for key,tensor in sorted(state.items()):
            value = tensor.detach().cpu().contiguous().numpy()
            sha.update((label+"/"+key+"/"+str(value.dtype)+"/"+str(value.shape)).encode())
            sha.update(value.tobytes())
    return sha.hexdigest()


def vector_env(config,training):
    return ParallelVectorEnv(config,training.workers,training.seed,
                             environment_factory=InterceptionEnv,
                             accumulator_factory=EpisodeAccumulator)


def evaluation_protocol(config,rows):
    unique = len({row["scenario_id"] for row in rows})
    return dict(kind="fixed_case" if config.initialization=="fixed" else "random_initial_positions",
        episodes=len(rows),unique_scenarios=unique,duplicate_scenarios=len(rows)-unique,
        randomized=[] if config.initialization=="fixed" else
            ["defender_initial_positions","intruder_initial_positions"],
        fixed=["intruder_endpoints","fixed_speeds","protected_region","task_assignment"],
        interpretation="geometric contact is an abstract event, not physical damage")


def evaluate(model,config,seeds,*,deterministic=True):
    require_cuda(str(model.device))
    rows,first_trace = [],[]
    for index,seed in enumerate(seeds):
        if not deterministic:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        env = InterceptionEnv(config)
        obs,info = env.reset(np.random.SeedSequence([1,seed]))
        tracker = EpisodeAccumulator()
        tracker.reset(info)
        frames = [serializable(dict(time=info["time"],step=info["step"],outcome=info["outcome"],defenders=info["defenders"],
            intruders=info["intruders"],defender_mask=info["defender_mask"],
            intruder_mask=info["intruder_mask"],events=[]))] if index==0 else []
        while True:
            action = model.act(obs,info["active_mask"],deterministic=deterministic)[0].cpu().numpy()
            obs,reward,terminated,truncated,info = env.step(action)
            tracker.add(info,reward)
            if index==0:
                frames.append(serializable(dict(time=info["time"],step=info["step"],outcome=info["outcome"],defenders=info["defenders"],
                    intruders=info["intruders"],defender_mask=info["defender_mask"],
                    intruder_mask=info["intruder_mask"],events=info["events"],
                    executed_velocities=info["executed_velocities"])))
            if terminated or truncated:
                rows.append(tracker.summary(info))
                break
        if index==0:
            first_trace = frames
    return serializable(dict(protocol=PROTOCOL,environment=asdict(config),
        model_sha256=model_digest(model.actor.state_dict(),model.critic.state_dict()),
        evaluation_protocol=evaluation_protocol(config,rows),
        summary=summarize(rows),episodes=rows,first_trace=first_trace))


def _recover(folder,payload):
    committed = payload["updates"]
    if any(int(path.stem.split("_")[-1])>committed for path in folder.glob("update_*.json")):
        raise ValueError("uncommitted update record ahead of checkpoint")
    rows = []
    for update in range(1,committed+1):
        path = folder/f"update_{update:06}.json"
        expected = payload["last_log"] if update==committed else None
        if not path.exists():
            if expected is None:
                raise ValueError("missing historical update record")
            atomic_json(path,expected)
        row = json.loads(path.read_text())
        if (row["update"]!=update or (expected is not None and
                row!=json.loads(json.dumps(expected,allow_nan=False))) or
                row.get("previous")!=(digest(rows[-1]) if rows else None)):
            raise ValueError("update record differs from checkpoint chain")
        rows.append(row)
    wave = payload["training"]["workers"]*payload["training"]["horizon"]
    if payload["steps"]!=committed*wave or payload["episodes"]!=sum(
            len(row["completed_episodes"]) for row in rows):
        raise ValueError("checkpoint counters differ from records")
    expected_evaluations = {entry["update"] for entry in payload["evaluations"]}
    if any(int(path.stem.split("_")[-1]) not in expected_evaluations
           for path in folder.glob("evaluation_*.json")):
        raise ValueError("evaluation record ahead of checkpoint")
    for entry in payload["evaluations"]:
        path = folder/f"evaluation_{entry['update']:06}.json"
        if path.exists():
            if digest(json.loads(path.read_text()))!=entry["sha256"]:
                raise ValueError("evaluation differs from checkpoint")
        elif "result" in entry:
            atomic_json(path,entry["result"])
        else:
            raise ValueError("missing historical evaluation record")
        if entry["update"]==committed and entry["result"]["model_sha256"]!=model_digest(
                payload["actor"],payload["critic"]):
            raise ValueError("evaluation model differs from checkpoint")
    lines = [json.dumps({k:v for k,v in row.items() if k!="completed_episodes"}) for row in rows]
    (folder/"training_metrics.jsonl").write_text(
        "".join(line+"\n" for line in lines),encoding="utf-8")
    return rows


def fit(folder,config=None,training=None,*,total_steps=None,until_steps=None,
        eval_interval=16,evaluation_seeds=None,device=None,resume=False):
    require_cuda(device or "cuda")
    folder = Path(folder)
    latest = folder/"latest.pt"
    if not resume:
        if config is None or training is None or not evaluation_seeds or eval_interval<1:
            raise ValueError("configuration, training and evaluation seeds required")
        wave = training.workers*training.horizon
        stop = total_steps if until_steps is None else until_steps
        if total_steps is None or not 0<stop<=total_steps or stop%wave or total_steps%wave:
            raise ValueError("budget must align with workers*horizon")
        folder.mkdir(parents=True,exist_ok=False)
        run = serializable(dict(id=uuid4().hex,protocol=PROTOCOL,environment=asdict(config),
            training=asdict(training),total_steps=total_steps,eval_interval=eval_interval,
            evaluation_seeds=list(evaluation_seeds),device=str(require_cuda(device or "cuda"))))
    with run_lock(folder):
        payload = read(latest,protocol=PROTOCOL) if resume else None
        if resume:
            run = payload["research_run"]
            if run["protocol"]!=PROTOCOL or run!=json.loads((folder/"config.json").read_text()):
                raise ValueError("run identity mismatch")
            config,training = InterceptionConfig(**run["environment"]),TrainConfig(**run["training"])
            total_steps,eval_interval,evaluation_seeds = (
                run["total_steps"],run["eval_interval"],run["evaluation_seeds"])
            if device is not None and str(require_cuda(device))!=run["device"]:
                raise ValueError("resume requires original CUDA device")
        wave = training.workers*training.horizon
        stop = total_steps if until_steps is None else until_steps
        if not 0<stop<=total_steps or stop%wave or stop<(payload["steps"] if payload else 0):
            raise ValueError("invalid resume target")
        random.seed(training.seed)
        np.random.seed(training.seed)
        torch.manual_seed(training.seed)
        model = InterceptionMAPPO(training,device=run["device"])
        with vector_env(config,training) as vector:
            if payload:
                restore(payload,model,config,vector,protocol=PROTOCOL)
                records = _recover(folder,payload)
            else:
                atomic_json(folder/"config.json",run)
                save(latest,model,config,vector,run=run,protocol=PROTOCOL,evaluations=[])
                records = []
            evaluations = payload["evaluations"].copy() if payload else []
            recent = [episode for row in records for episode in row["completed_episodes"]][-100:]
            terminal.training_start("Ideal interception",terminal.fields(
                experiment="ideal_interception",scale=config.scale,mode=config.intruder_mode,
                initialization=config.initialization),training,start_update=model.updates,
                planned_updates=total_steps//wave,
                evaluation=terminal.evaluation_setup(eval_interval,evaluation_seeds,
                    scenario=config.initialization,actions="Beta mean"),
                folder=folder,resume=resume,stop_steps=stop)
            while model.steps<stop:
                begin = time.perf_counter()
                try:
                    rollout,episodes = model.collect(vector)
                    sampled = time.perf_counter()
                    metrics = model.update(rollout)
                except (RuntimeError,FloatingPointError) as exc:
                    atomic_json(folder/"failure.json",dict(error=str(exc),
                        last_committed=str(latest),update=model.updates))
                    raise
                model.episodes += len(episodes)
                recent = (recent+episodes)[-100:]
                result = None
                if model.updates%eval_interval==0 or model.steps==total_steps:
                    result = evaluate(model,config,evaluation_seeds)
                    for entry in evaluations:
                        if "result" in entry:
                            path = folder/f"evaluation_{entry['update']:06}.json"
                            if not path.exists() or digest(json.loads(path.read_text()))!=entry["sha256"]:
                                raise ValueError("previous evaluation is not committed")
                            entry.pop("result")
                    evaluations.append(dict(update=model.updates,summary=result["summary"],
                                            sha256=digest(result),result=result))
                record = serializable(dict(update=model.updates,planned_updates=total_steps//wave,
                    previous=digest(records[-1]) if records else None,
                    joint_steps=model.steps,completed_episodes_total=model.episodes,
                    completed_episodes=episodes,sampling_speed=wave/(sampled-begin),**metrics))
                save(latest,model,config,vector,record,run=run,protocol=PROTOCOL,
                     evaluations=evaluations)
                atomic_json(folder/f"update_{model.updates:06}.json",record)
                with (folder/"training_metrics.jsonl").open("a",encoding="utf-8") as stream:
                    stream.write(json.dumps({k:v for k,v in record.items()
                                             if k!="completed_episodes"})+"\n")
                window = summarize(recent)
                terminal.training_progress("Ideal interception",record,total_steps=total_steps,
                    window=terminal.fields(win=f"{len(recent)}/100",
                        intercepted=terminal.percent(window["all_intercepted_rate"]),
                        breach=terminal.percent(window["breach_rate"]),
                        reward=terminal.number(window["mean_return"])))
                if result is not None:
                    atomic_json(folder/f"evaluation_{model.updates:06}.json",result)
                    terminal.emit("Ideal interception","EVAL",terminal.fields(upd=model.updates),
                        results(result["summary"]),terminal.fields(
                            scenario=result["evaluation_protocol"]["kind"],
                            unique=result["evaluation_protocol"]["unique_scenarios"]))
                records.append(record)
        terminal.training_end("Ideal interception",model,latest,paused=stop<total_steps)
    return latest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="operation",required=True)
    train,resume,assess = (subs.add_parser(name) for name in ("train","resume","evaluate"))
    for command in (train,resume,assess):
        command.add_argument("--out",required=True,type=Path)
        command.add_argument("--device")
    train.add_argument("--scale",choices=("1v1","3v1","5v2"),required=True)
    train.add_argument("--config",type=Path)
    train.add_argument("--total-steps",type=int,required=True)
    train.add_argument("--until-steps",type=int)
    train.add_argument("--seed",type=int,default=17)
    train.add_argument("--workers",type=int,default=4)
    train.add_argument("--horizon",type=int,default=128)
    train.add_argument("--eval-interval",type=int,default=16)
    train.add_argument("--eval-seed",type=int,default=2200)
    train.add_argument("--eval-episodes",type=int,default=10)
    resume.add_argument("--until-steps",type=int)
    assess.add_argument("checkpoint",type=Path)
    assess.add_argument("--eval-seed",type=int,default=2200)
    assess.add_argument("--eval-episodes",type=int,default=10)
    assess.add_argument("--sample-actions",action="store_true")
    assess.add_argument("--trace",action="store_true",help="export first evaluated episode as an ideal trace")
    args = parser.parse_args(argv)
    if args.operation=="train":
        options = json.loads(args.config.read_text()) if args.config else {}
        if options.get("scale",args.scale)!=args.scale:
            parser.error("scale differs from configuration")
        config = InterceptionConfig(**(options|dict(scale=args.scale)))
        training = TrainConfig(seed=args.seed,workers=args.workers,horizon=args.horizon)
        fit(args.out,config,training,total_steps=args.total_steps,
            until_steps=args.until_steps,eval_interval=args.eval_interval,
            evaluation_seeds=list(range(args.eval_seed,args.eval_seed+args.eval_episodes)),
            device=args.device)
    elif args.operation=="resume":
        fit(args.out,until_steps=args.until_steps,device=args.device,resume=True)
    else:
        payload = read(args.checkpoint,protocol=PROTOCOL)
        config = InterceptionConfig(**payload["environment"])
        model = InterceptionMAPPO(TrainConfig(**payload["training"]),device=args.device or "cuda")
        load_weights(payload,model)
        seeds = list(range(args.eval_seed,args.eval_seed+args.eval_episodes))
        args.out.mkdir(parents=True,exist_ok=False)
        result = evaluate(model,config,seeds,deterministic=not args.sample_actions)
        atomic_json(args.out/"evaluation.json",result|dict(checkpoint=str(args.checkpoint),
                                                           update=payload["updates"],seeds=seeds))
        if args.trace:
            from tools.ideal_interception_trace import from_evaluation
            atomic_json(args.out/"episode_000.json",from_evaluation(result))
        terminal.emit("Ideal interception","EVAL",terminal.fields(
            experiment="ideal_interception",scale=config.scale,upd=payload["updates"]),
            results(result["summary"]),terminal.fields(
                scenario=result["evaluation_protocol"]["kind"],
                unique=result["evaluation_protocol"]["unique_scenarios"],out=args.out))


if __name__=="__main__":
    from algorithm.ideal_interception import main as canonical_main
    terminal.run_cli("Ideal interception",canonical_main)
