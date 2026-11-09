"""Short paired MAPPO experiments on static, non-contact navigation."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle
import random
from statistics import mean
from types import SimpleNamespace

import numpy as np
import torch

from algorithm.common.interception_checkpoint import PROTOCOL as INTERCEPTION, read, save
from algorithm.common.research import atomic_json
from algorithm.common.runtime import require_cuda
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.mappo.interception import MAPPO, TrainConfig
from env.navigation import NavigationConfig, NavigationEnv, VERSION


PROTOCOL = INTERCEPTION | {"version": VERSION}
ARMS = {"off": None, "weight_0p1": .1, "weight_0p3": .3, "weight_1p0": 1.}


def validate_discount(config, training):
    if config.heading_enabled and config.heading_discount != training.gamma:
        raise ValueError("heading_discount must equal the MAPPO training gamma")


class NavigationAccumulator:
    def reset(self, info):
        self.initial = info["states"].tolist()
        self.previous = info["states"].copy()
        self.parts = {}
        self.total = self.discounted = 0.
        self.initial_phi = None
        self.factor = 1.
        self.angle = self.angle_samples = self.improvements = self.comparisons = 0
        self.cross_track = self.speed = self.speed_change = self.turn = 0.
        self.agent_seconds = self.turn_seconds = 0.
        self.constraints = 0
        self.positive = self.negative = self.zero = 0

    def add(self, info, rewards):
        h = info["heading"]
        self.total += info["team_reward"]
        self.constraints += int(info["outcome"] == "boundary")
        for key, value in info["reward_parts"].items():
            self.parts[key] = self.parts.get(key, 0.)+value
        reward = h["reward"]
        self.positive += reward > 0
        self.negative += reward < 0
        self.zero += reward == 0
        if self.initial_phi is None:
            self.initial_phi = h["potential"]
        self.discounted += self.factor*reward
        self.factor *= info["heading_discount"]
        self.next_phi = h["next_potential"]
        self.shaping_enabled = "heading" in info["reward_parts"]
        after = dict(zip(h["next"]["slots"],h["next"]["angle_deg"]))
        for i, angle in zip(h["current"]["slots"], h["current"]["angle_deg"]):
            if angle is not None:
                self.angle += angle
                self.angle_samples += 1
                if after.get(i) is not None:
                    self.comparisons += 1
                    self.improvements += after[i] < angle-1e-10
        slots = h["current"]["slots"]
        dt = info["durations"][slots]
        old = self.previous[slots]
        new = info["states"][slots]
        origin = info["origins"][slots]
        segment = np.asarray(info["waypoint"])-origin
        offset = new[:,:3]-origin
        projection = np.clip(np.sum(segment*offset,axis=1)/np.sum(segment**2,axis=1),0.,1.)
        error = np.linalg.norm(offset-segment*projection[:,None],axis=1)
        v0,v1 = old[:,3:],new[:,3:]
        speed0,speed1 = np.linalg.norm(v0,axis=1),np.linalg.norm(v1,axis=1)
        valid = (speed0 > 0) & (speed1 > 0)
        angle_change = np.degrees(np.arctan2(np.linalg.norm(np.cross(v0[valid],v1[valid]),axis=1),
                                             np.sum(v0[valid]*v1[valid],axis=1)))
        self.cross_track += float(error@dt)
        self.speed += float(((speed0+speed1)/2)@dt)
        self.speed_change += float(np.abs(speed1-speed0).sum())
        self.turn += float(angle_change.sum())
        self.agent_seconds += float(dt.sum())
        self.turn_seconds += float(dt[valid].sum())
        self.previous = info["states"].copy()

    def summary(self, info):
        def ratio(total,count):
            return total/count if count else None
        expected = self.factor*self.next_phi-self.initial_phi if self.shaping_enabled else 0.
        return dict(outcome=info["outcome"], time=info["time"], steps=info["step"],
            initial_states=self.initial, episode_return=self.total, reward_parts=self.parts.copy(),
            route_completion=info["arrivals"]/info["members"], shaping_total=self.parts.get("heading",0.),
            discounted_shaping=self.discounted, discounted_boundary_expression=expected,
            shaping_identity_error=self.discounted-expected,
            positive_heading_steps=self.positive, negative_heading_steps=self.negative, zero_heading_steps=self.zero,
            mean_heading_error_deg=ratio(self.angle,self.angle_samples),
            heading_improvement_rate=ratio(self.improvements,self.comparisons),
            tracking_error_m=ratio(self.cross_track,self.agent_seconds),
            mean_speed_mps=ratio(self.speed,self.agent_seconds),
            speed_change_mps2=ratio(self.speed_change,self.agent_seconds),
            turning_deg_per_s=ratio(self.turn,self.turn_seconds), safety_violations=self.constraints)


class _NavigationUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("__main__", "NavigationAccumulator"):
            return NavigationAccumulator
        return super().find_class(module, name)


def read_checkpoint(path):
    """Read trusted navigation archives, including the original CLI's tracker name."""
    compat = SimpleNamespace(__name__="pickle", Unpickler=_NavigationUnpickler, load=pickle.load)
    return read(path, protocol=PROTOCOL, pickle_module=compat)


def summarize(rows):
    keys = ("episode_return", "route_completion", "shaping_total", "mean_heading_error_deg",
            "heading_improvement_rate", "tracking_error_m", "mean_speed_mps", "speed_change_mps2",
            "turning_deg_per_s", "time")
    metrics = {}
    for key in keys:
        values = [row[key] for row in rows if row[key] is not None]
        metrics[key] = mean(values) if values else None
    return metrics | dict(episodes=len(rows), successes=sum(row["outcome"]=="success" for row in rows),
        timeouts=sum(row["outcome"]=="timeout" for row in rows),
        safety_violations=sum(row["safety_violations"] for row in rows),
        max_shaping_identity_error=max(abs(row["shaping_identity_error"]) for row in rows))


def evaluate(model, config, seeds, *, trace=False):
    require_cuda()
    rows, diagnostics = [], []
    for i, seed in enumerate(seeds):
        env = NavigationEnv(config)
        obs, info = env.reset(np.random.SeedSequence([1,seed]))
        tracker = NavigationAccumulator()
        tracker.reset(info)
        while True:
            action = model.act(obs,info["active_mask"],deterministic=True)[0].cpu().numpy()
            obs,rewards,terminated,truncated,info = env.step(action)
            tracker.add(info,rewards)
            if trace and i == 0:
                diagnostics.append({key:info[key] for key in ("time","step","heading","reward_parts","outcome","states")})
            if terminated or truncated:
                rows.append(tracker.summary(info) | {"seed":seed})
                break
    return dict(metrics=summarize(rows),episodes=rows,first_episode_trace=diagnostics)


def study(output, base, training, seeds, evaluation_seeds, updates, interval):
    require_cuda()
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    all_runs = []
    for name,weight in ARMS.items():
        config = replace(base,heading_enabled=weight is not None,heading_weight=base.heading_weight if weight is None else weight)
        validate_discount(config,training)
        for seed in seeds:
            settings = replace(training,seed=seed)
            folder = output/name/f"seed_{seed}"
            folder.mkdir(parents=True,exist_ok=False)
            atomic_json(folder/"config.json",dict(protocol=PROTOCOL,environment=asdict(config),
                         training=asdict(settings),updates=updates,evaluation_seeds=evaluation_seeds))
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = MAPPO(settings)
            initial = evaluate(model,config,evaluation_seeds)
            atomic_json(folder/"evaluation_0000.json",initial)
            curve = [dict(update=0,evaluation=initial["metrics"])]
            with ParallelVectorEnv(config,settings.workers,seed,environment_factory=NavigationEnv,
                                   accumulator_factory=NavigationAccumulator) as vector:
                with (folder/"train.jsonl").open("x",encoding="utf-8") as log:
                    for update in range(1,updates+1):
                        rollout,rows = model.collect(vector)
                        ppo = model.update(rollout)
                        model.episodes += len(rows)
                        record = dict(update=update,joint_steps=model.steps,ppo=ppo,episodes=rows)
                        log.write(json.dumps(record,allow_nan=False)+"\n")
                        log.flush()
                        if update % interval == 0 or update == updates:
                            final = evaluate(model,config,evaluation_seeds,trace=update==updates)
                            atomic_json(folder/f"evaluation_{update:04}.json",final)
                            curve.append(dict(update=update,evaluation=final["metrics"]))
                save(folder/"latest.pt",model,config,vector,record,protocol=PROTOCOL)
            item = dict(arm=name,seed=seed,curve=curve,joint_steps=model.steps,episodes=model.episodes)
            all_runs.append(item)
            atomic_json(output/"summary.json",dict(runs=all_runs))
            print(name,seed,final["metrics"],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True)
    parser.add_argument("--env",type=Path)
    parser.add_argument("--training",type=Path)
    parser.add_argument("--updates",type=int,default=12)
    parser.add_argument("--eval-interval",type=int,default=4)
    parser.add_argument("--seeds",type=int,nargs="+",default=[17,18])
    parser.add_argument("--evaluation-seed",type=int,default=2200)
    parser.add_argument("--evaluation-episodes",type=int,default=12)
    args = parser.parse_args()
    if min(args.updates,args.eval_interval,args.evaluation_episodes) < 1 or len(set(args.seeds))!=len(args.seeds):
        parser.error("positive budgets and distinct training seeds are required")
    config = NavigationConfig(**(json.loads(args.env.read_text()) if args.env else {}))
    training = TrainConfig(**(json.loads(args.training.read_text()) if args.training else {}))
    study(args.out,config,training,args.seeds,list(range(args.evaluation_seed,args.evaluation_seed+args.evaluation_episodes)),
          args.updates,args.eval_interval)


if __name__ == "__main__":
    # Pickled worker trackers must use the importable module name.
    from algorithm.navigation import main as canonical_main
    canonical_main()
