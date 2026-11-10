"""Versioned configuration and budget accounting for interception runs."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re

from algorithm.common.interception_checkpoint import PROTOCOL
from algorithm.common.research import EvaluationProtocol, atomic_json, read_json
from algorithm.common.serialization import serializable
from algorithm.mappo.interception import TrainConfig
from env.interception.config import InterceptionConfig, SCALES, TASKS


VERSION = 1


def positive_int(text):
    if not re.fullmatch(r"[0-9]+", text) or int(text) < 1:
        raise ValueError("expected a positive integer in decimal digits")
    return int(text)


def object_fields(value, name, allowed):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {', '.join(sorted(unknown))}")
    return value.copy()


def load_object(path):
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def budget(steps, training):
    if type(steps) is not int or steps < 1:
        raise ValueError("total_joint_steps must be a positive integer")
    wave = training.workers * training.horizon
    updates = (steps + wave - 1) // wave
    return dict(total_joint_steps=steps, aligned_joint_steps=updates * wave, updates=updates)


def environment_config(values, flags):
    values = values.copy()
    flags = {key: flags.get(key) for key in ("task", "scale", "evader_mode")}
    if flags.get("task") is not None:
        values.pop("scale", None)
    if flags.get("scale") is not None and flags.get("task") is None:
        values.pop("task", None)
    values.update({key: value for key, value in flags.items() if value is not None})
    if "task" in values and "scale" not in values:
        task = values["task"]
        if task not in TASKS:
            raise ValueError(f"unknown task {task}")
        values["scale"] = TASKS[task][0]
    return InterceptionConfig(**values)


@dataclass(frozen=True)
class TestProtocol:
    enabled: bool = True
    scenario: str = "random"
    seed: int = 10_010_000
    episodes: int = 20
    deterministic: bool = True
    checkpoint: str = "latest"

    def __post_init__(self):
        EvaluationProtocol(0, self.episodes, self.seed, self.scenario, self.deterministic)
        if type(self.enabled) is not bool or self.checkpoint not in ("latest", "best"):
            raise ValueError("invalid final test protocol")

    @property
    def seeds(self):
        return range(self.seed, self.seed + self.episodes)


@dataclass(frozen=True)
class Storage:
    trajectories: str = "none"
    trajectory_seeds: tuple = ()
    gzip: bool = True

    def __post_init__(self):
        if self.trajectories not in ("none", "selected", "all") or type(self.gzip) is not bool:
            raise ValueError("invalid trajectory storage policy")
        seeds = self.trajectory_seeds
        if not isinstance(seeds, (tuple, list)) or any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("trajectory_seeds must be nonnegative integers")
        if len(set(seeds)) != len(seeds) or bool(seeds) != (self.trajectories == "selected"):
            raise ValueError("selected trajectories require distinct seeds; other modes take no seeds")
        object.__setattr__(self, "trajectory_seeds", tuple(seeds))

    def validate_seeds(self, seeds):
        if any(seed not in seeds for seed in self.trajectory_seeds):
            raise ValueError("selected trajectory seed is outside the evaluation seeds")

    def traces(self, seed):
        return self.trajectories == "all" or seed in self.trajectory_seeds


@dataclass(frozen=True)
class Report:
    on_complete: bool = False
    plots: bool = True

    def __post_init__(self):
        if type(self.on_complete) is not bool or type(self.plots) is not bool:
            raise ValueError("report options must be booleans")


@dataclass(frozen=True)
class Experiment:
    name: str
    environment: InterceptionConfig
    training: TrainConfig
    total_joint_steps: int
    evaluation: EvaluationProtocol = field(default_factory=EvaluationProtocol)
    test: TestProtocol = field(default_factory=TestProtocol)
    storage: Storage = field(default_factory=Storage)
    report: Report = field(default_factory=Report)
    log_interval: int = 1
    evaluation_interval_joint_steps: int | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("experiment name must be nonempty")
        if self.environment.scale not in SCALES:
            raise ValueError("experiment requires an S0–S3 MAPPO environment")
        if type(self.log_interval) is not int or self.log_interval < 1:
            raise ValueError("log_interval must be a positive integer")
        plan = budget(self.total_joint_steps, self.training)
        interval = self.evaluation.evaluation_interval_updates
        if self.test.enabled:
            validation_seeds = range(self.evaluation.evaluation_seed_base,
                                     self.evaluation.evaluation_seed_base + self.evaluation.evaluation_episodes)
            if interval and max(validation_seeds.start, self.test.seeds.start) < min(validation_seeds.stop, self.test.seeds.stop):
                raise ValueError("validation and final test seed ranges overlap")
            if self.test.checkpoint == "best" and not 0 < interval <= plan["updates"]:
                raise ValueError("final test needs at least one scheduled validation to select Best")
            self.storage.validate_seeds(self.test.seeds)
        elif self.storage.trajectories != "none":
            raise ValueError("final test must be enabled to export its trajectories")

    def manifest(self):
        return serializable(dict(version=VERSION, algorithm=PROTOCOL["algorithm"], name=self.name,
            **budget(self.total_joint_steps, self.training), log_interval=self.log_interval,
            evaluation_interval_joint_steps=self.evaluation_interval_joint_steps,
            checkpoint="latest_best", test=asdict(self.test), storage=asdict(self.storage),
            report=asdict(self.report)))


def resolve(args):
    raw = load_object(args.config) if args.config else {"version": VERSION}
    raw = object_fields(raw, "experiment", ("version", "name", "algorithm", "environment", "training",
        "total_joint_steps", "log_interval", "evaluation", "checkpoint", "test", "storage", "report"))
    if type(raw.get("version")) is not int or raw["version"] != VERSION:
        raise ValueError("unsupported experiment config version")
    if raw.get("algorithm", PROTOCOL["algorithm"]) != PROTOCOL["algorithm"]:
        raise ValueError("unsupported interception algorithm protocol")
    if raw.get("checkpoint", "latest_best") != "latest_best":
        raise ValueError("checkpoint policy must be latest_best")
    env = object_fields(raw.get("environment", {}), "environment", InterceptionConfig.__dataclass_fields__)
    if args.env:
        overrides = object_fields(load_object(args.env), "environment", InterceptionConfig.__dataclass_fields__)
        for key, other in (("task", "scale"), ("scale", "task")):
            if key in overrides and other not in overrides:
                env.pop(other, None)
        env.update(overrides)
    config = environment_config(env, vars(args))
    settings = object_fields(raw.get("training", {}), "training", TrainConfig.__dataclass_fields__)
    if args.training:
        settings.update(object_fields(load_object(args.training), "training", TrainConfig.__dataclass_fields__))
    if args.seed is not None:
        settings["seed"] = args.seed
    training = TrainConfig(**settings)
    evaluation_fields = EvaluationProtocol.__dataclass_fields__
    validation = object_fields(raw.get("evaluation", {}), "evaluation", (*evaluation_fields, "interval_joint_steps"))
    step_interval = validation.pop("interval_joint_steps", None)
    if "evaluation_interval_updates" in validation and step_interval is not None:
        raise ValueError("configure the evaluation interval in updates or joint steps, not both")
    if args.evaluation_interval_updates is not None:
        step_interval = None
    if args.evaluation_interval_joint_steps is not None:
        validation.pop("evaluation_interval_updates", None)
        step_interval = args.evaluation_interval_joint_steps
    validation.update({key: getattr(args, key) for key in evaluation_fields if getattr(args, key) is not None})
    if step_interval is not None:
        validation["evaluation_interval_updates"] = budget(step_interval, training)["updates"]
    evaluation = EvaluationProtocol(**validation)
    log_interval = args.log_interval if args.log_interval is not None else raw.get("log_interval", 1)
    steps = args.total_steps if args.total_steps is not None else raw.get("total_joint_steps")
    if args.updates is not None and steps is not None and args.updates != budget(steps, training)["updates"]:
        raise ValueError("--updates conflicts with the aligned total step budget")
    if steps is None and (args.config or step_interval is not None):
        raise ValueError("unified experiments require total_joint_steps or --total-steps")
    if steps is None:
        return config, training, evaluation, log_interval, None
    def options(key, cls):
        return cls(**object_fields(raw.get(key, {}), key, cls.__dataclass_fields__))
    spec = Experiment(raw.get("name", config.task), config, training, steps, evaluation,
                      options("test", TestProtocol), options("storage", Storage), options("report", Report),
                      log_interval, step_interval)
    return config, training, evaluation, log_interval, spec


def planned_budget(folder, run, training, committed, extend=None):
    """Read the append-only sequence of atomically published budget decisions."""
    original = run["experiment"]
    current = budget(original["total_joint_steps"], training)
    if original.get("version") != VERSION or any(original.get(key) != value for key, value in current.items()):
        raise ValueError("invalid original experiment budget")
    paths = sorted(folder.glob("budget_*.json"))
    for index, path in enumerate(paths, 1):
        record = read_json(path)
        target = budget(record["target"]["total_joint_steps"], training)
        if (path.name != f"budget_{index:06}.json" or record !=
                dict(version=VERSION, run_id=run["id"], previous=current, target=target)
                or target["updates"] <= current["updates"]):
            raise ValueError("invalid experiment budget history")
        current = target
    if committed > current["updates"]:
        raise ValueError("budget history is behind the committed plan")
    if extend is not None:
        target = budget(extend, training)
        if target["updates"] <= current["updates"]:
            raise ValueError("extension must add at least one complete update to the saved budget")
        atomic_json(folder / f"budget_{len(paths)+1:06}.json",
                    dict(version=VERSION, run_id=run["id"], previous=current, target=target))
        current = target
    return current


def storage_usage(folder):
    files = [path.stat() for path in Path(folder).rglob("*") if path.is_file()]
    unique = {(stat.st_dev, stat.st_ino) if stat.st_ino else index: stat.st_size
              for index, stat in enumerate(files)}
    return dict(files=len(files), bytes=sum(stat.st_size for stat in files), unique_bytes=sum(unique.values()))
