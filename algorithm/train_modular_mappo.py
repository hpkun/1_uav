"""Formal CLI for modular MAPPO, including immutable run lineage and resume."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.train_mappo import (
    TeeOutput, ensure_fresh_output_directory, load_run_config,
    prepare_resume_rollback, reject_stale_resume_checkpoint,
    resolve_runtime_settings, validate_resume_config_snapshots,
)
from algorithm.common.protocol import config_sha256
from algorithm.mappo.trainer import MAPPO_IMPL_VERSION
from algorithm.modular_mappo.protocol import (
    checkpoint_architecture, validate_modular_checkpoint,
)
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.modular_mappo.trainer import MODULAR_MAPPO_IMPL_VERSION


def _merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = _merge(result.get(key, {}), value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def resolved(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_config(path: str | Path) -> dict:
    source = resolved(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if "extends" in data:
        parent = data.pop("extends")
        data = _merge(load_config(parent), data)
    return data


def default_output_dir(seed: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "outputs" / f"modular_{stamp}_seed{seed}"


def write_snapshots(output_dir: Path, env_config: dict, algorithm_config: dict) -> None:
    (output_dir / "env_config.yaml").write_text(yaml.safe_dump(env_config, sort_keys=False), encoding="utf-8")
    (output_dir / "algorithm_config.yaml").write_text(yaml.safe_dump(algorithm_config, sort_keys=False), encoding="utf-8")


def write_run_config(path: Path, runner: ModularMAPPOTrainingRunner,
                     env_path: Path, algorithm_path: Path) -> dict:
    protocol = runner.trainer.module_protocol()
    value = {
        "algorithm":"modular_mappo", "seed":runner.seed, "device":runner.device,
        "num_envs":runner.num_envs, "total_sampled_steps":runner.total_sampled_steps,
        "smoke":runner.smoke, "environment_variant":runner.env_config.get("environment_variant","direct_v2_3"),
        "environment_version":runner.env_config.get("environment_version"),
        "environment_config_path":str(env_path), "algorithm_config_path":str(algorithm_path),
        "environment_config_sha256":config_sha256(runner.env_config),
        "algorithm_config_sha256":config_sha256(runner.algorithm_config),
        "enabled_modules":protocol["enabled_modules"], "module_config_sha256":protocol["module_config_sha256"],
        "network_architecture":checkpoint_architecture(runner.trainer),
        "warm_start_provenance":runner.trainer.warm_start_provenance,
        "policy_anchor_provenance":runner.trainer.anchor_provenance,
        "curriculum_config":runner.algorithm_config.get("modules",{}).get("curriculum",{}),
        "output_dir":str(runner.output_dir.resolve()),
        "start_timestamp":datetime.now().astimezone().isoformat(),
        "baseline_mappo_impl_version":MAPPO_IMPL_VERSION,
        "modular_mappo_impl_version":MODULAR_MAPPO_IMPL_VERSION,
    }
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return value

def future_row_counts(run_dir: Path, checkpoint_steps: int) -> dict[str,int]:
    counts={}
    for name in ("training_metrics.jsonl","optimization_metrics.jsonl"):
        path=run_dir/name
        rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
        counts[name]=sum(int(row["sampled_steps"])>checkpoint_steps for row in rows)
    path=run_dir/"evaluation_history.csv"
    if path.exists():
        import csv
        with path.open(newline="",encoding="utf-8") as stream:counts["evaluation_history.csv"]=sum(int(row["sampled_steps"])>checkpoint_steps for row in csv.DictReader(stream))
    else:counts["evaluation_history.csv"]=0
    return counts


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--env-config",default="configs/persistent_wave_v2_environment.yaml")
    parser.add_argument("--algorithm-config",default="configs/modular_mappo_persistent.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--device",choices=("cpu","cuda"));parser.add_argument("--seed",type=int)
    parser.add_argument("--num-envs",type=int);parser.add_argument("--total-sampled-steps",type=int)
    parser.add_argument("--smoke",action="store_true",default=None);parser.add_argument("--resume")
    parser.add_argument("--warm-start-checkpoint");parser.add_argument("--reference-checkpoint")
    args=parser.parse_args()
    env_path, algorithm_path = resolved(args.env_config), resolved(args.algorithm_config)
    env_config=yaml.safe_load(env_path.read_text(encoding="utf-8"));algorithm_config=load_config(algorithm_path)
    resume_path=resolved(args.resume).resolve() if args.resume else None
    state=None;run_config=None;rollback={}
    if resume_path is None:
        seed=int(algorithm_config["training"]["seed"] if args.seed is None else args.seed)
        output_dir=default_output_dir(seed) if args.output_dir is None else resolved(args.output_dir)
        # This is deliberately the first write-capable operation.
        ensure_fresh_output_directory(output_dir)
    else:
        if not resume_path.is_file():raise FileNotFoundError(resume_path)
        output_dir=resume_path.parent if args.output_dir is None else resolved(args.output_dir).resolve()
        if output_dir!=resume_path.parent:raise RuntimeError("resume output_dir must equal checkpoint.parent")
        validate_resume_config_snapshots(output_dir,env_config,algorithm_config)
        run_config=load_run_config(output_dir)
        if run_config is None:raise RuntimeError("formal modular resume requires run_config.json")
        state=torch.load(resume_path,map_location="cpu",weights_only=False)
        reject_stale_resume_checkpoint(output_dir,resume_path)
    runtime=resolve_runtime_settings(algorithm_config,seed=args.seed,num_envs=args.num_envs,
        total_sampled_steps=args.total_sampled_steps,device=args.device,smoke=args.smoke,
        run_config=run_config,checkpoint_state=state)
    if state is not None:
        validate_modular_checkpoint(state,env_config,algorithm_config,{
            "training_seed":runtime["seed"],"training_num_envs":runtime["num_envs"],"training_smoke":runtime["smoke"]})
        if run_config["module_config_sha256"]!=state["module_config_sha256"]:raise RuntimeError("run_config/checkpoint module protocol mismatch")
        if run_config.get("network_architecture")!=state.get("extra",{}).get("network_architecture"):raise RuntimeError("run_config/checkpoint network architecture mismatch")
        if run_config.get("warm_start_provenance",{})!=state.get("warm_start_provenance",{}):raise RuntimeError("warm-start provenance mismatch")
        if run_config.get("policy_anchor_provenance",{})!=state.get("anchor_provenance",{}):raise RuntimeError("policy-anchor provenance mismatch")
        truncated_counts=future_row_counts(output_dir,int(state["sampled_steps"]))
        rollback=prepare_resume_rollback(output_dir,resume_path,int(state["sampled_steps"]))
    runner=ModularMAPPOTrainingRunner(env_config,algorithm_config,runtime["num_envs"],runtime["total_sampled_steps"],runtime["device"],runtime["seed"],output_dir,runtime["smoke"],args.warm_start_checkpoint,args.reference_checkpoint,resume_mode=state is not None)
    if state is None:
        write_snapshots(output_dir,env_config,algorithm_config)
        write_run_config(output_dir/"run_config.json",runner,env_path,algorithm_path)
    else:
        runner.resume(resume_path)
        original=runtime["original"]
        resume_record={"timestamp":datetime.now().astimezone().isoformat(),"resume_checkpoint":str(resume_path),
            "checkpoint_sampled_steps":int(state["sampled_steps"]),"original_target_budget":original["total_sampled_steps"],
            "new_target_budget":runtime["total_sampled_steps"],"extension":runtime["extended_training_target"],
            "seed":runtime["seed"],"num_envs":runtime["num_envs"],"old_device":original["device"],"new_device":runtime["device"],
            "logs_truncated":any(truncated_counts.values()),"truncated_row_counts":truncated_counts,"rollback":rollback,
            "best_restored_sampled_steps":runner.best_sampled_steps,"curriculum_stage":runner.current_stage,
            "enabled_modules":runner.trainer.module_protocol()["enabled_modules"]}
        with (output_dir/"resume_history.jsonl").open("a",encoding="utf-8") as stream:stream.write(json.dumps(resume_record)+"\n")
    with (output_dir/"train.log").open("a",encoding="utf-8") as log_stream:
        with redirect_stdout(TeeOutput(sys.stdout,log_stream)):
            runner.run()


if __name__=="__main__":main()
