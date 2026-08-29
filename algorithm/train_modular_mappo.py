"""CLI for independent modular MAPPO experiments."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner

def _merge(a,b):
 out=dict(a)
 for k,v in b.items():out[k]=_merge(out.get(k,{}),v) if isinstance(v,dict) and isinstance(out.get(k),dict) else v
 return out
def load_config(path):
 p=Path(path) if Path(path).is_absolute() else ROOT/path;data=yaml.safe_load(p.read_text(encoding="utf-8"))
 if "extends" in data:
  base=load_config(data.pop("extends"));data=_merge(base,data)
 return data
def main():
 p=argparse.ArgumentParser();p.add_argument("--env-config",default="configs/persistent_wave_v2_environment.yaml");p.add_argument("--algorithm-config",default="configs/modular_mappo_persistent.yaml");p.add_argument("--output-dir",required=True);p.add_argument("--device",choices=("cpu","cuda"));p.add_argument("--seed",type=int);p.add_argument("--num-envs",type=int);p.add_argument("--total-sampled-steps",type=int);p.add_argument("--smoke",action="store_true");p.add_argument("--resume");p.add_argument("--warm-start-checkpoint");p.add_argument("--reference-checkpoint");a=p.parse_args()
 envp=Path(a.env_config) if Path(a.env_config).is_absolute() else ROOT/a.env_config;env=yaml.safe_load(envp.read_text(encoding="utf-8"));cfg=load_config(a.algorithm_config)
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);(out/"env_config.yaml").write_text(yaml.safe_dump(env,sort_keys=False),encoding="utf-8");(out/"algorithm_config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
 reference=a.reference_checkpoint
 if a.resume and reference is None:
  resume_state=torch.load(a.resume,map_location="cpu",weights_only=False);reference=resume_state.get("anchor_provenance",{}).get("reference_checkpoint")
 runner=ModularMAPPOTrainingRunner(env,cfg,a.num_envs,a.total_sampled_steps,a.device,a.seed,out,a.smoke,a.warm_start_checkpoint,reference)
 if a.resume:runner.resume(a.resume)
 print(json.dumps(runner.run(),indent=2))
if __name__=="__main__":main()
