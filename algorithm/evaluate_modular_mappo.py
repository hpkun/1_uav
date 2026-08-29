"""Reconstruct modular architecture from checkpoint metadata and evaluate raw outcomes."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.evaluation import evaluate_modular
def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--env-config",required=True);p.add_argument("--episodes",type=int,default=50);p.add_argument("--seed-base",type=int,default=10000000);p.add_argument("--device",default="cpu",choices=("cpu","cuda"));a=p.parse_args();state=torch.load(a.checkpoint,map_location="cpu",weights_only=False);extra=state.get("extra",{});cfg=extra.get("algorithm_config")
 if cfg is None:raise RuntimeError("checkpoint lacks algorithm_config snapshot")
 envp=Path(a.env_config) if Path(a.env_config).is_absolute() else ROOT/a.env_config;env=yaml.safe_load(envp.read_text(encoding="utf-8"));arch=extra.get("network_architecture",{});hidden=int(arch.get("hidden_dim",state["actor"]["backbone.0.weight"].shape[0]));trainer=build_modular_mappo_trainer(cfg,a.device,hidden);trainer.load(a.checkpoint);result=evaluate_modular(trainer,env,range(a.seed_base,a.seed_base+a.episodes));result["module_metadata"]={"enabled_modules":state["enabled_modules"],"module_config":state["module_config"]};print(json.dumps(result,indent=2))
if __name__=="__main__":main()
