"""Print checkpoint metadata without constructing an environment."""
from __future__ import annotations
import argparse,torch,yaml
def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("checkpoint_positional",nargs="?"); p.add_argument("--checkpoint"); a=p.parse_args(); checkpoint=a.checkpoint or a.checkpoint_positional
    if not checkpoint: p.error("a checkpoint path is required")
    d=torch.load(checkpoint,map_location="cpu",weights_only=False)
    actor=sum(v.numel() for v in d["actor"].values()); critic=sum(v.numel() for v in d["critic"].values())
    print(yaml.safe_dump({"version":d["version"],"environment_steps":d["environment_steps"],"update_index":d["update_index"],"actor_parameters":actor,"critic_parameters":critic,"value_normalizer":{k:str(v) for k,v in d["value_normalizer"].items()},"best_evaluation":d["best_evaluation"],"config":d["config"]},sort_keys=False))
if __name__=="__main__": main()
