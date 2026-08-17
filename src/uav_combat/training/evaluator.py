"""Isolated deterministic evaluation on fixed disjoint seeds."""
from __future__ import annotations
import numpy as np
from ..environment.env import PaperUAVCombatEnv


def evaluate(actor,config="configs/paper_environment.yaml",seeds=range(10_000_000,10_000_020))->dict[str,float]:
    records=[]
    for seed in seeds:
        env=PaperUAVCombatEnv(config); obs,_=env.reset(int(seed)); total=0.0
        while True:
            actions=actor.act(obs,env.red_alive_mask,deterministic=True)
            obs,reward,terminated,truncated,info=env.step(actions); total+=float(reward[0])
            if terminated or truncated: break
        records.append({"episode_return":total,**info})
    mean=lambda key:float(np.mean([r[key] for r in records]))
    return {"episode_return":mean("episode_return"),"win_rate":mean("red_win"),"red_win_rate":mean("red_win"),"blue_win_rate":mean("blue_win"),"draw_timeout_rate":mean("draw_or_timeout"),"red_survivors":mean("red_survivors"),"blue_survivors":mean("blue_survivors"),"red_attack_kills":mean("red_attack_kills"),"blue_attack_kills":mean("blue_attack_kills"),"red_boundary_losses":mean("red_boundary_losses"),"blue_boundary_losses":mean("blue_boundary_losses"),"episode_length":mean("episode_length"),"evaluation_episodes":float(len(records))}
