"""Small pre-training numerical audit; never load/train a policy or alter data."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from algorithm.common.runtime import require_cuda
from algorithm.mappo.diagnostics import diagnose_actor, diagnose_critic, isolated_rng
from algorithm.mappo.networks import Actor, Critic, ProtocolCritic
from algorithm.mappo.paper_protocol import OBSERVATION_FIELDS, pack_centralized_slots
from env import EnvConfig
from env.factory import make_environment
from .numerical_cases import reward_boundaries, ppo_boundaries


def inspection_samples():
    actors, critics = [], []
    for scale in ('3U3P','5U5P','3U5P'):
        for seed in range(5):
            env=make_environment(EnvConfig(scale=scale))
            obs,info=env.reset(seed)
            for step in range(9):
                actors.append(obs[info['active_mask']].copy())
                critics.append(info['centralized_state'].copy())
                if step == 8: break
                action=np.tile([.5,1/3,1/8],(env.config.num_uavs,1))
                obs,_,terminal,truncated,info=env.step(action)
                if terminal or truncated: break
    return np.concatenate(actors),np.stack(critics)


def synthetic_physical_samples():
    """Independent physical-scale descriptors, NOT reconstructed task geometry."""
    rng=np.random.default_rng(20261022)
    slots=np.ones((128,5,12))
    for index in (1,6): slots[...,index]=rng.uniform(100,400,(128,5))
    for index in (2,7): slots[...,index]=rng.uniform(-1.2,1.2,(128,5))
    for index in (3,8): slots[...,index]=rng.uniform(-np.pi,np.pi,(128,5))
    for index in (4,9): slots[...,index]=rng.uniform(0,np.pi,(128,5))
    for index in (5,10): slots[...,index]=rng.uniform(500,12000,(128,5))
    slots[...,11]=rng.uniform(100,20000,(128,5))
    # Include exact endpoints, not just likely values from random sampling.
    slots[0,:,:]=[1,100,-1.2,-np.pi,0,500,100,-1.2,-np.pi,0,500,100]
    slots[-1,:,:]=[1,400,1.2,np.pi,np.pi,12000,400,1.2,np.pi,np.pi,12000,20000]
    return slots.reshape(-1,12),pack_centralized_slots(slots,np.ones((128,5)))


def network_report():
    device=require_cuda('cuda')
    ia,ic=inspection_samples(); sa,sc=synthetic_physical_samples()
    rows=[]
    with isolated_rng():
        for seed in (0,1,2,3,4):
            # Same Actor initialization for both input protocols at each seed.
            torch.manual_seed(seed); actor=Actor().to(device)
            torch.manual_seed(seed); c90=Critic().to(device)
            torch.manual_seed(seed); c60=ProtocolCritic().to(device)
            for label,a,c,critic in (('actual_inspection_environment',ia,ic,c90),
                                      ('synthetic_physical_inputs',sa,sc,c60)):
                rows.append(dict(initialization_seed=seed,source=label,
                    actor=diagnose_actor(actor,a,source=label),critic=diagnose_critic(critic,c,source=label)))
    return dict(initialization='current defaults; untrained models, not historical checkpoints',
                synthetic_warning='synthetic physical inputs; angles/distances sampled independently, no legal target/engagement data chain',
                inspection_sampling='3 scales x 5 reset seeds x (initial + 8 trim steps); original task/config/reward',
                synthetic_fields=[dict(index=i,name=n,unit=u) for i,n,u in OBSERVATION_FIELDS],
                initial_seeds=[0,1,2,3,4],rows=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    require_cuda('cuda')
    torch.set_num_threads(1)  # Same runtime default as TrainConfig; CLI process only.
    output=Path(args.output_dir)
    if output.exists(): raise FileExistsError('refusing to overwrite an existing audit output directory')
    with isolated_rng():
        networks=network_report()
        rewards=reward_boundaries()
        ppo=ppo_boundaries()
    output.mkdir(parents=True)
    for name,data in (('networks',networks),('reward_boundaries',rewards),('ppo_boundaries',ppo)):
        (output/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(output.resolve()),network_cases=len(networks['rows']),reward_cases=len(rewards['cases']),ppo_cases=len(ppo['cases'])),ensure_ascii=False))


if __name__=='__main__': main()
