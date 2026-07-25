import numpy as np
import torch
from uav_env.algorithms.mappo.config import load_mappo_config
from uav_env.algorithms.mappo.networks import SharedActor
from uav_env.algorithms.mappo.runner import MAPPORunner

def test_same_seed_network_identical():
 torch.manual_seed(7);a=SharedActor(11);torch.manual_seed(7);b=SharedActor(11);assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))


def test_two_independent_cpu_runners_match_rollout_returns_and_update(tmp_path):
 results=[]
 for run_id in ("independent_a","independent_b"):
  c=load_mappo_config("configs/mappo_smoke_1v1.yaml");c.update(seed=31,num_envs=1,rollout_length=2,ppo_epochs=1,num_mini_batches=1,device="cpu",run_id=run_id)
  runner=MAPPORunner(c,"determinism",tmp_path);buffer,_=runner.collect();metrics=runner.trainer.update(buffer)
  results.append((buffer.actions.copy(),buffer.rewards.copy(),buffer.advantages.copy(),buffer.returns.copy(),metrics,[p.detach().clone() for p in runner.actor.parameters()],[p.detach().clone() for p in runner.critic.parameters()]));runner.writer.close()
 for index in range(4): assert np.array_equal(results[0][index],results[1][index])
 assert results[0][4]==results[1][4]
 assert all(torch.equal(a,b) for a,b in zip(results[0][5],results[1][5]))
 assert all(torch.equal(a,b) for a,b in zip(results[0][6],results[1][6]))
