import torch
import numpy as np
import pytest
from uav_env.algorithms.mappo.checkpoint import save_checkpoint,load_checkpoint
from uav_env.algorithms.mappo.config import load_mappo_config
from uav_env.algorithms.mappo.networks import SharedActor,CentralizedCritic
from uav_env.algorithms.mappo.runner import MAPPORunner
from uav_env.algorithms.mappo.value_normalizer import ValueNormalizer

def test_checkpoint_actor_roundtrip(tmp_path):
 a=SharedActor(11);c=CentralizedCritic(10,1);ao=torch.optim.Adam(a.parameters());co=torch.optim.Adam(c.parameters());n=ValueNormalizer();x=torch.randn(2,11);before=a(x).detach().clone();p=tmp_path/"x.pt";save_checkpoint(p,a,c,ao,co,n,{},3,1,None)
 b=SharedActor(11);load_checkpoint(p,b,actor_only=True);assert torch.equal(before,b(x))


def test_runner_mid_episode_state_roundtrip(tmp_path):
 c=load_mappo_config("configs/mappo_smoke_1v1.yaml");c["num_envs"]=1;c["rollout_length"]=2;c["device"]="cpu";c["run_id"]="source"
 source=MAPPORunner(c,"checkpoint_test",tmp_path);source.collect();source.environment_steps=2;source.update_index=1;source._save("state.pt")
 restored_config=dict(c);restored_config["run_id"]="restored";restored=MAPPORunner(restored_config,"checkpoint_test",tmp_path);restored.resume(str(source.output_dir/"checkpoints"/"state.pt"))
 assert restored.environment_steps==2 and restored.update_index==1
 assert np.array_equal(source.current["local_obs"],restored.current["local_obs"])
 assert source.vector.envs[0].env.decision_step==restored.vector.envs[0].env.decision_step
 source.writer.close();restored.writer.close()


def test_full_resume_reproduces_next_update(tmp_path):
 c=load_mappo_config("configs/mappo_smoke_1v1.yaml");c.update(num_envs=1,rollout_length=2,ppo_epochs=1,num_mini_batches=1,device="cpu",run_id="origin")
 origin=MAPPORunner(c,"resume_update",tmp_path);origin._save("point.pt");checkpoint=origin.output_dir/"checkpoints"/"point.pt";origin.writer.close()
 results=[]
 for run_id in ("branch_a","branch_b"):
  branch_config=dict(c);branch_config["run_id"]=run_id;runner=MAPPORunner(branch_config,"resume_update",tmp_path);runner.resume(str(checkpoint));buffer,_=runner.collect();metrics=runner.trainer.update(buffer);results.append((buffer.actions.copy(),metrics,[p.detach().clone() for p in runner.actor.parameters()],[p.detach().clone() for p in runner.critic.parameters()]));runner.writer.close()
 assert np.array_equal(results[0][0],results[1][0])
 assert results[0][1]==results[1][1]
 assert all(torch.equal(a,b) for a,b in zip(results[0][2],results[1][2]))
 assert all(torch.equal(a,b) for a,b in zip(results[0][3],results[1][3]))


def test_v2_actor_only_allowed_but_full_resume_rejected(tmp_path):
 a=SharedActor(11);c=CentralizedCritic(10,1);ao=torch.optim.Adam(a.parameters());co=torch.optim.Adam(c.parameters());n=ValueNormalizer();p=tmp_path/"v2.pt"
 save_checkpoint(p,a,c,ao,co,n,{},0,0,None);data=torch.load(p,weights_only=False);data["version"]=2;torch.save(data,p)
 load_checkpoint(p,SharedActor(11),actor_only=True)
 with pytest.raises(ValueError,match="v2 critic value semantics are incompatible with v3 physical-value critic"):
  load_checkpoint(p,SharedActor(11),CentralizedCritic(10,1),normalizer=ValueNormalizer())
