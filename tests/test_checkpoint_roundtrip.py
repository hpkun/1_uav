import torch
import numpy as np
import pytest
from uav_env.algorithms.mappo.checkpoint import save_checkpoint,load_checkpoint,schema_metadata
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


def test_checkpoint_schema_metadata_rejects_legacy_to_v2_resume(tmp_path):
 a=SharedActor(63);c=CentralizedCritic(61,3);ao=torch.optim.Adam(a.parameters());co=torch.optim.Adam(c.parameters());n=ValueNormalizer();p=tmp_path/"schema.pt"
 cfg={"environment":{"environment_schema_version":"homogeneous_3v3_v2_timeaware","observation_schema":"fixed_id_body_time_63d","global_state_schema":"full_entity_time_61d","reward_profile":"project_3v3_v2","scenario_profile":"head_on_mirrored_jitter_v2"}}
 meta=schema_metadata(cfg,63,61,3)
 save_checkpoint(p,a,c,ao,co,n,cfg,0,0,None,metadata=meta)
 load_checkpoint(p,SharedActor(63),CentralizedCritic(61,3),normalizer=ValueNormalizer(),expected_metadata=meta)
 legacy={**meta,"observation_schema":"legacy","obs_dim":45}
 with pytest.raises(ValueError,match="checkpoint schema mismatch"):
  load_checkpoint(p,SharedActor(63),CentralizedCritic(61,3),normalizer=ValueNormalizer(),expected_metadata=legacy)


def test_legacy_v3_without_schema_metadata_full_resume_only_to_legacy(tmp_path):
 a=SharedActor(45);c=CentralizedCritic(87,3);ao=torch.optim.Adam(a.parameters());co=torch.optim.Adam(c.parameters());n=ValueNormalizer();p=tmp_path/"legacy_no_schema.pt"
 cfg={"environment":{"kind":"3v3","scenario":"head_on_formation","opponent":"pursuit"}}
 save_checkpoint(p,a,c,ao,co,n,cfg,123,7,{"score":1.0},metadata=schema_metadata({"environment":{}},45,87,3))
 data=torch.load(p,weights_only=False);del data["schema_metadata"];torch.save(data,p)
 legacy_expected={"environment_schema_version":"legacy","observation_schema":"legacy","global_state_schema":"legacy","reward_profile":"legacy","scenario_profile":"legacy","obs_dim":45,"state_dim":87,"num_agents":3}
 restored=load_checkpoint(p,SharedActor(45),CentralizedCritic(87,3),normalizer=ValueNormalizer(),expected_metadata=legacy_expected)
 assert restored["environment_steps"]==123
 assert restored["update_index"]==7
 v2_expected={"environment_schema_version":"homogeneous_3v3_v2_timeaware","observation_schema":"fixed_id_body_time_63d","global_state_schema":"full_entity_time_61d","reward_profile":"project_3v3_v2","scenario_profile":"head_on_mirrored_jitter_v2","obs_dim":63,"state_dim":61,"num_agents":3}
 with pytest.raises(ValueError,match="legacy checkpoint without schema metadata cannot resume into homogeneous_3v3_v2_timeaware"):
  load_checkpoint(p,SharedActor(63),CentralizedCritic(61,3),normalizer=ValueNormalizer(),expected_metadata=v2_expected)


def test_old_62d_60d_v2_checkpoint_rejected_by_timeaware_schema(tmp_path):
 a=SharedActor(62);c=CentralizedCritic(60,3);ao=torch.optim.Adam(a.parameters());co=torch.optim.Adam(c.parameters());n=ValueNormalizer();p=tmp_path/"old_v2.pt"
 old_cfg={"environment":{"environment_schema_version":"homogeneous_3v3_v2","observation_schema":"fixed_id_body_62d","global_state_schema":"full_entity_60d","reward_profile":"project_3v3_v2","scenario_profile":"head_on_mirrored_jitter_v2"}}
 save_checkpoint(p,a,c,ao,co,n,old_cfg,0,0,None,metadata=schema_metadata(old_cfg,62,60,3))
 expected={"environment_schema_version":"homogeneous_3v3_v2_timeaware","observation_schema":"fixed_id_body_time_63d","global_state_schema":"full_entity_time_61d","reward_profile":"project_3v3_v2","scenario_profile":"head_on_mirrored_jitter_v2","obs_dim":63,"state_dim":61,"num_agents":3}
 with pytest.raises(ValueError,match="checkpoint schema mismatch"):
  load_checkpoint(p,SharedActor(63),CentralizedCritic(61,3),normalizer=ValueNormalizer(),expected_metadata=expected)


def test_critic_dimension_error_is_wrapped(tmp_path):
 a=SharedActor(45);c=CentralizedCritic(87,3);ao=torch.optim.Adam(a.parameters());co=torch.optim.Adam(c.parameters());n=ValueNormalizer();p=tmp_path/"critic_bad.pt"
 save_checkpoint(p,a,c,ao,co,n,{},0,0,None)
 with pytest.raises(ValueError,match="Critic dimensions are incompatible"):
  load_checkpoint(p,SharedActor(45),CentralizedCritic(61,3),normalizer=ValueNormalizer())
