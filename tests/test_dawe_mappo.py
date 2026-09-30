from __future__ import annotations
import copy,json
from dataclasses import fields
from pathlib import Path
import numpy as np
import pytest
import torch
from algorithm.modules import DeploymentAlignedWaveExplorationModule
from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import validate_dawe_branch,validate_dawe_config_pair
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer,stable_ratio_terms
from algorithm.train_modular_mappo import load_config

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"outputs/diag_mappo_learnability/l3_seed5301/checkpoint_1505280.pt"
def configs():return (load_config(ROOT/"configs/dev_dawe_fixed10_control_300k.yaml"),load_config(ROOT/"configs/dev_dawe_fixed10_v1_300k.yaml"))
def module(enabled=True):return DeploymentAlignedWaveExplorationModule({"enabled":enabled,"mode":"fixed_wave_std_multiplier","wave1_multiplier":.25,"wave2_multiplier":.25,"wave3_multiplier":1.})
def trainer(enabled=True,device="cpu"):
 c=configs()[1 if enabled else 0];return build_modular_mappo_trainer(c,device,16,1_805_280)

def test_disabled_distribution_is_object_identity():
 base=torch.distributions.Normal(torch.zeros(2,4,3),torch.ones(2,4,3));assert module(False).effective_distribution(base,None) is base
def test_exact_v1_config_is_accepted():assert module(True).multipliers==(.25,.25,1.)
def test_enabled_extra_config_rejected():
 cfg={"enabled":True,"mode":"fixed_wave_std_multiplier","wave1_multiplier":.25,"wave2_multiplier":.25,"wave3_multiplier":1.,"schedule":True}
 with pytest.raises(ValueError):DeploymentAlignedWaveExplorationModule(cfg)
def test_nonpositive_multiplier_rejected():
 with pytest.raises(ValueError):DeploymentAlignedWaveExplorationModule({"enabled":False,"wave1_multiplier":0})
def test_invalid_wave_rejected():
 with pytest.raises(ValueError):module().multiplier(0)
def test_batch_wave_broadcast():
 base=torch.distributions.Normal(torch.zeros(3,4,3),torch.ones(3,4,3));effective=module().effective_distribution(base,torch.tensor([1,2,3]));assert effective.scale.shape==(3,4,3)
def test_wave1_std_quarter():
 base=torch.distributions.Normal(torch.zeros(1,4,3),torch.full((1,4,3),2.));assert torch.equal(module().effective_distribution(base,torch.tensor([1])).scale,torch.full((1,4,3),.5))
def test_wave2_std_quarter():
 base=torch.distributions.Normal(torch.zeros(1,4,3),torch.full((1,4,3),2.));assert torch.equal(module().effective_distribution(base,torch.tensor([2])).scale,torch.full((1,4,3),.5))
def test_wave3_std_identity():
 base=torch.distributions.Normal(torch.zeros(1,4,3),torch.full((1,4,3),2.));assert torch.equal(module().effective_distribution(base,torch.tensor([3])).scale,base.scale)
def test_wave_batch_mismatch_rejected():
 base=torch.distributions.Normal(torch.zeros(2,4,3),torch.ones(2,4,3))
 with pytest.raises(ValueError):module().effective_distribution(base,torch.tensor([1]))
def test_deterministic_action_parity():
 c,d=trainer(False),trainer(True);d.actor.load_state_dict(c.actor.state_dict());obs=np.ones((3,4,52),"f");alive=np.ones((3,4),"f");waves=np.array([1,2,3]);assert np.array_equal(c.act(obs,alive,True,True,wave_indices=waves)[0],d.act(obs,alive,True,True,wave_indices=waves)[0])
def test_same_rng_residual_scaling_cpu():
 c,d=trainer(False),trainer(True);d.actor.load_state_dict(c.actor.state_dict());obs=np.ones((3,4,52),"f");alive=np.ones((3,4),"f");waves=np.array([1,2,3]);state=torch.get_rng_state();torch.set_rng_state(state);cr=c.act(obs,alive,False,True,wave_indices=waves)[1];torch.set_rng_state(state);dr=d.act(obs,alive,False,True,wave_indices=waves)[1]
 with torch.no_grad():base,_=d.actor.distribution_step(torch.tensor(obs),None,None,None,torch.tensor(alive));mu=base.loc.numpy()
 assert np.allclose(dr[:2]-mu[:2],.25*(cr[:2]-mu[:2]),atol=1e-6);assert np.allclose(dr[2],cr[2],atol=1e-6)
def test_rollout_update_logprob_identity_all_waves():
 d=trainer(True);obs=torch.randn(3,4,52);alive=torch.ones(3,4);base,_=d.actor.distribution_step(obs,None,None,None,alive);eff=d._effective_actor_distribution(base,torch.tensor([1,2,3]));raw=eff.rsample();action=torch.tanh(raw);old=d.actor._squashed_log_prob(eff,raw,action);new=d.actor._squashed_log_prob(d._effective_actor_distribution(base,torch.tensor([1,2,3])),raw,action);_,ratio=stable_ratio_terms(new,old);assert torch.allclose(new,old) and torch.allclose(ratio,torch.ones_like(ratio))
def test_base_distribution_would_break_scaled_logprob_identity():
 d=trainer(True);obs=torch.randn(1,4,52);base,_=d.actor.distribution_step(obs,None,None,None,torch.ones(1,4));eff=d._effective_actor_distribution(base,torch.tensor([1]));raw=eff.rsample();act=torch.tanh(raw);assert not torch.allclose(d.actor._squashed_log_prob(eff,raw,act),d.actor._squashed_log_prob(base,raw,act))
def test_squashed_jacobian_uses_existing_actor_implementation():
 source=Path(__import__("algorithm.modular_mappo.trainer",fromlist=["x"]).__file__).read_text();assert "self.actor._squashed_log_prob(dist,raw,act)" in source
def test_effective_distribution_construction_does_not_consume_rng():
 base=torch.distributions.Normal(torch.zeros(3,4,3),torch.ones(3,4,3));state=torch.get_rng_state().clone();module().effective_distribution(base,torch.tensor([1,2,3]));assert torch.equal(state,torch.get_rng_state())
def test_rollout_schema_has_no_dawe_field():assert not any("dawe" in item.name for item in fields(ModularRolloutBatch))
def test_runner_uses_pre_wave_for_action():
 source=(ROOT/"algorithm/modular_mappo/runner.py").read_text();assert "wave_indices=pre_wave" in source
def test_loss_step_accepts_wave_indices():
 import inspect;assert "wave_indices" in inspect.signature(ModularMAPPOTrainer._loss_step).parameters
def test_training_cli_dispatches_both_dawe_interventions():
 source=(ROOT/"algorithm/train_modular_mappo.py").read_text();assert 'validate_dawe_branch if intervention in {"dawe_fixed10_control","deployment_aligned_wave_exploration"}' in source
def test_control_enabled_modules_exact():assert sorted(k for k,v in configs()[0]["modules"].items() if v.get("enabled"))==["actor_gradient_clipping","actor_lr_decay"]
def test_dawe_enabled_modules_exact():assert sorted(k for k,v in configs()[1]["modules"].items() if v.get("enabled"))==["actor_gradient_clipping","actor_lr_decay","deployment_aligned_wave_exploration"]
def test_config_pair_has_only_registered_differences():assert validate_dawe_config_pair(*configs())
def test_config_pair_unregistered_difference_rejected():
 c,d=configs();d["training"]["entropy_coefficient"]=.02
 with pytest.raises(RuntimeError):validate_dawe_config_pair(c,d)
def test_branch_validator_both_arms():
 state=torch.load(SOURCE,map_location="cpu",weights_only=False);env=load_config(ROOT/"configs/persistent_wave_v2_environment.yaml")
 for cfg in configs():assert validate_dawe_branch(state,env,cfg,{"training_seed":5301,"training_num_envs":24,"training_smoke":False})["target_sampled_steps"]==1_805_280
def test_branch_validator_rejects_wrong_source_step():
 state=torch.load(SOURCE,map_location="cpu",weights_only=False);state["sampled_steps"]-=1;env=load_config(ROOT/"configs/persistent_wave_v2_environment.yaml")
 with pytest.raises(RuntimeError):validate_dawe_branch(state,env,configs()[1],{"training_seed":5301,"training_num_envs":24,"training_smoke":False})
def test_branch_requires_optimizer_and_rng_restore():
 c=configs()[0];c["development_branch"]["rng_restore"]=False;state=torch.load(SOURCE,map_location="cpu",weights_only=False);env=load_config(ROOT/"configs/persistent_wave_v2_environment.yaml")
 with pytest.raises(RuntimeError):validate_dawe_branch(state,env,c,{"training_seed":5301,"training_num_envs":24,"training_smoke":False})
def test_45m_is_declared_untouched():
 manifest=json.loads((ROOT/"experiments/dawe_fixed10_300k_manifest.json").read_text());assert manifest["reserved_future_final_test"]=={"seed_start":45000000,"seed_end":45000199,"executed":False}
def test_expected_update_arithmetic_is_preregistered():
 source=(ROOT/"tools/preflight_dawe_fixed10_300k.py").read_text();assert '"expected_ppo_updates":49' in source and '"expected_actor_optimizer_steps":5860' in source
def test_incompatible_module_rejected():
 c=configs()[1];c["modules"]["wave_specific_mean_heads"]={"enabled":True}
 with pytest.raises(ValueError):build_modular_mappo_trainer(c,"cpu",16,1_805_280)
def test_checkpoint_roundtrip_is_self_describing(tmp_path):
 d=trainer(True);path=tmp_path/"dawe.pt";d.save(path);state=torch.load(path,map_location="cpu",weights_only=False);assert state["deployment_aligned_wave_exploration_state"]["version"]==1;trainer(True).load(path,strict_protocol=True,restore_rng=False)
@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
def test_cuda_same_rng_scaling_and_deterministic_parity():
 c,d=trainer(False,"cuda"),trainer(True,"cuda");d.actor.load_state_dict(c.actor.state_dict());obs=np.ones((3,4,52),"f");alive=np.ones((3,4),"f");waves=np.array([1,2,3]);assert np.array_equal(c.act(obs,alive,True,True,wave_indices=waves)[0],d.act(obs,alive,True,True,wave_indices=waves)[0]);state=torch.cuda.get_rng_state();torch.cuda.set_rng_state(state);cr=c.act(obs,alive,False,True,wave_indices=waves)[1];torch.cuda.set_rng_state(state);dr=d.act(obs,alive,False,True,wave_indices=waves)[1]
 with torch.no_grad():base,_=d.actor.distribution_step(torch.tensor(obs,device="cuda"),None,None,None,torch.tensor(alive,device="cuda"));mu=base.loc.cpu().numpy()
 assert np.allclose(dr[:2]-mu[:2],.25*(cr[:2]-mu[:2]),atol=2e-5) and np.allclose(dr[2],cr[2],atol=2e-5)
