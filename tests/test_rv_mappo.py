from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import torch
import pytest

from algorithm.modules import ReferenceVarianceModule
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import validate_rv_branch, validate_rv_config_pair
from algorithm.modular_mappo.trainer import stable_ratio_terms
from algorithm.train_modular_mappo import load_config

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"outputs/diag_mappo_learnability/l3_seed5301/checkpoint_1505280.pt"

def configs():
 return (load_config(ROOT/"configs/dev_rv_fixed10_control_300k.yaml"),load_config(ROOT/"configs/dev_rv_mappo_v1_300k.yaml"))
def trainer(enabled=True,device="cpu"):
 return build_modular_mappo_trainer(configs()[1 if enabled else 0],device,256,1_805_280)

def test_exact_module_protocol_and_invalid_values():
 ReferenceVarianceModule({"enabled":True,"mode":"frozen_source_state_dependent_variance","source_sampled_steps":1505280,"freeze_current_log_std_head":True,"behavior_uses_reference_mean":False})
 with pytest.raises(ValueError):ReferenceVarianceModule({"enabled":True,"mode":"other"})

def test_config_pair_and_exact_modules():
 c,r=configs();assert validate_rv_config_pair(c,r)
 assert sorted(k for k,v in c["modules"].items() if isinstance(v,dict) and v.get("enabled"))==["actor_gradient_clipping","actor_lr_decay"]
 assert sorted(k for k,v in r["modules"].items() if isinstance(v,dict) and v.get("enabled"))==["actor_gradient_clipping","actor_lr_decay","reference_variance"]

def test_pair_rejects_unregistered_difference():
 c,r=configs();r["training"]["entropy_coefficient"]=.02
 with pytest.raises(RuntimeError):validate_rv_config_pair(c,r)

def test_branch_validator_both_arms():
 state=torch.load(SOURCE,map_location="cpu",weights_only=False);env=load_config(ROOT/"configs/persistent_wave_v2_environment.yaml")
 for cfg in configs():assert validate_rv_branch(state,env,cfg,{"training_seed":5301,"training_num_envs":24,"training_smoke":False})["target_sampled_steps"]==1_805_280

def test_incompatible_dawe_rejected():
 cfg=configs()[1];cfg["modules"]["deployment_aligned_wave_exploration"]={"enabled":True,"mode":"fixed_wave_std_multiplier","wave1_multiplier":.25,"wave2_multiplier":.25,"wave3_multiplier":1.}
 with pytest.raises(ValueError):build_modular_mappo_trainer(cfg,"cpu",256,1_805_280)

def test_control_has_no_reference_and_identity_distribution():
 t=trainer(False);obs=torch.randn(3,4,52);alive=torch.ones(3,4);base,_=t.actor.distribution_step(obs,None,None,None,alive)
 assert t.reference_variance_actor is None and t._behavior_actor_distribution(base,obs,alive,torch.tensor([1,2,3])) is base

def test_branch_creates_frozen_reference_and_preserves_optimizer_membership():
 t=trainer(True);t.load(SOURCE,strict_protocol=False,restore_rng=False)
 assert t.reference_variance_actor_sha256()==t._module_state_sha256(t.actor)
 assert not t.reference_variance_actor.training and all(not p.requires_grad for p in t.reference_variance_actor.parameters())
 params={id(p) for g in t.actor_optimizer.param_groups for p in g["params"]}
 assert all(not p.requires_grad and id(p) in params for p in t.actor.log_std.parameters())

def test_behavior_uses_current_mean_reference_scale_and_ignores_wave():
 t=trainer(True);t.load(SOURCE,strict_protocol=False,restore_rng=False);obs=torch.randn(3,4,52);alive=torch.ones(3,4)
 current,_=t.actor.distribution_step(obs,None,None,None,alive);reference,_=t.reference_variance_actor.distribution_step(obs,None,None,None,alive)
 behavior=t._behavior_actor_distribution(current,obs,alive,torch.tensor([1,2,3]))
 assert torch.equal(behavior.loc,current.loc) and torch.equal(behavior.scale,reference.scale)
 assert torch.equal(behavior.scale,t._behavior_actor_distribution(current,obs,alive,torch.tensor([3,1,2])).scale)

def test_same_rng_control_rv_initial_stochastic_and_deterministic_parity():
 c=trainer(False);r=trainer(True);c.load(SOURCE,strict_protocol=False,restore_rng=False);r.load(SOURCE,strict_protocol=False,restore_rng=False)
 obs=torch.randn(3,4,52).numpy();alive=torch.ones(3,4).numpy();waves=torch.tensor([1,2,3]).numpy()
 state=torch.get_rng_state().clone();torch.set_rng_state(state);ca=c.act(obs,alive,False,True,wave_indices=waves);torch.set_rng_state(state);ra=r.act(obs,alive,False,True,wave_indices=waves)
 for i in range(3):assert (ca[i]==ra[i]).all()
 assert (c.act(obs,alive,True,True,wave_indices=waves)[0]==r.act(obs,alive,True,True,wave_indices=waves)[0]).all()

def test_loss_ratio_identity_and_gradient_semantics():
 t=trainer(True);t.load(SOURCE,strict_protocol=False,restore_rng=False);obs=torch.randn(3,4,52);alive=torch.ones(3,4);waves=torch.tensor([1,2,3])
 base,_=t.actor.distribution_step(obs,None,None,None,alive);dist=t._behavior_actor_distribution(base,obs,alive,waves);raw=dist.rsample();act=torch.tanh(raw);old=t.actor._squashed_log_prob(dist,raw,act);ones=torch.ones(3,4);zeros=torch.zeros(3,4)
 losses=t._loss_step(obs,act,raw,old,alive,ones,zeros,zeros,ones,torch.zeros(3,0),wave_indices=waves);_,ratio=stable_ratio_terms(losses[6],old);assert torch.allclose(ratio,torch.ones_like(ratio),atol=1e-6)
 t.actor_optimizer.zero_grad();(losses[0]-.01*losses[2]).backward()
 assert all(p.grad is None for p in t.actor.log_std.parameters()) and all(p.grad is None for p in t.reference_variance_actor.parameters())
 assert any(p.grad is not None and p.grad.abs().sum()>0 for p in list(t.actor.backbone.parameters())+list(t.actor.mean.parameters()))

def test_self_contained_checkpoint_and_missing_reference_fail_closed(tmp_path):
 t=trainer(True);t.load(SOURCE,strict_protocol=False,restore_rng=False);path=tmp_path/"rv.pt";t.save(path)
 restored=trainer(True);restored.load(path,strict_protocol=True,restore_rng=False);assert restored.reference_variance_actor_sha256()==t.reference_variance_actor_sha256()
 state=torch.load(path,map_location="cpu",weights_only=False);state["reference_variance_actor_state"]=None;bad=tmp_path/"bad.pt";torch.save(state,bad)
 with pytest.raises(RuntimeError):trainer(True).load(bad,strict_protocol=True,restore_rng=False)

def test_frozen_logstd_does_not_update():
 t=trainer(True);t.load(SOURCE,strict_protocol=False,restore_rng=False);before=[p.detach().clone() for p in t.actor.log_std.parameters()]
 t.actor_optimizer.zero_grad();sum(p.sum() for p in t.actor.mean.parameters()).backward();t.actor_optimizer.step()
 assert all(torch.equal(a,b) for a,b in zip(before,t.actor.log_std.parameters()))

def test_44m_and_45m_protocol_and_formal_dirs_absent():
 for cfg in configs():
  assert cfg["development_protocol"]["validation"]=={"seed_start":44000000,"seed_end":44000049,"episodes":50,"deterministic":True,"common_scenarios":True,"is_holdout":False}
  assert cfg["development_protocol"]["reserved_future_final_test"]=={"seed_start":45000000,"seed_end":45000199,"executed":False}
 assert not any((ROOT/f"outputs/dev_rv_{kind}_seed{seed}_300k").exists() for seed in (5301,5302,5303) for kind in ("control","v1"))

def test_runner_defers_reference_creation_until_after_rng_restore():
 text=(ROOT/"algorithm/modular_mappo/runner.py").read_text()
 assert text.index("self.trainer.restore_rng_state(state)") < text.index("self.trainer.finalize_reference_variance_branch()")
