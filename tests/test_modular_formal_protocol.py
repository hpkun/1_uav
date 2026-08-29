import copy,csv
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml

from env.factory import make_combat_environment
from algorithm.common.checkpoint import evaluation_selection_key
from algorithm.common.protocol import config_sha256
from algorithm.train_mappo import ensure_fresh_output_directory,resolve_runtime_settings
from algorithm.mappo.trainer import MAPPOTrainer,RolloutBatch
from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer,MODULAR_MAPPO_IMPL_VERSION
from algorithm.modular_mappo.protocol import validate_modular_checkpoint,is_formal_v2_checkpoint,checkpoint_architecture

ROOT=Path(__file__).resolve().parents[1]

def formal_state():
 config=yaml.safe_load((ROOT/"configs/modular_mappo_persistent.yaml").read_text());env=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text());trainer=ModularMAPPOTrainer(modules_config=config["modules"])
 extra={"environment_version":env["environment_version"],"environment_variant":env["environment_variant"],"environment_config_sha256":config_sha256(env),"algorithm_config_sha256":config_sha256(config),"network_architecture":checkpoint_architecture(trainer),"observation_dim":52,"action_dim":3,"num_agents":4,"training_seed":2023,"training_gamma":.999,"training_num_envs":24,"training_total_sampled_steps":1500000,"training_smoke":False,"algorithm_config":config}
 return trainer.checkpoint_state(extra),env,config

def test_fresh_checkpoint_is_formal_v2():
 state,env,config=formal_state();assert MODULAR_MAPPO_IMPL_VERSION==2 and state["modular_mappo_impl_version"]==2 and is_formal_v2_checkpoint(state);assert validate_modular_checkpoint(state,env,config)

def test_v2_resume_allowed_and_v1_formal_resume_rejected(tmp_path):
 trainer=ModularMAPPOTrainer();path=tmp_path/"v2.pt";trainer.save(path);ModularMAPPOTrainer().load(path)
 state,env,config=formal_state();state["modular_mappo_impl_version"]=1;assert not is_formal_v2_checkpoint(state)
 with pytest.raises(RuntimeError,match=r"checkpoint=1, current=2"):validate_modular_checkpoint(state,env,config)

def test_module_mismatch_and_baseline_checkpoint_rejected():
 state,env,config=formal_state();state["module_config_sha256"]="wrong"
 with pytest.raises(RuntimeError,match="module config mismatch"):validate_modular_checkpoint(state,env,config)
 state["algorithm"]="MAPPO"
 with pytest.raises(RuntimeError,match="algorithm mismatch"):validate_modular_checkpoint(state,env,config)

def test_modular_fresh_output_safety_before_artifacts(tmp_path):
 missing=tmp_path/"new";ensure_fresh_output_directory(missing);assert list(missing.iterdir())==[]
 empty=tmp_path/"empty";empty.mkdir();ensure_fresh_output_directory(empty)
 old=tmp_path/"old";old.mkdir();metric=old/"optimization_metrics.jsonl";metric.write_text("old\n")
 with pytest.raises(RuntimeError):ensure_fresh_output_directory(old)
 assert metric.read_text()=="old\n" and len(list(old.iterdir()))==1

def test_modular_resume_runtime_inherits_and_rejects_changes():
 config=yaml.safe_load((ROOT/"configs/modular_mappo_persistent.yaml").read_text())
 run={"seed":2024,"num_envs":3,"total_sampled_steps":100,"smoke":False,"device":"cpu"};state={"sampled_steps":60,"extra":{"training_total_sampled_steps":100}}
 inherited=resolve_runtime_settings(config,seed=None,num_envs=None,total_sampled_steps=None,device="cuda",smoke=None,run_config=run,checkpoint_state=state);assert (inherited["seed"],inherited["num_envs"],inherited["total_sampled_steps"],inherited["device"])==(2024,3,100,"cuda")
 for kwargs in ({"seed":2025},{"num_envs":4},{"total_sampled_steps":80}):
  values={"seed":None,"num_envs":None,"total_sampled_steps":None,"device":None,"smoke":None};values.update(kwargs)
  with pytest.raises(RuntimeError):resolve_runtime_settings(config,run_config=run,checkpoint_state=state,**values)
 extended=resolve_runtime_settings(config,seed=None,num_envs=None,total_sampled_steps=150,device=None,smoke=None,run_config=run,checkpoint_state=state);assert extended["extended_training_target"]

def test_best_selection_matches_baseline_for_both_variants():
 direct={"win_rate":.4,"average_return":5,"average_red_loss":2};assert evaluation_selection_key(direct,"direct_v2_3")==(.4,5.,-2.)
 persistent={"clear_wave_3_probability":.6,"average_waves_cleared":2.2,"average_return":10,"average_red_loss":1.5};assert evaluation_selection_key(persistent,"persistent_wave_v2")==(.6,2.2,10.,-1.5)

def test_resume_best_restored_from_history(tmp_path):
 rows=[{"sampled_steps":20,"clear_wave_3_probability":.3,"average_waves_cleared":1.,"average_return":1.,"average_red_loss":3.},
       {"sampled_steps":40,"clear_wave_3_probability":.6,"average_waves_cleared":2.,"average_return":2.,"average_red_loss":2.}]
 path=tmp_path/"evaluation_history.csv"
 with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 runner=object.__new__(ModularMAPPOTrainingRunner);runner.output_dir=tmp_path;runner.env_config={"environment_variant":"persistent_wave_v2"};runner.algorithm_config={};runner.evaluation_history=[];runner.best_evaluation=None;runner.best_sampled_steps=None
 runner.restore_best_from_disk(60);assert runner.best_sampled_steps==40
 worse={"clear_wave_3_probability":.2,"average_waves_cleared":3,"average_return":99,"average_red_loss":0};assert runner._evaluation_key(worse)<runner._evaluation_key(runner.best_evaluation)
 better={"clear_wave_3_probability":.7,"average_waves_cleared":0,"average_return":-99,"average_red_loss":4};assert runner._evaluation_key(better)>runner._evaluation_key(runner.best_evaluation)

def test_policy_anchor_checkpoint_is_self_contained(tmp_path):
 cfg={"policy_anchor":{"enabled":True,"coefficient":.01}}
 source=ModularMAPPOTrainer(hidden_dim=16,modules_config=cfg);reference=copy.deepcopy(source.actor);source.anchor.attach(reference,"deleted-source.pt");source.anchor_provenance={"reference_checkpoint":"deleted-source.pt"};path=tmp_path/"anchor.pt";source.save(path)
 restored=ModularMAPPOTrainer(hidden_dim=16,modules_config=cfg);restored.load(path);assert restored.anchor.reference_actor is not None and all(not p.requires_grad for p in restored.anchor.reference_actor.parameters())
 x=torch.randn(2,52);cur=restored.actor.distribution(x);ref=restored.anchor.reference_actor.distribution(x);_,metrics=restored.anchor.loss(cur,ref,0,torch.ones(2));assert metrics["anchor_kl"]<1e-7

def test_all_off_real_environment_chain_and_update_equivalence():
 config=yaml.safe_load((ROOT/"configs/combat_environment.yaml").read_text());env1=make_combat_environment(config);env2=make_combat_environment(config);o1,_=env1.reset(77);o2,_=env2.reset(77);assert np.array_equal(o1,o2)
 baseline=MAPPOTrainer(hidden_dim=16,ppo_epochs=1,minibatch_size=4,seed=5);modular=ModularMAPPOTrainer(hidden_dim=16,ppo_epochs=1,minibatch_size=4,seed=5,modules_config={});store={k:[] for k in ("observations","actions","raw_actions","old_log_probs","rewards","dones","alive_masks","next_observations","next_alive_masks")}
 for _ in range(4):
  alive=env1.red_alive_mask.copy();rng=torch.random.get_rng_state();a1,r1,l1=baseline.act(o1[None],alive[None],return_policy_data=True);torch.random.set_rng_state(rng);a2,r2,l2,_=modular.act(o2[None],alive[None],return_policy_data=True,context=np.zeros((1,0),np.float32),episode_mask=np.ones(1));assert np.array_equal(a1,a2) and np.array_equal(r1,r2) and np.array_equal(l1,l2)
  n1,re1,t1,tr1,i1=env1.step(a1[0]);n2,re2,t2,tr2,i2=env2.step(a2[0]);assert np.array_equal(re1,re2) and (t1,tr1)==(t2,tr2)
  for key,value in (("observations",o1),("actions",a1[0]),("raw_actions",r1[0]),("old_log_probs",l1[0]),("rewards",re1),("dones",float(t1 or tr1)),("alive_masks",alive),("next_observations",n1),("next_alive_masks",i1["red_alive_mask"])):store[key].append(value)
  o1,o2=n1,n2
 data={k:np.asarray(v)[:,None] for k,v in store.items()};base=RolloutBatch(**data);ctx=np.zeros((4,1,0),np.float32);mod=ModularRolloutBatch(**data,raw_environment_rewards=data["rewards"].copy(),wave_indices=np.ones((4,1),int),total_waves=np.ones((4,1),int),contexts=ctx,next_contexts=ctx,episode_masks=np.ones((4,1),np.float32));rng=torch.random.get_rng_state();baseline.update(base);torch.random.set_rng_state(rng);modular.update(mod)
 for p,q in zip(baseline.actor.parameters(),modular.actor.parameters()):assert torch.allclose(p,q,atol=2e-6,rtol=2e-6)
 for p,q in zip(baseline.critic.parameters(),modular.critic.parameters()):assert torch.allclose(p,q,atol=2e-6,rtol=2e-6)
