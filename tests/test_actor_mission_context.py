"""Focused protocol tests for the actor-only mission_markov development method."""
from __future__ import annotations

import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch

from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import checkpoint_architecture
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.modules.wave_survival_pbrs import mission_context_numpy,mission_progress_from_wave_state
from algorithm.train_modular_mappo import load_config
from tools.preflight_actor_mission_context import ENV,MANIFEST,load_yaml,resolve_protocol
from tools.analyze_actor_mission_context import development_label

ROOT=Path(__file__).resolve().parents[1]

def configs():return load_config(ROOT/"configs/diag_mappo_learnability_common_3m.yaml"),load_config(ROOT/"configs/dev_actor_mission_context_3m.yaml")

def test_actor_only_mission_markov_architecture_is_exact():
    base,method=configs();b=build_modular_mappo_trainer(base,"cpu",256,3_000_000);m=build_modular_mappo_trainer(method,"cpu",256,3_000_000)
    ba,ma=checkpoint_architecture(b),checkpoint_architecture(m)
    assert m.wave_context.context_dim==5 and ma["actor_context_dim"]==5 and ma["actor_input_dim"]==57
    assert ma["critic_context_dim"]==0 and ma["critic_input_dim"]==ba["critic_input_dim"]
    assert ma["critic_parameter_count"]==ba["critic_parameter_count"]
    assert m.actor.backbone[0].in_features==57 and m.actor.backbone[0].out_features==256
    assert m.actor.backbone[2].in_features==256 and m.actor.backbone[2].out_features==256
    assert not m.recurrent.enabled and not m.entity_attention_enabled and not m.curriculum.enabled

def test_actor_requires_context_but_actor_only_critic_does_not():
    _,method=configs();trainer=build_modular_mappo_trainer(method,"cpu",16,10)
    obs=np.zeros((1,4,52),np.float32);alive=np.ones((1,4),np.float32)
    with pytest.raises(ValueError,match="actor wave context is required"):trainer.act(obs,alive,True)
    values,_=trainer.values_step(obs,alive,context=None)
    assert values.shape==(1,4) and trainer._ctx(torch.ones(1,5),False) is None

def test_mission_context_one_hot_bounds_horizon_and_continuity():
    _,method=configs();trainer=build_modular_mappo_trainer(method,"cpu",16,10)
    total=np.asarray([3,3,3]);blue=np.ones((3,4),np.float32)
    ctx=mission_context_numpy(trainer,np.asarray([1,2,3]),total,blue,np.asarray([0,1500,3000]),3000)
    assert np.array_equal(ctx[:,:3],np.eye(3,dtype=np.float32))
    assert np.all((ctx[:,3:]>=0)&(ctx[:,3:]<=1)) and ctx[0,4]==1 and ctx[2,4]==0
    wave1_dead=np.zeros((1,4),np.float32);wave2_fresh=np.ones((1,4),np.float32)
    p1=mission_progress_from_wave_state(np.asarray([1]),wave1_dead,np.asarray([3]))
    p2=mission_progress_from_wave_state(np.asarray([2]),wave2_fresh,np.asarray([3]))
    assert p1[0]==pytest.approx(1/3) and p2[0]==pytest.approx(p1[0])
    progress=[]
    for alive_count in (4,3,2,1,0):
        mask=np.asarray([[1]*alive_count+[0]*(4-alive_count)],np.float32)
        progress.append(mission_progress_from_wave_state(np.asarray([1]),mask,np.asarray([3]))[0])
    assert progress==sorted(progress) and max(progress)<=1
    assert mission_progress_from_wave_state(np.asarray([99]),np.zeros((1,4),np.float32),np.asarray([3]))[0]<=1

def test_all_formal_context_paths_share_one_deterministic_definition():
    _,method=configs();trainer=build_modular_mappo_trainer(method,"cpu",16,10)
    args=(trainer,np.asarray([2]),np.asarray([3]),np.asarray([[1,0,1,0]],np.float32),np.asarray([900]),3000)
    values=[mission_context_numpy(*args) for _ in range(4)]
    assert all(np.array_equal(values[0],value) for value in values[1:]) and np.all(np.isfinite(values[0]))
    for relative in ("algorithm/modular_mappo/runner.py","algorithm/modular_mappo/evaluation.py",
                     "tools/record_combat_episode.py","tools/run_formal_holdout.py"):
        assert "mission_context_numpy" in (ROOT/relative).read_text(encoding="utf-8")

def test_method_changes_only_wave_context_and_not_environment_reward_or_blue_policy():
    base,method=configs();normalized=copy.deepcopy(method);normalized.pop("development_method")
    normalized["modules"]["wave_context"]=copy.deepcopy(base["modules"]["wave_context"])
    assert normalized==base
    env=load_yaml(ENV);snapshot=copy.deepcopy(env)
    trainer=build_modular_mappo_trainer(method,"cpu",16,10)
    raw=np.arange(4,dtype=np.float32)[None]
    adapted,_=trainer.reward_adapter.adapt(raw,[{}],np.asarray([1]))
    assert np.array_equal(adapted,raw) and env==snapshot
    assert env["reward"]==snapshot["reward"] and env["blue_policy"]==snapshot["blue_policy"]

def test_matched_seed_is_not_claimed_as_bit_identical_actor_initialization():
    base,method=configs();b=build_modular_mappo_trainer(base,"cpu",256,3_000_000);m=build_modular_mappo_trainer(method,"cpu",256,3_000_000)
    assert b.actor.backbone[0].weight.shape[1]==52 and m.actor.backbone[0].weight.shape[1]==57
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert "not bit-identical actor initialization" in manifest["paired_seed_note"]

def test_preflight_accepts_exact_protocol_without_output_check():
    result=resolve_protocol(check_outputs=False)
    assert result["status"]=="READY_FOR_ACTOR_MISSION_CONTEXT_DEVELOPMENT"
    assert result["architecture"]["actor_input_dim"]==57 and result["architecture"]["critic_context_dim"]==0

@pytest.mark.parametrize("target",["actor_critic","critic_only"])
def test_preflight_rejects_wrong_context_target(target):
    cfg=load_config(ROOT/"configs/dev_actor_mission_context_3m.yaml");cfg["modules"]["wave_context"]["context_target"]=target
    with pytest.raises(RuntimeError,match="actor_only mission_markov"):resolve_protocol(config=cfg,check_outputs=False)

@pytest.mark.parametrize("key,value",[("waves",2),("steps",2000)])
def test_preflight_rejects_nonfinal_environment(key,value):
    env=load_yaml(ENV)
    if key=="waves":env["persistent_waves"]["total_waves"]=value
    else:env["simulation"]["max_steps"]=value
    with pytest.raises(RuntimeError,match="environment must be"):resolve_protocol(env=env,check_outputs=False)

def test_preflight_rejects_future_final_execution():
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"));manifest["future_final"]["executed"]=True
    with pytest.raises(RuntimeError,match="future-final block"):resolve_protocol(manifest=manifest,check_outputs=False)

def test_preregistered_development_gate_labels_raw_aggregate():
    def agg(waves,wins,w1,q2,q3):return {"delta_average_waves":{"mean":waves,"paired_wins_of_3":wins},
        "delta_w1":{"mean":w1},"delta_q2":{"mean":q2},"delta_q3":{"mean":q3}}
    assert development_label(agg(.16,2,-.05,.01,-.01))=="ACTOR_MISSION_CONTEXT_PROMISING"
    assert development_label(agg(.10,2,0,.01,.01))=="ACTOR_MISSION_CONTEXT_WEAK_OR_MIXED"
    assert development_label(agg(.20,3,-.06,.1,.1))=="ACTOR_MISSION_CONTEXT_NOT_SUPPORTED"

def test_evaluation_log_q_ratios_are_safe_for_missing_or_zero_denominators():
    row={"sampled_steps":1,"clear_wave_1_probability":.5,"average_waves_cleared":.5,
         "average_return":0.,"average_red_loss":1.,"average_blue_loss":2.,
         "average_red_boundary_exits":0.,"average_red_ground_losses":0.}
    assert "Q2/Q3=nan/nan" in ModularMAPPOTrainingRunner.evaluation_log_line(row)
