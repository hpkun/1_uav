"""Focused tests for the Actor-only GRU History development protocol."""
from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest
import torch

from algorithm.modular_mappo.buffer import recurrent_batch_plan
from algorithm.modular_mappo.evaluation import evaluate_modular_episode
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import checkpoint_architecture
from algorithm.train_modular_mappo import load_config
from tools.preflight_actor_only_gru_history import CONFIG,ENV,MANIFEST,REGISTRY,ROOT,SEEDS,source_lifecycle_audit,synthetic_rollout,validate,_yaml

def config():return load_config(CONFIG)
def trainer(hidden=32,epochs=1):
    cfg=config();cfg["training"]["ppo_epochs"]=epochs;cfg["training"]["minibatch_size"]=64
    return build_modular_mappo_trainer(cfg,"cpu",hidden,1000)

def test_01_only_actor_lr_decay_and_recurrent_are_enabled():
    cfg=config();assert sorted(k for k,v in cfg["modules"].items() if v.get("enabled",False))==["actor_lr_decay","recurrent_memory"]

def test_02_actor_has_zero_context():assert checkpoint_architecture(trainer())["actor_context_dim"]==0
def test_03_critic_has_zero_context():assert checkpoint_architecture(trainer())["critic_context_dim"]==0

def test_04_actor_gru_is_128_after_formal_resolution():
    t=build_modular_mappo_trainer(config(),"cpu",256,3_000_000);assert isinstance(t.actor.gru,torch.nn.GRUCell) and t.actor.gru.input_size==256 and t.actor.gru.hidden_size==128

def test_05_critic_is_feed_forward_without_gru():
    t=trainer();assert not hasattr(t.critic,"gru") and t.critic.recurrent_hidden_dim==0 and checkpoint_architecture(t)["critic_gru_hidden_dim"]==0

def test_06_initial_actor_hidden_is_zero():
    actor,_=trainer().initial_hidden(3);assert actor.shape==(3,4,128) and not np.any(actor)

def test_07_initial_critic_hidden_is_none():assert trainer().initial_hidden(2)[1] is None

def test_08_survivor_hidden_is_retained_across_wave():
    t=trainer();hidden=np.ones((1,4,128),np.float32);result=t.recurrent.apply_alive(hidden,np.asarray([[1,1,0,1]],np.float32));assert np.all(result[0,[0,1,3]]==1) and not np.any(result[0,2])

def test_09_dead_agent_hidden_is_zero_and_stays_zero():
    t=trainer();hidden=np.ones((1,4,128),np.float32);dead=np.asarray([[1,0,1,1]],np.float32);first=t.recurrent.apply_alive(hidden,dead);second=t.recurrent.apply_alive(first,dead);assert not np.any(first[0,1]) and not np.any(second[0,1])

def test_10_episode_done_resets_environment_hidden():
    t=trainer();hidden=np.ones((2,4,128),np.float32);t.recurrent.reset_for_episode(hidden,np.asarray([False,True]));assert np.all(hidden[0]==1) and not np.any(hidden[1])

def test_11_sequence_length_is_frozen_to_32():assert config()["modules"]["recurrent_memory"]["sequence_length"]==32

def test_12_chunk_and_minibatch_arithmetic():assert recurrent_batch_plan(256,24,32,512,10)=={"sequence_chunks":192,"sequences_per_minibatch":16,"recurrent_minibatches_per_epoch":12,"optimizer_steps":120}

def test_13_recurrent_update_is_finite_and_actor_gru_learns():
    t=trainer();before={k:v.clone() for k,v in t.actor.gru.state_dict().items()};m=t.update(synthetic_rollout(t));assert all(np.isfinite(float(v)) for v in m.values()) and m["actor_gru_grad_norm"]>0 and any(not torch.equal(v,t.actor.gru.state_dict()[k]) for k,v in before.items())

def test_14_critic_gru_gradient_is_exactly_zero():
    t=trainer();m=t.update(synthetic_rollout(t));assert m["critic_gru_grad_norm"]==0

def test_15_checkpoint_roundtrip_preserves_gru_and_protocol(tmp_path):
    t=trainer();path=tmp_path/"gru.pt";t.save(path);restored=trainer();restored.load(path);state=torch.load(path,map_location="cpu",weights_only=False);assert all(torch.equal(v,restored.actor.gru.state_dict()[k]) for k,v in t.actor.gru.state_dict().items()) and state["module_config"]["recurrent_memory"]==config()["modules"]["recurrent_memory"]

def test_16_deterministic_recurrent_evaluation_runs_without_mission_context():
    t=trainer();record=evaluate_modular_episode(t,_yaml(ENV),88_500_091,include_trace=True);assert len(record["action_trace"])==len(record["wave_trace"])>0 and np.all(np.isfinite(record["action_trace"]))

def test_17_source_lifecycle_and_lr_contract():
    assert all(source_lifecycle_audit().values());t=trainer()
    for step,expected in ((0,.0003),(600000,.0003),(750000,.0002),(900000,.0001),(3000000,.0001)):assert t.actor_lr_decay.learning_rate(step,.0003)==pytest.approx(expected)

def test_18_manifest_45m_launcher_analyzer_and_preflight():
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"));registry=json.loads(REGISTRY.read_text(encoding="utf-8"));assert manifest["training"]["seeds"]==list(SEEDS) and manifest["future_final"]["executed"] is False and registry["evaluation_ranges"]["45000000..45000199"]["executed"] is False
    launcher=(ROOT/"tools/run_actor_only_gru_history_3m.sh").read_text(encoding="utf-8");analyzer=(ROOT/"tools/analyze_actor_only_gru_history.py").read_text(encoding="utf-8");assert "for seed in 5301 5302 5303" in launcher and "nohup" not in launcher and "evaluate_modular" not in analyzer and "torch" not in analyzer
    assert validate(check_outputs=False)["status"]=="READY_FOR_ACTOR_ONLY_GRU_HISTORY_3M"
