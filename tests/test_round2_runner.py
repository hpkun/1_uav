from pathlib import Path
import numpy as np
import yaml

from uav_combat.training.runner import PaperTrainingRunner
from uav_combat.training.evaluator import evaluate
from uav_combat.training.vector_env import SyncVectorEnv


ROOT=Path(__file__).resolve().parents[1]


def configs():
    return yaml.safe_load((ROOT/"configs/paper_environment.yaml").read_text(encoding="utf-8")),yaml.safe_load((ROOT/"configs/madsac.yaml").read_text(encoding="utf-8"))


def test_formal_configuration_declares_real_24_env_protocol():
    _,a=configs(); assert a["training"]["num_train_envs"]==24 and a["training"]["evaluation_episodes"]==20


def test_24_env_shapes_steps_replay_and_distinct_seeds(tmp_path):
    e,a=configs(); runner=PaperTrainingRunner(e,a,num_envs=24,total_env_steps=24,output_dir=tmp_path,smoke=True)
    assert runner.observations.shape==(24,4,45) and len(set(runner.vector.last_reset_seeds))==24
    result=runner.vector_step()
    assert runner.trainer.sampled_env_steps==24 and runner.trainer.vector_steps==1 and runner.trainer.replay.size==24 and result["new_transitions"]==24


def test_deterministic_seed_allocation_and_independent_reset():
    e,_=configs(); v=SyncVectorEnv(2,e,base_seed=10,seed_stride=100)
    assert v.seed_for(0,0)==10 and v.seed_for(1,0)==11 and v.seed_for(0,1)==110
    v.reset(); v.envs[0].max_steps=1; result=v.step_batch(np.zeros((2,4,3),np.float32))
    assert v.episode_indices.tolist()==[1,0] and v.last_reset_seeds.tolist()==[110,11] and result.observations.shape==(2,4,45)


def test_update_data_schedule_invariant_to_env_count(tmp_path):
    e,a=configs(); one=PaperTrainingRunner(e,a,num_envs=1,total_env_steps=1,output_dir=tmp_path/"one",smoke=True); many=PaperTrainingRunner(e,a,num_envs=24,total_env_steps=24,output_dir=tmp_path/"many",smoke=True)
    total_one=sum(one._scheduled_updates(1) for _ in range(240)); total_many=sum(many._scheduled_updates(24) for _ in range(10))
    assert total_one==total_many==12


def test_five_run_output_and_initialization_separation(tmp_path):
    e,a=configs(); r0=PaperTrainingRunner(e,a,num_envs=1,total_env_steps=1,seed=3,run_id=0,output_dir=tmp_path,smoke=True); r1=PaperTrainingRunner(e,a,num_envs=1,total_env_steps=1,seed=4,run_id=1,output_dir=tmp_path,smoke=True)
    assert r0.output_dir!=r1.output_dir and r0.vector.base_seed!=r1.vector.base_seed
    assert not np.array_equal(next(r0.trainer.actor.parameters()).detach().numpy(),next(r1.trainer.actor.parameters()).detach().numpy())


def test_evaluation_seed_set_disjoint_and_fixed(tmp_path):
    e,a=configs(); r=PaperTrainingRunner(e,a,num_envs=4,total_env_steps=4,output_dir=tmp_path,smoke=True)
    assert len(r.evaluation_seeds)==20 and not set(r.vector.last_reset_seeds).intersection(r.evaluation_seeds)


def test_evaluation_does_not_change_trainer_rng_or_training_env(tmp_path):
    e,a=configs(); r=PaperTrainingRunner(e,a,num_envs=1,total_env_steps=1,output_dir=tmp_path,smoke=True)
    rng_before=repr(r.trainer.rng.bit_generator.state); position_before=r.vector.envs[0].red[0].as_array().copy()
    evaluate(r.trainer,e,[999999])
    assert repr(r.trainer.rng.bit_generator.state)==rng_before and np.array_equal(r.vector.envs[0].red[0].as_array(),position_before)


def test_learning_starts_cannot_be_less_than_batch(tmp_path):
    e,a=configs(); a["reproduction_assumptions"]["learning_starts"]=10
    import pytest
    with pytest.raises(ValueError): PaperTrainingRunner(e,a,num_envs=1,total_env_steps=1,output_dir=tmp_path,smoke=False)
