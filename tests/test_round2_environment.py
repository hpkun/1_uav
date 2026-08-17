import copy
from pathlib import Path
import numpy as np
import pytest
import yaml

from uav_combat.environment.death import DeathCause, death_summary
from uav_combat.environment.env import PaperUAVCombatEnv
from uav_combat.models import AircraftState


ROOT=Path(__file__).resolve().parents[1]


def configured_env(sensor_noise=False):
    env=PaperUAVCombatEnv(ROOT/"configs/paper_environment.yaml",sensor_noise=sensor_noise); env.reset(1); return env


def set_deaths(env,red_causes,blue_causes):
    env.red_death_causes[:]=red_causes; env.blue_death_causes[:]=blue_causes
    for states,causes in ((env.red,red_causes),(env.blue,blue_causes)):
        for s,c in zip(states,causes): s.alive=c==DeathCause.NONE


@pytest.mark.parametrize("blue_causes",[
    [DeathCause.ATTACK]*4,
    [DeathCause.ATTACK]*3+[DeathCause.BOUNDARY],
    [DeathCause.BOUNDARY]*4,
])
def test_all_blue_death_causes_produce_red_win(blue_causes):
    env=configured_env(); set_deaths(env,[DeathCause.NONE]*4,blue_causes)
    assert env._outcome(True,False)=="red_win"


def test_mutual_elimination_draw():
    env=configured_env(); set_deaths(env,[DeathCause.ATTACK]*4,[DeathCause.ATTACK]*4)
    assert env._outcome(True,False)=="draw_mutual_elimination"


def test_all_red_dead_produces_blue_win():
    env=configured_env(); set_deaths(env,[DeathCause.ATTACK]*4,[DeathCause.NONE]*4)
    assert env._outcome(True,False)=="blue_win"


def test_timeout_is_not_red_win():
    env=configured_env(); assert env._outcome(False,True)=="timeout"


def test_boundary_death_does_not_increment_attack_kills():
    env=configured_env(); env.blue[0].x=6000; before=env.red_attack_kills; env._resolve_boundaries()
    assert env.blue_death_causes[0]==DeathCause.BOUNDARY and env.red_attack_kills==before


def test_death_ledger_conservation_no_duplicate_and_symmetry():
    env=configured_env(); assert env._record_death("blue",0,DeathCause.ATTACK); assert not env._record_death("blue",0,DeathCause.BOUNDARY)
    env.red_attack_kills=1; summary=death_summary(env.blue_death_causes)
    assert sum(summary.values())==4 and env.red_attack_kills==summary["attack_deaths"]


def test_simultaneous_mutual_hit_both_apply():
    env=configured_env(); red_hits,blue_hits=env._apply_simultaneous_hits([(0,0)],[(0,0)])
    assert red_hits==[(0,0)] and blue_hits==[(0,0)]
    assert env.blue_death_causes[0]==DeathCause.ATTACK and env.red_death_causes[0]==DeathCause.ATTACK


def test_duplicate_hit_proposals_create_one_death_and_one_kill():
    env=configured_env(); credited,_=env._apply_simultaneous_hits([(0,0),(1,0)],[])
    assert credited==[(0,0)] and env.red_attack_kills==1 and death_summary(env.blue_death_causes)["attack_deaths"]==1


def test_both_team_attack_kill_death_symmetry():
    env=configured_env(); env._apply_simultaneous_hits([(0,0),(1,1)],[(0,0)])
    assert env.red_attack_kills==death_summary(env.blue_death_causes)["attack_deaths"]
    assert env.blue_attack_kills==death_summary(env.red_death_causes)["attack_deaths"]


def test_pre_attack_reward_target_survives_current_step_then_switches(monkeypatch):
    env=configured_env();
    for i in range(1,4): env._record_death("red",i,DeathCause.BOUNDARY)
    for i in (2,3): env._record_death("blue",i,DeathCause.BOUNDARY)
    env.red[0]=AircraftState(0,0,-3000,200,0,0,True); env.blue[0]=AircraftState(1000,0,-3000,200,0,np.pi,True); env.blue[1]=AircraftState(-1500,0,-3000,200,0,0,True)
    monkeypatch.setattr(env,"_hit_proposals",lambda r,b: ([(0,0)],[]))
    _,_,_,_,info=env.step(np.zeros((4,3)),np.zeros((4,3)))
    # The reciprocal geometry also activates R42=-0.15, so the frozen
    # pre-attack R41+R42 value is -0.05 and the kill makes 9.95.
    assert info["reward_target_indices"][0]==0 and info["local_rewards"][0]==pytest.approx(9.95)
    monkeypatch.setattr(env,"_hit_proposals",lambda r,b: ([],[])); _,_,_,_,next_info=env.step(np.zeros((4,3)),np.zeros((4,3)))
    assert next_info["reward_target_indices"][0]==1


def test_pre_attack_threat_reward_combines_with_own_death(monkeypatch):
    env=configured_env(); env.red[0]=AircraftState(0,0,-3000,200,0,0,True); env.blue[0]=AircraftState(1000,0,-3000,200,0,np.pi,True)
    monkeypatch.setattr(env,"_hit_proposals",lambda r,b: ([],[(0,0)])); _,_,_,_,info=env.step(np.zeros((4,3)),np.zeros((4,3)))
    assert info["red_death_causes"][0]==DeathCause.ATTACK and info["local_rewards"][0] < -10


def test_blue_observation_supplied_action_and_fixed_fallback():
    env=configured_env(); _,info0=env.reset(9); assert info0["blue_observations"].shape==(4,45)
    supplied=np.full((4,3),.25,np.float32); _,_,_,_,info=env.step(np.zeros((4,3)),supplied); assert np.allclose(info["executed_blue_actions"],supplied)
    a=configured_env(); b=configured_env(); a.reset(12); b.reset(12); fixed=np.stack([b.fixed_policy.action(s,b.red)[0] for s in b.blue])
    out_a=a.step(np.zeros((4,3)),None); out_b=b.step(np.zeros((4,3)),fixed); assert np.allclose(out_a[0],out_b[0])


def test_supplied_blue_action_changes_motion_from_fixed_policy():
    a=configured_env(); b=configured_env(); a.reset(33); b.reset(33)
    a.step(np.zeros((4,3)),np.ones((4,3))); b.step(np.zeros((4,3)),None)
    assert not np.allclose([s.psi for s in a.blue],[s.psi for s in b.blue])


def test_mirrored_observation_and_self_play_smoke():
    from uav_combat.madsac import MADSACTrainer
    env=configured_env(); obs,info=env.reset(5); trainer=MADSACTrainer(hidden_dim=32,replay_capacity=16,batch_size=2)
    blue=info["blue_observations"]
    for _ in range(5):
        red_actions=trainer.act(obs,env.red_alive_mask,True); blue_actions=trainer.act(blue,env.blue_alive_mask,True)
        obs,_,term,trunc,step_info=env.step(red_actions,blue_actions); blue=step_info["blue_observations"]
        if term or trunc: break
    assert obs.shape==blue.shape==(4,45) and np.isfinite(obs).all() and np.isfinite(blue).all()


def test_controller_config_is_runtime_source_of_truth():
    cfg=yaml.safe_load((ROOT/"configs/paper_environment.yaml").read_text(encoding="utf-8")); modified=copy.deepcopy(cfg); modified["reproduction_assumptions"]["controller"]["k_yaw"]=.1
    normal=PaperUAVCombatEnv(cfg,sensor_noise=False); slow=PaperUAVCombatEnv(modified,sensor_noise=False); s=AircraftState(0,0,0,200,0,0)
    target=normal.controller.action_to_target(s,np.array([.5,0,0]),normal.spec)
    assert abs(slow.controller.compute_control(s,target,slow.spec).phi)<abs(normal.controller.compute_control(s,target,normal.spec).phi)


def test_formal_sensor_enabled_and_dt_paper_audit():
    env=PaperUAVCombatEnv(ROOT/"configs/paper_environment.yaml"); assert env.sensor.enabled and not env.sensor_noise_test_override and env.dt==.1
    assert '("dt", env["simulation"]["dt"], "PAPER")' in (ROOT/"scripts/audit_paper_environment.py").read_text(encoding="utf-8")
