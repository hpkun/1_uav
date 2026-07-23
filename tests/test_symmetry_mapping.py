from math import pi
import pytest
from conftest import make_state
from uav_env.actions.discrete_15 import DiscreteAction15
from uav_env.actions.discrete_15 import get_control
from uav_env.combat.attack_geometry import AttackZoneConfig,compute_combat_geometry
from uav_env.combat.damage import DamageConfig,apply_damage
from uav_env.core.symmetry import mirror_action_xz,mirror_state_xz
from uav_env.dynamics.propagation import propagate_state

def test_action_and_state_mirror(profile):
 s=make_state(profile,y=7,heading=pi/3); m=mirror_state_xz(s)
 assert m.y==-7 and m.heading_angle==pytest.approx(5*pi/3)
 assert mirror_action_xz(DiscreteAction15.LEFT_ACCELERATE)==DiscreteAction15.RIGHT_ACCELERATE
 assert mirror_action_xz(DiscreteAction15.CLIMB_HOLD)==DiscreteAction15.CLIMB_HOLD

def test_left_right_dynamics_are_mirrored(profile):
 s=make_state(profile,y=13,heading=pi/5);m=mirror_state_xz(s);a=DiscreteAction15.LEFT_HOLD
 next_a=propagate_state(s,get_control(a),profile,.1,9.81);next_b=propagate_state(m,get_control(mirror_action_xz(a)),profile,.1,9.81);expected=mirror_state_xz(next_a)
 assert next_b.to_kinematic_vector().tolist()==pytest.approx(expected.to_kinematic_vector().tolist(),abs=1e-10)

def test_attack_and_escape_angles_are_reflection_invariant(profile,experiment_config):
 attacker=make_state(profile,x=10,y=20,heading=pi/7);target=make_state(profile,x=400,y=-80,heading=5*pi/6)
 config=AttackZoneConfig.from_config(experiment_config);a=compute_combat_geometry(attacker,target,config);b=compute_combat_geometry(mirror_state_xz(attacker),mirror_state_xz(target),config)
 assert b.distance==pytest.approx(a.distance);assert b.attacker_attack_angle==pytest.approx(a.attacker_attack_angle);assert b.target_escape_angle==pytest.approx(a.target_escape_angle)

def test_mirrored_health_trajectory_uses_same_damage_samples(profile):
 a=make_state(profile);b=mirror_state_xz(a);config=DamageConfig()
 for sample in (.05,.2,.6,.95):
  a,_=apply_damage(a,config,sample);b,_=apply_damage(b,config,sample);assert a.health==b.health and a.alive==b.alive
