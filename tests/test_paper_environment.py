import numpy as np
import pytest

from uav_combat.controller import TargetStateController
from uav_combat.dynamics import PointMassDynamics
from uav_combat.environment.env import PaperUAVCombatEnv
from uav_combat.environment.fixed_policy import NearestTargetPursuitPolicy
from uav_combat.environment.geometry import PaperAirCombatGeometry, compute_paper_geometry
from uav_combat.environment.observation import build_observation
from uav_combat.environment.reward import equation25_reward
from uav_combat.environment.scenario import random_diameter_states
from uav_combat.environment.sensor import ObservedState, SensorModel
from uav_combat.environment.weapon import WeaponModel
from uav_combat.models import AircraftSpec, AircraftState, ControlCommand


def state(x=0, y=0, z=0, v=200, theta=0, psi=0, alive=True):
    return AircraftState(x, y, z, v, theta, psi, alive)


def test_equations_1_and_2():
    s = state(v=200, theta=0.2, psi=-0.3); c = ControlCommand(nx=1.2, nz=2.0, phi=0.4); g = 9.81
    got = PointMassDynamics(g).derivatives(s, c)
    expected = [s.v*np.cos(s.theta)*np.cos(s.psi), s.v*np.cos(s.theta)*np.sin(s.psi), -s.v*np.sin(s.theta), g*(c.nx-np.sin(s.theta)), g/s.v*(c.nz*np.cos(c.phi)-np.cos(s.theta)), g*c.nz*np.sin(c.phi)/(s.v*np.cos(s.theta))]
    assert np.allclose(got, expected)
    assert np.all(np.isfinite(PointMassDynamics().derivatives(state(v=0, theta=np.pi/2), c)))


def test_equation23_action_mapping_and_limits():
    spec = AircraftSpec(150,300,-np.pi/3,np.pi/3,-3,3,-6,6,-np.pi/2,np.pi/2,1,.7,50,1,1,1)
    controller = TargetStateController(); target = controller.action_to_target(state(v=280, theta=.9, psi=0), np.array([1,1,1]), spec)
    assert target.desired_psi == pytest.approx(-np.pi)
    assert target.desired_theta == pytest.approx(np.pi/3)
    assert target.desired_v == 300
    control = controller.compute_control(state(), target, spec)
    assert spec.phi_min <= control.phi <= spec.phi_max and spec.nx_min <= control.nx <= spec.nx_max and spec.nz_min <= control.nz <= spec.nz_max


@pytest.mark.parametrize("target, expected", [(state(100,0), (0, np.pi, 0)), (state(-100,0), (-np.pi, 0, 0)), (state(0,100), (np.pi/2,-np.pi/2,0))])
def test_ata_aa_horizontal_cases(target, expected):
    g = compute_paper_geometry(state(), target)
    assert g.ata == pytest.approx(expected[0]); assert abs(g.aa) == pytest.approx(abs(expected[1])); assert g.ha == pytest.approx(expected[2])


def test_ha_hca_and_zero_distance_stability():
    g = compute_paper_geometry(state(), state(100,0,-100,psi=np.pi))
    assert g.ha == pytest.approx(np.pi/4); assert abs(g.hca) == pytest.approx(np.pi)
    same = compute_paper_geometry(state(), state())
    assert same.distance == 0 and np.all(np.isfinite([same.ata,same.aa,same.ha,same.hca]))


def test_sensor_seed_clip_and_disabled():
    model = SensorModel(10,.1,2,1,1,1,True); s = state()
    a = model.observe(s,.2,np.random.default_rng(7)); b = model.observe(s,.2,np.random.default_rng(7)); c = model.observe(s,.2,np.random.default_rng(8))
    assert a == b and a != c
    assert abs(a.x-s.x) <= 10 and abs(a.psi-s.psi) <= .1 and abs(a.v-s.v) <= 2
    exact = SensorModel(10,.1,2,1,1,1,False).observe(s,.2,np.random.default_rng(1))
    assert exact == ObservedState(s.x,s.y,s.z,s.v,.2,s.psi,s.theta)


def test_weapon_equations_7_and_8_seeded():
    w = WeaponModel(100,4000,np.pi/6,np.pi/6,2000,.05,.05)
    assert w.can_fire(PaperAirCombatGeometry(4000,np.pi/6,0,np.pi/6,0))
    assert not w.can_fire(PaperAirCombatGeometry(4000,np.pi/6+1e-6,0,0,0))
    g = PaperAirCombatGeometry(2000,.1,0,.1,0)
    assert w.sample_hit(g,np.random.default_rng(4)) == w.sample_hit(g,np.random.default_rng(4))


def test_random_diameter_symmetry_and_4v4_reset():
    red, blue, _ = random_diameter_states(np.random.default_rng(2))
    assert len(red)==len(blue)==4
    assert np.allclose(np.mean([[s.x,s.y] for s in red],0), -np.mean([[s.x,s.y] for s in blue],0))
    env = PaperUAVCombatEnv(sensor_noise=False); obs, info = env.reset(3)
    assert obs.shape == (4,45) and info["red_ids"] == [f"red_{i}" for i in range(4)]
    next_obs, rewards, terminated, truncated, info = env.step(np.zeros((4,3)))
    assert next_obs.shape == (4,45) and rewards.shape == (4,) and not (terminated and truncated)


def test_nearest_policy_switches_and_dead_slot_zero():
    own = state(); targets = [state(100),state(50)]
    policy = NearestTargetPursuitPolicy(); assert policy.nearest_target_index(own,targets)==1
    targets[1].alive=False; assert policy.nearest_target_index(own,targets)==0
    observed = [ObservedState(i,0,0,200,0,0,0) for i in range(8)]
    obs = build_observation(0, observed[:4], observed[4:], [True,False,True,True], [True]*4)
    assert obs.shape==(45,) and np.all(obs[7:13] == 0)


@pytest.mark.parametrize("angle,reward", [(30,.01),(15,.02),(5,.10)])
def test_reward_r41_tiers_and_4000_boundary(angle,reward):
    a=np.deg2rad(angle); g=PaperAirCombatGeometry(4000,a,0,a,0)
    # At exactly 4000 m, Equation (25) includes both R3 (d>=4000)
    # and R41 (d<=4000).
    assert equation25_reward(g,None) == pytest.approx(reward + .001)
    far=PaperAirCombatGeometry(4001,np.deg2rad(30),0,np.deg2rad(30),0)
    assert equation25_reward(far,None)==pytest.approx(.001)


@pytest.mark.parametrize("angle,reward", [(30,-.015),(15,-.025),(5,-.15)])
def test_reward_r42_tiers(angle,reward):
    a=np.deg2rad(angle); g=PaperAirCombatGeometry(3999,a,0,a,0)
    assert equation25_reward(None,g)==pytest.approx(reward)


def test_reward_r1_r2_and_reverse_threat():
    assert equation25_reward(None,None,destroyed_blue=1)==10
    assert equation25_reward(None,None,red_destroyed=True)==-10
    assert equation25_reward(None,None,red_boundary_loss=True)==-10
    attack=PaperAirCombatGeometry(3999,0,0,0,0); threat=PaperAirCombatGeometry(3999,0,0,0,0)
    assert equation25_reward(attack,threat)==pytest.approx(-.05)


def test_action_lower_physical_ranges():
    spec = AircraftSpec(150,300,-np.pi/3,np.pi/3,-3,3,-6,6,-np.pi/2,np.pi/2,1,.7,50,1,1,1)
    target = TargetStateController().action_to_target(state(v=170), -np.ones(3), spec)
    assert target.desired_psi == pytest.approx(-np.pi) and target.desired_theta == pytest.approx(-np.pi/3) and target.desired_v == 150


def test_same_heading_hca_zero():
    assert compute_paper_geometry(state(), state(10, 0)).hca == 0


def test_opposite_heading_hca_pi():
    assert abs(compute_paper_geometry(state(), state(10, 0, psi=np.pi)).hca) == pytest.approx(np.pi)


def test_weapon_launch_minimum_boundary():
    w = WeaponModel(100,4000,np.pi/6,np.pi/6,2000,.05,.05)
    assert w.can_fire(PaperAirCombatGeometry(100,0,0,0,0)) and not w.can_fire(PaperAirCombatGeometry(99.999,0,0,0,0))


def test_reward_3999_has_no_r3():
    assert equation25_reward(PaperAirCombatGeometry(3999,0,np.pi,0,0),None) == 0


def test_reward_4001_has_no_r41():
    assert equation25_reward(PaperAirCombatGeometry(4001,0,0,0,0),None) == pytest.approx(.001)


def test_reset_seed_reproducibility():
    env=PaperUAVCombatEnv(); a,_=env.reset(17); b,_=env.reset(17)
    assert np.array_equal(a,b)


def test_paper_attack_range_and_counts():
    env=PaperUAVCombatEnv(sensor_noise=False)
    assert env.weapon.distance_max == 4000 and env.team_size == 4 and env.observation_dim == 45
