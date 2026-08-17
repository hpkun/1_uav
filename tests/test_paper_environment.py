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


def spec():
    return AircraftSpec(150, 300, -np.pi/3, np.pi/3, -3, 3, -6, 6, -np.pi/2, np.pi/2, 1, .7, 50, 1, 1, 1)


def test_equations_1_and_2():
    s = state(v=200, theta=.2, psi=-.3)
    c = ControlCommand(nx=1.2, nz=2.0, phi=.4)
    g = 9.81
    expected = [
        s.v*np.cos(s.theta)*np.cos(s.psi), s.v*np.cos(s.theta)*np.sin(s.psi), -s.v*np.sin(s.theta),
        g*(c.nx-np.sin(s.theta)), g/s.v*(c.nz*np.cos(c.phi)-np.cos(s.theta)),
        g*c.nz*np.sin(c.phi)/(s.v*np.cos(s.theta)),
    ]
    assert np.allclose(PointMassDynamics(g).derivatives(s, c), expected)


def test_table2_and_equation23_action_mapping():
    controller = TargetStateController()
    target = controller.action_to_target(state(v=280, theta=.9), np.ones(3), spec())
    assert abs(target.desired_psi) == pytest.approx(np.pi)
    assert target.desired_theta == pytest.approx(np.pi/3)
    assert target.desired_v == 300


@pytest.mark.parametrize(
    "red,blue,expected",
    [
        (state(0, 0), state(100, 0), (0, 0, np.pi, np.pi, 0)),
        (state(0, 0), state(100, 0, psi=np.pi), (0, np.pi, 0, np.pi, 0)),
        (state(0, -100), state(0, 0), (np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, 0)),
        (state(100, 0), state(0, 0), (np.pi, np.pi, 0, 0, 0)),
        (state(0, 0), state(100, 0, z=-100), (0, 0, np.pi, np.pi, np.pi/4)),
    ],
    ids=["A_red_behind", "B_head_on", "C_side", "D_blue_behind", "E_height"],
)
def test_figure2_equation6_physical_truth_table(red, blue, expected):
    red_geometry = compute_paper_geometry(red, blue)
    blue_geometry = compute_paper_geometry(blue, red)
    got = (red_geometry.ata, red_geometry.aa, blue_geometry.ata, blue_geometry.aa, red_geometry.ha)
    assert np.allclose(np.abs(got[:4]), np.abs(expected[:4]))
    assert got[4] == pytest.approx(expected[4])


def test_same_heading_hca_zero_and_opposite_hca_pi():
    assert compute_paper_geometry(state(), state(10, 0)).hca == 0
    assert abs(compute_paper_geometry(state(), state(10, 0, psi=np.pi)).hca) == pytest.approx(np.pi)


def test_sensor_equations_3_to_5_use_printed_shared_samples():
    observed = SensorModel(10, .1, 2, 3, 3, 3, True).observe(state(x=1, y=2, z=3, theta=.2, psi=.3), .4, np.random.default_rng(7))
    position_offsets = np.array([observed.x-1, observed.y-2, observed.z-3])
    attitude_offsets = np.array([observed.phi-.4, observed.psi-.3, observed.theta-.2])
    assert np.allclose(position_offsets, position_offsets[0])
    assert np.allclose(attitude_offsets, attitude_offsets[0])


class FixedNormal:
    def __init__(self, value): self.value = value
    def normal(self): return self.value


def test_weapon_equations_7_and_8_threshold_and_shared_epsilon():
    weapon = WeaponModel(100, 4000, np.pi/6, np.pi/6, 2000, .05, .05)
    assert weapon.can_fire(PaperAirCombatGeometry(4000, np.pi/6, 0, np.pi/6, 0))
    assert not weapon.can_fire(PaperAirCombatGeometry(4000, np.pi/6+1e-6, 0, 0, 0))
    threshold = np.pi*np.exp(-2000/2000)
    assert weapon.sample_hit(PaperAirCombatGeometry(2000, threshold-.01, 0, threshold-.01, 0), FixedNormal(0))
    assert not weapon.sample_hit(PaperAirCombatGeometry(2000, threshold+.01, 0, 0, 0), FixedNormal(0))


def test_fixed_blue_policy_switches_to_nearest_survivor():
    own = state()
    targets = [state(100), state(50)]
    policy = NearestTargetPursuitPolicy()
    assert policy.nearest_target_index(own, targets) == 1
    targets[1].alive = False
    assert policy.nearest_target_index(own, targets) == 0


def test_random_diameter_initialization_is_4v4_and_opposite():
    red, blue, _ = random_diameter_states(np.random.default_rng(2))
    assert len(red) == len(blue) == 4
    assert np.allclose(np.mean([[s.x, s.y] for s in red], axis=0), -np.mean([[s.x, s.y] for s in blue], axis=0))


def test_equation24_natural_45_dimensions_and_enemy_geometry():
    observed = [ObservedState(i*100, 0, 0, 200, 0, 0, 0) for i in range(8)]
    observation = build_observation(0, observed[:4], observed[4:], [True]*4, [True]*4)
    assert observation.shape == (45,)
    assert observation[27] == pytest.approx(0.0)  # first enemy AA: same-heading target ahead
    assert observation[28] == pytest.approx(0.0)  # first enemy ATA


@pytest.mark.parametrize("angle,reward", [(30, .01), (15, .02), (5, .10)])
def test_equation25_r41_tiers(angle, reward):
    radians = np.deg2rad(angle)
    geometry = PaperAirCombatGeometry(3999, radians, 0, radians, 0)
    assert equation25_reward(geometry, None) == pytest.approx(reward)


@pytest.mark.parametrize("angle,reward", [(30, -.015), (15, -.025), (5, -.15)])
def test_equation25_r42_tiers(angle, reward):
    radians = np.deg2rad(angle)
    geometry = PaperAirCombatGeometry(3999, radians, 0, radians, 0)
    assert equation25_reward(None, geometry) == pytest.approx(reward)


def test_equation25_r4_is_piecewise_not_additive():
    advantageous = PaperAirCombatGeometry(1000, 0, 0, 0, 0)
    threat = PaperAirCombatGeometry(1000, 0, 0, 0, 0)
    assert equation25_reward(advantageous, threat) == pytest.approx(.10)


def test_equation25_inclusive_4000_and_r1_r2():
    geometry = PaperAirCombatGeometry(4000, 0, 0, 0, 0)
    assert equation25_reward(geometry, None) == pytest.approx(.101)
    assert equation25_reward(None, None, destroyed_blue=1) == 10
    assert equation25_reward(None, None, red_destroyed=True) == -10
    assert equation25_reward(None, None, red_boundary_loss=True) == -10


def test_paper_binary_success_and_mutual_edge():
    env = PaperUAVCombatEnv(sensor_noise=False)
    env.reset(1)
    for aircraft in env.red + env.blue:
        aircraft.alive = False
    info = env._info(np.zeros(4), np.zeros((4, 3)), [None]*4, truncated=False)
    assert info["red_success"] and info["red_win"] and not info["blue_win"]
    assert info["termination_reason"] == "all_blue_destroyed"


def test_timeout_with_blue_survivors_is_red_failure():
    env = PaperUAVCombatEnv(sensor_noise=False)
    env.reset(1)
    info = env._info(np.zeros(4), np.zeros((4, 3)), [None]*4, truncated=True)
    assert not info["red_success"] and info["blue_win"]


def test_boundary_means_dead_and_r2_event():
    env = PaperUAVCombatEnv(sensor_noise=False)
    env.reset(1)
    env.red[0].x = 5000.01
    red_losses, _ = env._resolve_boundaries()
    assert red_losses == [0] and not env.red[0].alive and env.red_boundary_losses == 1


def test_environment_step_contract():
    env = PaperUAVCombatEnv(sensor_noise=False)
    observation, _ = env.reset(3)
    next_observation, reward, terminated, truncated, info = env.step(np.zeros((4, 3)))
    assert observation.shape == next_observation.shape == (4, 45)
    assert reward.shape == (4,) and np.isfinite(reward).all()
    assert not (terminated and truncated) and "red_success" in info
