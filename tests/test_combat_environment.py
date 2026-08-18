from pathlib import Path
import copy
import numpy as np
import pytest

from uav_combat.config import load_config
from uav_combat.dynamics import PointMassDynamics
from uav_combat.environment.control import action_to_control
from uav_combat.environment.env import MultiUAVCombatEnv
from uav_combat.environment.geometry import engagement_geometry
from uav_combat.environment.observation import build_team_observations
from uav_combat.integrator import RK4Integrator
from uav_combat.models import AircraftState, AircraftSpec


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/combat_environment.yaml"


def state(x=0.0, y=0.0, altitude=3000.0, v=225.0, theta=0.0, psi=0.0, alive=True):
    return AircraftState(x, y, -altitude, v, theta, psi, alive)


def one_alive(primary: AircraftState) -> list[AircraftState]:
    return [primary] + [state(alive=False) for _ in range(3)]


def test_zero_action_is_trim_for_100_steps():
    config = load_config(CONFIG_PATH)
    dynamics = PointMassDynamics()
    integrator = RK4Integrator(config["simulation"]["dt"])
    spec = AircraftSpec(**config["aircraft"])
    aircraft = state()
    control = action_to_control(np.zeros(3), config["action"])
    for _ in range(100):
        aircraft = integrator.step(aircraft, control, dynamics, spec)
    assert aircraft.v == pytest.approx(225.0, abs=1e-10)
    assert aircraft.theta == pytest.approx(0.0, abs=1e-10)
    assert aircraft.psi == pytest.approx(0.0, abs=1e-10)


def test_head_on_at_1km_has_attack_zero_escape_pi_and_is_not_attackable():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    geometry = engagement_geometry(state(), state(x=1000.0, psi=np.pi))
    assert geometry.attack_angle == pytest.approx(0.0)
    assert geometry.escape_angle == pytest.approx(np.pi)
    assert not env.weapon.attackable(geometry)


def test_tail_chase_at_1km_has_both_angles_zero_and_is_attackable():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    geometry = engagement_geometry(state(), state(x=1000.0))
    assert geometry.attack_angle == pytest.approx(0.0)
    assert geometry.escape_angle == pytest.approx(0.0)
    assert env.weapon.attackable(geometry)


def test_ninety_degree_crossing_is_not_a_rear_shot():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    geometry = engagement_geometry(state(), state(x=1000.0, psi=np.pi / 2))
    assert geometry.escape_angle == pytest.approx(np.pi / 2)
    assert not env.weapon.attackable(geometry)


def test_lock_dwell_proposes_kill_on_third_consecutive_step():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    env.reset(1)
    red = one_alive(state())
    blue = one_alive(state(x=1000.0))
    assert env._lock_proposals(red, blue, env.red_locks) == []
    assert env._lock_proposals(red, blue, env.red_locks) == []
    assert env._lock_proposals(red, blue, env.red_locks) == [(0, 0)]
    assert env.red_locks[0].lock_steps == 3


def test_lock_break_resets_and_requires_three_new_steps():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    env.reset(2)
    red = one_alive(state())
    blue = one_alive(state(x=1000.0))
    env._lock_proposals(red, blue, env.red_locks)
    env._lock_proposals(red, blue, env.red_locks)
    blue[0].y = 3000.0
    assert env._lock_proposals(red, blue, env.red_locks) == []
    assert env.red_locks[0].current_lock_target == -1
    assert env.red_locks[0].lock_steps == 0
    blue[0].y = 0.0
    assert env._lock_proposals(red, blue, env.red_locks) == []
    assert env._lock_proposals(red, blue, env.red_locks) == []
    assert env._lock_proposals(red, blue, env.red_locks) == [(0, 0)]


def test_simultaneous_resolution_and_mutual_destruction_draw():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    env.reset(3)
    env.red = one_alive(state())
    env.blue = one_alive(state(x=1000.0))
    red_credited, blue_credited = env._resolve_combat([(0, 0)], [(0, 0)])
    assert red_credited == {0: [0]}
    assert blue_credited == {0: [0]}
    assert not env.red[0].alive and not env.blue[0].alive
    assert env._outcome(False) == (False, False, True, "draw_mutual_destruction")


def test_win_loss_and_timeout_outcomes_are_mutually_exclusive():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    env.reset(31)
    env.red = one_alive(state())
    env.blue = [state(alive=False) for _ in range(4)]
    assert env._outcome(False) == (True, False, False, "red_win")
    env.red = [state(alive=False) for _ in range(4)]
    env.blue = one_alive(state())
    assert env._outcome(False) == (False, True, False, "blue_win")
    env.red = one_alive(state())
    env.blue = one_alive(state())
    assert env._outcome(True) == (False, False, True, "draw_timeout")


def test_two_attackers_split_one_kill_reward_equally():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    env.reset(4)
    rewards = env._event_rewards([], {}, {0: [0, 1]})
    assert rewards.tolist() == [5.0, 5.0, 0.0, 0.0]


def test_boundary_death_penalty_occurs_once():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    env.reset(5)
    env.red = one_alive(state(x=env.radius - 1.0))
    env.blue = [state(alive=False) for _ in range(4)]
    _, first_reward, _, _, _ = env.step(np.zeros((4, 3)), np.zeros((4, 3)))
    _, second_reward, _, _, _ = env.step(np.zeros((4, 3)), np.zeros((4, 3)))
    assert first_reward[0] == pytest.approx(-10.0)
    assert second_reward[0] == pytest.approx(0.0)
    assert env.red_boundary_losses == 1


def test_observation_shape_dead_masks_finiteness_and_dead_self():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    observation, _ = env.reset(6)
    assert observation.shape == (4, 54)
    env.red[1].alive = False
    env.blue[2].alive = False
    observation = env._observations()
    assert np.all(np.isfinite(observation))
    assert np.all(observation[1] == 0.0)
    assert observation[0, 11] == 0.0  # first ally slot's alive mask
    assert observation[0, 32] == 1.0
    assert observation[0, 46] == 0.0  # third enemy slot's alive mask


def test_observations_are_invariant_to_common_horizontal_rotation():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    env.reset(7)
    baseline = env._observations()
    angle = 1.234
    cosine, sine = np.cos(angle), np.sin(angle)
    rotated = copy.deepcopy(env)
    for aircraft in rotated.red + rotated.blue:
        aircraft.x, aircraft.y = (
            cosine * aircraft.x - sine * aircraft.y,
            sine * aircraft.x + cosine * aircraft.y,
        )
        aircraft.psi = (aircraft.psi + angle + np.pi) % (2 * np.pi) - np.pi
    assert np.allclose(rotated._observations(), baseline, atol=2e-6)


def test_seeded_initialization_has_only_documented_perturbations():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    env.reset(8)
    for aircraft in env.red + env.blue:
        assert 215.0 <= aircraft.v <= 235.0
        assert 2900.0 <= aircraft.altitude <= 3100.0
        assert aircraft.theta == 0.0
    assert np.allclose(sorted(np.linalg.norm([s.x, s.y]) for s in env.red),
                       sorted(np.linalg.norm([s.x, s.y]) for s in env.blue))
