from pathlib import Path
import copy
import numpy as np
import pytest

from uav_combat.config import load_config
from uav_combat.dynamics import PointMassDynamics
from uav_combat.environment.control import action_to_control, trim_normal_load
from uav_combat.environment.env import MultiUAVCombatEnv
from uav_combat.environment.geometry import EngagementGeometry, engagement_geometry, engagement_score
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
    for _ in range(100):
        control = action_to_control(aircraft, np.zeros(3), config["action"])
        aircraft = integrator.step(aircraft, control, dynamics, spec)
    assert aircraft.v == pytest.approx(225.0, abs=1e-10)
    assert aircraft.theta == pytest.approx(0.0, abs=1e-10)
    assert aircraft.psi == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("theta_deg,phi_deg", [
    (0, 0), (0, 30), (0, 45), (0, 60), (30, 45), (-30, 45),
])
def test_trim_relative_mapping_is_vertical_neutral(theta_deg, phi_deg):
    config = load_config(CONFIG_PATH)
    own = state(theta=np.deg2rad(theta_deg))
    action = np.array([0.0, 0.0, phi_deg / 60.0])
    control = action_to_control(own, action, config["action"])
    assert control.nz == pytest.approx(
        trim_normal_load(own.theta, np.deg2rad(phi_deg)), abs=1e-12
    )
    assert control.nz * np.cos(control.phi) - np.cos(own.theta) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("theta_deg,phi_deg", [(0, 0), (0, 60), (30, 45), (-30, 45)])
def test_vertical_action_sign_always_controls_pitch_rate_direction(theta_deg, phi_deg):
    config = load_config(CONFIG_PATH)
    own = state(theta=np.deg2rad(theta_deg))
    positive = action_to_control(own, [0, 0.5, phi_deg / 60], config["action"])
    negative = action_to_control(own, [0, -0.5, phi_deg / 60], config["action"])
    assert positive.nz * np.cos(positive.phi) - np.cos(own.theta) > 0
    assert negative.nz * np.cos(negative.phi) - np.cos(own.theta) < 0


def test_constant_bank_neutral_vertical_preserves_theta_and_altitude_for_100_steps():
    config = load_config(CONFIG_PATH)
    dynamics = PointMassDynamics()
    integrator = RK4Integrator(config["simulation"]["dt"])
    spec = AircraftSpec(**config["aircraft"])
    aircraft = state()
    initial_altitude = aircraft.altitude
    for _ in range(100):
        control = action_to_control(aircraft, [0.0, 0.0, 0.5], config["action"])
        aircraft = integrator.step(aircraft, control, dynamics, spec)
    assert aircraft.theta == pytest.approx(0.0, abs=1e-10)
    assert aircraft.altitude == pytest.approx(initial_altitude, abs=1e-8)
    assert abs(aircraft.psi) > 0.1


def test_v14_config_has_only_altitude_envelope_and_single_relative_position_scale():
    config = load_config(CONFIG_PATH)
    assert set(config["action"]) == {"nx_scale", "nz_delta_scale", "phi_max"}
    assert "elevation_gain" not in config["blue_policy"]
    assert "elevation_action_scale" not in config["blue_policy"]
    assert config["blue_policy"]["pitch_load_gain"] == 4.0
    assert set(config["flight_envelope"]) == {"altitude_min", "altitude_max"}
    assert set(config) == {
        "simulation", "action", "aircraft", "flight_envelope", "scenario",
        "weapon", "reward", "observation", "blue_policy",
    }
    assert set(config["blue_policy"]) == {
        "desired_speed", "speed_error_scale", "heading_gain", "pitch_load_gain",
    }
    assert set(config["reward"]) == {
        "kill_reward", "death_penalty", "shaping_lambda", "engagement_distance_scale",
    }
    assert set(config["observation"]) == {
        "speed_center", "speed_scale", "relative_position_scale",
        "relative_velocity_scale",
    }


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


def test_ninety_degree_escape_is_attackable_at_the_inclusive_boundary():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    geometry = engagement_geometry(state(), state(x=1000.0, psi=np.pi / 2))
    assert geometry.escape_angle == pytest.approx(np.pi / 2)
    assert env.weapon.attackable(geometry)


def test_forty_five_degree_attack_is_attackable_at_the_inclusive_boundary():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    geometry = engagement_geometry(state(psi=np.pi / 4), state(x=1000.0))
    assert geometry.attack_angle == pytest.approx(np.pi / 4)
    assert env.weapon.attackable(geometry)


def test_escape_angle_above_ninety_degrees_is_not_attackable():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    geometry = engagement_geometry(state(), state(x=1000.0, psi=np.deg2rad(91.0)))
    assert geometry.escape_angle > np.pi / 2
    assert not env.weapon.attackable(geometry)


def test_engagement_score_uses_fixed_8km_scale_not_arena_radius():
    config = load_config(CONFIG_PATH)
    geometry = EngagementGeometry(distance=2000.0, attack_angle=0.3, escape_angle=0.5)
    scale = config["reward"]["engagement_distance_scale"]
    score_at_15km_arena = engagement_score(geometry, scale)
    modified = copy.deepcopy(config)
    modified["flight_envelope"]["altitude_max"] = 99_000.0
    score_at_other_arena = engagement_score(
        geometry, modified["reward"]["engagement_distance_scale"]
    )
    assert scale == 8000.0
    assert score_at_other_arena == pytest.approx(score_at_15km_arena)


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


def test_altitude_death_penalty_occurs_once():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    env.reset(5)
    env.red = one_alive(state(altitude=env.altitude_min + 1.0, theta=-np.pi / 3.0))
    env.blue = [state(alive=False) for _ in range(4)]
    _, _, _, _, first_info = env.step(np.zeros((4, 3)), np.zeros((4, 3)))
    _, _, _, _, second_info = env.step(np.zeros((4, 3)), np.zeros((4, 3)))
    assert first_info["event_rewards"][0] == pytest.approx(-10.0)
    assert second_info["event_rewards"][0] == pytest.approx(0.0)
    assert env.red_altitude_losses == 1


def test_altitude_causes_are_classified_once_per_aircraft():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    env.reset(51)
    env.red = [
        state(x=100_000.0),
        state(altitude=env.altitude_min - 1.0),
        state(altitude=env.altitude_max + 1.0),
        state(),
    ]
    red_losses, blue_losses = env._resolve_altitude_limits()
    assert red_losses == [1, 2]
    assert blue_losses == []
    assert env.red_altitude_losses == 2
    assert env._cause_count(env.red_altitude_causes, "altitude_low") == 1
    assert env._cause_count(env.red_altitude_causes, "altitude_high") == 1


def test_pure_pursuit_matches_target_direction():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    own, target = state(), state(x=1000.0, y=2000.0)
    direction = env.fixed_policy.desired_horizontal_direction(own, target)
    expected = np.array([1.0, 2.0]) / np.sqrt(5.0)
    assert np.allclose(direction, expected)


def test_blue_action_uses_public_helper_and_common_physical_mapping(monkeypatch):
    env = MultiUAVCombatEnv(CONFIG_PATH)
    own, target = state(), state(x=1000.0)
    calls = []
    original = env.fixed_policy.action_toward
    def recording_helper(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(env.fixed_policy, "action_toward", recording_helper)
    action = env.fixed_policy.action(own, [target])
    assert len(calls) == 1
    common_control = action_to_control(own, action, env.config["action"])
    states = [own.copy()]
    env._advance(states, np.asarray([action]))
    direct = env.integrator.step(own, common_control, env.dynamics, env.spec)
    assert np.allclose(states[0].as_array(), direct.as_array())


def test_observation_shape_dead_masks_finiteness_and_dead_self():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    observation, _ = env.reset(6)
    assert observation.shape == (4, 52)
    env.red[1].alive = False
    env.blue[2].alive = False
    observation = env._observations()
    assert np.all(np.isfinite(observation))
    assert np.all(observation[1] == 0.0)
    assert observation[0, 9] == 0.0  # first ally slot's alive mask
    assert observation[0, 30] == 1.0
    assert observation[0, 44] == 0.0  # third enemy slot's alive mask


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


def test_observations_and_blue_actions_are_invariant_to_horizontal_translation():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    env.reset(71)
    baseline_observation = env._observations()
    baseline_actions = env.fixed_policy.team_actions(env.blue, env.red)
    for aircraft in env.red + env.blue:
        aircraft.x += 50_000.0
        aircraft.y -= 30_000.0
    assert np.allclose(env._observations(), baseline_observation, atol=2e-6)
    assert np.allclose(env.fixed_policy.team_actions(env.blue, env.red), baseline_actions)


def test_multistep_environment_is_horizontally_translation_invariant():
    first, shifted = MultiUAVCombatEnv(CONFIG_PATH), MultiUAVCombatEnv(CONFIG_PATH)
    first.reset(72); shifted.reset(72)
    translation = np.array([50_000.0, -30_000.0])
    for aircraft in shifted.red + shifted.blue:
        aircraft.x += translation[0]; aircraft.y += translation[1]
    for _ in range(25):
        red_actions = first.fixed_policy.team_actions(first.red, first.blue)
        shifted_actions = shifted.fixed_policy.team_actions(shifted.red, shifted.blue)
        assert np.allclose(shifted_actions, red_actions, atol=1e-7)
        blue_actions = first.fixed_policy.team_actions(first.blue, first.red)
        shifted_blue = shifted.fixed_policy.team_actions(shifted.blue, shifted.red)
        first_result = first.step(red_actions, blue_actions)
        shifted_result = shifted.step(shifted_actions, shifted_blue)
        assert np.allclose(first_result[0], shifted_result[0], atol=2e-6)
        assert np.allclose(first_result[1], shifted_result[1], atol=2e-6)
        assert first_result[2:4] == shifted_result[2:4]
        for original, translated in zip(first.red + first.blue, shifted.red + shifted.blue):
            assert translated.x == pytest.approx(original.x + translation[0], abs=1e-8)
            assert translated.y == pytest.approx(original.y + translation[1], abs=1e-8)
            assert translated.altitude == pytest.approx(original.altitude, abs=1e-8)
            assert translated.v == pytest.approx(original.v, abs=1e-10)
            assert translated.theta == pytest.approx(original.theta, abs=1e-10)
            assert translated.psi == pytest.approx(original.psi, abs=1e-10)


def test_seeded_initialization_has_only_documented_perturbations():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    env.reset(8)
    for aircraft in env.red + env.blue:
        assert 215.0 <= aircraft.v <= 235.0
        assert 2900.0 <= aircraft.altitude <= 3100.0
        assert aircraft.theta == 0.0
    assert np.allclose(sorted(np.linalg.norm([s.x, s.y]) for s in env.red),
                       sorted(np.linalg.norm([s.x, s.y]) for s in env.blue))
