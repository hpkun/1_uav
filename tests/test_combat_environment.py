from pathlib import Path
import copy
import numpy as np
import pytest

from uav_combat.config import load_config
from uav_combat.dynamics import PointMassDynamics
from uav_combat.environment.control import action_to_control, action_to_target
from uav_combat.environment.env import MultiUAVCombatEnv
from uav_combat.environment.geometry import engagement_geometry
from uav_combat.environment.reward import combat_reward_components, relation_score
from uav_combat.integrator import RK4Integrator
from uav_combat.models import AircraftState, AircraftSpec


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/combat_environment.yaml"


def state(
    x=0.0, y=0.0, altitude=3000.0, v=225.0,
    theta=0.0, psi=0.0, alive=True,
):
    return AircraftState(x, y, -altitude, v, theta, psi, alive)


def one_alive(primary: AircraftState) -> list[AircraftState]:
    return [primary] + [state(alive=False) for _ in range(3)]


def test_level_center_speed_command_is_stable_for_100_steps():
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
    assert aircraft.altitude == pytest.approx(3000.0, abs=1e-8)


def test_action_decodes_high_level_heading_pitch_and_speed_commands():
    config = load_config(CONFIG_PATH)["action"]["command"]
    own = state(psi=0.2)
    low = action_to_target(own, [-1, -1, -1], config)
    high = action_to_target(own, [1, 1, 1], config)
    assert low.pitch == pytest.approx(-np.pi / 6)
    assert high.pitch == pytest.approx(np.pi / 6)
    assert low.speed == pytest.approx(170.0)
    assert high.speed == pytest.approx(280.0)
    assert abs(((high.heading - own.psi + np.pi) % (2 * np.pi)) - np.pi) == pytest.approx(np.pi)


def test_response_mapping_turns_and_climbs_in_commanded_direction():
    config = load_config(CONFIG_PATH)["action"]
    own = state()
    left = action_to_control(own, [0.25, 0, 0], config)
    right = action_to_control(own, [-0.25, 0, 0], config)
    climb = action_to_control(own, [0, 0.5, 0], config)
    dynamics = PointMassDynamics()
    assert dynamics.derivatives(own, left)[5] > 0.0
    assert dynamics.derivatives(own, right)[5] < 0.0
    assert dynamics.derivatives(own, climb)[4] > 0.0
    assert abs(left.phi) <= config["controller"]["bank_max"]


def test_sustained_level_turn_preserves_altitude_and_speed():
    config = load_config(CONFIG_PATH)
    dynamics = PointMassDynamics()
    integrator = RK4Integrator(config["simulation"]["dt"])
    spec = AircraftSpec(**config["aircraft"])
    aircraft = state()
    for _ in range(100):
        control = action_to_control(aircraft, [0.25, 0.0, 0.0], config["action"])
        aircraft = integrator.step(aircraft, control, dynamics, spec)
    assert aircraft.altitude == pytest.approx(3000.0, abs=1e-7)
    assert aircraft.v == pytest.approx(225.0, abs=1e-10)
    assert abs(aircraft.psi) > 0.5


def test_random_commands_remain_finite_and_inside_state_limits():
    config = load_config(CONFIG_PATH)
    env = MultiUAVCombatEnv(config)
    env.reset(21)
    rng = np.random.default_rng(22)
    for _ in range(300):
        observation, reward, terminated, truncated, _ = env.step(
            rng.uniform(-1, 1, (4, 3)).astype(np.float32)
        )
        assert np.all(np.isfinite(observation))
        assert np.all(np.isfinite(reward))
        for aircraft in env.red + env.blue:
            assert config["aircraft"]["v_min"] <= aircraft.v <= config["aircraft"]["v_max"]
            assert config["aircraft"]["theta_min"] <= aircraft.theta <= config["aircraft"]["theta_max"]
        if terminated or truncated:
            break


def test_head_on_inside_range_is_not_a_fire_opportunity():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    geometry = engagement_geometry(state(), state(x=1000.0, psi=np.pi))
    assert geometry.attack_angle == pytest.approx(0.0)
    assert geometry.target_aspect == pytest.approx(np.pi)
    assert not env.weapon.in_fire_window(geometry)


def test_tail_and_side_aspect_inside_range_are_fire_opportunities():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    tail = engagement_geometry(state(), state(x=1000.0))
    side = engagement_geometry(state(), state(x=1000.0, psi=np.pi / 2))
    assert env.weapon.in_fire_window(tail)
    assert env.weapon.in_fire_window(side)


@pytest.mark.parametrize("distance", [299.0, 2001.0])
def test_fire_window_respects_both_range_limits(distance):
    env = MultiUAVCombatEnv(CONFIG_PATH)
    assert not env.weapon.in_fire_window(
        engagement_geometry(state(), state(x=distance))
    )


def test_lock_requires_five_consecutive_steps_and_resets_on_break():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    env.reset(2)
    red, blue = one_alive(state()), one_alive(state(x=1000.0))
    for _ in range(4):
        assert env._lock_proposals(red, blue, env.red_locks, "red") == []
    blue[0].y = 3000.0
    assert env._lock_proposals(red, blue, env.red_locks, "red") == []
    assert env.red_locks[0].lock_steps == 0
    blue[0].y = 0.0
    for _ in range(4):
        assert env._lock_proposals(red, blue, env.red_locks, "red") == []
    assert env._lock_proposals(red, blue, env.red_locks, "red") == [(0, 0)]


def test_reward_components_are_finite_and_approach_has_positive_progress():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    current_red = one_alive(state())
    current_blue = one_alive(state(x=4000.0, v=200.0))
    next_red = one_alive(state(x=25.0, v=250.0))
    next_blue = one_alive(state(x=4020.0, v=200.0))
    components = combat_reward_components(
        current_red, current_blue, next_red, next_blue,
        env.weapon, env.config["reward"], env.dt,
    )
    assert set(components) == {"progress", "tactical", "fire"}
    assert all(np.all(np.isfinite(value)) for value in components.values())
    assert components["progress"][0] > 0.0


def test_tactical_relation_prefers_tail_position_over_head_on():
    config = load_config(CONFIG_PATH)["reward"]
    attacker = state()
    tail_target = state(x=1000.0)
    head_on_target = state(x=1000.0, psi=np.pi)
    assert relation_score(attacker, tail_target, config) > relation_score(
        attacker, head_on_target, config
    )


def test_kill_credit_is_shared_and_death_penalty_is_finite():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    env.reset(4)
    rewards = env._event_rewards([2], {}, {0: [0, 1]})
    assert rewards.tolist() == [5.0, 5.0, -10.0, 0.0]
    assert np.all(np.isfinite(rewards))


def test_observation_is_52d_finite_bounded_and_masks_dead_slots():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    observation, _ = env.reset(6)
    env.red[1].alive = False
    env.blue[2].alive = False
    env.blue[0].x += 100_000.0
    observation = env._observations()
    assert observation.shape == (4, 52)
    assert np.all(np.isfinite(observation))
    assert np.max(np.abs(observation)) <= 1.0
    assert np.all(observation[1] == 0.0)
    assert observation[0, 9] == 0.0
    assert observation[0, 44] == 0.0


def test_observation_is_translation_and_rotation_invariant():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    env.reset(7)
    baseline = env._observations()
    transformed = copy.deepcopy(env)
    angle = 1.234
    cosine, sine = np.cos(angle), np.sin(angle)
    for aircraft in transformed.red + transformed.blue:
        aircraft.x += 50_000.0
        aircraft.y -= 30_000.0
        aircraft.x, aircraft.y = (
            cosine * aircraft.x - sine * aircraft.y,
            sine * aircraft.x + cosine * aircraft.y,
        )
        aircraft.psi = (aircraft.psi + angle + np.pi) % (2 * np.pi) - np.pi
    assert np.allclose(transformed._observations(), baseline, atol=2e-6)


def test_initialization_covers_all_modes_and_starts_outside_fire_range():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    modes = set()
    for seed in range(100):
        _, info = env.reset(seed)
        modes.add(info["scenario_mode"])
        minimum = min(
            engagement_geometry(red, blue).distance
            for red in env.red for blue in env.blue
        )
        assert minimum > env.weapon.range_max
    assert modes == {"head_on", "offset", "flank"}


def test_only_altitude_is_a_battlefield_limit():
    config = load_config(CONFIG_PATH)
    assert set(config["flight_envelope"]) == {"altitude_min", "altitude_max"}
    assert not any("boundary" in key or "center" in key for key in config["reward"])


def test_blue_policy_uses_nearest_target_and_common_high_level_mapping():
    env = MultiUAVCombatEnv(CONFIG_PATH)
    own = state()
    targets = [state(x=2000.0), state(x=1000.0)] + [state(alive=False)] * 2
    assert env.fixed_policy.nearest_target_index(own, targets) == 1
    action = env.fixed_policy.action(own, targets)
    control = action_to_control(own, action, env.config["action"])
    direct = env.integrator.step(own, control, env.dynamics, env.spec)
    states = [own.copy()]
    env._advance(states, np.asarray([action]))
    assert np.allclose(states[0].as_array(), direct.as_array())
