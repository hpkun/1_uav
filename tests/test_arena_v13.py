from pathlib import Path

import numpy as np
import pytest

from uav_combat.config import load_config
from uav_combat.environment.arena import (
    arena_constrained_direction, boundary_cost, horizontal_safety_severity,
    vertical_safety_severity,
)
from uav_combat.environment.control import action_to_control
from uav_combat.environment.env import MultiUAVCombatEnv
from uav_combat.environment.observation import build_team_observations
from uav_combat.environment.reward import combined_potentials
from uav_combat.models import AircraftState


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs/combat_environment.yaml")
BATTLEFIELD = CONFIG["battlefield"]
RADIUS = BATTLEFIELD["horizontal_radius"]


def state(x=0.0, y=0.0, altitude=3000.0, psi=0.0, alive=True):
    return AircraftState(x, y, -altitude, 225.0, 0.0, psi, alive)


@pytest.mark.parametrize("fraction,expected", [(0.5, 0.0), (0.65, 0.0), (1.0, 1.0)])
def test_horizontal_soft_boundary_endpoints(fraction, expected):
    own = state(x=fraction * RADIUS)
    assert horizontal_safety_severity(own, BATTLEFIELD) == pytest.approx(expected)
    assert boundary_cost(own, BATTLEFIELD) == pytest.approx(expected * expected)


def test_boundary_cost_is_strictly_between_endpoints_in_soft_region():
    assert 0.0 < boundary_cost(state(x=0.8 * RADIUS), BATTLEFIELD) < 1.0


@pytest.mark.parametrize("fraction", [0.66, 0.8, 0.95])
def test_outward_target_is_immediately_corrected_inward(fraction):
    own = state(x=fraction * RADIUS)
    direction = arena_constrained_direction(own, np.array([1.0, 0.0]), BATTLEFIELD)
    assert np.dot(direction, np.array([1.0, 0.0])) < -0.99


def test_inward_and_tangential_requests_remain_safe_and_interpretable():
    own = state(x=0.8 * RADIUS)
    inward = arena_constrained_direction(own, [-1.0, 0.0], BATTLEFIELD)
    tangent = arena_constrained_direction(own, [0.0, 1.0], BATTLEFIELD)
    assert inward[0] < -0.99
    assert tangent[0] < 0.0 and tangent[1] > 0.0


@pytest.mark.parametrize("altitude", [500.0, 1000.0, 7500.0, 8000.0])
def test_vertical_soft_severity_and_cost_are_bounded(altitude):
    severity = vertical_safety_severity(state(altitude=altitude), BATTLEFIELD)
    assert 0.0 <= severity <= 1.0
    if altitude in (500.0, 8000.0):
        assert severity == pytest.approx(1.0)


def test_combined_potential_decreases_outward_and_increases_inward():
    empty = [state(alive=False) for _ in range(4)]
    def potential(radius):
        team = [state(x=radius)] + [state(alive=False) for _ in range(3)]
        return combined_potentials(team, empty, 8000.0, BATTLEFIELD, 1.0)[2][0]
    middle = potential(0.8 * RADIUS)
    assert potential(0.81 * RADIUS) < middle
    assert potential(0.79 * RADIUS) > middle


def test_vertical_combined_potential_has_correct_direction():
    empty = [state(alive=False) for _ in range(4)]
    def potential(altitude):
        team = [state(altitude=altitude)] + [state(alive=False) for _ in range(3)]
        return combined_potentials(team, empty, 8000.0, BATTLEFIELD, 1.0)[2][0]
    assert potential(550.0) < potential(650.0)
    assert potential(7950.0) < potential(7850.0)


def test_observation_scales_derive_from_arena_and_keep_legal_geometry_bounded():
    team = [
        state(x=RADIUS, psi=0.0), state(x=-RADIUS),
        state(y=RADIUS), state(y=-RADIUS),
    ]
    opponents = [
        state(x=-RADIUS, altitude=8000.0), state(x=RADIUS, altitude=500.0),
        state(y=-RADIUS), state(y=RADIUS),
    ]
    obs = build_team_observations(team, opponents, CONFIG["observation"], BATTLEFIELD)
    assert obs.shape == (4, 54)
    assert np.all(np.isfinite(obs))
    assert np.max(np.abs(obs[:, 3:5])) <= 1.0 + 1e-6
    position_indices = [5 + 7 * slot + axis for slot in range(7) for axis in range(3)]
    assert np.max(np.abs(obs[:, position_indices])) <= 1.0 + 1e-6


def test_recovery_speed_interpolates_260_to_225():
    env = MultiUAVCombatEnv(ROOT / "configs/combat_environment.yaml")
    assert env.fixed_policy.recovery_speed(state(x=0.65 * RADIUS), 260.0) == pytest.approx(260.0)
    assert env.fixed_policy.recovery_speed(state(x=RADIUS), 260.0) == pytest.approx(225.0)


def recovery_probe(fraction, speed, angle=0.0):
    env = MultiUAVCombatEnv(ROOT / "configs/combat_environment.yaml")
    outward = np.array([np.cos(angle), np.sin(angle)])
    own = AircraftState(*(fraction * RADIUS * outward), -3000.0, speed, 0.0, angle)
    target = AircraftState(*(20_000.0 * outward), -3000.0, speed, 0.0, angle)
    maximum = fraction * RADIUS
    first_non_outward = None
    crossed = False
    for step in range(1, 1001):
        action = env.fixed_policy.action(own, [target])
        control = action_to_control(own, action, env.config["action"])
        own = env.integrator.step(own, control, env.dynamics, env.spec)
        radius = float(np.hypot(own.x, own.y))
        maximum = max(maximum, radius)
        radial_velocity = np.dot([own.x, own.y], own.velocity_vector()[:2]) / radius
        if first_non_outward is None and radial_velocity <= 0:
            first_non_outward = step
        if radius > RADIUS:
            crossed = True
            break
        if first_non_outward is not None and radius <= 0.65 * RADIUS:
            break
    return maximum, crossed, first_non_outward, own


@pytest.mark.parametrize("fraction", [0.7, 0.8, 0.9])
@pytest.mark.parametrize("speed", [225.0, 260.0, 300.0])
def test_outward_recovery_trajectory_is_finite_and_turns_when_space_permits(fraction, speed):
    maximum, crossed, first_non_outward, own = recovery_probe(fraction, speed)
    assert np.all(np.isfinite(own.as_array()))
    assert maximum <= RADIUS + speed * CONFIG["simulation"]["dt"] + 1.0
    if fraction == 0.7:
        assert not crossed
        assert first_non_outward is not None
