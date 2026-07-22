from __future__ import annotations

from math import pi

import pytest

from conftest import make_state
from uav_env.combat.attack_geometry import AttackZoneConfig, compute_combat_geometry
from uav_env.core.enums import Team
from uav_env.entities.type_profiles import UAVTypeProfile


def config() -> AttackZoneConfig:
    return AttackZoneConfig(40.0, 900.0, pi / 6.0, pi / 3.0, pi / 4.0, 40.0, 1300.0, pi / 3.0)


def test_target_ahead_has_zero_attack_and_escape_angles(profile: UAVTypeProfile) -> None:
    attacker = make_state(profile, x=0.0, heading=0.0)
    target = make_state(profile, x=500.0, heading=0.0, team=Team.BLUE)
    geometry = compute_combat_geometry(attacker, target, config())
    assert geometry.distance == pytest.approx(500.0)
    assert geometry.line_of_sight.tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert geometry.attacker_attack_angle == pytest.approx(0.0)
    assert geometry.target_escape_angle == pytest.approx(0.0)
    assert geometry.can_attack


def test_target_flying_toward_attacker_has_pi_escape_angle(profile: UAVTypeProfile) -> None:
    attacker = make_state(profile)
    target = make_state(profile, x=500.0, heading=pi, team=Team.BLUE)
    geometry = compute_combat_geometry(attacker, target, config())
    assert geometry.target_escape_angle == pytest.approx(pi)
    assert not geometry.can_attack


def test_zero_distance_is_finite(profile: UAVTypeProfile) -> None:
    attacker = make_state(profile)
    target = make_state(profile, team=Team.BLUE)
    geometry = compute_combat_geometry(attacker, target, config())
    assert geometry.distance == 0.0
    assert geometry.attacker_attack_angle == 0.0
    assert not geometry.can_attack
