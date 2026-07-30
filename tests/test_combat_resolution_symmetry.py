from __future__ import annotations

from dataclasses import replace
from math import pi

import numpy as np
import pytest

from conftest import make_state
from uav_env.combat.attack_geometry import AttackZoneConfig, compute_combat_geometry
from uav_env.combat.damage import DamageConfig, damage_for_random_value
from uav_env.combat.multi_combat import resolve_multi_attacks
from uav_env.core.enums import Team
from uav_env.entities.uav import UAV


class SequenceRNG:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)
        self.index = 0

    def random(self) -> float:
        value = self.values[self.index]
        self.index += 1
        return value


def _wide_attack() -> AttackZoneConfig:
    return AttackZoneConfig(40.0, 900.0, pi, pi, pi, 40.0, 1300.0, pi)


def _tail_attack() -> AttackZoneConfig:
    return AttackZoneConfig(40.0, 900.0, 0.5, 0.5, 0.8, 40.0, 1300.0, 0.5)


def _uav(uav_id: str, team: Team, state, profile) -> UAV:
    return UAV(uav_id, int(team), state, profile)


def test_damage_probability_boundaries_cover_miss_and_hit() -> None:
    cfg = DamageConfig()
    assert damage_for_random_value(0.0, cfg) == 51.0
    assert damage_for_random_value(0.999999, cfg) == 0.0


def test_resolve_multi_attacks_allows_both_sides_in_same_decision_step(profile) -> None:
    aircraft = [
        _uav("red_0", Team.RED, make_state(profile, x=0.0, y=0.0, z=1500.0, heading=0.0, team=Team.RED), profile),
        _uav("red_1", Team.RED, make_state(profile, x=1500.0, y=0.0, z=1500.0, heading=pi, team=Team.RED), profile),
        _uav("blue_0", Team.BLUE, make_state(profile, x=500.0, y=0.0, z=1500.0, heading=0.0, team=Team.BLUE), profile),
        _uav("blue_1", Team.BLUE, make_state(profile, x=2000.0, y=0.0, z=1500.0, heading=pi, team=Team.BLUE), profile),
    ]
    result = resolve_multi_attacks(aircraft, _wide_attack(), DamageConfig(), SequenceRNG([0.0, 0.0, 0.0, 0.0]))
    attackers = {attempt.attacker_id for attempt in result.attack_attempts}
    assert any(item.startswith("red_") for item in attackers)
    assert any(item.startswith("blue_") for item in attackers)
    assert all(attack.hit for attack in result.resolved_attacks)


def test_multiple_attackers_same_target_share_effective_damage_and_single_destroy_credit(profile) -> None:
    aircraft = [
        _uav(f"red_{i}", Team.RED, make_state(profile, x=float(i * 10), z=1500.0, heading=0.0, team=Team.RED), profile)
        for i in range(3)
    ]
    aircraft.append(_uav("blue_0", Team.BLUE, make_state(profile, x=500.0, z=1500.0, heading=0.0, team=Team.BLUE, health=40.0), profile))
    result = resolve_multi_attacks(aircraft, _tail_attack(), DamageConfig(), SequenceRNG([0.0, 0.0, 0.0]))
    target_attacks = [attack for attack in result.resolved_attacks if attack.target_id == "blue_0"]
    assert len(target_attacks) == 3
    assert sum(attack.effective_damage for attack in target_attacks) == pytest.approx(40.0)
    assert sum(attack.overkill_damage for attack in target_attacks) == pytest.approx(3 * 51.0 - 40.0)
    assert sum(attack.destroy_credit for attack in target_attacks) == 1
    assert result.updated_states["blue_0"].health == pytest.approx(0.0)
    assert not result.updated_states["blue_0"].alive


def test_exact_kill_and_overkill_are_clamped_to_remaining_health(profile) -> None:
    exact = [
        _uav("red_0", Team.RED, make_state(profile, x=0.0, z=1500.0, heading=0.0, team=Team.RED), profile),
        _uav("blue_0", Team.BLUE, make_state(profile, x=500.0, z=1500.0, heading=0.0, team=Team.BLUE, health=21.0), profile),
    ]
    exact_result = resolve_multi_attacks(exact, _tail_attack(), DamageConfig(), SequenceRNG([0.1]))
    red_hit = next(attack for attack in exact_result.resolved_attacks if attack.attacker_id == "red_0")
    assert red_hit.nominal_damage == pytest.approx(21.0)
    assert red_hit.effective_damage == pytest.approx(21.0)
    assert red_hit.overkill_damage == pytest.approx(0.0)
    assert red_hit.destroy_credit

    overkill = [
        _uav("red_0", Team.RED, make_state(profile, x=0.0, z=1500.0, heading=0.0, team=Team.RED), profile),
        _uav("blue_0", Team.BLUE, make_state(profile, x=500.0, z=1500.0, heading=0.0, team=Team.BLUE, health=10.0), profile),
    ]
    overkill_result = resolve_multi_attacks(overkill, _tail_attack(), DamageConfig(), SequenceRNG([0.0]))
    red_overkill = next(attack for attack in overkill_result.resolved_attacks if attack.attacker_id == "red_0")
    assert red_overkill.effective_damage == pytest.approx(10.0)
    assert red_overkill.overkill_damage == pytest.approx(41.0)


def test_dead_aircraft_cannot_attack_or_be_selected_as_target(profile) -> None:
    aircraft = [
        _uav("red_0", Team.RED, make_state(profile, x=0.0, z=1500.0, heading=0.0, team=Team.RED), profile),
        _uav("red_1", Team.RED, replace(make_state(profile, x=10.0, z=1500.0, heading=0.0, team=Team.RED), health=0.0, alive=False, damaged=True), profile),
        _uav("blue_0", Team.BLUE, make_state(profile, x=500.0, z=1500.0, heading=0.0, team=Team.BLUE), profile),
    ]
    result = resolve_multi_attacks(aircraft, _wide_attack(), DamageConfig(), SequenceRNG([0.0, 0.0]))
    assert all(attack.attacker_id != "red_1" for attack in result.attack_attempts)
    assert all(attack.target_id != "red_1" for attack in result.attack_attempts)


def test_mirrored_red_blue_attack_probability_geometry_is_identical(profile) -> None:
    cfg = _wide_attack()
    red = make_state(profile, x=-300.0, y=80.0, z=1500.0, heading=0.0, team=Team.RED)
    blue = make_state(profile, x=200.0, y=80.0, z=1500.0, heading=0.0, team=Team.BLUE)
    mirrored_red = replace(red, x=-blue.x, heading_angle=pi - blue.heading_angle, team_id=int(Team.RED))
    mirrored_blue = replace(blue, x=-red.x, heading_angle=pi - red.heading_angle, team_id=int(Team.BLUE))
    assert compute_combat_geometry(red, blue, cfg).can_attack == compute_combat_geometry(mirrored_red, mirrored_blue, cfg).can_attack
