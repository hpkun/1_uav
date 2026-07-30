from __future__ import annotations

from dataclasses import replace
from math import pi

import numpy as np
import pytest

from conftest import make_state
from uav_env.combat.attack_geometry import AttackZoneConfig, compute_combat_geometry
from uav_env.core.enums import Team


def _cfg() -> AttackZoneConfig:
    return AttackZoneConfig(
        attack_distance_min=40.0,
        attack_distance_max=900.0,
        attack_angle_max=0.5,
        escape_angle_max=1.1,
        attack_area_angle_max=0.8,
        advantage_distance_min=40.0,
        advantage_distance_max=1300.0,
        advantage_escape_angle_max=1.1,
    )


def test_forward_tail_chase_geometry_oracle(profile) -> None:
    attacker = make_state(profile, x=0.0, y=0.0, z=1500.0, heading=0.0, team=Team.RED)
    target = make_state(profile, x=500.0, y=0.0, z=1500.0, heading=0.0, team=Team.BLUE)
    geometry = compute_combat_geometry(attacker, target, _cfg())
    assert geometry.distance == pytest.approx(500.0)
    assert geometry.line_of_sight == pytest.approx(np.asarray([1.0, 0.0, 0.0]))
    assert geometry.attacker_attack_angle == pytest.approx(0.0)
    assert geometry.target_escape_angle == pytest.approx(0.0)
    assert geometry.in_attack_area
    assert geometry.in_advantage_area
    assert geometry.can_attack


def test_geometry_rejects_behind_too_far_and_too_close(profile) -> None:
    attacker = make_state(profile, x=0.0, z=1500.0, heading=0.0)
    behind = make_state(profile, x=-500.0, z=1500.0, heading=0.0, team=Team.BLUE)
    far = make_state(profile, x=901.0, z=1500.0, heading=0.0, team=Team.BLUE)
    close = make_state(profile, x=39.0, z=1500.0, heading=0.0, team=Team.BLUE)
    cfg = _cfg()
    assert not compute_combat_geometry(attacker, behind, cfg).can_attack
    assert not compute_combat_geometry(attacker, far, cfg).can_attack
    assert not compute_combat_geometry(attacker, close, cfg).can_attack


def test_attack_distance_and_angle_thresholds_are_inclusive(profile) -> None:
    cfg = _cfg()
    low = compute_combat_geometry(
        make_state(profile, x=0.0, z=1500.0, heading=0.0),
        make_state(profile, x=40.0, z=1500.0, heading=0.0, team=Team.BLUE),
        cfg,
    )
    high = compute_combat_geometry(
        make_state(profile, x=0.0, z=1500.0, heading=0.0),
        make_state(profile, x=900.0, z=1500.0, heading=0.0, team=Team.BLUE),
        cfg,
    )
    angle_edge = compute_combat_geometry(
        make_state(profile, x=0.0, z=1500.0, heading=cfg.attack_angle_max),
        make_state(profile, x=500.0, z=1500.0, heading=0.0, team=Team.BLUE),
        cfg,
    )
    assert low.can_attack
    assert high.can_attack
    assert angle_edge.attacker_attack_angle == pytest.approx(cfg.attack_angle_max)
    assert angle_edge.can_attack


def test_mirrored_red_blue_geometry_is_symmetric(profile) -> None:
    cfg = _cfg()
    red = make_state(profile, x=-200.0, y=100.0, z=1500.0, heading=0.25, team=Team.RED)
    blue = make_state(profile, x=300.0, y=100.0, z=1500.0, heading=0.25, team=Team.BLUE)
    mirrored_red = replace(red, x=-red.x, heading_angle=pi - red.heading_angle, team_id=int(Team.RED))
    mirrored_blue = replace(blue, x=-blue.x, heading_angle=pi - blue.heading_angle, team_id=int(Team.BLUE))
    original = compute_combat_geometry(red, blue, cfg)
    mirrored = compute_combat_geometry(mirrored_red, mirrored_blue, cfg)
    assert mirrored.distance == pytest.approx(original.distance)
    assert mirrored.attacker_attack_angle == pytest.approx(original.attacker_attack_angle)
    assert mirrored.target_escape_angle == pytest.approx(original.target_escape_angle)
    assert mirrored.can_attack == original.can_attack


def test_swap_attacker_target_matches_directional_definition(profile) -> None:
    cfg = _cfg()
    attacker = make_state(profile, x=0.0, z=1500.0, heading=0.0, team=Team.RED)
    target = make_state(profile, x=500.0, z=1500.0, heading=0.0, team=Team.BLUE)
    forward = compute_combat_geometry(attacker, target, cfg)
    reverse = compute_combat_geometry(target, attacker, cfg)
    assert forward.distance == pytest.approx(reverse.distance)
    assert forward.can_attack
    assert not reverse.can_attack
    assert reverse.attacker_attack_angle == pytest.approx(pi)


def test_vertical_offset_uses_three_dimensional_distance(profile) -> None:
    cfg = _cfg()
    attacker = make_state(profile, x=0.0, z=1500.0, heading=0.0)
    target = make_state(profile, x=300.0, z=1900.0, heading=0.0, team=Team.BLUE)
    geometry = compute_combat_geometry(attacker, target, cfg)
    assert geometry.distance == pytest.approx(500.0)
    assert geometry.attacker_attack_angle > 0.0
    assert np.isfinite(geometry.line_of_sight).all()
