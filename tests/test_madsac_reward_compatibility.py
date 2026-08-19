import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.diagnose_madsac_reward_compatibility import (
    DEG_30,
    EXPECTED_ACTIVE_REWARD_SHA256,
    paper_geometry,
    paper_r1,
    paper_r2_v1_4,
    paper_r3,
    paper_r4,
)
from uav_combat.models import AircraftState


ROOT = Path(__file__).resolve().parents[1]


def state(x=0.0, y=0.0, z=-3000.0, psi=0.0):
    return AircraftState(x, y, z, 225.0, 0.0, psi)


def test_paper_geometry_head_on_tail_same_heading_and_crossing():
    head_red, head_blue = state(), state(x=1000.0, psi=np.pi)
    head = paper_geometry(head_red, head_blue)
    assert head.ata == pytest.approx(0.0)
    assert abs(head.aa) == pytest.approx(np.pi)

    tail_red, tail_blue = state(), state(x=1000.0)
    tail = paper_geometry(tail_red, tail_blue)
    assert tail.ata == pytest.approx(0.0)
    assert tail.aa == pytest.approx(0.0)

    same_red, same_blue = state(), state(y=1000.0)
    same = paper_geometry(same_red, same_blue)
    assert same.ata == pytest.approx(np.pi / 2)
    assert same.aa == pytest.approx(-np.pi / 2)

    cross_red, cross_blue = state(), state(x=1000.0, psi=np.pi / 2)
    cross = paper_geometry(cross_red, cross_blue)
    assert cross.ata == pytest.approx(0.0)
    assert cross.aa == pytest.approx(np.pi / 2)


def test_paper_geometry_height_angle_uses_ned_and_radians():
    geometry = paper_geometry(state(), state(x=1000.0, z=-4000.0))
    assert geometry.ha == pytest.approx(np.pi / 4)


def test_r3_thresholds_are_inclusive_as_printed():
    exact = paper_geometry(state(), state(x=4000.0, psi=0.0))
    assert paper_r3(exact) == pytest.approx(0.001)
    too_close = type(exact)(3999.999, exact.ata, exact.aa, exact.ha)
    assert paper_r3(too_close) == 0.0
    edge_angle = type(exact)(4000.0, DEG_30, exact.aa, DEG_30)
    assert paper_r3(edge_angle) == pytest.approx(0.001)


def test_r4_strongest_first_tiers_and_printed_branch_precedence():
    red = paper_geometry(state(), state(x=4000.0))
    blue = paper_geometry(state(x=4000.0), state())
    assert paper_r4(red, blue) == pytest.approx(0.1)
    red15 = type(red)(4000.0, np.deg2rad(15), 0.0, np.deg2rad(15))
    assert paper_r4(red15, blue) == pytest.approx(0.02)
    red30 = type(red)(4000.0, np.deg2rad(30), 0.0, np.deg2rad(30))
    assert paper_r4(red30, blue) == pytest.approx(0.01)
    red_out = type(red)(4000.0, np.deg2rad(31), 0.0, 0.0)
    assert paper_r4(red_out, blue) == 0.0


def test_r4_negative_blue_centered_tiers():
    red = paper_geometry(state(), state(x=1000.0, psi=np.pi))
    blue_adv = type(red)(1000.0, 0.0, 0.0, 0.0)
    assert paper_r4(red, blue_adv) == pytest.approx(-0.15)


def test_r1_and_v1_4_r2_adaptation():
    assert paper_r1(1, 0) == 10.0
    assert paper_r1(0, 1) == -10.0
    assert paper_r1(2, 3) == -10.0
    value, status = paper_r2_v1_4()
    assert value == 0.0
    assert status == "not applicable in V1.4 horizontal-unbounded setting"


def test_diagnostic_is_isolated_and_active_reward_unchanged():
    active_files = [*ROOT.joinpath("src").rglob("*.py"), ROOT / "scripts/train_madsac.py"]
    for path in active_files:
        assert "diagnose_madsac_reward_compatibility" not in path.read_text(encoding="utf-8")
    reward = ROOT / "src/uav_combat/environment/reward.py"
    assert hashlib.sha256(reward.read_bytes()).hexdigest() == EXPECTED_ACTIVE_REWARD_SHA256
