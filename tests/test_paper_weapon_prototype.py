from pathlib import Path

import numpy as np
import pytest

from uav_combat.diagnostics.paper_weapon_prototype import (
    D_HIT, EntryTriggeredAttempt, PaperWeaponGeometry, fire_gate, hit_samples,
    hit_threshold,
)


ROOT = Path(__file__).resolve().parents[1]


def geometry(distance=1000.0, ata=0.0, ha=0.0):
    return PaperWeaponGeometry(distance, ata, ha)


def test_eq7_inclusive_boundaries_and_negative_controls():
    assert fire_gate(geometry(distance=0.0))
    assert fire_gate(geometry(distance=4000.0, ata=np.pi / 6, ha=-np.pi / 6))
    assert not fire_gate(geometry(distance=4000.0001))
    assert not fire_gate(geometry(ata=np.deg2rad(30.0001)))
    assert not fire_gate(geometry(ha=np.deg2rad(-30.0001)))


def test_eq8_threshold_decreases_strictly_with_distance():
    values = [hit_threshold(distance) for distance in (0, 1000, 2000, 3000, 4000)]
    assert all(left > right for left, right in zip(values, values[1:]))
    assert hit_threshold(4000.0) == pytest.approx(np.pi / 6)
    assert D_HIT == pytest.approx(4000.0 / np.log(6.0))


def test_seeded_monte_carlo_is_reproducible():
    first = hit_samples(geometry(4000), np.random.default_rng(123), 1000, "shared")
    second = hit_samples(geometry(4000), np.random.default_rng(123), 1000, "shared")
    assert np.array_equal(first, second)


def test_entry_trigger_only_attempts_on_rising_edge():
    trigger = EntryTriggeredAttempt()
    gates = [False, True, True, True, False, False, True, True, False]
    assert [trigger.update(value) for value in gates] == [
        False, True, False, False, False, False, True, False, False
    ]


def test_weapon_prototype_is_not_imported_by_active_runtime():
    active = [*ROOT.joinpath("src/uav_combat/environment").rglob("*.py"),
              *ROOT.joinpath("src/uav_combat/training").rglob("*.py"),
              *ROOT.joinpath("src/uav_combat/madsac").rglob("*.py")]
    for path in active:
        assert "paper_weapon_prototype" not in path.read_text(encoding="utf-8")
