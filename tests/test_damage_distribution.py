from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from conftest import make_state
from uav_env.combat.damage import DamageConfig, damage_for_random_value, sample_damage
from uav_env.entities.type_profiles import UAVTypeProfile


def test_damage_boundaries() -> None:
    config = DamageConfig()
    assert damage_for_random_value(0.0, config) == 51.0
    assert damage_for_random_value(0.1, config) == 21.0
    assert damage_for_random_value(0.4, config) == 11.0
    assert damage_for_random_value(0.8, config) == 0.0


def test_monte_carlo_damage_probabilities() -> None:
    config = DamageConfig()
    samples = np.random.default_rng(12345).random(100_000)
    counts = Counter(damage_for_random_value(float(value), config) for value in samples)
    empirical = [counts[value] / len(samples) for value in config.damage_values]
    assert empirical == pytest.approx([0.1, 0.3, 0.4, 0.2], abs=0.006)


def test_seed_controls_damage_sample_and_health_floor(profile: UAVTypeProfile) -> None:
    target = make_state(profile, health=10.0)
    updated_a, result_a = sample_damage(target, DamageConfig(), np.random.default_rng(1), True)
    _, result_b = sample_damage(target, DamageConfig(), np.random.default_rng(2), True)
    assert result_a.random_value != result_b.random_value
    if result_a.damage > 0.0:
        assert updated_a.health == 0.0
        assert not updated_a.alive
        assert updated_a.damaged
