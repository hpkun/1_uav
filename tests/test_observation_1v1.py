from __future__ import annotations

import numpy as np
import pytest

from conftest import make_state
from uav_env.core.enums import Team
from uav_env.entities.type_profiles import UAVTypeProfile
from uav_env.observations.normalization import NormalizationConfig
from uav_env.observations.single_observation import (
    actor_observation_raw_1v1,
    build_actor_observation_1v1,
    build_critic_state_1v1,
    critic_state_raw_1v1,
)


def test_actor_observation_order_and_values(profile: UAVTypeProfile) -> None:
    own = make_state(profile, x=0.0, y=0.0, z=1000.0, speed=100.0)
    enemy = make_state(profile, x=300.0, y=400.0, z=1000.0, speed=80.0, team=Team.BLUE)
    raw = actor_observation_raw_1v1(own, enemy)
    assert raw.shape == (11,)
    assert raw[:4] == pytest.approx([-300.0, -400.0, 1000.0, 500.0])
    assert raw[4] == pytest.approx(0.0)
    assert raw[5] == pytest.approx(np.arctan2(400.0, 300.0))
    assert raw[6:9] == pytest.approx([20.0, 0.0, 0.0])
    normalization = NormalizationConfig(mode="paper_linear", speed_difference_reference=150.0, clip_observation=False)
    normalized = build_actor_observation_1v1(own, enemy, normalization)
    assert normalized[0] == pytest.approx(2.0 * -300.0 / 5000.0 - 1.0)
    assert normalized[2] == pytest.approx(2.0 * 1000.0 / 5000.0 - 1.0)


def test_actor_normalization_clips_when_enabled(profile: UAVTypeProfile) -> None:
    own = make_state(profile, x=-10_000.0)
    enemy = make_state(profile, x=10_000.0, team=Team.BLUE)
    observation = build_actor_observation_1v1(own, enemy, NormalizationConfig(speed_difference_reference=150.0))
    assert observation.shape == (11,)
    assert np.max(np.abs(observation)) <= 1.0


def test_critic_state_order_and_angle_differences(profile: UAVTypeProfile) -> None:
    own = make_state(profile, x=0.0, z=1000.0, speed=100.0, heading=6.1)
    enemy = make_state(profile, x=100.0, z=1200.0, speed=80.0, heading=0.1, team=Team.BLUE)
    raw = critic_state_raw_1v1(own, enemy)
    assert raw.shape == (10,)
    assert raw[0] == pytest.approx(np.sqrt(100.0**2 + 200.0**2))
    assert raw[3] == 200.0
    assert -np.pi <= raw[5] < np.pi
    assert raw[7:] == pytest.approx([20.0, 300.0, 0.0])
    normalization = NormalizationConfig(mode="paper_linear", speed_difference_reference=150.0, clip_observation=False)
    normalized = build_critic_state_1v1(own, enemy, normalization)
    assert normalized[0] == pytest.approx(2.0 * raw[0] / 5000.0 - 1.0)
    assert normalized[8] == pytest.approx(1.0)
