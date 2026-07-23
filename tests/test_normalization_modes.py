from math import pi

import numpy as np
import pytest

from conftest import make_state
from uav_env.core.enums import Team
from uav_env.entities.type_profiles import UAVTypeProfile
from uav_env.observations.normalization import NormalizationConfig
from uav_env.observations.single_observation import build_actor_observation_1v1


def test_signed_positions_do_not_collapse_and_are_symmetric(profile: UAVTypeProfile) -> None:
    cfg = NormalizationConfig(mode="symmetric_training", speed_difference_reference=150.0)
    own = make_state(profile, x=0.0)
    enemy_1200 = make_state(profile, x=1200.0, team=Team.BLUE)
    enemy_1800 = make_state(profile, x=1800.0, team=Team.BLUE)
    assert build_actor_observation_1v1(own, enemy_1200, cfg)[0] == pytest.approx(-1200/5000)
    assert build_actor_observation_1v1(own, enemy_1800, cfg)[0] == pytest.approx(-1800/5000)
    reverse = build_actor_observation_1v1(enemy_1200, own, cfg)[0]
    assert reverse == pytest.approx(1200/5000)


def test_paper_yaw_wrap_and_symmetric_bounds(profile: UAVTypeProfile) -> None:
    own = make_state(profile)
    enemy = make_state(profile, x=100.0, y=-100.0, team=Team.BLUE)
    paper = build_actor_observation_1v1(own, enemy, NormalizationConfig(mode="paper_linear"))
    expected_yaw = 7.0 * pi / 4.0
    assert paper[5] == pytest.approx(2.0 * expected_yaw / (2.0*pi) - 1.0)
    symmetric = build_actor_observation_1v1(own, enemy, NormalizationConfig(mode="symmetric_training"))
    assert np.all(np.isfinite(symmetric))
    assert np.max(np.abs(symmetric)) <= 1.0
