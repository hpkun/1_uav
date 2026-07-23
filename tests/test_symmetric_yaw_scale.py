from math import pi

import numpy as np
import pytest

from uav_env.observations.normalization import FeatureSpec, NormalizationConfig, normalize_by_specs
from uav_env.observations.single_observation import actor_feature_specs, critic_feature_specs
from uav_env.observations.multi_observation import multi_observation_specs


def yaw(value, mode):
    config = NormalizationConfig(mode=mode)
    return normalize_by_specs([value], [FeatureSpec("yaw", 2*pi, "yaw", pi)], config).values[0]


@pytest.mark.parametrize(("value", "expected"), [(7*pi/4, .75), (pi, 0.0)])
def test_paper_yaw(value, expected):
    assert yaw(value, "paper_linear") == pytest.approx(expected)


@pytest.mark.parametrize(("value", "expected"), [(pi, 1.0), (-pi, -1.0), (pi/2, .5)])
def test_symmetric_yaw(value, expected):
    assert yaw(value, "symmetric_training") == pytest.approx(expected)


def test_small_signed_yaws_are_opposites_and_all_tables_use_pi():
    assert yaw(.1, "symmetric_training") == pytest.approx(-yaw(-.1, "symmetric_training"))
    config = NormalizationConfig()
    tables = (actor_feature_specs(config), critic_feature_specs(config), multi_observation_specs(config))
    for table in tables:
        assert all(spec.symmetric_reference == pytest.approx(pi) for spec in table if spec.kind == "yaw")
