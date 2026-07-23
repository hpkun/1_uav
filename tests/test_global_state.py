import pytest

from test_multi_observation import aircraft
from uav_env.entities.type_profiles import UAVTypeProfile
from uav_env.observations.global_state import build_global_state_2v2
from uav_env.observations.normalization import NormalizationConfig


def test_global_state_shape_order_and_actions(profile: UAVTypeProfile) -> None:
    reds,blues=aircraft(profile)
    result=build_global_state_2v2(reds,blues,NormalizationConfig())
    assert result.raw.shape==(40,) and result.normalized.shape==(40,)
    assert result.feature_names[2].startswith("red_0_blue_0")
    assert result.feature_names[11].startswith("red_0_blue_1")
    assert result.raw[-2:].tolist()==[0.0,0.0]
    paper=build_global_state_2v2(reds,blues,NormalizationConfig(mode="paper_linear"))
    assert paper.normalized[-2:].tolist()==[0.0,0.0]
