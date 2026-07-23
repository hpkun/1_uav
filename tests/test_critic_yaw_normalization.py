from math import pi
import pytest
from uav_env.observations.normalization import NormalizationConfig,normalize_by_specs
from uav_env.observations.single_observation import critic_feature_specs

def test_critic_paper_yaw_reference():
 c=NormalizationConfig(mode="paper_linear")
 spec=critic_feature_specs(c)[4]
 assert spec.reference==pytest.approx(2*pi)
 assert normalize_by_specs([7*pi/4],[spec],c).values[0]==pytest.approx(0.75)
 assert normalize_by_specs([pi],[spec],c).values[0]==pytest.approx(0.0)

def test_symmetric_yaw_sign():
 c=NormalizationConfig(mode="symmetric_training"); spec=critic_feature_specs(c)[4]
 assert normalize_by_specs([0.2],[spec],c).values[0]==pytest.approx(-normalize_by_specs([-0.2],[spec],c).values[0])

