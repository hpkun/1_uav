import numpy as np

from conftest import make_state
from uav_env.core.enums import Team
from uav_env.entities.uav import UAV
from uav_env.entities.type_profiles import UAVTypeProfile
from uav_env.observations.multi_observation import build_multi_observations
from uav_env.observations.normalization import NormalizationConfig


def aircraft(profile: UAVTypeProfile):
    reds=[UAV(f"red_{i}",1,make_state(profile,x=i*100),profile) for i in range(2)]
    blues=[UAV("blue_0",0,make_state(profile,x=500,team=Team.BLUE),profile),UAV("blue_1",0,make_state(profile,x=1000,team=Team.BLUE),profile)]
    return reds,blues


def test_multi_observation_shape_sort_and_dead_slot(profile: UAVTypeProfile) -> None:
    reds,blues=aircraft(profile)
    result=build_multi_observations(reds,blues,NormalizationConfig())
    assert result.raw.shape==(2,28) and result.normalized.shape==(2,28)
    assert result.raw[0,9] < result.raw[0,20]
    blues[0].state.alive=False; blues[0].state.damaged=True
    result=build_multi_observations(reds,blues,NormalizationConfig())
    assert result.enemy_alive_masks.tolist()==[[1,0],[1,0]]
    assert np.all(result.normalized[:,17:28]==0.0)
