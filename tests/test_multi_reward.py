from conftest import make_state
from uav_env.core.enums import Team
from uav_env.entities.type_profiles import UAVTypeProfile
from uav_env.rewards.multi_reward import pair_situation_reward
from uav_env.utils.config import load_multi_experiment_config


def test_multi_situation_reward_is_finite(profile: UAVTypeProfile) -> None:
    red=make_state(profile)
    blue=make_state(profile,x=1000,team=Team.BLUE)
    value=pair_situation_reward(red,blue,red,blue,load_multi_experiment_config())
    assert value==value
