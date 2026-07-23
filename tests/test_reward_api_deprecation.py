import pytest
from uav_env.rewards.components import piecewise_distance_reward,height_reward

def test_ambiguous_reward_aliases_warn_and_use_paper():
 with pytest.warns(DeprecationWarning): assert piecewise_distance_reward(40,40,900,1300,5000)==0
 with pytest.warns(DeprecationWarning): assert height_reward(100)==1

