import pytest

from uav_env.rewards.multi_reward import assign_dense_rewards


def test_dense_assignment_hand_calculation() -> None:
    result=assign_dense_rewards({"red_0":0.2,"red_1":0.1},{"red_0":True,"red_1":True},0.01)
    factor=0.01*2+0.003*2/2+0.007*0.3/2
    assert result["red_0"]==pytest.approx(factor*0.2/0.3)
    assert result["red_1"]==pytest.approx(factor*0.1/0.3)
    assert assign_dense_rewards({"a":0.0,"b":-0.2},{"a":True,"b":True})=={"a":0.0,"b":-0.2}
