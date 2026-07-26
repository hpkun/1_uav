import pytest

from uav_env.rewards.multi_reward import assign_dense_rewards


def test_dense_assignment_hand_calculation() -> None:
    result=assign_dense_rewards({"red_0":0.2,"red_1":0.1},{"red_0":False,"red_1":False},2,0.01)
    factor=0.01*2+0.003*2/2+0.007*0.3/2
    assert result["red_0"]==pytest.approx(factor*0.2/0.3)
    assert result["red_1"]==pytest.approx(factor*0.1/0.3)
    assert assign_dense_rewards({"a":0.0,"b":-0.2},{"a":False,"b":False},2)=={"a":0.0,"b":-0.2}


def test_dense_assignment_uses_fixed_team_size_and_active_minimum() -> None:
    result = assign_dense_rewards({"red_1": 0.2, "red_2": -0.4}, {"red_1": True, "red_2": False}, 3, 0.01)
    assert set(result) == {"red_1", "red_2"}
    assert result["red_1"] == pytest.approx(-0.03)
    assert result["red_1"] != pytest.approx(0.37)
    assert result["red_2"] == pytest.approx(-0.4)


def test_dense_assignment_keeps_negative_literal_damage_branch() -> None:
    result = assign_dense_rewards({"red_0": 0.2, "red_1": 0.1}, {"red_0": True, "red_1": False}, 3, 0.01)
    assert result["red_0"] == pytest.approx(-0.13)
