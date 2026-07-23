import pytest

from uav_env.rewards.components import paper_distance_approach_reward, paper_distance_reward, paper_piecewise_distance_reward


MIN, MAX, ADV, DESIRED = 40.0, 900.0, 1300.0, 5000.0
MID = (MIN + MAX) / 2.0


@pytest.mark.parametrize("distance,expected", [(MIN,0.0),(MID,0.75),(MAX,0.5),(ADV,0.25),(DESIRED,0.0)])
def test_paper_piecewise_formula_values(distance: float, expected: float) -> None:
    assert paper_piecewise_distance_reward(distance, MIN, MAX, ADV, DESIRED) == pytest.approx(expected)


def test_paper_approach_sign_and_total() -> None:
    assert paper_distance_approach_reward(1000.0, 999.0, MID) == 0.25
    assert paper_distance_approach_reward(999.0, 1000.0, MID) == 0.0
    assert paper_distance_approach_reward(1000.0, 1000.0, MID) == 0.0
    total, approach, piecewise = paper_distance_reward(1001.0, 1000.0, MIN, MAX, ADV, DESIRED)
    assert total == pytest.approx(approach + piecewise)
