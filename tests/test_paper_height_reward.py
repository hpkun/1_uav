import pytest

from uav_env.rewards.components import paper_height_reward


@pytest.mark.parametrize("height,expected", [(-300.0,0.0),(100.0,1.0),(300.0,1.0),(500.0,0.1),(700.0,0.1),(-301.0,0.0),(901.0,0.0)])
def test_paper_height_formula_values(height: float, expected: float) -> None:
    assert paper_height_reward(height, 900.0) == pytest.approx(expected)
