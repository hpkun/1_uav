"""Current-target pure pursuit changes only guidance, not assignment or avoidance."""
import numpy as np

from algorithm.ideal_collision_avoidance import AvoidanceConfig
from env.ideal_interception.large_scale_20v20_env import TwentyVTwentyEnv
from tools.ideal_20v20_nearest_pp_cpa import decide


def test_pure_pursuit_uses_nearest_assignment_and_cpa_direction():
    env = TwentyVTwentyEnv()
    _, info = env.reset(17)
    actions, count = decide(info, AvoidanceConfig(mode="predictive_cpa_knn"))
    assert actions.shape == (20, 3) and np.isfinite(actions).all()
    assert np.all((0 <= actions) & (actions <= 1))
    assert count >= 0
    for i, j in info["assignment"].items():
        raw = info["intruders"][j, :3]-info["defenders"][i, :3]
        raw /= np.linalg.norm(raw)
        np.testing.assert_allclose(actions[i], .5+.5*raw)
