from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from uav_env.envs import make_1v1_env


def test_reset_and_step_follow_gymnasium_api() -> None:
    env = make_1v1_env(seed=5)
    check_env(env, skip_render_check=True)
    observation, info = env.reset(seed=5)
    assert observation.shape == (11,)
    assert info["critic_state"].shape == (10,)
    observation, reward, terminated, truncated, info = env.step(0)
    assert observation.shape == (11,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["reward_breakdown"].total == reward
    assert info["reward_breakdown"].additive_total() == reward


def test_complete_episode_terminates_or_truncates() -> None:
    env = make_1v1_env(scenario="tail_chase", opponent="straight", seed=1)
    env.reset(seed=1)
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(0)
        assert np.isfinite(reward)
        assert np.all(np.isfinite(info["red_state"].to_kinematic_vector()))
        assert np.all(np.isfinite(info["blue_state"].to_kinematic_vector()))
    assert info["outcome"].termination_reason != "ongoing"
    assert len(env.get_trajectory()) == info["decision_step"] + 1
