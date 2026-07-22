from __future__ import annotations

from uav_env.envs import make_1v1_env


def summarize(seed: int) -> list[tuple[float, float, float, int]]:
    env = make_1v1_env(scenario="balanced_random", opponent="random", seed=seed)
    env.reset(seed=seed)
    summary = []
    for action in [0, 1, 9, 12, 3, 6, 2, 10]:
        _, reward, terminated, truncated, info = env.step(action)
        summary.append((info["red_state"].x, info["blue_state"].y, reward, info["blue_action"]))
        if terminated or truncated:
            break
    return summary


def test_same_seed_and_actions_are_identical() -> None:
    assert summarize(17) == summarize(17)


def test_different_seeds_change_randomized_episode() -> None:
    assert summarize(17) != summarize(18)
