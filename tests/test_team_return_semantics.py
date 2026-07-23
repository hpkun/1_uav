import numpy as np

from uav_env.algorithms.mappo.adapter import AdapterStep, SyncCombatVectorEnv


def test_two_agent_step_and_multistep_return_semantics():
    steps = []
    for rewards in ([2.0, 4.0], [1.0, 3.0]):
        values = np.asarray(rewards, dtype=np.float32)
        steps.append(AdapterStep(np.zeros((2, 28), np.float32), np.zeros(40, np.float32), values,
                                 float(values.mean()), float(values.sum()), np.ones(2, np.float32),
                                 np.ones((2, 15), bool), False, False, {}))
    stacked = SyncCombatVectorEnv._stack(steps)
    assert np.allclose(stacked["team_rewards"], [3.0, 2.0])
    assert np.allclose(stacked["agent_reward_sums"], [6.0, 4.0])
    assert stacked["team_rewards"].sum() == 5.0
    assert stacked["agent_reward_sums"].sum() == 10.0


def test_one_agent_definitions_coincide():
    rewards = np.asarray([2.5], dtype=np.float32)
    step = AdapterStep(np.zeros((1, 11), np.float32), np.zeros(10, np.float32), rewards, 2.5, 2.5,
                       np.ones(1, np.float32), np.ones((1, 15), bool), False, False, {})
    assert step.team_reward == step.agent_reward_sum == float(step.agent_rewards[0])
