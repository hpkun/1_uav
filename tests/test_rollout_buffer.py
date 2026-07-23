from uav_env.algorithms.mappo.rollout_buffer import RolloutBuffer

def test_rollout_shapes():
 b=RolloutBuffer(4,2,2,28,40);assert b.observations.shape==(5,2,2,28) and b.values.shape==(5,2,2) and b.rewards.shape==(4,2,2)

