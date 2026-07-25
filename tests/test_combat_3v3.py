from dataclasses import replace

import numpy as np
import torch

from scripts.run_3v3_episode import run_3v3_episode
from uav_env.algorithms.mappo.adapter import CombatEnvDescription, ParallelCombatVectorEnv
from uav_env.algorithms.mappo.config import load_mappo_config
from uav_env.algorithms.mappo.runner import MAPPORunner
from uav_env.combat.multi_combat import assign_nearest_targets_independently
from uav_env.envs import make_3v3_env


def test_3v3_reset_and_step_shapes() -> None:
    env = make_3v3_env(seed=7)
    observation, info = env.reset(seed=7)
    assert env.red_count == env.blue_count == env.num_red_agents == 3
    assert env.local_observation_dim == 45
    assert env.global_state_dim == 87
    assert observation.shape == (3, 45)
    assert info["global_state"].shape == (87,)
    assert info["available_action_mask"].shape == (3, 15)
    observation, reward, _, _, info = env.step(np.zeros(3, dtype=np.int64))
    assert observation.shape == (3, 45)
    assert set(info["agent_rewards"]) == {"red_0", "red_1", "red_2"}
    assert np.isfinite(reward) and np.all(np.isfinite(observation))


def test_independent_nearest_targets_allow_reuse_and_break_ties_by_id() -> None:
    env = make_3v3_env(seed=9)
    env.reset(seed=9)
    for index, blue in enumerate(env.blue_aircraft):
        blue.state = replace(blue.state, x=0.0, y=float(index), z=1800.0)
    env.red_aircraft[0].state = replace(env.red_aircraft[0].state, x=10.0, y=0.0, z=1800.0)
    env.red_aircraft[1].state = replace(env.red_aircraft[1].state, x=-10.0, y=0.0, z=1800.0)
    env.red_aircraft[2].state = replace(env.red_aircraft[2].state, x=100.0, y=0.0, z=1800.0)
    assignments = assign_nearest_targets_independently(env.blue_aircraft, env.red_aircraft)
    assert [item.attacker_id for item in assignments] == ["blue_0", "blue_1", "blue_2"]
    assert [item.target_id for item in assignments] == ["red_0", "red_0", "red_0"]


def test_3v3_rule_episode_reaches_a_boundary() -> None:
    _, summary = run_3v3_episode(seed=3)
    assert summary.termination_reason != "ongoing"
    assert 0 <= summary.red_survivors <= 3
    assert 0 <= summary.blue_survivors <= 3


def test_parallel_3v3_shapes_terminal_retention_and_close() -> None:
    description = CombatEnvDescription("3v3", "head_on_formation", "pursuit", "paper_2024_exact")
    vector = ParallelCombatVectorEnv(description, 4, 21)
    try:
        reset = vector.reset()
        assert reset["local_obs"].shape == (4, 3, 45)
        assert reset["global_state"].shape == (4, 87)
        assert reset["available_actions"].shape == (4, 3, 15)
        result = None
        for _ in range(400):
            result = vector.step(np.zeros((4, 3), dtype=np.int64))
            if np.any(result["terminated"] | result["truncated"]):
                break
        assert result is not None
        completed = np.flatnonzero(result["terminated"] | result["truncated"])
        assert completed.size > 0
        index = int(completed[0])
        assert result["terminal_steps"][index].global_state.shape == (87,)
        assert result["reset_steps"][index] is not None
        assert result["next_global_state"][index].shape == (87,)
    finally:
        vector.close()
    assert not any(vector.workers_alive)


def test_3v3_parallel_rollout_and_update_change_networks(tmp_path) -> None:
    config = load_mappo_config("configs/mappo_smoke_3v3.yaml")
    config.update(rollout_length=2, ppo_epochs=1, num_mini_batches=1, validation_episodes=1, test_episodes=1, device="cpu", run_id="test")
    runner = MAPPORunner(config, "3v3_update", tmp_path)
    actor_before = [parameter.detach().clone() for parameter in runner.actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in runner.critic.parameters()]
    try:
        buffer, _ = runner.collect()
        metrics = runner.trainer.update(buffer)
    finally:
        runner.close()
    assert any(not torch.equal(before, after) for before, after in zip(actor_before, runner.actor.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(critic_before, runner.critic.parameters()))
    assert np.all(np.isfinite(list(metrics.values())))
