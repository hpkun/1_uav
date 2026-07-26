from dataclasses import replace

import numpy as np
import pytest
import torch

from scripts.run_3v3_episode import run_3v3_episode
from uav_env.algorithms.mappo.adapter import CombatEnvDescription, ParallelCombatVectorEnv
from uav_env.algorithms.mappo.config import load_mappo_config
from uav_env.algorithms.mappo.runner import MAPPORunner
from uav_env.combat.multi_combat import MultiCombatStepResult, ResolvedAttack, assign_nearest_targets_independently
from uav_env.envs import make_3v3_env
from uav_env.rewards.multi_reward import individual_situation_reward


def _zero_dense_breakdown(breakdown) -> None:
    assert breakdown.situation == 0.0
    assert breakdown.event == 0.0
    assert breakdown.raw_dense == 0.0
    assert breakdown.assigned_dense == 0.0
    assert breakdown.terminal == 0.0
    assert breakdown.total == 0.0
    assert breakdown.contribution_score == 0.0


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


def test_3v3_situation_reward_uses_living_blue_set(monkeypatch) -> None:
    env = make_3v3_env(seed=13, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=13)
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, health=0.0, alive=False, damaged=True)
    values = {"blue_0": 99.0, "blue_1": 2.0, "blue_2": 1.0}

    def fake_pair(previous_red, previous_blue, red, blue, config):
        return values[blue.type_id]

    monkeypatch.setattr("uav_env.rewards.multi_reward.pair_situation_reward", fake_pair)
    for blue in env.blue_aircraft:
        blue.state.type_id = blue.uav_id
    previous = {u.uav_id: u.state.copy() for u in env.all_aircraft}
    assert individual_situation_reward(env.red_aircraft[0], env.blue_aircraft, previous, env.config) == 2.0


def test_3v3_previously_dead_red_slot_stays_zero_for_nonterminal_steps() -> None:
    env = make_3v3_env(seed=17, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=17)
    env.red_aircraft[0].state = replace(env.red_aircraft[0].state, health=0.0, alive=False, damaged=True)
    for _ in range(2):
        _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.int64))
        assert not terminated
        assert not truncated
        _zero_dense_breakdown(info["agent_reward_breakdowns"]["red_0"])


def test_3v3_red_destroyed_this_step_receives_one_event_then_zero_afterward() -> None:
    env = make_3v3_env(seed=19, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=19)
    min_altitude = float(env.config["min_altitude"])
    env.red_aircraft[0].state = replace(env.red_aircraft[0].state, z=min_altitude, health=300.0, alive=True, damaged=False, crashed=False)
    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.int64))
    assert not terminated
    assert not truncated
    breakdown = info["agent_reward_breakdowns"]["red_0"]
    active_raw = [item.raw_dense for item in info["agent_reward_breakdowns"].values() if item.raw_dense != 0.0 or item.event != 0.0 or item.situation != 0.0]
    assert breakdown.situation == 0.0
    assert breakdown.event == pytest.approx(-0.5)
    assert breakdown.raw_dense == pytest.approx(-0.5)
    expected = min(-float(env.config["r_den0"]) * 3 - min(active_raw), -float(env.config["r_den0"]) * 3)
    assert breakdown.assigned_dense == pytest.approx(expected)
    assert breakdown.assigned_dense < 0.0
    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.int64))
    assert not terminated
    assert not truncated
    _zero_dense_breakdown(info["agent_reward_breakdowns"]["red_0"])


def test_3v3_red_destroyed_by_attack_this_step_never_gets_positive_dense(monkeypatch) -> None:
    env = make_3v3_env(seed=29, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=29)

    def fake_resolve_multi_attacks(aircraft, attack_config, damage_config, rng, sample_team_order=None):
        states = {u.uav_id: u.state.copy() for u in aircraft}
        states["red_0"] = replace(states["red_0"], health=0.0, alive=False, damaged=True, ever_hit=True)
        return MultiCombatStepResult(
            states,
            [],
            [ResolvedAttack("blue_0", "red_0", 100.0, 0.0, 300.0, 300.0, 0.0, True, True)],
        )

    monkeypatch.setattr("uav_env.envs.combat_multi_env.resolve_multi_attacks", fake_resolve_multi_attacks)
    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.int64))
    assert not terminated
    assert not truncated
    breakdown = info["agent_reward_breakdowns"]["red_0"]
    active_raw = [item.raw_dense for item in info["agent_reward_breakdowns"].values() if item.raw_dense != 0.0 or item.event != 0.0 or item.situation != 0.0]
    expected = min(-float(env.config["r_den0"]) * 3 - min(active_raw), -float(env.config["r_den0"]) * 3)
    assert breakdown.situation == 0.0
    assert breakdown.event == pytest.approx(-2.5)
    assert breakdown.assigned_dense == pytest.approx(expected)
    assert breakdown.assigned_dense < 0.0
    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.int64))
    assert not terminated
    assert not truncated
    _zero_dense_breakdown(info["agent_reward_breakdowns"]["red_0"])


def test_3v3_fixed_slots_zero_after_prior_death_but_terminal_allocation_remains() -> None:
    env = make_3v3_env(seed=23, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=23)
    env.red_aircraft[0].state = replace(env.red_aircraft[0].state, health=0.0, alive=False, damaged=True)
    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.int64))
    assert not terminated
    assert not truncated
    _zero_dense_breakdown(info["agent_reward_breakdowns"]["red_0"])

    for blue in env.blue_aircraft:
        blue.state = replace(blue.state, health=0.0, alive=False, damaged=True)
    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.int64))
    assert terminated
    assert not truncated
    breakdown = info["agent_reward_breakdowns"]["red_0"]
    assert breakdown.situation == 0.0
    assert breakdown.event == 0.0
    assert breakdown.raw_dense == 0.0
    assert breakdown.assigned_dense == 0.0
    assert breakdown.terminal != 0.0
    assert breakdown.total == pytest.approx(breakdown.terminal)


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


def test_parallel_run_saves_periodic_step_and_one_final_last(tmp_path) -> None:
    config = load_mappo_config("configs/mappo_smoke_3v3.yaml")
    config.update(
        rollout_length=1, total_env_steps=8, checkpoint_interval=4, evaluation_interval=1000,
        ppo_epochs=1, num_mini_batches=1, validation_episodes=1, test_episodes=1,
        device="cpu", run_id="checkpoint_schedule",
    )
    runner = MAPPORunner(config, "3v3_schedule", tmp_path)
    output = runner.run()
    checkpoints = output / "checkpoints"
    assert (checkpoints / "initial.pt").is_file()
    assert (checkpoints / "step_4.pt").is_file()
    assert not (checkpoints / "step_8.pt").exists()
    assert (checkpoints / "best.pt").is_file()
    assert (checkpoints / "last.pt").is_file()
    assert not any(runner.vector.workers_alive)


def test_keyboard_interrupt_does_not_create_extra_checkpoint_and_closes_workers(tmp_path, monkeypatch, capsys) -> None:
    config = load_mappo_config("configs/mappo_smoke_3v3.yaml")
    config.update(rollout_length=1, total_env_steps=8, device="cpu", run_id="interrupted")
    runner = MAPPORunner(config, "3v3_interrupt", tmp_path)

    def interrupt_collect():
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "collect", interrupt_collect)
    with pytest.raises(KeyboardInterrupt):
        runner.run()
    checkpoints = runner.output_dir / "checkpoints"
    assert (checkpoints / "initial.pt").is_file()
    assert not (checkpoints / "last.pt").exists()
    assert "no step checkpoint is available" in capsys.readouterr().out
    assert not any(runner.vector.workers_alive)
