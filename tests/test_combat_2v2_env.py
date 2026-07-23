import numpy as np
from dataclasses import replace

from uav_env.envs.combat_multi_env import CombatMultiEnv
from uav_env.utils.config import load_multi_experiment_config


def test_2v2_api_and_complete_short_episode() -> None:
    config=load_multi_experiment_config(); config["max_decision_steps"]=10; config["max_episode_seconds"]=5.0
    env=CombatMultiEnv(config,opponent="straight",seed=1)
    obs,info=env.reset(seed=1)
    assert obs.shape==(2,28) and info["global_state"].shape==(40,)
    terminated=truncated=False
    while not (terminated or truncated):
        obs,reward,terminated,truncated,info=env.step(np.array([0,0]))
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
        assert reward==np.mean(list(info["agent_rewards"].values()))
    assert len(env.get_trajectory())==env.decision_step+1


def test_timeout_winner_uses_survivor_count() -> None:
    config = load_multi_experiment_config()
    config["max_decision_steps"] = 1
    config["max_episode_seconds"] = config["decision_dt"]
    env = CombatMultiEnv(config, opponent="straight", seed=4)
    env.reset(seed=4)
    blue = env.blue_aircraft[0]
    blue.state = replace(blue.state, health=0.0, alive=False, damaged=True)

    _, _, terminated, truncated, info = env.step(np.array([0, 0]))

    assert not terminated and truncated
    assert info["outcome"].winner == "red"
    assert info["outcome"].red_survivors == 2
    assert info["outcome"].blue_survivors == 1


def test_all_living_aircraft_advance_in_same_decision_step() -> None:
    config = load_multi_experiment_config()
    env = CombatMultiEnv(config, opponent="straight", seed=5)
    env.reset(seed=5)
    before = {u.uav_id: u.state.position_vector().copy() for u in env.all_aircraft}

    env.step(np.array([0, 0]))

    for aircraft in env.all_aircraft:
        assert not np.allclose(aircraft.state.position_vector(), before[aircraft.uav_id])
