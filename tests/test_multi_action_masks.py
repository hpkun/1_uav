import numpy as np

from uav_env.envs import make_2v2_env


def test_dead_agent_action_is_ignored_and_masked() -> None:
    env=make_2v2_env(seed=2); env.reset(seed=2)
    env.red_aircraft[0].state.alive=False; env.red_aircraft[0].state.damaged=True
    _,_,_,_,info=env.step(np.array([14,0]))
    assert info["red_actions"][0]==0
    assert info["available_action_mask"][0].tolist()==[1]+[0]*14
