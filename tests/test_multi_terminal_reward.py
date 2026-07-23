from test_multi_observation import aircraft
from uav_env.combat.events import EpisodeOutcome
from uav_env.entities.type_profiles import UAVTypeProfile
from uav_env.rewards.multi_reward import multi_terminal_rewards
from uav_env.utils.config import load_multi_experiment_config


def test_terminal_reward_zero_safe_and_draw(profile: UAVTypeProfile) -> None:
    reds,_=aircraft(profile); config=load_multi_experiment_config()
    draw=EpisodeOutcome("draw",True,True,"timeout",400,200,2,2)
    assert multi_terminal_rewards(draw,reds,{"red_0":0,"red_1":0},config)=={"red_0":0.0,"red_1":0.0}
    win=EpisodeOutcome("red",True,False,"blue_eliminated",10,5,2,0)
    values=multi_terminal_rewards(win,reds,{"red_0":0,"red_1":0},config)
    assert all(value>0 for value in values.values())
