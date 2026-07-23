from dataclasses import replace
import pytest
from conftest import make_state
from uav_env.entities.uav import UAV
from uav_env.combat.events import EpisodeOutcome
from uav_env.rewards.multi_reward import multi_terminal_reward_allocations

def aircraft(uid, profile):
 state=make_state(profile);return UAV(uid, state.team_id, state, profile)

def test_win_formula_complete_hand_value(profile,experiment_config):
 config=experiment_config;config["multi_terminal_reward_profile"]="paper_2024_exact"; reds=[aircraft("red_0",profile),aircraft("red_1",profile)];reds[1].state=replace(reds[1].state,health=150)
 out=EpisodeOutcome("red",True,False,"blue_eliminated",1,.5,2,0);a=multi_terminal_reward_allocations(out,reds,{"red_0":3,"red_1":1},config);base=50*2*(1+399/400)
 assert a["red_0"].team_base==pytest.approx(base);assert a["red_0"].allocation_factor==pytest.approx(1/6+1/4+1/6)
 assert a["red_1"].allocation_factor==pytest.approx(1/6+1/12+1/12)

def test_loss_reverse_contribution_and_last_step(profile,experiment_config):
 config=experiment_config;config["multi_terminal_reward_profile"]="paper_2024_exact";reds=[aircraft("red_0",profile),aircraft("red_1",profile)];reds[0].state=replace(reds[0].state,health=0,alive=False,damaged=True);reds[1].state=replace(reds[1].state,health=0,alive=False,damaged=True)
 out=EpisodeOutcome("blue",False,True,"red_eliminated",400,200,0,2);a=multi_terminal_reward_allocations(out,reds,{"red_0":4,"red_1":1},config);assert a["red_0"].team_base==pytest.approx(-80);assert a["red_1"].contribution_component>a["red_0"].contribution_component

def test_draw_configured(profile,experiment_config):
 reds=[aircraft("red_0",profile),aircraft("red_1",profile)];out=EpisodeOutcome("draw",True,True,"timeout",400,200,2,2);assert all(x.reward==0 for x in multi_terminal_reward_allocations(out,reds,{},experiment_config).values())
