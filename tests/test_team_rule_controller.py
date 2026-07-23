from uav_env.opponents.straight import StraightOpponent
from uav_env.opponents.team_controller import TeamRuleController
from uav_env.envs import make_2v2_env

def test_team_controller_unique_targets():
 env=make_2v2_env(seed=1);env.reset(seed=1);c=TeamRuleController("straight",StraightOpponent(),9);actions,assign=c.select_actions(env.red_aircraft,env.blue_aircraft)
 assert len(actions)==2 and len({x.target_id for x in assign})==2

