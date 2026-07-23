from uav_env.core.symmetry import mirror_action_xz,mirror_state_xz
from uav_env.envs import make_1v1_env
from uav_env.opponents.pursuit import PursuitOpponent

def test_pursuit_mirror_action():
 env=make_1v1_env(opponent="pursuit",seed=1);env.reset(seed=1);p=env.opponent_policy;a=p.select_action(env.blue.state,env.red.state);b=p.select_action(mirror_state_xz(env.blue.state),mirror_state_xz(env.red.state));assert b==mirror_action_xz(a)

def test_exact_collinear_pursuit_avoids_chiral_tie():
 env=make_1v1_env(scenario="head_on",opponent="pursuit",seed=1);env.reset(seed=1);p=env.opponent_policy;a=p.select_action(env.red.state,env.blue.state);assert int(a)<=8
