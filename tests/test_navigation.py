import numpy as np
import pytest
from env import EnvConfig
from env.factory import make_environment
from tools.navigation_baseline import NavigationConfig,NavigationController,reachable_scenario
from tools.audit_navigation import run_episode,aggregate


def test_navigation_legal_and_physical_inverse():
    c=EnvConfig(); env=make_environment(c); _,info=env.reset(3)
    actions,diag=NavigationController().actions(info,c)
    assert actions.shape==(3,3) and np.all((actions>=0)&(actions<=1))
    np.testing.assert_allclose(diag['desired_rates'],diag['achieved_rates'],atol=1e-12)
    np.testing.assert_array_equal(env.task.states,info['states'])


def test_reachable_multitask_navigation():
    result=run_episode(EnvConfig(scale='3U5P',random_scenario=False),1,'navigation',options=reachable_scenario())
    assert result['success'] and result['completed_waypoints']==5
    assert sum(result['per_uav_completed_waypoints'])==5
    assert result['later_assignments'] and result['reassignment_completions']
    assert result['episode_length']<250


def test_navigation_saturation_recorded():
    c=EnvConfig(); env=make_environment(c); _,info=env.reset(1)
    ctrl=NavigationController(NavigationConfig(max_psi_rate=10,k_psi=10))
    info['states'][:,5]=np.pi
    _,diag=ctrl.actions(info,c)
    assert diag['normal_load_saturation'].any()
    assert not np.allclose(diag['desired_rates'],diag['achieved_rates'])


def test_clip_rates_use_active_uav_steps():
    rows=[dict(success=False,active_uav_steps=10,gamma_clips=2,speed_clips=1),dict(success=True,active_uav_steps=5,gamma_clips=1,speed_clips=2)]
    for r in rows:
        for k in ('completion_rate','completed_waypoints','episode_length','initial_nearest_waypoint_distance','final_nearest_waypoint_distance','episode_progress','total_travel_distance','average_abs_gamma','average_speed','minimum_inter_uav_distance','waypoint_assignment_switches'): r[k]=1
    summary=aggregate(rows)
    assert summary['gamma_clip_rate']==pytest.approx(3/15)
    assert summary['speed_clip_rate']==pytest.approx(3/15)
