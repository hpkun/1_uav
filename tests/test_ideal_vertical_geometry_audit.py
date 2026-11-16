from dataclasses import replace

import pytest

from algorithm.ideal_predictive_guidance import run_episode
from env.ideal_interception import InterceptionConfig,InterceptionEnv
from tools.ideal_vertical_geometry_audit import audit


def test_seed17_strict_zuo_first_prediction_changes_only_with_geometry():
    base=InterceptionConfig()
    old=replace(base,intruder_goal_altitude_mode="asset_z")
    previous,_=run_episode(old,"zuo_tp",17)
    current,_=run_episode(base,"zuo_tp",17)
    a=previous["first_zuo_ready_prediction"]
    b=current["first_zuo_ready_prediction"]
    assert a["decision_time_s"]==b["decision_time_s"]==pytest.approx(.05)
    assert a["defender_position_m"]==b["defender_position_m"]
    assert a["predicted_point_m"][2]<0<b["predicted_point_m"][2]
    assert not a["predicted_point_in_bounds"] and b["predicted_point_in_bounds"]
    assert a["alignment"]<0 and b["alignment"]<0
    assert abs(a["horizon_s"]-b["horizon_s"])<.01
    assert previous["predicted_point_below_floor_count"]>0
    assert current["predicted_point_below_floor_count"]==0
    assert previous["outcome"]==current["outcome"]=="defender_boundary"


def test_vertical_goal_clips_to_protected_height():
    base=InterceptionConfig(protected_height=50.)
    info=InterceptionEnv(base).reset(17)[1]
    plan=info["intruder_plans"][0]
    assert plan["start_altitude_m"]>50.
    assert plan["goal_altitude_m"]==50.
    assert plan["altitude_change_m"]==pytest.approx(50.-plan["start_altitude_m"])
    assert info["scene_diagnostics"]["scenario_validity"]["region_crossing"]


def test_old_new_ablation_keeps_same_start_and_reports_geometry():
    result=audit([17,18])
    assert len(result["trials"])==2
    for row in result["trials"]:
        a=row["modes"]["asset_z"]
        b=row["modes"]["preserve_start_altitude"]
        assert a["start"]==b["start"]
        assert a["path_sha256"]!=b["path_sha256"]
    assert result["aggregates"]["asset_z"]["prediction_below_floor_fraction"]>0
    assert result["aggregates"]["preserve_start_altitude"]["prediction_below_floor_fraction"]==0
