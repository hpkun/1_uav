from dataclasses import replace
import json

import numpy as np
import pytest

from algorithm.ideal_interception import reset_evaluation_scene
from algorithm.ideal_predictive_guidance import run_episode
from env.ideal_interception import InterceptionConfig,InterceptionEnv
from env.ideal_interception.seeding import scenario_seed
from tools.ideal_interception_replay import ROLE_COLORS,intruder_reference_color,render_html
from tools.ideal_interception_trace import record
from tools.ideal_intruder_policy_benchmark import compare


def test_role_colors_and_reference_match_paper(tmp_path):
    assert ROLE_COLORS=={"defender":"#d64040","intruder":"#2378bf"}
    assert intruder_reference_color()==ROLE_COLORS["intruder"]
    source=tmp_path/"episode.json"
    from tools.ideal_interception_trace import write_trace
    write_trace(source,record(InterceptionConfig(max_steps=2,intruder_mode="straight"),17))
    output=tmp_path/"episode.html"
    render_html(source,output)
    html=output.read_text(encoding="utf-8")
    assert "Red Defender" in html and "Blue Intruder" in html
    assert "I0 (Blue Intruder) reference" in html
    assert "#d64040" in html and "#2378bf" in html


def test_formal_scenario_identity_ignores_legacy_endpoint():
    base=InterceptionConfig()
    changed=replace(base,intruder_ends=((-80.,90.,30.),base.intruder_ends[1]))
    a=InterceptionEnv(base).reset(17)[1]
    b=InterceptionEnv(changed).reset(17)[1]
    assert a["intruder_plans"]==b["intruder_plans"]
    assert a["intruder_paths"]==b["intruder_paths"]
    assert a["scenario_id"]==b["scenario_id"]
    for modified in (replace(base,intruder_bezier_jitter_ratio=.06),
                     replace(base,intruder_speeds=(21.,20.)),
                     replace(base,protected_center=(-210.,0.,0.))):
        other=InterceptionEnv(modified).reset(17)[1]
        assert other["scenario_id"]!=a["scenario_id"]


@pytest.mark.parametrize("seed",[17,18])
def test_seed_is_shared_across_guidance_mappo_scene_benchmark_and_trace(seed):
    config=InterceptionConfig()
    _,_,mappo_info=reset_evaluation_scene(config,seed)
    two,two_trace=run_episode(config,"two_point_pip",seed)
    pip,pip_trace=run_episode(config,"three_point_pip",seed)
    benchmark=compare(config,[seed])
    trace=record(config,seed)
    expected=mappo_info["intruder_paths"][0]
    for candidate in (two_trace,pip_trace,trace):
        assert candidate["references"][0]["specification"]==expected
        assert candidate["scenario_id"]==mappo_info["scenario_id"]
    assert two["intruder_path_sha256"]==pip["intruder_path_sha256"]
    assert benchmark["trials"][0]["path_sha256"]==expected["polyline_sha256"]
    assert np.array_equal(scenario_seed(seed).generate_state(4),
                          np.random.SeedSequence(seed).generate_state(4))


@pytest.mark.parametrize("method",["two_point_pip","three_point_pip"])
def test_future_truth_coverage_and_beyond_reference_are_explicit(method):
    summary,trace=run_episode(InterceptionConfig(),method,17)
    rows=[f["offline_evaluation"] for f in trace["frames"][1:]]
    total=summary["predictive_steps_total"]
    scored=summary["future_truth_scored_steps"]
    beyond=summary["prediction_beyond_reference_steps"]
    assert total==summary["predictive_steps"]
    assert total==scored+summary["future_truth_unscored_steps"]
    assert scored==sum("future_position_error_m" in row for row in rows)
    assert beyond==sum(row.get("prediction_beyond_reference",False) for row in rows)
    assert summary["future_truth_coverage"]==pytest.approx(scored/total)
    assert summary["prediction_beyond_reference_fraction"]==pytest.approx(beyond/total)
    if beyond:
        assert all("future_position_error_m" not in row for row in rows
                   if row.get("prediction_beyond_reference"))


def test_paired_future_error_aggregates_use_scored_samples_only():
    result=compare(InterceptionConfig(),[17,18])
    assert result["shared_intruder_paths"]
    for method,aggregate in result["aggregates"].items():
        scores=[row["methods"][method] for row in result["trials"]]
        assert aggregate["predictive_steps_total"]==sum(s["predictive_steps_total"] for s in scores)
        assert aggregate["future_truth_scored_steps"]==sum(s["future_truth_scored_steps"] for s in scores)
        assert aggregate["future_truth_coverage"]==pytest.approx(
            aggregate["future_truth_scored_steps"]/aggregate["predictive_steps_total"])
        assert aggregate["future_position_error_p95_m"] is not None
        assert aggregate["future_position_error_max_m"]>=aggregate["future_position_error_p95_m"]
