import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest

from env.ideal_kinematic import IdealConfig, IdealEnv
from env.ideal_kinematic.feasibility import analyze

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("task,expected", [("IK2", 3), ("IK3", 5)])
def test_original_v11_defaults_are_necessarily_infeasible(task, expected):
    report = analyze(IdealConfig(task=task))
    assert len(report["members"]) == expected
    assert not report["necessary_reachable"]
    assert all(not row["necessary_reachable"] for row in report["members"])
    assert all(row["latest_possible_fraction"] < .7 for row in report["members"])
    assert report["interpretation"].startswith("necessary_only")


@pytest.mark.parametrize("task", ["IK2", "IK3"])
def test_separate_validation_fixture_has_deterministic_success(task):
    path = ROOT/f"configs/ideal_{task.lower()}_feasibility_v12.json"
    config = IdealConfig(**json.loads(path.read_text()))
    report = analyze(config)
    assert report["necessary_reachable"]
    env = IdealEnv(config)
    env.reset(100)
    for _ in range(config.max_steps):
        _,_,terminated,truncated,info = env.step(np.full((5,3),.5))
    assert terminated and not truncated and info["outcome"] == "success"
    assert [row["valid_fraction"] for row in info["tracking"]] == pytest.approx(
        [1.]*config.active_count)
    assert info["contact_count"] == 0


def test_one_unreachable_member_does_not_get_team_pass():
    config = IdealConfig(**json.loads(
        (ROOT/"configs/ideal_ik3_feasibility_v12.json").read_text()))
    starts = list(config.starts)
    starts[4] = (-25.,-5.5,20.)
    bad = replace(config, starts=tuple(starts))
    rows = analyze(bad)["members"]
    assert all(row["necessary_reachable"] for row in rows[:4])
    assert not rows[4]["necessary_reachable"]


def test_obviously_unreachable_and_exact_fraction_threshold():
    config = IdealConfig(**json.loads(
        (ROOT/"configs/ideal_ik2_feasibility_v12.json").read_text()))
    impossible = replace(config, starts=((-25.,-2.5,20.),)+config.starts[1:])
    assert not analyze(impossible)["members"][0]["necessary_reachable"]
    base = IdealConfig()
    starts = ((-4.,-2.5,20.),(-4.,0.,22.5),(-4.,2.5,20.))+base.starts[3:]
    trial = replace(base, task="IK2", dt=1., max_steps=4, random_range=0.,
        starts=starts, speeds=(1.,)*5, target_starts=((0.,0.,20.),base.target_starts[1]),
        target_ends=((1.,0.,20.),base.target_ends[1]), reference_end="stop")
    def run(c):
        env = IdealEnv(c)
        env.reset()
        for _ in range(c.max_steps):
            _,_,_,_,info = env.step(np.full((5,3),.5))
        return info
    fractions = [row["valid_fraction"] for row in run(trial)["tracking"]]
    threshold = min(fractions)
    assert 0 < threshold < .7
    assert run(replace(trial, tracking_success_fraction=threshold))["outcome"] == "success"
    assert run(replace(trial, tracking_success_fraction=threshold+1e-6))["outcome"] == "tracking_failure"
