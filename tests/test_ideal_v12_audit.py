import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from algorithm.ideal_kinematic import scenario_protocol
from env.ideal_kinematic import IdealConfig, IdealEnv

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return IdealConfig(**json.loads(
        (ROOT/"configs/ideal_ik2_feasibility_v12.json").read_text()))


def run(config, direction):
    env = IdealEnv(config)
    env.reset(3)
    total, parts = 0., {}
    while True:
        actions = np.full((5,3), .5)
        direction(actions)
        _,_,terminated,truncated,info = env.step(actions)
        total += info["team_reward"]
        for key,value in info["reward_parts"].items():
            parts[key] = parts.get(key,0.)+value
        if terminated or truncated:
            return dict(outcome=info["outcome"], time=info["time"],
                        total=float(total), parts=parts)


def test_reward_audit_success_deadline_boundary_and_early_contacts():
    base = fixture()
    cases = dict(
        success=(base,lambda a: None),
        poor_safe=(base,lambda a: a.__setitem__(0,[.5,1.,.5])),
        poor_deadline=(base,lambda a: a.__setitem__(0,[0.,.5,.5])),
        boundary=(replace(base,starts=((-28.,-2.5,20.),)+base.starts[1:]),
                  lambda a: a.__setitem__(0,[0.,.5,.5])),
        friendly_contact=(base,lambda a: (a.__setitem__(0,[.5,1.,.5]),
                                          a.__setitem__(2,[.5,0.,.5]))),
        target_contact=(replace(base,target_ends=((.1,0.,20.),base.target_ends[1]),
                                reference_end="stop"),
                        lambda a: a.__setitem__(1,[.5,.5,0.])))
    results = {name:run(*case) for name,case in cases.items()}
    assert [results[name]["outcome"] for name in cases] == [
        "success","success","tracking_failure","boundary","contact","target_contact"]
    assert results["success"]["time"] == results["poor_deadline"]["time"] == results["poor_safe"]["time"] == 4.
    assert results["boundary"]["time"] == 2.
    assert results["friendly_contact"]["time"] < 4.
    assert results["target_contact"]["time"] < 4.
    assert results["success"]["parts"]["completion"] == base.completion_weight
    assert results["friendly_contact"]["parts"]["contact_failure"] == -base.boundary_weight
    assert results["target_contact"]["parts"]["contact_failure"] == -base.boundary_weight
    assert all(results["success"]["total"] > results[name]["total"]
               for name in cases if name != "success")
    assert results["poor_safe"]["total"] > results["poor_deadline"]["total"]
    assert results["poor_deadline"]["total"] > results["friendly_contact"]["total"]
    assert results["poor_deadline"]["total"] > results["target_contact"]["total"]
    assert results["poor_deadline"]["total"] > results["boundary"]["total"]
    for result in results.values():
        assert result["total"] == pytest.approx(sum(result["parts"].values()))


def test_fixed_and_random_scenario_identity_and_report():
    fixed = fixture()
    one, two = IdealEnv(fixed), IdealEnv(fixed)
    first = one.reset(10)[1]
    second = two.reset(11)[1]
    np.testing.assert_array_equal(first["actual_states"], second["actual_states"])
    assert first["scenario_id"] == second["scenario_id"]
    rows = [dict(scenario_id=first["scenario_id"]),dict(scenario_id=second["scenario_id"])]
    report = scenario_protocol(fixed, rows)
    assert report["kind"] == "fixed_case" and report["unique_scenarios"] == 1
    assert report["duplicate_scenarios"] == 1 and report["randomized"] == []
    random_config = replace(fixed, initialization="random", random_range=.5)
    a,b,c = (IdealEnv(random_config) for _ in range(3))
    alpha = a.reset(10)[1]
    beta = b.reset(10)[1]
    gamma = c.reset(11)[1]
    np.testing.assert_array_equal(alpha["actual_states"], beta["actual_states"])
    assert alpha["scenario_id"] == beta["scenario_id"]
    assert gamma["scenario_id"] != alpha["scenario_id"]
    report = scenario_protocol(random_config,[dict(scenario_id=alpha["scenario_id"]),
                                              dict(scenario_id=gamma["scenario_id"])])
    assert report["kind"] == "random_initial_positions"
    assert report["unique_scenarios"] == 2
    assert report["randomized"] == ["uav_initial_positions"]
