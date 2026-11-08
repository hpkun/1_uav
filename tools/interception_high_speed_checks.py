"""Deterministic kinematic and swept-contact evidence for the high-speed profile."""
import argparse
import json
from pathlib import Path

import numpy as np

from algorithm.common.serialization import write_json
from env.interception import InterceptionConfig, InterceptionEnv
from env.interception.motion import advance, execute_velocity
from tools.interception_high_speed_audit import PROFILE


def config(**overrides):
    return InterceptionConfig(**(json.loads(PROFILE.read_text(encoding="utf-8")) | overrides))


def motion_checks():
    c = config()
    designs = {
        "accelerate": ([17,0,0], [25,0,0]), "decelerate": ([25,0,0], [0,0,0]),
        "reverse": ([25,0,0], [-25,0,0]), "right_angle": ([25,0,0], [0,25,0]),
        "three_dimensional": ([24,0,3], [0,-25,-12]),
        "climb": ([20,0,0], [0,0,25]), "descend": ([20,0,0], [0,0,-25]),
        "switch_at_speed": ([25,0,0], [25,0,0]),
    }
    result = {}
    for name, (velocity, desired) in designs.items():
        state = np.r_[[100., 100., 50.], velocity]
        rows = [dict(time=0., state=state.tolist())]
        for step in range(50):
            request = [0,25,0] if name == "switch_at_speed" and step >= 5 else desired
            executed = execute_velocity(request, state[3:], c.pursuer_vmax, c.pursuer_tau, c)
            for dt in (0., .02, .05, .1):
                actual = advance(state, executed, c.pursuer_tau, dt)
                acceleration = (executed-actual[3:])/c.pursuer_tau
                assert np.linalg.norm(actual[3:]) <= 25+1e-12
                assert np.linalg.norm(acceleration[:2]) <= 6+1e-12
                assert -4-1e-12 <= actual[5] <= 5+1e-12
                np.testing.assert_array_equal(advance(state, executed, c.pursuer_tau, 0.), state)
                rows.append(dict(time=step*c.dt+dt, request=request, executed=executed.tolist(),
                                 state=actual.tolist(), acceleration=acceleration.tolist()))
            state = actual
        result[name] = dict(max_speed=max(np.linalg.norm(x["state"][3:]) for x in rows),
                            max_horizontal_acceleration=max(np.linalg.norm(x["acceleration"][:2]) for x in rows[1:]),
                            min_vertical_speed=min(x["state"][5] for x in rows),
                            max_vertical_speed=max(x["state"][5] for x in rows), samples=rows)
    return result


def contact_checks():
    # Velocity is held by commanding the same value as the initial state.
    designs = {
        "head_on": ([2.65,0,0], [25,0,0], [-28,0,0]),
        "same_direction": ([1.30,0,0], [25,0,0], [20,0,0]),
        "oblique_3d": ([1.2,-1.2,-.4], [12,0,4], [0,24,0]),
        "height_separation": ([2.65,0,1.02], [25,0,0], [-28,0,0]),
        "exact_tangent": ([2.65,1.,0], [25,0,0], [-28,0,0]),
        "inside_tangent": ([2.65,.9999,0], [25,0,0], [-28,0,0]),
        "outside_tangent": ([2.65,1.0001,0], [25,0,0], [-28,0,0]),
    }
    rows = []
    for name, (offset, vp, ve) in designs.items():
        offset, vp, ve = map(np.asarray, (offset, vp, ve))
        delta = ve-vp
        speed = np.linalg.norm(ve)
        c = config(scale="1v1", evader_mode="constant_velocity",
                   constant_direction=ve, evader_speed_fraction=speed/28)
        env = InterceptionEnv(c, diagnostics=True)
        start = np.array([100., 100., 50.])
        env.reset(1, dict(scenario="custom", pursuers=[np.r_[start,vp]],
                          evaders=[np.r_[start+offset,ve]]))
        action = np.full((5,3), .5)
        action[0] += vp/50
        center = -float(offset@delta)/(delta@delta)
        radial = np.linalg.norm(offset+center*delta)
        analytic = center-np.sqrt(max(0., 1-radial**2))/np.linalg.norm(delta)
        expected = analytic if radial <= 1+1e-13 and 0 <= analytic <= .1 else None
        _, _, _, _, info = env.step(action)
        events = [event for event in info["events"] if event["kind"] == "interception_contact"]
        observed = events[0]["time"] if events else None
        assert (observed is None) == (expected is None), name
        if observed is not None:
            assert abs(observed-expected) <= 1e-7, name
        rows.append(dict(case=name, start_distance=np.linalg.norm(offset),
                         unconstrained_end_distance=np.linalg.norm(offset+.1*delta),
                         continuous_minimum=np.linalg.norm(offset+np.clip(center,0,.1)*delta),
                         observed_minimum=info["closest_targets"][5]["center_distance"],
                         expected_time=expected, actual_time=observed,
                         time_error=observed-expected if observed is not None else None,
                         outcome=info["outcome"]))
    # Relative speed is 5 m/s in these fixtures. The protected cylinder
    # is reached at t=.05 s by the intruder starting at x=-26 m.
    cases = []
    for t in (.04, .05, .06):
        cases.append((f"breach_priority_{t}", "1v1",
                      [[-27-5*t,0,25,25,0,0]], [[-26,0,25,20,0,0]],
                      [(0, 5, min(t,.05), "interception_contact" if t < .05 else "breach")]))
    cases.append(("competing_defenders", "3v1",
                  [[98.9,99.4,50,25,0,0], [98.9,100.6,50,25,0,0], [40,60,50,0,0,0]],
                  [[100,100,50,20,0,0]], [(0,5,.06,"interception_contact")]))
    cases.append(("two_contacts", "5v2",
                  [[98.8,100,50,25,0,0], [98.3,130,55,25,0,0], [40,60,50,0,0,0],
                   [40,90,50,0,0,0], [40,120,50,0,0,0]],
                  [[100,100,50,20,0,0], [100,130,55,20,0,0]],
                  [(0,5,.04,"interception_contact"), (1,6,.14,"interception_contact")]))
    for name, scale, p, e, expected in cases:
        env = InterceptionEnv(config(scale=scale, evader_mode="constant_velocity",
                                     constant_direction=(1,0,0), evader_speed_fraction=20/28),
                              diagnostics=True)
        env.reset(1, dict(scenario="custom", pursuers=p, evaders=e))
        actions = np.full((5,3), .5)
        actions[:len(p)] += np.asarray(p)[:,3:]/50
        observed = []
        for _ in range(2):
            _, _, terminated, _, info = env.step(actions)
            observed.extend(x for x in info["events"] if x["kind"] in ("interception_contact", "breach"))
            if terminated:
                break
        assert len(observed) == len(expected), name
        for event, (i, j, t, kind) in zip(observed, expected):
            assert event["kind"] == kind and abs(event["time"]-t) <= 1e-7, name
            if kind == "interception_contact":
                assert event["entity_ids"] == [i,j] and not info["active_mask"][i], name
            offset = np.subtract(e[j-5][:3],p[i][:3])
            delta = np.subtract(e[j-5][3:],p[i][3:])
            step_start = .1*int(t/.1)
            nearest = np.clip(-(offset@delta)/(delta@delta),step_start,step_start+.1)
            rows.append(dict(case=name, start_distance=np.linalg.norm(offset+step_start*delta),
                             unconstrained_end_distance=np.linalg.norm(offset+(step_start+.1)*delta),
                             continuous_minimum=np.linalg.norm(offset+nearest*delta),
                             expected_time=t, actual_time=event["time"], time_error=event["time"]-t,
                             event=event, outcome=info["outcome"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    write_json(args.out, dict(motion=motion_checks(), contact=contact_checks()))


if __name__ == "__main__":
    main()
