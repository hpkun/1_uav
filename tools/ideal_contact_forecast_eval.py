"""Offline, conditional contact forecasts from a validated bilateral trace.

Only complete future windows with active assigned members are scored. The
recorded defender motion is an explicit hindsight trajectory assumption.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from prediction.contact_forecast import estimate_contact
from prediction.forecast import METHODS, forecast_assigned
from prediction.history import IntruderHistory
from tools.ideal_interception_timeline import Timeline


def evaluate(source, horizon_steps=2):
    if type(horizon_steps) is not int or horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    timeline = Timeline(source) if not isinstance(source, Timeline) else source
    data, dt = timeline.data, timeline.config.dt
    count = timeline.config.intruder_count
    assignment = {i: i % count for i in range(timeline.config.defender_count)}
    history = IntruderHistory(dt, slots=2)
    history.reset(data["scenario_id"])
    frames = timeline.frames
    rows = []
    for frame in frames:
        time = float(frame["time"])
        if not np.isclose(time, frame["step"]*dt, rtol=1e-9, atol=1e-9):
            continue
        positions = np.asarray(frame["intruders"], float)[:, :3]
        history.observe(time, positions, frame["intruder_mask"], data["scenario_id"])
        # The formal comparison starts only when both methods have three observations.
        if all(len(history.observed(j)[0]) < 3 for j in set(assignment.values())):
            continue
        forecasts = forecast_assigned(history, assignment, [h*dt for h in range(1, horizon_steps+1)])
        for i, j in assignment.items():
            if not frame["defender_mask"][i] or not frame["intruder_mask"][j]:
                continue
            for method in ("zuo_linear", "tan_quadratic"):
                target = forecasts.get((j, method))
                if target is None or target.status != "READY" or len(history.observed(j)[0]) < 3:
                    continue
                future = list(target.future_times)
                real = [event for event in timeline.events if event["kind"] == "interception_contact"
                        and event["members"] == [i, 5+j] and time < event["time"] <= future[-1]+1e-9]
                event = real[0] if real else None
                if event is not None and event["time"] < future[-1]:
                    future = [t for t in future if t < event["time"]-1e-9] + [float(event["time"])]
                    _, observed = history.observed(j)
                    required, predictor = METHODS[method]
                    estimated = predictor(observed[-required:], dt, [t-time for t in future])
                    target = replace(target, future_times=tuple(future),
                        future_positions=tuple(tuple(float(x) for x in p) for p in estimated))
                complete = future[-1] <= timeline.duration+1e-9
                if complete:
                    states = [timeline.at(t) for t in (time, *future) if event is None or t < event["time"]-1e-9]
                    by_id = [{entity["id"]: entity for entity in state["entities"]} for state in states]
                    complete = all(row[i]["active"] and row[5+j]["active"] for row in by_id)
                if complete:
                    track = [row[i]["position"] for row in by_id]
                    if event is not None:
                        track.append(event["positions"][0])
                    estimate = estimate_contact(i, j, target, (time, *future), track,
                        timeline.config.contact_distance, "recorded_virtual")
                else:
                    estimate = estimate_contact(i, j, target, (time, future[-1]),
                        (frame["defenders"][i][:3], frame["defenders"][i][:3]),
                        timeline.config.contact_distance, "recorded_virtual")
                    # A partial/terminal replay must never count as a negative case.
                    estimate = asdict(estimate)
                    estimate.update(status="NOT_EVALUABLE", contact_predicted=False,
                                    predicted_contact_time=None, predicted_contact_position=None)
                record = asdict(estimate) if not isinstance(estimate, dict) else estimate
                record.update(actual_contact=bool(event) if complete else None,
                    actual_contact_time=float(event["time"]) if event and complete else None,
                    actual_contact_position=tuple(float(x) for x in event["positions"][0]) if event and complete else None)
                if complete and event and record["contact_predicted"]:
                    record["time_error_s"] = record["predicted_contact_time"]-event["time"]
                    record["position_error_m"] = float(np.linalg.norm(
                        np.asarray(record["predicted_contact_position"])-event["positions"][0]))
                else:
                    record["time_error_s"] = record["position_error_m"] = None
                rows.append(record)
    metrics = {}
    for method in ("zuo_linear", "tan_quadratic"):
        subset = [row for row in rows if row["prediction_method"] == method]
        valid = [row for row in subset if row["status"] != "NOT_EVALUABLE"]
        positives = [row for row in valid if row["actual_contact"]]
        negatives = [row for row in valid if not row["actual_contact"]]
        matched = [row for row in positives if row["contact_predicted"]]
        metrics[method] = dict(valid_samples=len(valid),not_evaluable=len(subset)-len(valid),
            actual_contacts=len(positives),contact_detection_rate=len(matched)/len(positives) if positives else None,
            false_positive_rate=sum(row["contact_predicted"] for row in negatives)/len(negatives) if negatives else None,
            no_contact_rate=sum(not row["contact_predicted"] for row in valid)/len(valid) if valid else None,
            mean_abs_time_error_s=float(np.mean([abs(row["time_error_s"]) for row in matched])) if matched else None,
            mean_position_error_m=float(np.mean([row["position_error_m"] for row in matched])) if matched else None)
    return dict(protocol="uav-ideal-contact-forecast-eval-v1",scenario_id=data["scenario_id"],
        trace_version=data["trace_version"],assignment=assignment,assignment_source="fixed_scenario_rule",
        defender_trajectory_source="recorded_virtual",horizon_steps=horizon_steps,dt=dt,
        methods=metrics,records=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon-steps", type=int, default=2)
    args = parser.parse_args(argv)
    result = evaluate(args.trace, args.horizon_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result["methods"], ensure_ascii=False))


if __name__ == "__main__":
    main()
