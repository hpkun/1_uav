# Conditional ideal contact forecast

`prediction.forecast.forecast_assigned` consumes the existing defender-to-intruder
assignment and `IntruderHistory`. It predicts each distinct active assigned
intruder once. Zuo needs two observed positions; Tan needs three. Both receive
the same positive offsets. The legacy two-step functions and their experiment
outputs remain unchanged. `TargetForecast` includes absolute future times and
positions. Neither predictor reads a `Reference` or future truth.

`prediction.contact_forecast.estimate_contact` accepts a forecast plus an
explicit defender trajectory. It linearly interpolates both timestamped
tracks and uses the existing 3D sphere `proximity` calculation on every common
time segment. A reported contact is conditional on the supplied defender
track. The reported contact position is the defender center at first sphere
contact, matching the first position in a real `interception_contact` event.
`NO_CONTACT_IN_FORECAST_WINDOW` says nothing about target reachability.
Insufficient history or missing defender coverage yields `NOT_EVALUABLE`.
This module never changes an environment event, action, member mask or reward.

## Distinct time quantities for 1v1

`prediction.intercept_time.zuo_lookahead_time` returns Zuo's distance-based
**prediction horizon** (seconds). It is not a contact-time solution.
`estimate_constant_velocity_contact` returns the earliest geometric contact
time under an explicit ideal assumption: a target maintains its currently
estimated velocity and a defender picks one straight direction immediately
and travels at its configured fixed speed. `estimate_from_history` obtains
that target velocity causally from the latest two real observations. It
returns `INSUFFICIENT_HISTORY`, `INACTIVE`, `NO_CONTACT_IN_HORIZON`, or
`NO_CONSTANT_VELOCITY_SOLUTION` where appropriate. The resulting
`time_to_contact_s` is relative to the latest observation; the separate
`predicted_contact_time_s` is absolute simulation time. A future turn or a
different defender action invalidates the assumption, so neither field is
treated as a real event. `ContactForecast.predicted_contact_time` instead
describes the first contact of two explicitly supplied virtual trajectories.

`prediction.intercept_prediction` now exposes two additional 1v1 estimates
using only `IntruderHistory` positions. `estimate_zuo_prediction_from_history`
reuses the paper look-ahead time and the existing two-point linear position
predictor. Its `horizon_kind` is `paper_lookahead`; it is never presented as
a physical contact time. It also records the paper's unit direction toward
that point as a diagnostic (`paper_direction_unit`), without generating an
environment action. `estimate_three_point_adaptive_from_history` uses the
existing three-point quadratic position predictor with a project-designed
rolling look-ahead. With target range R, contact distance Rc, line-of-sight
unit vector u, estimated velocity v and acceleration a, it computes
`T_range=(R-Rc)/max(defender_speed-u·v, V_min)` and
`T_maneuver=eta*||v||/||a||` (`inf` when `||a|| <= epsilon_acceleration`).
The final horizon is `clip(min(T_range,T_maneuver), history.dt, T_max)`;
already-contacting members use zero instead. Defaults are `V_min=1 m/s`,
`eta=0.1`, `epsilon_acceleration=1e-6 m/s²`, `T_max=1 s`, and
`T_min=history.dt`. These are initial engineering choices, not Tan or Zuo
paper parameters or a calibrated optimum. The horizon is not a strict
impact time; it is recomputed each control period. The old four-root
three-point contact-time interface has been removed. The independent
two-point constant-velocity contact diagnostic remains available.

The offline tool reads validated `uav-ideal-trace-v1` and v2 replays. New v2
frames store the environment's actual runtime assignment, which the evaluator
uses directly. Historical V2-environment/v1-trace files retain the fixed
`D_i -> I_(i % intruder_count)` rule. V3-environment/v1-trace files can be
replayed geometrically but are rejected by assignment-dependent evaluation
because the exact event-time association is unavailable. Recorded
defender motion is a hindsight **virtual trajectory assumption**, not a
forecast of future defender actions. A completed window is scored only while
both members are active; a recorded assigned contact can close the window at
its actual event time. Other partial windows are not scored as negative cases.
Reported event metrics are conditional on this case selection. The replay
protocol and the real `events.py` state machine are unchanged.

```bash
python -m tools.ideal_contact_forecast_eval --trace outputs/ideal_prediction/straight/trace.json.gz --out outputs/ideal_contact_forecast/straight.json --horizon-steps 2
python -m tools.ideal_prediction_compare --observations observations.json.gz --out new_results --horizons 1 2 5 10
```

The second command also writes `horizons.json`. Its truth comes from regular
observation grid points only; it does not interpolate future Reference truth.
Each horizon reports its own valid count, MAE, RMSE, P95 and maximum error.
Prediction horizons longer than available future observations have no score.

## Independent 1v1 deterministic guidance

The CPU-only runner samples the current environment state each 0.05 s,
updates `IntruderHistory`, reads the current assignment and chooses a unit
direction toward the newly predicted target position. It maps that direction
through the existing Beta-compatible `[0,1]` action interface. Zuo needs one
warmup pure-pursuit step; ThreePoint-Adaptive needs two. On a genuinely
invalid prediction or direction, the current observed target position is
used for that step. No previous prediction is reused. The runner is separate
from MAPPO, training, reward and event detection.

```bash
python -m algorithm.ideal_predictive_guidance --method zuo_tp --intruder-mode straight --seed 17 --out outputs/ideal_guidance/zuo_straight_seed17
python -m algorithm.ideal_predictive_guidance --method three_point_adaptive --intruder-mode straight --seed 17 --out outputs/ideal_guidance/three_point_straight_seed17
```

Each output directory has `summary.json` and a replay-compatible
`episode.json` (v2 trace). The latter puts a step-start `control` decision
on the resulting step-end frame and records the actual environment
assignment on every frame. The reported minimum separation is the continuous
defender-target center distance along the executed motion, including
sub-step positions. Success or failure is reported as observed; the default
adaptive parameters are not tuned to improve contact outcomes.
