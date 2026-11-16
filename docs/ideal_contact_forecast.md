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

`prediction.intercept_prediction` exposes two additional 1v1 estimates
using only `IntruderHistory` positions. `estimate_zuo_prediction_from_history`
reuses the paper look-ahead time and the existing two-point linear position
predictor. Its `horizon_kind` is `paper_lookahead`; it is never presented as
a physical contact time. It also records the paper's unit direction toward
that point as a diagnostic (`paper_direction_unit`), without generating an
environment action. `estimate_three_point_pip_from_history` uses three
positions to estimate current target velocity by the second-order backward
difference. It assumes that velocity remains constant and solves the earliest
positive root of `A*T²+B*T+C=0`, where `A=v·v-Vd²`, `B=2*(pt-pd)·v`, and
`C=||pt-pd||²`. Near equal speeds it solves the linear equation `B*T+C=0`.
The predicted point is `pt+v*T`. With no positive solution, the online
controller uses the observed target position for that step. Acceleration is
recorded for diagnosis but never extrapolated in online PIP. This is
center-coincidence geometry; actual 0.1 m contact is decided by the unchanged
continuous event detector. The independent two-point constant-velocity contact
diagnostic and offline fixed-horizon quadratic prediction remain available.

The offline tool reads validated `uav-ideal-trace-v1` and v2 replays. New v2
frames store the environment's actual runtime assignment, which the evaluator
uses directly. Historical V2-environment/v1-trace files retain the fixed
`D_i -> I_(i % intruder_count)` rule. V3.0/V3.1-environment/v1-trace files can be
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
warmup pure-pursuit step; ThreePoint-PIP needs two. On a genuinely
invalid prediction or direction, the current observed target position is
used for that step. No previous prediction is reused. The runner is separate
from MAPPO, training, reward and event detection.

```bash
python -m algorithm.ideal_predictive_guidance --method zuo_tp --intruder-mode straight --seed 17 --out outputs/ideal_guidance/zuo_pip_canonical_straight_seed17
python -m algorithm.ideal_predictive_guidance --method three_point_pip --intruder-mode straight --seed 17 --out outputs/ideal_guidance/three_point_pip_canonical_straight_seed17
python -m algorithm.ideal_predictive_guidance --method zuo_tp --intruder-mode bezier --seed 17 --out outputs/ideal_guidance/zuo_pip_canonical_bezier_seed17
python -m algorithm.ideal_predictive_guidance --method three_point_pip --intruder-mode bezier --seed 17 --out outputs/ideal_guidance/three_point_pip_canonical_bezier_seed17
```

Each output directory has `summary.json` and a replay-compatible
`episode.json` (v2 trace). The latter puts a step-start `control` decision
on the resulting step-end frame and records the actual environment
assignment on every frame. The reported minimum separation is the continuous
defender-target center distance along the executed motion, including
sub-step positions. The dedicated canonical 1v1 configuration places the
defender at `(-100,-30,50)` m and the target at `(100,0,70)` m, with endpoint
`(-220,0,70)` m and 20 m/s speeds. Training defaults remain unchanged. Its
initial straight center PIP is `T=5.1625 s` at `(-3.25,0,70)` m. After the
episode, each step-end frame receives `offline_evaluation`: range dynamics and,
for straight, an oracle PIP using real reference velocity. For bezier, future
reference truth is sampled only if the predicted time is within path duration.
Neither truth source is available to the online decision callback. Summary
metrics include per-step horizon, point, angle, and range diagnostics.

The straight and deterministic Bézier cases above are analytical/controlled
guidance fixtures. They are not the formal intruder strategy. For the V3.1
formal randomized `zuo_bezier` policy, use the same seed for paired controller
runs, or run the built-in 20-trial comparison. It asserts the exact same
intruder path hash for both controllers on every seed and records all trial
hashes, contact outcomes, fallback counts and future-position errors.

```bash
python -m algorithm.ideal_predictive_guidance --method zuo_tp --intruder-policy zuo_bezier --seed 17 --out outputs/ideal_guidance/formal_zuo_seed17
python -m algorithm.ideal_predictive_guidance --method three_point_pip --intruder-policy zuo_bezier --seed 17 --out outputs/ideal_guidance/formal_pip_seed17
python -m tools.ideal_intruder_policy_benchmark --seed-start 17 --trials 20 --out outputs/ideal_guidance/formal_paired_17_36.json
```
