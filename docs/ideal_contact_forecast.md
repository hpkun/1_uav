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
environment action. `estimate_three_point_contact_from_history` computes
the local quadratic velocity and acceleration from three observations and
solves the earliest nonnegative root of
`||target_now - defender_now + velocity*T + acceleration*T²/2|| =
defender_speed*T + contact_distance`. It rescales polynomial coefficients,
filters complex and negative roots, and verifies candidates against this
unsquared distance equation. Near-zero acceleration reuses the existing
constant-velocity solver. An optional explicit maximum horizon only filters
solutions; `recommended_horizon()` does not participate. Both estimates
return predicted target positions and absolute simulation times, and neither
changes control or real events. Their assumptions need re-evaluation after
each new observation.

The offline tool reads a validated `uav-ideal-trace-v1` replay. The old trace
does not serialize assignment, so it reconstructs the current fixed scenario
rule `D_i -> I_(i % intruder_count)` and labels that source in output. Recorded
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
