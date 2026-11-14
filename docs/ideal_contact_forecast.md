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
