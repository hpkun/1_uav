# Offline three-point intruder prediction

This module is independent of MAPPO, defender control, assignment, rewards and
the ideal trajectory protocol. It consumes the positions of each active
intruder from a validated `uav-ideal-trace-v1` episode. It never reads reference
control points or future positions while predicting. Future recorded states
are used only by the offline error evaluator.

`prediction.three_point.predict_two_steps` uses the oldest-to-newest positions
`P[k-2], P[k-1], P[k]` at equal interval `dt` and evaluates the quadratic
through them at `dt` and `2*dt`. The history manager keeps separate three-point
buffers for I0 and I1, restarts a buffer after a sampling gap, and retires an
inactive target until scenario reset. The evaluator uses only complete control
steps; a step-internal terminal frame is never treated as another regular
observation. A future position is valid only if the target remains active at
that future recorded time.

```bash
python -m tools.ideal_interception_trace outputs/ideal_prediction/straight/trace.json.gz --scale 1v1 --mode straight --seed 17
python -m tools.ideal_prediction_demo --trace outputs/ideal_prediction/straight/trace.json.gz --out outputs/ideal_prediction/straight --plot
```

The CLI writes `predictions.json`, `predictions.csv`, `metrics.json`, and,
with `--plot`, `error_curve.png`. It will not overwrite existing output files.
`--threshold METERS` optionally reports the fraction of valid position errors
at or below that distance. This is a prediction accuracy measure, not an
interception or geometric contact success rate.

The supplied experiment cases use `1v1 straight fixed`, `1v1 bezier fixed`,
and `5v2 bezier random`, all with requested initialization seed 17. The
three-point baseline is exact up to floating-point arithmetic on constant
velocity straight segments. Errors on Bézier paths reflect local curvature
and changes of direction at piecewise-linear knots. The defaults are
simulation fixtures, not validated physical flight parameters. These offline
results make no claim about control performance.
