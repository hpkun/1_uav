# Complete intruder reference observations and paired prediction comparison

This is an offline experiment separate from bilateral interception episodes.
`tools.ideal_blue_observations` calls `InterceptionEnv.reset(seed)` once and
samples each existing `env.paths[j]` with `Reference.sample(t)` over that
target's full reference duration. It never calls `env.step()`. The file uses
`uav-ideal-blue-observations-v1`, not the bilateral replay protocol. Regular
samples occur at `k * config.dt` only while the target's path exists. A
non-grid endpoint is stored separately and excluded from predictions.

`prediction.zuo_linear` estimates velocity from the latest two positions and
extrapolates one or two control intervals. `prediction.three_point` remains
the existing Tan quadratic baseline. Both are evaluated only after the third
observation, on the same target, time, horizons and future truth. The
comparison tool reads saved observations; it does not regenerate the path.
No predictor receives control points, reference objects or future positions.

```bash
python -m tools.ideal_blue_observations --out observations.json.gz --scale 5v2 --mode bezier --initialization random --seed 17
python -m tools.ideal_prediction_compare --observations observations.json.gz --out results --plot
```

The observation generator also accepts `--config` with the same JSON fields
as `InterceptionConfig`; scene flags cannot be mixed with it. The comparison
output contains `predictions.csv`, `comparison.csv`, `metrics.json`,
`comparison.png`, and `error_curve.png`. Zuo and Tan rows with the same target,
observation time and horizon share the exact same future truth. Invalid
future horizons have empty truth and error fields and are excluded from the
statistics. Relative MAE improvement is omitted when Zuo MAE is at floating
point noise scale (at most `1e-9 m`).

The four requested seed-17 experiments are under
`outputs/ideal_blue_prediction/`. These are complete ideal reference paths,
so their results should not be interpreted as interception success or as
performance of a learned defender policy. The straight paths give only
floating-point residuals for both predictors. On these deterministic Bézier
polylines, local quadratic extrapolation has lower short-horizon error than
linear extrapolation; this is an observed property of the specified paths,
not a claim for other motion distributions.
