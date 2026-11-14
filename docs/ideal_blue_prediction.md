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

## Paired positions across longer horizons

The original two-step CLI and its files remain available. For one compact
multi-horizon report from an existing observation file:

```bash
python -m tools.ideal_prediction_compare --observations outputs/ideal_blue_prediction/bezier_1v1_seed17/observations.json.gz --out outputs/ideal_extended_prediction/bezier_1v1_seed17 --extended --horizons 1 5 10 20 --plot-horizon 10 --tan-eta 1 --tan-min 0.1 --tan-max 1.0 --tan-epsilon-acceleration 0.000001
```

This mode writes only `comparison.csv`, `summary.json`,
`trajectory_overlay.png`, and `error_curve.png`. Each CSV row contains the
same target, observation time and future truth for both methods, plus their
3D predicted positions, separate errors and the distance between predictions.
The overlay shows a small selection of forecast points on the complete truth
path in 3D and XY. `--plot-horizon` selects the common horizon displayed.
Rows require three past observations and a future regular-grid observation;
the non-grid endpoint is not used as truth.

The optional Tan cap is an **exploratory flag**, not a validated confidence
bound. It uses eta * estimated speed / (estimated acceleration magnitude +
epsilon acceleration), clipped to the supplied minimum and maximum seconds.
The epsilon parameter has units m/s². `HORIZON_LIMITED` rows retain their raw
prediction and error in all aggregates; the cap never improves reported MAE
by dropping long-range cases. Omit `--tan-eta` to disable this flag entirely.
The strict Zuo linear baseline uses the same fixed horizons; Zuo's separate
distance-dependent rule requires synchronized defender positions, which the
blue-only observation file does not contain.
