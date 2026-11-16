# Ideal contact forecasts and deterministic guidance

`prediction.forecast.forecast_assigned` and
`prediction.contact_forecast.estimate_contact` are independent conditional
forecast utilities. They consume recorded target observations and an
explicit defender trajectory, then search continuous 3D line segments for
first geometric contact. `NO_CONTACT_IN_FORECAST_WINDOW` describes only the
supplied forecast window. Neither utility changes environment actions,
events, or rewards. The separate historical Zuo/Tan fixed-horizon prediction
comparison remains available for replaying earlier prediction studies.

The active deterministic interception experiment uses
`estimate_two_point_pip_from_history` and
`estimate_three_point_pip_from_history`. TwoPoint-PIP estimates current
target velocity as `(p[k]-p[k-1]) / (t[k]-t[k-1])`; ThreePoint-PIP uses
`(3p[k]-4p[k-1]+p[k-2]) / (2dt)`. Both pass the resulting velocity to
`solve_constant_velocity_pip`, which solves the earliest positive root of
`||p_target-p_defender+v_target*T|| = speed_defender*T`. The predicted point
is `p_target+v_target*T`. This is a constant-velocity center-coincidence
prediction, while the actual 0.1 m geometric contact uses the environment's
continuous event detector. Warmup and no-positive-root steps guide toward
the latest observed target position.

The target's known reference path is never read by online guidance. Offline
diagnostics after the episode compare predicted future positions with the
reference only when the predicted time lies within its duration. The 6v6
experiment also compares the two velocity estimators and PIP solutions at
identical recorded states. True reference velocity and central-difference
local acceleration appear only in these offline diagnostics. See
`docs/ideal_multi_interception.md` for Level 1 and Level 2 profiles and
current commands.

V2 and V3.0–V3.3 trace files remain playable. A historical control field
`method="zuo_tp"` is accepted as recorded metadata; the current formal
guidance CLI does not run that strict baseline.
