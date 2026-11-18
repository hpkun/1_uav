# Independent A/B kinematic interception study

Protocol: `uav-kinematic-interception-v1.0.0`. This package does not alter the
existing ideal, acceleration limited, 20v20, prediction, or MAPPO protocols.
It is a deterministic research simulator, not a new reinforcement learning
environment or a calibrated aircraft model.

## Motion and guidance

- **A** stores position and 3D velocity. A command requests a unit direction
  and speed in `[0,20]` m/s. The velocity change is limited to `10 dt` m/s;
  position uses constant acceleration over each executed substep.
- **B** stores position, speed, yaw, and flight path angle. It constrains speed
  to `[12,20]` m/s, yaw rate to 0.5 rad/s, flight path rate to 0.3 rad/s,
  flight path angle to ±20°, combined lateral acceleration to 10 m/s², and
  tangential acceleration to +3/−4 m/s². Position uses midpoint integration.
- Both models start at 20 m/s along +X. These settings are exploratory and are
  not validated parameters for any particular UAV.
- All guides use the same `(desired direction, desired speed)` command. G0 is
  pure pursuit; G1 uses a causal 3 point CV estimate and the old analytic PIP
  equation; G2 adds a finite candidate forward simulation through the selected
  A/B dynamics; G3 uses the same simulation with a causal 7 point quadratic
  fit. CA is used for the first 2 s, then joins continuously to CV, with a
  20 m/s target forecast speed cap. K-FIP reports failure of its *candidate
  search*, never global physical unreachability.
- G2/G3 replan every 0.25 s. Execution and prediction call the same `step`
  function using 0.01 s substeps by default. Multi-member scenarios can use
  the existing CPA-kNN direction modifier before the A/B execution layer.

## Scenarios and events

The 1v1 fixture has a fixed straight or Bézier reference plus a seeded
lateral offset. The 3v3, 6v6, and 20v20 fixtures read the historical 20v20
scene generator once, selecting its first 3, 6, or all 20 pairs. Target
reference paths are identical across models and guides for each seed. The
historical environment's `step` is never called.

Defender paths are subdivided at 0.01 s or finer and at all active reference
path knots. Continuous segment intersection detects contact, friendly contact,
boundary exit, protected region entry, and reference completion. Events are
ordered by time and then by the explicit event priority in `environment.py`.
The 0.1 m contact radius is an abstract geometric threshold. Curved path
intersection is an approximation whose substep convergence should be checked
on each new scenario set. Assigned contact, unassigned contact, breach, and
path completion are reported separately.

## Ubuntu commands

Run after `conda activate uav` from the repository root. Evaluation commands
require CUDA per the repository runtime policy.

```bash
python -m kinematic_interception.run --seed 17
python -m kinematic_interception.run --seed 17 19 23 --offline-prediction
python -m kinematic_interception.run --scale 20 --seed 17 --model A --method G2 --avoidance predictive_cpa_knn
python -m kinematic_interception.run --seed 17 --model B --method G3 --out outputs/kinematic_b_g3_seed17.json
python -m kinematic_interception.visualization outputs/kinematic_b_g3_seed17.json outputs/kinematic_b_g3_seed17.png
```

Supplying a single model and method with `--out` writes the entire episode's
states, commands, events, and reference specifications to one JSON file. Without
`--out`, no experiment file is written. The offline predictor reads future
reference positions only to score errors, after each causal prediction has
been made. ADE averages errors along each forecast horizon, FDE measures its
endpoint, and RMSE uses squared position errors. The summary gives each path
equal weight; adjacent time samples are not treated as independent trials.

## Interpretation

Compare G1/G2 and G2/G3 **within** each motion model on identical scenarios.
An A/B difference is also a difference in reachable sets, especially A's
ability to stop versus B's minimum speed. The default single seed is an
interface check, not a performance claim. Run multiple paired seeds and report
variation before drawing a research conclusion. This V1 uses fixed target
reference trajectories; it does not simulate target-side A/B execution or
learn a policy.
