# Ideal point-mass experiments

Protocol: `uav-ideal-kinematic-v1.0.0`. This environment and its checkpoints are
separate from the interception and preset-tracking protocols. Run commands in
WSL Ubuntu after `conda activate uav`; training and checkpoint evaluation require CUDA.

## Tasks

| Task | Active UAVs | Reference targets | Completion |
| --- | ---: | ---: | --- |
| IK0 | 1 | 0 | Reach its static waypoint |
| IK1 | 3 | 0 | All three reach their respective waypoints |
| IK2 | 3 | 1 | At the time horizon, all active UAVs remain within the tracking band, outside contact distance |
| IK3 | 5 | 2 | Same tracking criterion for both assigned targets |

IK2 and IK3 use fixed assignment `UAV i -> target i % target_count`.
This assignment supplies reference information only. The Actor alone chooses
the direction of motion. A boundary exit or moving-target contact ends an
episode as a failure. A time horizon without meeting the tracking criterion is
`time_limit` (truncation); a successful horizon is a true termination.

Each active UAV has `[position_m, velocity_m_per_s]`. Each Beta action is a
three-component vector in `[0,1]`; subtracting 0.5 and normalizing gives a
direction. An all-0.5 action retains the previous direction. Position changes
by `fixed_speed * direction * dt` with no acceleration or response lag.
An inactive UAV has zero velocity. The Bézier reference is a polyline with
1024 segments, advanced by cumulative arc length; its speed along each segment
is exactly the configured reference speed. The path is an approximation to
the mathematical Bézier curve, not an analytic arc-length solution.

The 64-dimensional local observation contains own state, assigned reference,
relative peer and target states, masks, task identity and remaining time.
The 59-dimensional centralized state contains full normalized states,
assignment indices, masks, target states, task and remaining time. Position and
velocity use the configured normalization scales. The original MAPPO
optimizer, Beta policy, GAE and parallel vector sampler are reused with a
separate model instance for these dimensions. Reward components are stored
individually: progress or tracking, completion, danger, boundary and time.
Contact events are recorded for safety analysis without collision shaping.

## Commands

```bash
conda activate uav
python -m algorithm.ideal_kinematic train --task IK0 --out outputs/ideal_ik0 --total-steps 1024
python -m algorithm.ideal_kinematic resume --out outputs/ideal_ik0
python -m algorithm.ideal_kinematic evaluate outputs/ideal_ik0/latest.pt --out outputs/ideal_ik0_eval
```

`--config` accepts a JSON object with `IdealConfig` fields. Use
`--eval-seed`, `--eval-episodes`, `--eval-interval`, `--workers`,
`--horizon`, `--seed` and `--until-steps` to control a run.
Evaluation uses Beta mean actions by default; `--sample-actions` requests
stochastic actions. `evaluation.json` includes the first episode's full 3D
trajectory with SI timestamps, reference states and continuous events.
Stochastic policy sampling uses the evaluation seed as its PyTorch seed.

The default geometry and reward weights are simple engineering fixtures for
interface validation. They are not calibrated UAV physics or benchmark
parameters. No long training run or learned task performance is claimed.
