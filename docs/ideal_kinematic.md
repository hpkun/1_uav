# Ideal point-mass experiments

Protocol: `uav-ideal-kinematic-v1.1.0`. This environment and its checkpoints are
separate from the interception and preset-tracking protocols. Run commands in
WSL Ubuntu after `conda activate uav`; training and checkpoint evaluation require CUDA.

## Tasks

| Task | Active UAVs | Reference targets | Completion |
| --- | ---: | ---: | --- |
| IK0 | 1 | 0 | Reach its static waypoint |
| IK1 | 3 | 0 | All three reach their respective waypoints |
| IK2 | 3 | 1 | Each UAV reaches the required valid tracking time fraction |
| IK3 | 5 | 2 | Same criterion for both assigned targets |

IK2 and IK3 use fixed assignment `UAV i -> target i % target_count`.
This assignment supplies reference information only. The Actor alone chooses
the direction of motion. IK1 is coordinated waypoint navigation; it does not
measure strict area coverage. A boundary exit, friendly contact or contact with
**any** moving reference target ends the episode at the physical event time.
The configured deadline is an environment termination, including failure:
`deadline_failure` for IK0/IK1 and `tracking_failure` for IK2/IK3.
It is not an external truncation and GAE does not bootstrap across it.

Each active UAV has `[position_m, velocity_m_per_s]`. Each Beta action is a
three-component vector in `[0,1]`; subtracting 0.5 and normalizing gives a
direction. An all-0.5 action retains the previous direction. Position changes
by `fixed_speed * direction * dt` with no acceleration or response lag.
An inactive UAV has zero velocity. The Bézier reference is a polyline with
1024 segments, advanced by cumulative arc length; its speed along each segment
is exactly the configured reference speed. The path is an approximation to
the mathematical Bézier curve, not an analytic arc-length solution.
The reference end behavior is `ping_pong` by default: it reverses instantly
at either end and continues moving for the full episode. `stop` preserves
the previous stationary-after-arrival behavior. At every polyline knot and
turnaround, position is continuous and velocity is right-continuous (the
velocity of the segment starting at that instant).

For dynamic tasks, each UAV's valid interval is exactly
`contact_distance < distance(UAV, assigned reference) <= tracking_distance`.
The valid time fraction is the union length of these intervals divided by
elapsed episode time, using all linear pieces and turnarounds. Team success at
the deadline requires every UAV's fraction to meet
`tracking_success_fraction` (default 0.7), with no earlier safety termination.
The default threshold is an engineering test value, not a calibrated criterion.
Mean tracking error is the time integral of Euclidean distance divided by
elapsed time. Its numerical quadrature compares 16 and 32 point Gauss rules
and recursively bisects until their difference is below `1e-9 * max(1, integral)`
or raises a numerical error at depth 10. Error percentiles are duration
weighted percentiles of each linear piece's midpoint sample, rather than
exact continuous-time quantiles.
Longest valid duration is the longest merged valid interval; departure count
counts valid-to-invalid transitions; outside time is elapsed minus valid time.

Friendly warning and safety violation entries use continuous threshold
intervals with per-pair enter/exit state. A sustained interval counts once;
re-entry counts again. Safety duration sums pairwise time below the safety
threshold and can exceed episode time when several pairs violate at once.
Contact counts are distinct from warnings. Minimum friendly separation is
minimized over the continuous active trajectories. The reference motion
fraction is the mean, across reference targets, of each target's moving time
divided by episode time; per-target fractions are also recorded. For
`ping_pong` each fraction is 1. Initial random positions are redrawn
as a whole until all geometry checks pass, up to `init_max_attempts`.

Within a step, event candidates are ordered by physical time. At a tie the
priority is boundary, friendly contact, target contact, all-waypoint success,
deadline, warning, safety violation, arrival; member IDs break remaining ties.
Member waypoint arrival stops that member alone. A global event stops all
members at its time. Observational events after the global stop are discarded.

The 64-dimensional local observation contains own state, assigned reference,
relative peer and target states, masks, task identity and remaining time.
Inactive peer slots are zero; the centralized critic also masks their
associated state and assignment fields.
The 59-dimensional centralized state contains full normalized states,
assignment indices, masks, target states, task and remaining time. Position and
velocity use the configured normalization scales. The original MAPPO
optimizer, Beta policy, GAE and parallel vector sampler are reused with a
separate model instance for these dimensions. Reward components are stored
individually: progress or tracking, completion, danger, boundary and time.
Contact events are recorded for safety analysis without collision shaping.
Per-episode output includes completion time, actual travel, continuous
tracking and safety metrics; evaluation summaries aggregate them across seeds.

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
Checkpoints embed the V1.1 protocol and configuration. V1.0 checkpoints and
snapshots are rejected rather than interpreted under the new task rules.
On resume, the committed checkpoint can recreate a missing last update record,
evaluation file and derived JSONL view; a conflicting record is rejected.

The default geometry and reward weights are simple engineering fixtures for
interface validation. They are not calibrated UAV physics or benchmark
parameters. No long training run or learned task performance is claimed.
