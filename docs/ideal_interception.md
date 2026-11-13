# Ideal three-dimensional interception simulation

Protocol: `uav-ideal-interception-v2.0.0`. This is an abstract geometric
contact simulation for multi-agent decision research. A contact is a
continuous-time distance event; the model does not simulate airframe shape,
impulse, structural damage or real UAV collision control.

The former `uav-ideal-kinematic-v1.2.0` IK0–IK3 navigation and non-contact
tracking tasks have been removed from active entry points. Their historical
Checkpoint and experiment files are not converted or overwritten. The new
Checkpoint protocol rejects them and V1 interception Checkpoints. Historical
Checkpoints and experiment files are retained. The v4 interception environment
is separate and unchanged.

## Entities and scenarios

| Scale | Defenders | Intruders |
| --- | ---: | ---: |
| 1v1 | 1 | 1 |
| 3v1 | 3 | 1 |
| 5v2 | 5 | 2 |

Every active entity has SI state `[x,y,z,vx,vy,vz]`. Each defender's three
Beta actions specify a direction after subtracting 0.5 and normalizing.
Its speed is fixed by configuration; an all-0.5 action retains the previous
valid heading. There is no acceleration, inertia, attitude or PID model.
Intruders are independent entities with status and activity masks. Their
straight or piecewise-linear Bézier paths supply preset constant-speed motion;
an active intruder never waits at a path endpoint: reaching it emits
`path_complete`, deactivates that intruder and ends the episode with
`intruder_path_complete`. Inactive entities have zero velocity.
The 5-defender and 2-intruder slots keep network dimensions fixed across scales.
An inactive slot has no Actor action or subsequent movement.

The protected region is a vertical cylinder. Event candidates use continuous
line or path segments, including sub-step contacts and region entry. The
environment sorts candidates by physical time, then by priority: region
breach, friendly contact, defender-intruder contact, intruder-intruder contact,
boundary exit, path completion. Member IDs break remaining ties.
Events involving an exited member are ignored. A defender-intruder contact
records an `interception_contact` and, **as a simulation rule**, removes
those two members. If another intruder remains, the episode continues.
All intruders intercepted ends in `all_intercepted`. Region entry, friendly
contact, intruder contact, member boundary exit, path completion and task deadline have distinct
terminal outcomes. `info.time` is the actual final event time; all task
deadlines are true terminations rather than external truncations.

The fixed association `defender i -> intruder i % active_intruders` is task
context only. It does not choose a defender action. The Actor receives a
68-dimensional local observation with own state, masked friendly and intruder
relative states, association, region and time context. The Critic receives a
66-dimensional bilateral centralized state; inactive entity slots and their
associations are masked. The original MAPPO Beta/PPO/GAE code is reused through
a distinct model instance.

Rewards are deliberately sparse and separate from event metrics:
`interception` for each simulated defender-intruder contact,
`protection` when every intruder is intercepted, penalties for breach,
friendly/intruder contact, boundary exit, exhausted defenders, path completion
and deadline, plus elapsed-time cost. There is only one distance threshold:
`contact_distance`. Initial overlap is rejected using that threshold. The
contact term denotes an abstract
task event, not a physical damage model. Episode and evaluation reports record
outcome, event categories and timestamps, member exits, protected-region
breaches and minimum friendly separation. Each episode records the actual
intruder start, end, speed, Bézier control points, polyline point count,
polyline hash and path duration. A fixed scenario
with multiple seeds is identified as one physical scenario.

V2 defaults use `dt=0.1 s`, a 30 s episode budget and 20 m/s fixed speed for
both sides. The space spans 600 m horizontally and 120 m vertically; the
protected cylinder has radius 20 m. `position_scale=300 m` and
`speed_scale=20 m/s` match these scales. These values are engineering test
settings, not calibrated physical parameters or reproduced results from Zuo
et al. Random initialization perturbs starting positions while keeping the
configured paths and fixed task association. A terminal event can end an
episode before the nominal 30 s budget.

## Commands

Run in WSL Ubuntu with `conda activate uav`; training and Checkpoint
evaluation require CUDA:

```bash
python -m algorithm.ideal_interception train --scale 1v1 --out outputs/ideal_interception_1v1 --total-steps 1024
python -m algorithm.ideal_interception resume --out outputs/ideal_interception_1v1
python -m algorithm.ideal_interception evaluate outputs/ideal_interception_1v1/latest.pt --out outputs/ideal_interception_eval
```

`--config` accepts a JSON object with `InterceptionConfig` fields.
`--seed`, `--eval-seed`, `--eval-episodes`, `--workers`, `--horizon`,
`--eval-interval` and `--until-steps` configure short or full runs.
Evaluation stores per-episode records, a summary and the first episode's
timestamped bilateral trajectory. The newest full evaluation is recoverable
from the committed Checkpoint; older evaluations have indexed checksums.

Defaults are engineering fixtures for interface verification, not calibrated
physical or approved research benchmark parameters. Short CUDA smoke tests
verify compatibility and recovery; no long training or convergence claim is
made.
