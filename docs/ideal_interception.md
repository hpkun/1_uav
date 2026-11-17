# Ideal three-dimensional interception simulation

Current protocol: `uav-ideal-interception-v3.5.0`. The formal 15v15 Level-2
scenario, paired avoidance benchmark, audit commands, and historical playback
matrix are documented in [ideal_15v15.md](ideal_15v15.md). The V3.2 details
below describe the historical smaller-scale protocol and its provenance.

This is an abstract geometric
contact simulation for multi-agent decision research. A contact is a
continuous-time distance event; the model does not simulate airframe shape,
impulse, structural damage or real UAV collision control.

The former `uav-ideal-kinematic-v1.2.0` IK0–IK3 navigation and non-contact
tracking tasks have been removed from active entry points. Their historical
Checkpoint and experiment files are not converted or overwritten. The new
Checkpoint protocol rejects them and V1 interception Checkpoints. Historical
Checkpoints and experiment files are retained. The v4 interception environment
is separate and unchanged.

Project replay colors follow the requested team convention: **Red = defender**
(`D0…`, `#d64040`) and **Blue = attacker/intruder** (`I0…`, `#2378bf`).
The intruder Bézier reference is a blue dashed line. Colors and display labels
do not change the internal roles, events, guidance formulas or simulation state.

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
Intruders are independent entities with status and activity masks. By default,
each reset samples a complete `zuo_bezier` plan with a random start near the
configured +x anchor and two random cubic control points. Its goal `P3` lies
on the protected cylinder axis: `x,y = protected_center.x,y` and, by default,
`z = clip(start.z, protected_center.z, protected_center.z + protected_height)`.
At the default start altitudes no clipping occurs. The cylinder geometry and
`region_breach` rule are unchanged; `P3` is a reference waypoint inside the
region, while its boundary is the success condition. The explicit diagnostic
`intruder_goal_altitude_mode="asset_z"` retains the previous `P3.z =
protected_center.z` geometry. This altitude rule is a project-specific
adaptation of a point asset to a 3D protected cylinder, not a Zuo paper parameter.
The reference is fixed before any defender action and followed at configured
constant speed using polyline arc length. An explicit `intruder_mode=straight`
or `bezier` selects a legacy/test reference fixture; its `intruder_ends` do not
define the formal policy goal. These preset paths supply constant-speed motion;
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

Association uses nearest 3D Euclidean distance among active intruders for
each active defender. It allows many defenders to select one target; exact
ties choose the lower target index. It is neither Hungarian matching nor
load balancing and does not use threat scores. Association is computed on
reset and recomputed when the active member set changes while an episode is
ongoing; normal control steps do not trigger reassignment. It is task
context only and does not choose a defender action. The Actor receives a
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
polyline hash and path duration. The formal policy additionally records its
actual start, asset goal, control points, generation seed and policy identity
in reset information and scenario identity. Different seeds produce different
formal paths even when `initialization="fixed"` keeps defenders fixed.

Each initialized scene also records `reference_length`, dimensionless
`step_ratio` (reference speed × control period / horizontal scene span),
`protected_radius_ratio`, each path's duration fraction of the episode budget,
whether the path stays within bounds, and whether it enters the protected
region. The last distinction separates bypass paths from region-crossing paths
without using a learned policy's contact result to label the geometry.

V3.2 defaults use `dt=0.05 s`, `max_steps=600`, a 30 s episode budget,
`contact_distance=0.1 m`, and 20 m/s fixed speed for
both sides. The space spans 600 m horizontally and 120 m vertically; the
protected cylinder has radius 20 m. `position_scale=300 m` and
`speed_scale=20 m/s` match these scales. These values are engineering test
settings, not calibrated physical parameters or reproduced results from Zuo
et al. Formal intruder starts always vary by seed; `initialization="random"`
also perturbs defender starts. A terminal event can end an
episode before the nominal 30 s budget.

V3.2 changes the default formal path geometry, so new snapshots and Checkpoints
use V3.2 and reject V3.1/V3.0 training state. Historical V3.1, V3.0 and V2
trajectory files remain readable for offline playback; they are not rewritten
or silently treated as V3.2 training data. Observation and centralized state
dimensions remain 68 and 66.

### Intruder policy provenance and sampling

Zuo et al., Section 5.1.2 supports a pre-programmed cubic Bézier path, a
random start and new random control points per independent trial, the
protected asset as destination, constant speed, and no defender awareness,
evasion, reactive steering or intruder cooperation. Our arena coordinates,
uniform start zone, control-point distribution, `1–2` intruder two-anchor
wedge approximation, preserved-altitude goal, and V3.2 default `0.05 s / 20 m/s` profile are
project-specific adaptations. The paper does not specify a control-point
sampling distribution.

The project defaults are `intruder_start_jitter_m=5` and
`intruder_bezier_jitter_ratio=0.04`. For each start `P0`, `P1` and `P2` begin
at one and two thirds of the span to the asset, then receive independent
bounded uniform offsets in two perpendicular lateral/vertical directions.
The amplitude is `0.04 × ||P3-P0||` in each direction. Rejection sampling
checks initial contact, arena bounds, nondegenerate segments and entry into
the protected cylinder within the episode budget; it raises after
`init_max_attempts` failures. These numerical values are engineering choices,
not parameters reported by Zuo et al. `zuo_paper_profile()` is an optional
paper-like numerical profile (`dt=0.1 s`, defender `25 m/s`, intruder `30 m/s`,
contact distance `0.1 m`) and does not replace the V3.2 defaults.

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

## Ideal trajectory and replay

The ideal trace protocol is `uav-ideal-trace-v2`, separate from v4 replay.
Each trace records the environment protocol, scene geometry and diagnostics,
seed and scenario ID, exact reference polyline and hash, entity states and
masks, executed defender velocity per control period, step-internal events,
runtime assignment in every frame, and final result. Deterministic guidance
traces also record each step-start decision in a `control` field on the
corresponding step-end frame. Historical v1 traces remain playable. The player verifies the reference hash and reconstructs
motion between control boundaries from the recorded velocity and polyline.
Event times are keyframes, including times between control steps. A physical
event is marked at its recorded position; `member_inactive` and `episode_end`
only change state. The visuals show abstract geometric contact, not damage.
The time-axis reader checks reference speed, duration and geometry against
the environment configuration and validates every recorded step against the
executed velocity, inactive transitions and final state. The executed velocity
comes directly from the environment's step result.

Generate a deterministic trace without a model or CUDA, then render the same
file as an interactive HTML player or an MP4 (FFmpeg required):

```bash
python -m tools.ideal_interception_trace ideal_episode.json.gz --scale 5v2 --seed 17
python -m tools.ideal_interception_replay ideal_episode.json.gz ideal_episode.html --view 3d
python -m tools.ideal_interception_replay ideal_episode.json.gz ideal_episode.mp4 --view top --fps 30
```

`--view` accepts `3d`, `top`, or `altitude`. For a trained model, pass
`--trace` to `algorithm.ideal_interception evaluate`; it writes
`episode_000.json` alongside `evaluation.json` without another evaluation
run. The trace contains the first evaluated episode. The HTML and MP4
exporters consume that same trace; neither runs the policy or environment.

The `altitude` view is an XZ spatial projection, not an altitude-over-time
chart. HTML playback uses elapsed physical time and supports 0.5×, 1×, 2×,
4×, arbitrary time seeking, and precise event keyframes. MP4 samples physical
time at a fixed FPS. Both exporters refuse an existing output by default;
pass `--overwrite` to replace it after a successful temporary export.
Video encoding writes into a temporary directory before publishing, so a
failed encoder leaves an existing output intact. MP4 uses the available H.264
or MPEG-4 FFmpeg encoder. Plotly and Matplotlib use the same protected-cylinder
geometry; the MP4 exporter retains static artists across frames.

The model-free trace command also accepts `--config scene.json` with the same
`InterceptionConfig` fields as training; scene flags cannot be mixed with it.
`--scenario-type region_crossing` requires every reference to remain in the
airspace and enter the protected region within the episode budget. The default
`geometry` type permits deliberate bypass and boundary test cases. The trace
stores the complete effective configuration and the declared scenario type.
`python -m tools.ideal_interception_benchmark` measures reset and diagnosis
cost without training.

An independent offline three-position intruder predictor and recorded-trace
accuracy evaluator are documented in `docs/ideal_prediction.md`.

Defaults are engineering fixtures for interface verification, not calibrated
physical or approved research benchmark parameters. Short CUDA smoke tests
verify compatibility and recovery; no long training or convergence claim is
made.
