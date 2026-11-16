# Ideal interception V3.3: project 6v6 guidance experiment

The protocol is `uav-ideal-interception-v3.3.0`. It has six defender and six
intruder slots, 112 local Actor features, and 128 centralized Critic features.
V3.2 checkpoints cannot be resumed under V3.3. Historical V2 and V3.0–V3.2
traces retain their recorded 5/2 slot schema in offline playback; no extra
historical member is constructed.

The [Zuo et al. paper](https://www.mdpi.com/2504-446X/10/6/418) motivates a
circular defender holding formation, wedge intruder formation, preprogrammed
cubic Bézier paths with new random control points per independent trial,
constant-speed motion, pre-guidance task allocation, and no target reassignment
after engagement. The paper's full framework also includes load-balanced,
threat-aware allocation and cooperative obstacle avoidance. Its simulations
use different force ratios, coordinates, deployment sequence, and colors.
This repository keeps **Red = Defender** and **Blue = Intruder**. It does not
reproduce the paper's coordinates or full framework.

The project-specific `formal_6v6_config()` fixes the six defenders on a radius
50 m circle centered at `(-120,0,60)` m:

| Member | Initial position (m) |
| --- | --- |
| D0 | (-70, 0, 60) |
| D1 | (-95, 43.3012701892, 60) |
| D2 | (-145, 43.3012701892, 60) |
| D3 | (-170, 0, 60) |
| D4 | (-145, -43.3012701892, 60) |
| D5 | (-95, -43.3012701892, 60) |

The intruder wedge anchors are `I0=(190,0,60)`, `I1=(210,-30,60)`,
`I2=(210,30,60)`, `I3=(230,-60,60)`, `I4=(230,0,60)`, and
`I5=(230,60,60)` m. Each actual start receives the existing seeded ±5 m
jitter. The existing `sample_zuo_bezier_plan()` creates two control points
with the existing 0.04 displacement ratio and preserves each sampled start
altitude at the goal. Path motion is constant speed by arc length and does not
react to defenders. The unified profile uses 20 m/s on both sides, `dt=0.05 s`,
600 steps, 0.1 m geometric contact distance, the existing protected cylinder,
and the existing arena. The separate paper-like sensitivity profile changes
only speeds to 25/30 m/s and `dt` to 0.1 s.

At reset, `hungarian_fixed` minimizes the total 3D Euclidean distance over a
strict one-to-one pairing. The O(n³) implementation visits defender rows and
intruder columns in ascending order; this gives a stable deterministic tie
rule (an all-equal matrix maps `D_i` to `I_i`). The initial pairing is kept
for the entire episode. Runtime assignment only filters out inactive pairs.
When an assigned target breaches, its defender retires with
`target_breached_no_reassignment`; other resolved targets likewise retire
their assigned defender. No defender is assigned a new target.

For 6v6, a contact, breach, boundary exit, or same-side geometric contact
resolves only the affected members. An episode ends when all six intruders
are resolved or the 30 s deadline is reached. `all_intercepted` requires six
contacts and no breach; all other fully resolved combinations use
`mixed_resolution`. Friendly contact remains observable; no obstacle
avoidance, repulsion, or path deconfliction is added. The 1v1, 3v1, and 5v2
scales retain nearest-distance reassignment after membership events and their
previous terminal semantics.

The independent runner uses one `IntruderHistory` for all six observed targets
and independently computes each active defender's fixed-pair guidance. It
supports `zuo_tp`, `zuo_tp_pip_time`, and `three_point_pip`, preserving the
respective one-, one-, and two-step pure-pursuit warmups. A PIP with no
positive root uses the current observed target position for that step and
retries on the next. Reference future positions are used only after the
episode to score conditional prediction errors; coverage and predictions
beyond path duration are reported separately. A prediction is not extended
beyond the reference endpoint to create artificial truth.

Run these in WSL Ubuntu after `conda activate uav`:

```bash
python -m algorithm.ideal_multi_predictive_guidance --method zuo_tp --seed 17 --out outputs/ideal_guidance/multi6_zuo_seed17
python -m algorithm.ideal_multi_predictive_guidance --method zuo_tp_pip_time --seed 17 --out outputs/ideal_guidance/multi6_hybrid_seed17
python -m algorithm.ideal_multi_predictive_guidance --method three_point_pip --seed 17 --out outputs/ideal_guidance/multi6_threepoint_seed17
python -m tools.ideal_multi_guidance_benchmark --seed-start 17 --trials 50 --profile unified --out outputs/ideal_guidance/multi6_unified_17_66.json
python -m tools.ideal_multi_guidance_benchmark --seed-start 17 --trials 1 --profile paper_like --out outputs/ideal_guidance/multi6_paper_like_seed17.json
```

Each single-run directory contains `summary.json` and replayable
`episode.json`. The benchmark contains per-seed three-method summaries and
aggregates. Each seed is checked for identical starts, path specifications,
six path hashes, scenario ID, and initial assignment across methods.
Interception success ratio (ISR), breach ratio (BR), and all-intercepted
mission success rate (MSR) remain separate metrics. The standard Hungarian
allocation isolates guidance differences and is not Zuo's load-balanced,
threat-aware allocation.
