# Ideal interception V3.4: 6v6 motion profiles and PIP guidance

The protocol is `uav-ideal-interception-v3.4.0`. Six red defenders and six blue
intruders use 112 local Actor features and 128 centralized Critic features.
V3.3 checkpoints cannot resume a V3.4 run. V2 and V3.0–V3.3 traces remain
replayable; historical `zuo_tp` control labels are presentation metadata.

## Geometry and profiles

`formal_6v6_config()` preserves the V3.3 **Level 1 Smooth Bézier** scene,
including its seeded ±5 m starts, 0.04 control perturbation ratio, paths,
scenario IDs, and reference hashes. The historical config value
`intruder_policy="zuo_bezier"` remains an input alias, but it does not mean
the project-specific distribution reproduces a paper's exact scenario.

`formal_6v6_level2_config()` selects **Level 2 Maneuvering Bézier**. It keeps
the same six defender circle positions, six intruder wedge anchors, protected
region, arena, speeds, `dt=0.05 s`, contact distance, and fixed Hungarian
assignment. Each intruder start `P0` uses the same seed and ±5 m draw as
Level 1. `P3.xy` is the protected center and `P3.z=P0.z`. For chord
`span=P3-P0` of length `L`, the controls are

```text
Q1 = P0 + span/3                         Q2 = P0 + 2 span/3
P1 = Q1 + sign A1 e_lateral + V e_vertical
P2 = Q2 - sign A2 e_lateral - V e_vertical
A1/L, A2/L ~ Uniform(0.08, 0.14); V/L ~ Uniform(-0.04, 0.04)
sign ∈ {-1,+1}
```

The opposite lateral offsets produce a single cubic S-turn. `e_parallel`,
`e_lateral`, and `e_vertical` form a local orthonormal basis. The values
0.08, 0.14, and 0.04 are project-specific geometry parameters. Rejection
requires a maximum chord deviation of at least 8 m and maximum heading
deviation of at least 8°, in addition to finite nondegenerate path geometry,
arena containment, protected-region entry within the episode, and valid
initial positions. These thresholds define the Level 2 geometry; they are
not selected from guidance outcomes. Invalid controls are resampled without
falling back to Level 1.

Both levels pre-generate the entire target path before the episode. The
`Reference` advances along its polyline by arc length, preserving configured
speed magnitude. The target never reads defender state and never replans
in response to defender actions. Level 2 curvature changes velocity
*direction*; `v²κ` is an offline equivalent normal-acceleration diagnostic.
No reactive evasion or obstacle avoidance is implemented.

The scene diagnostics include chord length, path length and ratio, chord
deviation, heading variation, curvature, and equivalent normal acceleration.
They do not enter the online controller. For each episode, the recorded trace
also supports offline future-position errors and a same-state comparison of
two-point and three-point velocity estimates, PIP horizons, and PIP points.
Reference velocity and central-difference local acceleration are used only
after the episode. Low/Medium/High acceleration bins use tercile thresholds
from the scored trace samples.

## Formal guidance

The active deterministic method namespace is exactly `two_point_pip` and
`three_point_pip`. TwoPoint-PIP estimates velocity from the latest two real
positions. ThreePoint-PIP uses the unchanged three-point backward difference.
Both call the same unchanged constant-velocity positive-root PIP solver.
An unavailable or nonpositive PIP solution falls back to the current observed
target position for that step. The Hungarian assignment and continuous
geometric contact rules are unchanged.

Run in WSL Ubuntu with the `uav` Conda environment:

```bash
python -m algorithm.ideal_multi_predictive_guidance --method two_point_pip --motion-level level2_maneuvering --seed 17 --out outputs/ideal_guidance/v34_level2_two_point_seed17
python -m algorithm.ideal_multi_predictive_guidance --method three_point_pip --motion-level level2_maneuvering --seed 17 --out outputs/ideal_guidance/v34_level2_three_point_seed17
python -m tools.ideal_multi_guidance_benchmark --motion-level level1_smooth --seed-start 17 --trials 50 --out outputs/ideal_guidance/v34_level1_6v6_17_66.json
python -m tools.ideal_multi_guidance_benchmark --motion-level level2_maneuvering --seed-start 17 --trials 50 --out outputs/ideal_guidance/v34_level2_6v6_17_66.json
```

Each single-run directory contains `summary.json` and replayable
`episode.json`. The benchmark enforces identical scenarios, path hashes,
starts, and assignment between methods for each seed. Its MSR is the fraction
of episodes with all targets intercepted, ISR is intercepted members divided
by initial intruder members, and BR is breached members divided by initial
intruder members. These are different quantities. The independent MAPPO
evaluation summary uses the same member denominators for 6v6; earlier
smaller scales keep their established summary semantics.

The 6v6 formation, path profile, and deterministic fixed assignment are
project choices motivated by the broad interception setting of
[Zuo et al.](https://www.mdpi.com/2504-446X/10/6/418). They do not reproduce
the paper's full allocation or cooperative obstacle-avoidance framework.

The later, independent friendly-only controller experiment is documented in
`docs/ideal_collision_avoidance.md`.
