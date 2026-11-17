# V3.4 friendly collision avoidance experiment

This controller experiment keeps `uav-ideal-interception-v3.4.0` and the
`uav-ideal-trace-v2` schema. It fixes **ThreePoint-PIP**, Level 2 pre-generated
intruder trajectories, one-to-one Hungarian assignment, continuous physical
contact detection, and all environment settings except the audited defender
formation radius. Red is Defender; Blue is Intruder. No MAPPO training is
performed. The avoidance layer receives only active defender states and the
raw PIP heading. Intruders are never treated as avoidance obstacles.

## Three controller modes

The independent `AvoidanceConfig` holds all fixed controller parameters.
`none` returns the raw unit direction exactly. `knn_repulsion` selects the
four nearest active defenders by current distance; only neighbors closer
than 6 m contribute `0.40 max(0,1-d/6)^2.5` times the outward unit direction.
If the sum is nonzero, the normalized avoidance direction is blended with
raw guidance at fixed weight 0.40. This mode does not use velocity or CPA.

`predictive_cpa_knn` evaluates every active friendly pair. With relative
position `r=p_j-p_i` and velocity `v=v_j-v_i`, it clips
`t_cpa=-(r·v)/(v·v)` to `[0,3 s]`, then computes `d_cpa=||r+v t_cpa||`.
Strictly diverging pairs have zero risk. Equal-velocity pairs use `t_cpa=0`:
they have zero risk outside 8 m, and current-distance risk inside 8 m.
Distance risk is `clip(1-(d_cpa-3)/5,0,1)`; time risk is
`clip(1-t_cpa/3,0,1)`. Their product orders positive-risk neighbors by
descending risk, then ascending CPA time and defender index. At most four
contribute. Their risk-weighted outward CPA directions are normalized and
blended with guidance at `0.40 × maximum pair risk`. If the CPA separation
is zero, a deterministic orthogonal escape direction is used. If contributions
cancel, raw guidance is retained. The final unit direction still maps to
the configured constant defender speed.

The 3 m safety and 8 m alert radii are controller and audit definitions.
Physical geometric collision remains at `contact_distance=0.1 m`.
Offline safety analysis uses continuous within-step friendly-pair segments.
Each entry into `<3 m` or `<8 m` is counted once, including an entry and exit
within one control step. Alert exposure sums time inside 8 m across unique
friendly pairs, in **pair-seconds**. The minimum actual pair distance is
computed continuously. CPA risk audit uses only recorded decision states.
The trace records raw and adjusted headings, selected neighbors, risks,
avoidance activity, weights, and angular deviation per active defender.
`predictive_early_activation_fraction` is the fraction of active CPA
defender decisions for which every selected risk neighbor was still farther
than 8 m at the decision instant.

## Nominal and audited stress scenes

`formal_6v6_level2_config()` remains the nominal 50 m defender circle.
`formal_6v6_avoidance_stress_config(radius)` changes only the defender circle
radius. Its `scenario_id` includes the actual radius and
`avoidance_stress_v1` profile version, and excludes avoidance mode. The
nominal Level 2 identity no longer includes the unused Level 1 control
perturbation ratio. This corrects the identity without changing any path.

Before comparing controllers, the no-avoidance radius audit runs seeds
17–66 at 30, 20, 15, 12, and 10 m. A radius is invalid if the initial
minimum defender separation is at most 8 m. Among valid radii with 20%–60%
of episodes reaching `<8 m`, the largest is chosen. If none qualify, choose
the rate closest to 40%, breaking ties toward the larger radius. If even
the smallest valid candidate is below 20%, the audit reports inadequate
pressure and formal stress benchmarking must stop. The rule cannot use
controller performance.

Run in Ubuntu after `conda activate uav` from the repository root:

```bash
python -m tools.ideal_collision_risk_audit --seed-start 17 --trials 50 --out outputs/ideal_guidance/v34_avoidance_radius_audit_17_66.json
python -m tools.ideal_collision_avoidance_benchmark --scenario-profile nominal --seed-start 17 --trials 50 --out outputs/ideal_guidance/v34_avoidance_nominal_17_66.json
python -m tools.ideal_collision_avoidance_benchmark --scenario-profile stress --radius-audit outputs/ideal_guidance/v34_avoidance_radius_audit_17_66.json --seed-start 17 --trials 50 --out outputs/ideal_guidance/v34_avoidance_stress_17_66.json
python -m algorithm.ideal_collision_avoidance_guidance --scenario-profile stress --radius-audit outputs/ideal_guidance/v34_avoidance_radius_audit_17_66.json --avoidance-mode predictive_cpa_knn --seed 17 --out outputs/ideal_guidance/v34_avoidance_stress_seed17_cpa
python -m tools.ideal_interception_replay outputs/ideal_guidance/v34_avoidance_stress_seed17_cpa/episode.json outputs/ideal_guidance/v34_avoidance_stress_seed17_cpa/episode_full_3d.mp4 --fps 20
```

For the other single-run videos, change `--avoidance-mode` to `none` or
`knn_repulsion` and use a new output directory. The replay keeps the same
arena axes and labels the method and avoidance mode. The benchmark enforces
identical seed, scene ID, path hashes, starts, path specifications, and
Hungarian assignment across modes; defender trajectories may differ.
