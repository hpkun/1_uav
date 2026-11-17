# Ideal interception V3.5: formal 15v15 nominal experiment

The active protocol is `uav-ideal-interception-v3.5.0`. Slot counts are 15
defenders and 15 intruders. The shared Actor receives 265 features per defender,
the centralized Critic receives 444 features and returns 15 values, and the
action tensor has shape `(15, 3)`. Inactive slots store neutral actions and
do not move. Existing 1v1, 3v1, 5v2 and 6v6 scenes retain their active counts.
V3.4 and earlier training Checkpoints cannot be resumed as V3.5; the loader
requires an exact protocol match. No MAPPO training is part of this experiment.

## Frozen nominal geometry

Use `formal_15v15_level2_config()`. The arena is
`(-300,-300,0)` to `(300,300,120)` metres, the protected cylinder is centered
at `(-220,0,0)` with radius 20 m and height 100 m, and geometric contact
distance is 0.1 m. Both teams fly at 20 m/s, with a 0.05 s control period and
600-step (30 s) budget. Defenders start at
`(-120+60 cos(2πi/15), 60 sin(2πi/15), 60)` for `i=0,…,14`, heading in
the positive x direction. Their nearest initial spacing is
`120 sin(π/15) = 24.9494028981 m`. Intruder anchors are in
`FIFTEEN_INTRUDER_WEDGE` in `env/ideal_interception/config.py`; I0–I5 match
the formal 6v6 anchors exactly. Each seed draws 15 independent, pre-programmed
Level-2 cubic Bézier paths with the existing start jitter, control-point
distribution, rejection tests and arc-length speed parameterization.

Hungarian assignment minimizes initial 3D distance and is computed once at
reset. It is strictly one-to-one and is filtered when a member exits. The
ThreePoint-PIP solver and warmup/fallback rules are unchanged. The avoidance
modes are `none`, `knn_repulsion`, and `predictive_cpa_knn`, with their V3.4
parameters unchanged. The 15v15 nominal radius remains 60 m regardless of
audit results. There is no 15v15 stress profile.

For the kNN mode, every neighbor contribution is multiplied by the same
positive `repulsion_lambda=0.40` and the resulting vector is normalized.
Consequently, that common multiplier does not change the final repulsion
direction; `fixed_weight=0.40` controls its contribution to the blended
guidance direction. The CPA metrics named
`minimum_preavoidance_predicted_cpa_distance_m` and
`maximum_preavoidance_cpa_risk` use step-start velocities before avoidance
acts. Historical field names remain as aliases in episode summaries.

Friendly safety is measured on continuous motion segments, including minimum
distance, time spent below 8 m, crossings below 3 m and geometric contacts.
These are abstract point-mass events, not physical airframe collisions.

## Reproducible commands

Run in WSL Ubuntu after `conda activate uav`, from the repository root:

```bash
python -m tools.ideal_15v15_path_audit --out outputs/ideal_guidance/v35_15v15_path_audit_0_99.json
python -m tools.ideal_collision_risk_audit --scale 15v15 --scenario-profile nominal --seed-start 17 --trials 50 --out outputs/ideal_guidance/v35_15v15_nominal_risk_audit_17_66.json
python -m tools.ideal_collision_avoidance_benchmark --scale 15v15 --scenario-profile nominal --seed-start 17 --trials 50 --out outputs/ideal_guidance/v35_15v15_avoidance_nominal_17_66.json
python -m algorithm.ideal_collision_avoidance_guidance --scale 15v15 --scenario-profile nominal --avoidance-mode predictive_cpa_knn --seed 17 --out outputs/ideal_guidance/v35_15v15_seed17_cpa
python -m tools.ideal_interception_replay outputs/ideal_guidance/v35_15v15_seed17_cpa/episode.json outputs/ideal_guidance/v35_15v15_seed17_cpa/episode_full_3d.mp4 --fps 20
```

The episode command is also available with `none` and `knn_repulsion`.
Outputs are write-once by default. Each paired trial checks the same
`scenario_id`, initial states, assignment, Bézier controls and path hashes
across all avoidance modes. The only intended difference is the closed-loop
defender trajectory. For 6v6 regression, retain the 6v6 nominal and historical
stress helpers; no old trace or Checkpoint is overwritten.

Timeline and replay read slots by environment protocol: V3.5 uses 15/15,
V3.4 and V3.3 use 6/6, and V3.2, V3.1, V3.0 and V2 use 5/2. Replay colors
are red for defenders and blue for intruders. The trace schema remains
`uav-ideal-trace-v2`; the slot schema is selected using the environment
protocol. Historical playback does not upgrade a training Checkpoint.
For 15v15, standalone HTML uses the recorded 0.05 s control frames and exact
event keyframes, avoiding redundant 30 FPS copies of all 30 trails. MP4 uses
its requested export FPS.
