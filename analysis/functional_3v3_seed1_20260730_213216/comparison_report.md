# Functional 3v3 MAPPO Comparison, Revised Seed-1 Analysis

Batch: `20260730_213216`

## Scope

This report reads the three completed result directories only. It does not retrain, reevaluate checkpoints, or modify environment, reward, algorithm, configuration, output, log, or checkpoint files. The canonical analysis products are regenerated in `analysis/functional_3v3_seed1_20260730_213216/`.

## Research Questions and Judgments

H1: whether functional role differentiation forms. Evidence includes Combat vs Support structural role differences, Support being unarmed with longer-range sensing, support coverage, support position error, incoming threat, support survival, mission success, and whether stable rear support, survival, and information coverage behavior appears. Judgment: **partially supported** at the structural and sensing levels, **not supported** at the successful behavioral level. Support metrics exist and coverage is nonzero, but held-out Support survival and mission success are 0.

H2: whether relay compensates for the firepower loss caused by removing one armed Combat UAV. The primary comparison is heterogeneous relay vs homogeneous control, with relay vs no-relay as a mechanism check. Judgment: **not supported**. Relay is accompanied by nonzero extra visibility and, at the combat-ranked validation best checkpoint, more attacks/hits/damage than no-relay, but it does not reduce blue survivors, create red wins, produce elimination wins, preserve Support survival, or produce mission success.

H3: whether the experiment successfully isolates functional heterogeneity under identical maneuverability. Judgment: **design condition satisfied**. The three runs keep dynamics, network dimensions, initial scenario, MAPPO hyperparameters, seed, opponent, rollout length, environment steps, observation/state schemas, and training schedule fixed. The intended differences are role/weapon/sensing/reward/relay semantics.

## Checkpoint Selection

`checkpoint_selection: combat` uses the code path in `src/uav_env/algorithms/mappo/metrics.py::evaluation_key`. Validation checkpoints are ranked lexicographically by:

1. elimination red win rate, higher is better
2. overall red win rate, higher is better
3. red effective damage through `mean_effective_damage` or `mean_red_effective_damage`, higher is better
4. survivor difference, higher is better
5. red hits through `mean_hits` or `mean_red_hits`, higher is better
6. attack-area steps through `mean_attack_area_steps` or `mean_red_attack_area_steps`, higher is better
7. mean team episode return or mean episode return, higher is better
8. red crash rate, lower is better by negation
9. timeout rate, lower is better by negation

The selected checkpoint should be called **combat-ranked validation best checkpoint**. It is not the best held-out test checkpoint, not the best return checkpoint, not the best survival checkpoint, and not a global optimum. Because all validation red win rates are 0 here, later tuple fields such as effective damage determine the selected `best`.

| mode | best environment steps | ranking tuple |
| --- | --- | --- |
| Homogeneous | 51200 | combat-ranked validation best checkpoint selected by tuple (0.000, 0.000, 10.050, -2.700, 0.550, 16.950, -134.598, -0.100, -0.250) |
| Heterogeneous No Relay | 151552 | combat-ranked validation best checkpoint selected by tuple (0.000, 0.000, 6.300, -2.550, 0.300, 14.350, -117.999, -1.000, -0.250) |
| Heterogeneous Relay | 51200 | combat-ranked validation best checkpoint selected by tuple (0.000, 0.000, 65.250, -2.750, 3.250, 18.000, -128.986, -0.150, -0.250) |

Homogeneous best vs last reflects an attack/survival trade-off: `best` has more combat damage (4.80 vs 0.00 held-out) but much worse return and red survival. Relay best vs last reflects a stronger attack/avoidance trade-off: `best` has 74.00 damage and 3.50 hits but lower return and fewer red survivors than `last`. No-relay best is selected by validation combat ranking despite held-out combat output remaining 0. Validation combat ranking and held-out test behavior can diverge.

## Artifact and Metadata Integrity

| mode | complete | steps | updates | episodes | best step | ckpt metadata ok | schema | obs/state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| homogeneous_control | True | 301056 | 147 | 893 | 51200 | True | functional_heterogeneous_3v3_v1 | 69/64 |
| heterogeneous_no_relay | True | 301056 | 147 | 915 | 151552 | True | functional_heterogeneous_3v3_v1 | 69/64 |
| heterogeneous_relay | True | 301056 | 147 | 1027 | 51200 | True | functional_heterogeneous_3v3_v1 | 69/64 |

All three runs reach 301,056 rollout-aligned environment steps. All required artifacts are present. Checkpoint metadata is readable for `initial.pt`, `last.pt`, and `best.pt`; checkpoint schemas are `functional_heterogeneous_3v3_v1`, obs dim is 69, and state dim is 64. `last.pt` records 301,056 steps in all runs, while `best.pt` steps match `final_summary.yaml` and validation records.

## Held-Out Test Core Table

`timeout_rate` is a termination condition, not a winner class; it should not be added to red/blue/draw rates. Homogeneous Support performance fields are N/A because no Support agent exists.

| mode | checkpoint | overall_red_win_rate | blue_win_rate | draw_rate | timeout_rate | elimination_red_win_rate | mean_episode_return | mean_red_survivors | mean_blue_survivors | combat_attack_attempts_mean | combat_hits_mean | combat_effective_damage_mean | support_survival_rate | support_detection_coverage_mean | relay_visible_enemy_count_mean | support_incoming_threat_mean | support_position_error_mean | mission_success_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Homogeneous | last | 0.000 | 0.850 | 0.150 | 1.000 | 0.000 | -53.254 | 2.050 | 3.000 | 0.000 | 0.000 | 0.000 | N/A | N/A | N/A | N/A | N/A | N/A |
| Homogeneous | best | 0.000 | 0.950 | 0.050 | 0.550 | 0.000 | -124.587 | 0.750 | 3.000 | 0.350 | 0.300 | 4.800 | N/A | N/A | N/A | N/A | N/A | N/A |
| Heterogeneous No Relay | last | 0.000 | 1.000 | 0.000 | 0.950 | 0.000 | -57.869 | 1.000 | 3.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.292 | 1038.227 | 0.000 |
| Heterogeneous No Relay | best | 0.000 | 1.000 | 0.000 | 0.250 | 0.000 | -113.267 | 0.450 | 3.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.990 | 0.000 | 0.315 | 630.141 | 0.000 |
| Heterogeneous Relay | last | 0.000 | 1.000 | 0.000 | 0.450 | 0.000 | -88.648 | 0.450 | 3.000 | 0.600 | 0.400 | 8.400 | 0.000 | 0.812 | 1.046 | 0.259 | 531.500 | 0.000 |
| Heterogeneous Relay | best | 0.000 | 1.000 | 0.000 | 0.200 | 0.000 | -119.564 | 0.200 | 3.000 | 4.450 | 3.500 | 74.000 | 0.000 | 0.811 | 1.268 | 0.384 | 1810.117 | 0.000 |

Homogeneous test-last is a strong timeout survival/delay policy: overall red win 0.00, blue win 0.85, draw 0.15, timeout 1.00, red survivors 2.05, blue survivors 3.00, combat damage 0.00. It partially forms draws through timeout but never red victory.

## Support Behavior

Support position tolerance is 900 m.

| mode | checkpoint | position error m | <=900m | incoming threat | coverage | relay-visible enemies | support survival | mission success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Heterogeneous No Relay | last | 1038.227 | False | 0.292 | 1.000 | 0.000 | 0.000 | 0.000 |
| Heterogeneous No Relay | best | 630.141 | True | 0.315 | 0.990 | 0.000 | 0.000 | 0.000 |
| Heterogeneous Relay | last | 531.500 | True | 0.259 | 0.812 | 1.046 | 0.000 | 0.000 |
| Heterogeneous Relay | best | 1810.117 | False | 0.384 | 0.811 | 1.268 | 0.000 | 0.000 |

No-relay last has perfect coverage (1.000) but position error is 1038.23 m, above tolerance, and Support survival is 0. No-relay best has position error 630.14 m and coverage 0.990, but still has Support survival 0 and mission success 0.

Relay last has lower position error (531.50 m) than relay best and is within tolerance, yet Support still does not survive; the likely descriptive interpretation is that rear positioning and visibility alone did not prevent lethal exposure under the current policy/opponent. Relay best has much stronger combat activity, but this is accompanied by much larger position error (1810.12 m), higher incoming threat (0.384), Support survival 0, and mission success 0. Coverage and relay-visible information are meaningful, but current held-out results prove sensing capability, not stable support-role behavior.

## Relay vs No-Relay Absolute Deltas

At held-out last, relay minus no-relay has return delta -30.78, red-survivor delta -0.55, attack-attempt delta 0.60, hit delta 0.40, damage delta 8.40, coverage delta -0.188, relay-visible delta 1.046, support-survival delta 0.00, and mission-success delta 0.00.

At the combat-ranked validation best checkpoint, relay minus no-relay has attack-attempt delta 4.45, hit delta 3.50, damage delta 74.00, red-survivor delta -0.25, blue-survivor delta 0.00, win delta 0.00, elimination-win delta 0.00, Support-survival delta 0.00, and mission-success delta 0.00. These are absolute descriptive deltas; no relative percentages are reported for returns, survivor difference, position error, incoming threat, or zero-denominator cases.

| checkpoint | contrast | metric | absolute delta |
| --- | --- | --- | --- |
| last | heterogeneous_no_relay - homogeneous_control | overall_red_win_rate | 0.000 |
| last | heterogeneous_no_relay - homogeneous_control | elimination_red_win_rate | 0.000 |
| last | heterogeneous_no_relay - homogeneous_control | blue_win_rate | 0.150 |
| last | heterogeneous_no_relay - homogeneous_control | draw_rate | -0.150 |
| last | heterogeneous_no_relay - homogeneous_control | timeout_rate | -0.050 |
| last | heterogeneous_no_relay - homogeneous_control | mean_episode_return | -4.616 |
| last | heterogeneous_no_relay - homogeneous_control | mean_survivor_difference | -1.050 |
| last | heterogeneous_no_relay - homogeneous_control | mean_red_survivors | -1.050 |
| last | heterogeneous_no_relay - homogeneous_control | mean_blue_survivors | 0.000 |
| last | heterogeneous_no_relay - homogeneous_control | combat_attack_attempts_mean | 0.000 |
| last | heterogeneous_no_relay - homogeneous_control | combat_hits_mean | 0.000 |
| last | heterogeneous_no_relay - homogeneous_control | combat_effective_damage_mean | 0.000 |
| last | heterogeneous_no_relay - homogeneous_control | support_survival_rate | N/A |
| last | heterogeneous_no_relay - homogeneous_control | mission_success_rate | N/A |
| last | heterogeneous_no_relay - homogeneous_control | support_detection_coverage_mean | N/A |
| last | heterogeneous_no_relay - homogeneous_control | relay_visible_enemy_count_mean | N/A |
| last | heterogeneous_no_relay - homogeneous_control | support_incoming_threat_mean | N/A |
| last | heterogeneous_no_relay - homogeneous_control | support_position_error_mean | N/A |
| last | heterogeneous_relay - heterogeneous_no_relay | overall_red_win_rate | 0.000 |
| last | heterogeneous_relay - heterogeneous_no_relay | elimination_red_win_rate | 0.000 |
| last | heterogeneous_relay - heterogeneous_no_relay | blue_win_rate | 0.000 |
| last | heterogeneous_relay - heterogeneous_no_relay | draw_rate | 0.000 |
| last | heterogeneous_relay - heterogeneous_no_relay | timeout_rate | -0.500 |
| last | heterogeneous_relay - heterogeneous_no_relay | mean_episode_return | -30.779 |
| last | heterogeneous_relay - heterogeneous_no_relay | mean_survivor_difference | -0.550 |
| last | heterogeneous_relay - heterogeneous_no_relay | mean_red_survivors | -0.550 |
| last | heterogeneous_relay - heterogeneous_no_relay | mean_blue_survivors | 0.000 |
| last | heterogeneous_relay - heterogeneous_no_relay | combat_attack_attempts_mean | 0.600 |
| last | heterogeneous_relay - heterogeneous_no_relay | combat_hits_mean | 0.400 |
| last | heterogeneous_relay - heterogeneous_no_relay | combat_effective_damage_mean | 8.400 |
| last | heterogeneous_relay - heterogeneous_no_relay | support_survival_rate | 0.000 |
| last | heterogeneous_relay - heterogeneous_no_relay | mission_success_rate | 0.000 |
| last | heterogeneous_relay - heterogeneous_no_relay | support_detection_coverage_mean | -0.188 |
| last | heterogeneous_relay - heterogeneous_no_relay | relay_visible_enemy_count_mean | 1.046 |
| last | heterogeneous_relay - heterogeneous_no_relay | support_incoming_threat_mean | -0.033 |
| last | heterogeneous_relay - heterogeneous_no_relay | support_position_error_mean | -506.727 |
| last | heterogeneous_relay - homogeneous_control | overall_red_win_rate | 0.000 |
| last | heterogeneous_relay - homogeneous_control | elimination_red_win_rate | 0.000 |
| last | heterogeneous_relay - homogeneous_control | blue_win_rate | 0.150 |
| last | heterogeneous_relay - homogeneous_control | draw_rate | -0.150 |
| last | heterogeneous_relay - homogeneous_control | timeout_rate | -0.550 |
| last | heterogeneous_relay - homogeneous_control | mean_episode_return | -35.394 |
| last | heterogeneous_relay - homogeneous_control | mean_survivor_difference | -1.600 |
| last | heterogeneous_relay - homogeneous_control | mean_red_survivors | -1.600 |
| last | heterogeneous_relay - homogeneous_control | mean_blue_survivors | 0.000 |
| last | heterogeneous_relay - homogeneous_control | combat_attack_attempts_mean | 0.600 |
| last | heterogeneous_relay - homogeneous_control | combat_hits_mean | 0.400 |
| last | heterogeneous_relay - homogeneous_control | combat_effective_damage_mean | 8.400 |
| last | heterogeneous_relay - homogeneous_control | support_survival_rate | N/A |
| last | heterogeneous_relay - homogeneous_control | mission_success_rate | N/A |
| last | heterogeneous_relay - homogeneous_control | support_detection_coverage_mean | N/A |
| last | heterogeneous_relay - homogeneous_control | relay_visible_enemy_count_mean | N/A |
| last | heterogeneous_relay - homogeneous_control | support_incoming_threat_mean | N/A |
| last | heterogeneous_relay - homogeneous_control | support_position_error_mean | N/A |
| best | heterogeneous_no_relay - homogeneous_control | overall_red_win_rate | 0.000 |
| best | heterogeneous_no_relay - homogeneous_control | elimination_red_win_rate | 0.000 |
| best | heterogeneous_no_relay - homogeneous_control | blue_win_rate | 0.050 |
| best | heterogeneous_no_relay - homogeneous_control | draw_rate | -0.050 |
| best | heterogeneous_no_relay - homogeneous_control | timeout_rate | -0.300 |
| best | heterogeneous_no_relay - homogeneous_control | mean_episode_return | 11.320 |
| best | heterogeneous_no_relay - homogeneous_control | mean_survivor_difference | -0.300 |
| best | heterogeneous_no_relay - homogeneous_control | mean_red_survivors | -0.300 |
| best | heterogeneous_no_relay - homogeneous_control | mean_blue_survivors | 0.000 |
| best | heterogeneous_no_relay - homogeneous_control | combat_attack_attempts_mean | -0.350 |
| best | heterogeneous_no_relay - homogeneous_control | combat_hits_mean | -0.300 |
| best | heterogeneous_no_relay - homogeneous_control | combat_effective_damage_mean | -4.800 |
| best | heterogeneous_no_relay - homogeneous_control | support_survival_rate | N/A |
| best | heterogeneous_no_relay - homogeneous_control | mission_success_rate | N/A |
| best | heterogeneous_no_relay - homogeneous_control | support_detection_coverage_mean | N/A |
| best | heterogeneous_no_relay - homogeneous_control | relay_visible_enemy_count_mean | N/A |
| best | heterogeneous_no_relay - homogeneous_control | support_incoming_threat_mean | N/A |
| best | heterogeneous_no_relay - homogeneous_control | support_position_error_mean | N/A |
| best | heterogeneous_relay - heterogeneous_no_relay | overall_red_win_rate | 0.000 |
| best | heterogeneous_relay - heterogeneous_no_relay | elimination_red_win_rate | 0.000 |
| best | heterogeneous_relay - heterogeneous_no_relay | blue_win_rate | 0.000 |
| best | heterogeneous_relay - heterogeneous_no_relay | draw_rate | 0.000 |
| best | heterogeneous_relay - heterogeneous_no_relay | timeout_rate | -0.050 |
| best | heterogeneous_relay - heterogeneous_no_relay | mean_episode_return | -6.297 |
| best | heterogeneous_relay - heterogeneous_no_relay | mean_survivor_difference | -0.250 |
| best | heterogeneous_relay - heterogeneous_no_relay | mean_red_survivors | -0.250 |
| best | heterogeneous_relay - heterogeneous_no_relay | mean_blue_survivors | 0.000 |
| best | heterogeneous_relay - heterogeneous_no_relay | combat_attack_attempts_mean | 4.450 |
| best | heterogeneous_relay - heterogeneous_no_relay | combat_hits_mean | 3.500 |
| best | heterogeneous_relay - heterogeneous_no_relay | combat_effective_damage_mean | 74.000 |
| best | heterogeneous_relay - heterogeneous_no_relay | support_survival_rate | 0.000 |
| best | heterogeneous_relay - heterogeneous_no_relay | mission_success_rate | 0.000 |
| best | heterogeneous_relay - heterogeneous_no_relay | support_detection_coverage_mean | -0.178 |
| best | heterogeneous_relay - heterogeneous_no_relay | relay_visible_enemy_count_mean | 1.268 |
| best | heterogeneous_relay - heterogeneous_no_relay | support_incoming_threat_mean | 0.069 |
| best | heterogeneous_relay - heterogeneous_no_relay | support_position_error_mean | 1179.976 |
| best | heterogeneous_relay - homogeneous_control | overall_red_win_rate | 0.000 |
| best | heterogeneous_relay - homogeneous_control | elimination_red_win_rate | 0.000 |
| best | heterogeneous_relay - homogeneous_control | blue_win_rate | 0.050 |
| best | heterogeneous_relay - homogeneous_control | draw_rate | -0.050 |
| best | heterogeneous_relay - homogeneous_control | timeout_rate | -0.350 |
| best | heterogeneous_relay - homogeneous_control | mean_episode_return | 5.023 |
| best | heterogeneous_relay - homogeneous_control | mean_survivor_difference | -0.550 |
| best | heterogeneous_relay - homogeneous_control | mean_red_survivors | -0.550 |
| best | heterogeneous_relay - homogeneous_control | mean_blue_survivors | 0.000 |
| best | heterogeneous_relay - homogeneous_control | combat_attack_attempts_mean | 4.100 |
| best | heterogeneous_relay - homogeneous_control | combat_hits_mean | 3.200 |
| best | heterogeneous_relay - homogeneous_control | combat_effective_damage_mean | 69.200 |
| best | heterogeneous_relay - homogeneous_control | support_survival_rate | N/A |
| best | heterogeneous_relay - homogeneous_control | mission_success_rate | N/A |
| best | heterogeneous_relay - homogeneous_control | support_detection_coverage_mean | N/A |
| best | heterogeneous_relay - homogeneous_control | relay_visible_enemy_count_mean | N/A |
| best | heterogeneous_relay - homogeneous_control | support_incoming_threat_mean | N/A |
| best | heterogeneous_relay - homogeneous_control | support_position_error_mean | N/A |

## Training Windows

Training-window statistics use `last_10`, `last_20`, and 10-update sliding windows. `best_10_mission_success` is marked `tied_all_zero` for all three modes because mission success is zero throughout training; there is no meaningful best mission-success window.

| mode | window | status | start update | end update | start steps | end steps | selection value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| homogeneous_control | all | selected | 1.000 | 147.000 | 2048.000 | 301056.000 | N/A |
| homogeneous_control | last_10 | selected | 138.000 | 147.000 | 282624.000 | 301056.000 | N/A |
| homogeneous_control | last_20 | selected | 128.000 | 147.000 | 262144.000 | 301056.000 | N/A |
| homogeneous_control | best_10_return | selected | 137.000 | 146.000 | 280576.000 | 299008.000 | -77.377 |
| homogeneous_control | best_10_red_damage | tied_best | N/A | N/A | N/A | N/A | 2.092 |
| homogeneous_control | best_10_mission_success | tied_all_zero | N/A | N/A | N/A | N/A | 0.000 |
| heterogeneous_no_relay | all | selected | 1.000 | 147.000 | 2048.000 | 301056.000 | N/A |
| heterogeneous_no_relay | last_10 | selected | 138.000 | 147.000 | 282624.000 | 301056.000 | N/A |
| heterogeneous_no_relay | last_20 | selected | 128.000 | 147.000 | 262144.000 | 301056.000 | N/A |
| heterogeneous_no_relay | best_10_return | selected | 47.000 | 56.000 | 96256.000 | 114688.000 | -59.257 |
| heterogeneous_no_relay | best_10_red_damage | tied_all_zero | N/A | N/A | N/A | N/A | 0.000 |
| heterogeneous_no_relay | best_10_mission_success | tied_all_zero | N/A | N/A | N/A | N/A | 0.000 |
| heterogeneous_relay | all | selected | 1.000 | 147.000 | 2048.000 | 301056.000 | N/A |
| heterogeneous_relay | last_10 | selected | 138.000 | 147.000 | 282624.000 | 301056.000 | N/A |
| heterogeneous_relay | last_20 | selected | 128.000 | 147.000 | 262144.000 | 301056.000 | N/A |
| heterogeneous_relay | best_10_return | selected | 50.000 | 59.000 | 102400.000 | 120832.000 | -66.675 |
| heterogeneous_relay | best_10_red_damage | tied_best | N/A | N/A | N/A | N/A | 7.355 |
| heterogeneous_relay | best_10_mission_success | tied_all_zero | N/A | N/A | N/A | N/A | 0.000 |

Final cumulative SPS is the last cumulative `samples_per_second` value, not the mean of cumulative SPS values.

| mode | final cumulative SPS | min cumulative SPS | max cumulative SPS |
| --- | --- | --- | --- |
| Homogeneous | 133.518 | 123.098 | 229.939 |
| Heterogeneous No Relay | 145.358 | 131.529 | 287.819 |
| Heterogeneous Relay | 156.277 | 139.751 | 280.479 |

## Validation Trajectory Summary

Validation contains six scheduled evaluations per mode. All validation red win rates are 0. Homogeneous/no-relay late checkpoints mostly move toward timeout survival and low combat output. Relay has a combat-active validation best at 51,200 steps, but later training reduces combat activity while not creating held-out wins.

## Evidence Boundary

The following statements are supported: relay produced nonzero extra visible enemies; relay combat-ranked best was accompanied by more attack attempts, hits, and damage than no-relay best; these phenomena appeared together in seed 1; the extra information did not convert to red win rate, elimination, lower blue survival, Support survival, or mission success.

The following stronger statements are not supported by this dataset: relay caused an attack capability improvement; relay significantly improved attack; relay proved effective; relay compensated for the lost Combat UAV; functional heterogeneity improved mission performance.

## Final Conclusions

1. All three groups have zero held-out red wins and zero elimination wins.
2. Homogeneous and no-relay last checkpoints mainly show survival/avoidance local optima.
3. The relay link actually provides additional visible targets.
4. Relay combat-ranked best shows more combat activity than no-relay best.
5. Attack activity does not convert into lower blue survival, red wins, eliminations, Support survival, or mission success.
6. H1 is partially supported only structurally and perceptually; behavioral role formation is not supported.
7. H2 is not supported.
8. H3's controlled design condition is satisfied.
9. Results are descriptive for seed 1 and 20 held-out test episodes; statistical significance is not claimed.
10. The highest-priority next step is to keep configuration unchanged and add independent seed replications.
