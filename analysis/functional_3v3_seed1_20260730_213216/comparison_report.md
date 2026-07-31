# Functional 3v3 MAPPO comparison — seed 1, batch 20260730_213216

## 1. Executive Summary

All three 300k experiments are complete and internally consistent at **301,056 environment steps** (147 × 2,048). None learned a held-out red win: test `last` and `best` overall/elimination win rates and heterogeneous mission-success rates are all **0**. Training mainly moved toward survival/avoidance. Relay produced real visible information (`relay_visible_enemy_count_mean`: last **1.046**, best **1.268**) and, at `best`, more combat activity than no-relay (attempts **4.45 vs 0.00**, damage **74.00 vs 0.00**), but it did not convert into wins or mission success. These are preliminary seed-1 aggregates, not significance claims.

## 2. Experiment Integrity

| mode | complete | steps | updates | episodes | best step | NaN | Inf |
| --- | --- | --- | --- | --- | --- | --- | --- |
| homogeneous_control | True | 301056 | 147 | 893 | 51200 | 0 | 0 |
| heterogeneous_no_relay | True | 301056 | 147 | 915 | 151552 | 0 | 0 |
| heterogeneous_relay | True | 301056 | 147 | 1027 | 51200 | 0 | 0 |

Required artifacts and all initial/last/best test blocks exist. Schema is `functional_heterogeneous_3v3_v1`, Actor=69D, Critic=64D, seed=1, 16 parallel envs, rollout=128, opponent=`greedy_combat`. CSV numeric columns contain no NaN or Inf. Homogeneous support fields are encoded as zeros upstream but are treated as **N/A** here because `support_metrics_applicable=0`.

## 3. Experimental Design and Controlled Variables

- Homogeneous: 3 armed Combat UAVs; no Support, no relay.
- No relay: 2 armed Combat UAVs + 1 unarmed long-range sensing Support UAV; no information sharing.
- Relay: same heterogeneous roles/weapons/reward as no-relay; Support shares information while alive.
- Controlled: dynamics, initial scenario, 69D Actor input, 64D Critic state, MAPPO hyperparameters, 300k target, seed, and GreedyCombat opponent.
- Contrasts: no-relay−homogeneous combines role differentiation and loss of one weapon platform; relay−no-relay isolates relay sharing; relay−homogeneous measures the complete heterogeneous scheme.

## 4. Validation Trajectories

- **Homogeneous**: steps [51200, 100352, 151552, 200704, 251904, 301056]; selected best=51200. Return -134.60 → -58.58, timeout 0.25 → 1.00, red survivors 0.30 → 1.95, combat damage 10.05 → 0.00.
- **Heterogeneous, no relay**: steps [51200, 100352, 151552, 200704, 251904, 301056]; selected best=151552. Return -91.47 → -58.82, timeout 0.70 → 1.00, red survivors 0.85 → 1.00, combat damage 0.00 → 0.00.
- **Heterogeneous, relay**: steps [51200, 100352, 151552, 200704, 251904, 301056]; selected best=51200. Return -128.99 → -88.26, timeout 0.25 → 0.45, red survivors 0.25 → 0.45, combat damage 65.25 → 8.45.

No group shows validation win improvement. Returns improve largely alongside rising timeout/survival and collapsing combat output, indicating an attack-to-avoidance transition. See `validation_*.png` and `validation_trajectory.csv`.

## 5. Held-Out Test Results

| mode | checkpoint | overall_red_win_rate | elimination_red_win_rate | mean_episode_return | timeout_rate | mean_red_survivors | mean_blue_survivors | combat_attack_attempts_mean | combat_hits_mean | combat_effective_damage_mean | support_survival_rate | support_detection_coverage_mean | relay_visible_enemy_count_mean | mission_success_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Homogeneous | last | 0.000 | 0.000 | -53.254 | 1.000 | 2.050 | 3.000 | 0.000 | 0.000 | 0.000 | N/A | N/A | N/A | N/A |
| Homogeneous | best | 0.000 | 0.000 | -124.587 | 0.550 | 0.750 | 3.000 | 0.350 | 0.300 | 4.800 | N/A | N/A | N/A | N/A |
| Heterogeneous, no relay | last | 0.000 | 0.000 | -57.869 | 0.950 | 1.000 | 3.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| Heterogeneous, no relay | best | 0.000 | 0.000 | -113.267 | 0.250 | 0.450 | 3.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.990 | 0.000 | 0.000 |
| Heterogeneous, relay | last | 0.000 | 0.000 | -88.648 | 0.450 | 0.450 | 3.000 | 0.600 | 0.400 | 8.400 | 0.000 | 0.812 | 1.046 | 0.000 |
| Heterogeneous, relay | best | 0.000 | 0.000 | -119.564 | 0.200 | 0.200 | 3.000 | 4.450 | 3.500 | 74.000 | 0.000 | 0.811 | 1.268 | 0.000 |

## 6. Homogeneous vs Heterogeneous No Relay

At test-last, no-relay has the same zero wins, a slightly lower return (-57.87 vs -53.25), fewer red survivors (1.00 vs 2.05), and zero combat output in both. Removing one armed Combat UAV plus adding Support therefore yields no demonstrated combat benefit. Support survival is 0 at last/best, so the role reward did not produce a surviving support role on held-out tests.

## 7. No Relay vs Relay

Relay is operationally nonzero, but outcome conversion is absent. At test-best it adds 4.45 attempts, 3.50 hits and 74.00 damage per episode, while both have 0 wins/mission success and 0 support survival. At test-last it retains modest combat (8.40 damage vs 0.00) but has lower return and fewer red survivors. Classification: **B — relay increases usable visibility and some attack activity, but not wins/mission success**; it is not evidence of a successful information advantage.

## 8. Full Heterogeneous Scheme vs Homogeneous Control

At last, homogeneous attains the highest red survival (2.05) and best return (-53.25), but does so with zero combat damage and 100% timeout. Relay is more active but less survivable, still with zero wins. Thus the full heterogeneous design is not supported on outcome performance in this seed.

## 9. Support Role Analysis

Support is present and its sensing coverage is meaningful (no-relay last 1.000; relay last 0.812), but held-out support survival is 0 for both last and best and mission success remains 0. Role differentiation exists structurally and in diagnostics, but a successful behavioral Support role is not established.

## 10. Relay Information Value

No-relay relay-visible count is strictly 0; relay has positive count across initial/last/best. The channel therefore produces additional visibility. The best checkpoint converts it to attacks/hits/damage, but neither checkpoint converts it to victory, mission success, or Support survival.

## 11. Behavioral Local Optima

All groups exhibit survival/avoidance local optimization late in training: timeout rises, return and red survival improve relative to initial behavior, blue survival remains near 3, and red hits/damage approach zero. It is strongest in homogeneous and no-relay final validation/test; relay retains some attack activity and therefore shows a weaker but still outcome-failing version.

## 12. Best vs Last Checkpoint

Validation-selected best does not generalize as a winner: all held-out win rates remain zero. Homogeneous/no-relay `last` are more avoidant and higher-return than `best`; relay `best` is much more aggressive (damage 74.00 vs last 8.40) but suffers lower return and survival. This is behavioral trade-off and validation/test instability, not proof of statistical overfitting with only 20 aggregate test episodes.

## 13. Evidence for Research Hypotheses

- H1 (functional role differentiation): **partial structural evidence only**; sensing/role metrics exist, but Support never survives held-out tests.
- H2 (relay supplies information): **supported descriptively in seed 1** by positive relay-visible count versus strict zero without relay.
- H3 (relay improves combat/mission outcome): **not supported**; some best-checkpoint combat activity rises, but wins and mission success remain zero.

## 14. Limitations

One seed, 20 validation episodes and 20 test episodes per checkpoint; files provide aggregate metrics, not episode-level samples. Therefore no valid confidence intervals or hypothesis tests can be computed. Differences are descriptive and cannot be called statistically significant or stable. `mission_success` and Support metrics are applicable only to heterogeneous groups. Total/mean reward fields are not interchanged; this report uses `mean_episode_return` consistently.

### Absolute and relative deltas

Relative percentages are omitted when the denominator is zero or the metric is N/A.

| checkpoint | contrast | metric | absolute delta | relative % |
| --- | --- | --- | --- | --- |
| last | heterogeneous_no_relay - homogeneous_control | overall_red_win_rate | 0.000 | N/A |
| last | heterogeneous_no_relay - homogeneous_control | elimination_red_win_rate | 0.000 | N/A |
| last | heterogeneous_no_relay - homogeneous_control | mission_success_rate | N/A | N/A |
| last | heterogeneous_no_relay - homogeneous_control | support_survival_rate | N/A | N/A |
| last | heterogeneous_no_relay - homogeneous_control | mean_episode_return | -4.616 | -8.667 |
| last | heterogeneous_no_relay - homogeneous_control | timeout_rate | -0.050 | -5.000 |
| last | heterogeneous_no_relay - homogeneous_control | mean_red_survivors | -1.050 | -51.220 |
| last | heterogeneous_no_relay - homogeneous_control | mean_blue_survivors | 0.000 | 0.000 |
| last | heterogeneous_no_relay - homogeneous_control | combat_attack_attempts_mean | 0.000 | N/A |
| last | heterogeneous_no_relay - homogeneous_control | combat_hits_mean | 0.000 | N/A |
| last | heterogeneous_no_relay - homogeneous_control | combat_effective_damage_mean | 0.000 | N/A |
| last | heterogeneous_no_relay - homogeneous_control | support_detection_coverage_mean | N/A | N/A |
| last | heterogeneous_no_relay - homogeneous_control | relay_visible_enemy_count_mean | N/A | N/A |
| last | heterogeneous_relay - heterogeneous_no_relay | overall_red_win_rate | 0.000 | N/A |
| last | heterogeneous_relay - heterogeneous_no_relay | elimination_red_win_rate | 0.000 | N/A |
| last | heterogeneous_relay - heterogeneous_no_relay | mission_success_rate | 0.000 | N/A |
| last | heterogeneous_relay - heterogeneous_no_relay | support_survival_rate | 0.000 | N/A |
| last | heterogeneous_relay - heterogeneous_no_relay | mean_episode_return | -30.779 | -53.187 |
| last | heterogeneous_relay - heterogeneous_no_relay | timeout_rate | -0.500 | -52.632 |
| last | heterogeneous_relay - heterogeneous_no_relay | mean_red_survivors | -0.550 | -55.000 |
| last | heterogeneous_relay - heterogeneous_no_relay | mean_blue_survivors | 0.000 | 0.000 |
| last | heterogeneous_relay - heterogeneous_no_relay | combat_attack_attempts_mean | 0.600 | N/A |
| last | heterogeneous_relay - heterogeneous_no_relay | combat_hits_mean | 0.400 | N/A |
| last | heterogeneous_relay - heterogeneous_no_relay | combat_effective_damage_mean | 8.400 | N/A |
| last | heterogeneous_relay - heterogeneous_no_relay | support_detection_coverage_mean | -0.188 | -18.755 |
| last | heterogeneous_relay - heterogeneous_no_relay | relay_visible_enemy_count_mean | 1.046 | N/A |
| last | heterogeneous_relay - homogeneous_control | overall_red_win_rate | 0.000 | N/A |
| last | heterogeneous_relay - homogeneous_control | elimination_red_win_rate | 0.000 | N/A |
| last | heterogeneous_relay - homogeneous_control | mission_success_rate | N/A | N/A |
| last | heterogeneous_relay - homogeneous_control | support_survival_rate | N/A | N/A |
| last | heterogeneous_relay - homogeneous_control | mean_episode_return | -35.394 | -66.464 |
| last | heterogeneous_relay - homogeneous_control | timeout_rate | -0.550 | -55.000 |
| last | heterogeneous_relay - homogeneous_control | mean_red_survivors | -1.600 | -78.049 |
| last | heterogeneous_relay - homogeneous_control | mean_blue_survivors | 0.000 | 0.000 |
| last | heterogeneous_relay - homogeneous_control | combat_attack_attempts_mean | 0.600 | N/A |
| last | heterogeneous_relay - homogeneous_control | combat_hits_mean | 0.400 | N/A |
| last | heterogeneous_relay - homogeneous_control | combat_effective_damage_mean | 8.400 | N/A |
| last | heterogeneous_relay - homogeneous_control | support_detection_coverage_mean | N/A | N/A |
| last | heterogeneous_relay - homogeneous_control | relay_visible_enemy_count_mean | N/A | N/A |
| best | heterogeneous_no_relay - homogeneous_control | overall_red_win_rate | 0.000 | N/A |
| best | heterogeneous_no_relay - homogeneous_control | elimination_red_win_rate | 0.000 | N/A |
| best | heterogeneous_no_relay - homogeneous_control | mission_success_rate | N/A | N/A |
| best | heterogeneous_no_relay - homogeneous_control | support_survival_rate | N/A | N/A |
| best | heterogeneous_no_relay - homogeneous_control | mean_episode_return | 11.320 | 9.086 |
| best | heterogeneous_no_relay - homogeneous_control | timeout_rate | -0.300 | -54.545 |
| best | heterogeneous_no_relay - homogeneous_control | mean_red_survivors | -0.300 | -40.000 |
| best | heterogeneous_no_relay - homogeneous_control | mean_blue_survivors | 0.000 | 0.000 |
| best | heterogeneous_no_relay - homogeneous_control | combat_attack_attempts_mean | -0.350 | -100.000 |
| best | heterogeneous_no_relay - homogeneous_control | combat_hits_mean | -0.300 | -100.000 |
| best | heterogeneous_no_relay - homogeneous_control | combat_effective_damage_mean | -4.800 | -100.000 |
| best | heterogeneous_no_relay - homogeneous_control | support_detection_coverage_mean | N/A | N/A |
| best | heterogeneous_no_relay - homogeneous_control | relay_visible_enemy_count_mean | N/A | N/A |
| best | heterogeneous_relay - heterogeneous_no_relay | overall_red_win_rate | 0.000 | N/A |
| best | heterogeneous_relay - heterogeneous_no_relay | elimination_red_win_rate | 0.000 | N/A |
| best | heterogeneous_relay - heterogeneous_no_relay | mission_success_rate | 0.000 | N/A |
| best | heterogeneous_relay - heterogeneous_no_relay | support_survival_rate | 0.000 | N/A |
| best | heterogeneous_relay - heterogeneous_no_relay | mean_episode_return | -6.297 | -5.560 |
| best | heterogeneous_relay - heterogeneous_no_relay | timeout_rate | -0.050 | -20.000 |
| best | heterogeneous_relay - heterogeneous_no_relay | mean_red_survivors | -0.250 | -55.556 |
| best | heterogeneous_relay - heterogeneous_no_relay | mean_blue_survivors | 0.000 | 0.000 |
| best | heterogeneous_relay - heterogeneous_no_relay | combat_attack_attempts_mean | 4.450 | N/A |
| best | heterogeneous_relay - heterogeneous_no_relay | combat_hits_mean | 3.500 | N/A |
| best | heterogeneous_relay - heterogeneous_no_relay | combat_effective_damage_mean | 74.000 | N/A |
| best | heterogeneous_relay - heterogeneous_no_relay | support_detection_coverage_mean | -0.178 | -18.027 |
| best | heterogeneous_relay - heterogeneous_no_relay | relay_visible_enemy_count_mean | 1.268 | N/A |
| best | heterogeneous_relay - homogeneous_control | overall_red_win_rate | 0.000 | N/A |
| best | heterogeneous_relay - homogeneous_control | elimination_red_win_rate | 0.000 | N/A |
| best | heterogeneous_relay - homogeneous_control | mission_success_rate | N/A | N/A |
| best | heterogeneous_relay - homogeneous_control | support_survival_rate | N/A | N/A |
| best | heterogeneous_relay - homogeneous_control | mean_episode_return | 5.023 | 4.031 |
| best | heterogeneous_relay - homogeneous_control | timeout_rate | -0.350 | -63.636 |
| best | heterogeneous_relay - homogeneous_control | mean_red_survivors | -0.550 | -73.333 |
| best | heterogeneous_relay - homogeneous_control | mean_blue_survivors | 0.000 | 0.000 |
| best | heterogeneous_relay - homogeneous_control | combat_attack_attempts_mean | 4.100 | 1171.429 |
| best | heterogeneous_relay - homogeneous_control | combat_hits_mean | 3.200 | 1066.667 |
| best | heterogeneous_relay - homogeneous_control | combat_effective_damage_mean | 69.200 | 1441.667 |
| best | heterogeneous_relay - homogeneous_control | support_detection_coverage_mean | N/A | N/A |
| best | heterogeneous_relay - homogeneous_control | relay_visible_enemy_count_mean | N/A | N/A |

## 15. Recommended Next Experiment

**Run additional independent seeds with the unchanged three-way design. All win and mission-success estimates are zero in seed 1, while checkpoint-level combat differences are large and directionally unstable; replication is therefore more informative than reward or code changes.**

### Training-process summary

- **Homogeneous**: 147 updates, final step 301056, episodes=893, mean SPS=149.1; entropy initial/max/final=2.708/2.708/2.164; last-30 return=-79.73±10.52, timeout=0.97, red damage=0.00.
- **Heterogeneous, no relay**: 147 updates, final step 301056, episodes=915, mean SPS=164.2; entropy initial/max/final=2.708/2.708/2.047; last-30 return=-73.63±12.08, timeout=0.79, red damage=0.00.
- **Heterogeneous, relay**: 147 updates, final step 301056, episodes=1027, mean SPS=171.9; entropy initial/max/final=2.708/2.708/2.256; last-30 return=-86.97±10.97, timeout=0.52, red damage=0.36.
