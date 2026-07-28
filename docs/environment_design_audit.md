# Environment design and provenance audit

Scope: fixed homogeneous 3v3 UAV close-combat project, especially the time-aware V2 path used by `configs/mappo_3v3_v2.yaml` and `configs/mappo_learnability_3v3.yaml`. This audit is read-only with respect to environment, reward, configuration, tests, and MAPPO source. It writes only this audit document and `docs/reward_implementation_audit.md`.

## 1. Source material availability

Primary paper requested by the user:

| Paper | Availability in project root | Audit status |
|---|---:|---|
| Chen, Luo, Guo, "A deep reinforcement learning cooperative air combat method with temporal feature and attention enhancement for heterogeneous flight vehicles", Aerospace Science and Technology, 2026 | Not found | insufficient_evidence |

Available PDFs actually read:

| Paper | Pages | Role in this audit |
|---|---:|---|
| Zheng, Wei, Duan, "UAV swarm air combat maneuver decision-making method based on multi-agent reinforcement learning and transferring", Science China Information Sciences, 2024 | 18 | Main available source for multi-UAV model, 15 actions, 3v2 swarm setting, state spaces, blue script, reward assignment, terminal reward |
| Zheng, Duan, "UAV maneuver decision-making via deep reinforcement learning for short-range air combat", Intelligence & Robotics, 2023 | 19 | Auxiliary source for 1v1 action library, actor/critic state split, dense/event/end-game reward definitions |

Because the requested Chen/Luo/Guo 2026 paper is not present, any claim that the current project implements that 2026 environment is not provable from current materials. The current repository is primarily traceable to Zheng/Duan 2023 and Zheng/Wei/Duan 2024, plus project-specific V2 engineering.

## 2. Paper fact table

### Zheng, Wei, Duan 2024 facts

| Topic | Paper location | Terms / formulas / tables | Fact |
|---|---|---|---|
| Scenario | p.4, Sec. 2.2; p.8, Sec. 3.2; p.14-16, Sec. 4.2 | red side `Omega_r`, blue side `Omega_b`; 3-on-2 training; 3-on-3 confrontation test | Red UAVs are MAPPO agents; blue UAVs use a maneuver strategy script. Main swarm training focuses on 3v2, not fixed homogeneous 3v3. |
| Homogeneity | p.7, Sec. 3.1 | "agents are isomorphic" | Red agents share actor/critic networks and have the same physical properties and role. |
| Motion model | p.3-4, Sec. 2.1, Eq. (1)-(2), Fig. 2 | 3-DoF overload model, `u=[nx,nz,gamma]^T` | Three-degree-of-freedom point-mass model in ENU coordinates; Runge-Kutta integration; no JSBSim or 6-DoF model. |
| Dynamics timing | p.13, Sec. 4.1 | `Tmax=200s`, `tstep=0.1s`, decision step `0.5s` | A decision action is held for five 0.1s physics steps; maximum 400 decision steps. |
| Attack geometry | p.5, Sec. 2.2, Eq. (3)-(4) | `Datt,min/max`, `phi_att`, `phi_esp` | Attack requires distance interval, attack angle threshold, and escape angle threshold. |
| Damage | p.5, Sec. 2.2, Eq. (5) | `B`, `Delta B`, `patt1/2/3`, `B1/2/3` | Health/blood decreases by probabilistic finite damage; destroyed when health <= 0; collision/ground damage flag. No explicit missile model. |
| Attack/advantage areas | p.5, Sec. 2.2, Eq. (6)-(7) | attack area, advantage area | Attack area is looser than true attack condition; advantage area is behind enemy tail. |
| Action space | p.6, Sec. 2.3, Table 1 | 15 basic maneuvers | Discrete 15-action library using `[nx,nz,gamma]`, including forward/up/down/left/right with maintain/accelerate/decelerate. |
| Local state | p.6, Sec. 2.4, Eq. (8)-(9) | `Sred`, `Sblue`, `Sblue,k` | For each red UAV: relative features to friendly red UAVs and blue UAVs; no own health/time field in Eq. (9). |
| Global state | p.6, Sec. 2.4, Eq. (10) | `Sg`, damage flags, pair blocks, red actions | Critic state includes red damage flags, red-blue pair features, and red action indices. |
| Blue rule | p.7, Sec. 2.5, Algorithm 1, Eq. (11) | target selection, prediction, decision; threat value | Blue chooses targets, predicts target action by evaluating all actions, then chooses own action by threat value. |
| MAPPO | p.7-12, Sec. 3.1-3.4, Fig. 3-4, Eq. (26)-(27) | CTDE, shared actor/critic | Red agents are learning agents; critic uses global state; actor uses local state; no self-play is described for the main setting. |
| Reward structure | p.8-11, Sec. 3.3, Eq. (12)-(25), Table 2, Algorithm 2 | situation, event, dense assignment, end-game | Reward has situation reward, event reward, assigned dense reward, and terminal/end-game reward. |
| Parameters | p.13, Sec. 4.1 | Table-free parameter paragraph | Speed 30-180 m/s; altitude ceiling implied by normalization/constraints; attack 40-900m; angles pi/6, pi/3, pi/4; health 300; reward weights and base rewards. |
| Initial scenarios | p.12 Table 3; p.13-14 Sec. 4.1-4.2 | C1/C2/C3/C4/C5/C6 | Paper uses random initialization ranges or initial point sets, staged training, and blue health curriculum. |
| Timeout result | p.5 Sec. 2.2; p.16 Eq. (28) | side with more survivors wins; tie if equal survivors | Timeout is an episode end; paper defines winner by survivor counts. It does not publish a fixed per-agent timeout reward of -4. |

### Zheng, Duan 2023 facts

| Topic | Paper location | Terms / formulas / tables | Fact |
|---|---|---|---|
| Scenario | p.4-6, Sec. 2.2 | one-to-one red/blue | Red is PPO learner; blue uses a prediction/decision rule; not a multi-agent swarm paper. |
| Motion model | p.3-4, Sec. 2.1, Eq. (1)-(2), Fig. 1 | 3-DoF model | Same simplified overload point-mass model and Runge-Kutta propagation. |
| Actor observation | p.10-11, Eq. (14)-(19) | 11D actor observation | Relative position and velocity features, normalized. |
| Critic state | p.11, Eq. (20)-(21) | 10D critic state | Includes health/blood information unavailable to actor. |
| Action space | p.11-12, Sec. 3.3, Table 1 | 15 basic actions | Same 15 `[nx,nz,gamma]` actions. |
| Dense reward | p.12-13, Eq. (22)-(29) | angle/distance/height/speed and `wdense` | 1v1 dense reward includes a `-1` step penalty scaled by `wdense=0.05`; this differs from the 2024 situation reward form. |
| Event reward | p.13-14, Eq. (30) | advantage and attack events | Event reward includes advantage area and attack success/penalties. |
| End-game reward | p.14, Eq. (31) | win/loss terminal | Draw is treated as loss in 2023 1v1. |

## 3. Current project environment fact table

| Module / config | Runtime role | Current behavior |
|---|---|---|
| `configs/base.yaml` + `configs/paper_2024_homogeneous.yaml` | Base physical/reward constants | 3-DoF model constants, attack/damage/reward constants, 400 decisions over 200s. |
| `configs/scenario_3v3_v2.yaml` | Formal V2 scenario | Fixed homogeneous 3v3, mirrored head-on jitter, 1800m team distance, 500m lateral spacing, time-aware schema. |
| `configs/scenario_3v3_learnability_v1.yaml` | Learnability scenario | Fixed homogeneous 3v3, closer 1200m distance, smaller jitter, straight opponent in MAPPO config. |
| `src/uav_env/actions/discrete_15.py` | Action table | Implements 0-based version of paper Table 1. |
| `src/uav_env/dynamics/point_mass_3d.py`, `propagation.py`, `rk4.py` | Dynamics | Implements 3-DoF overload derivative, RK4 integration, clipping/numerical protection. |
| `src/uav_env/combat/attack_geometry.py` | Attack/area geometry | Implements attack, attack area, and advantage area inequalities. |
| `src/uav_env/combat/damage.py`, `multi_combat.py` | Damage/attack resolution | Implements probabilistic damage, nearest target attack, simultaneous aggregation, effective/overkill accounting. |
| `src/uav_env/envs/combat_multi_env.py` | Main 2v2/3v3 env | Fixed equal-team 2v2/3v3, red learning agents, blue rule policy, V2 3v3 schema gate, reward assembly. |
| `src/uav_env/observations/multi_observation.py` | Actor observation | Legacy 2v2/3v3 paper-shaped relative observation; V2 63D fixed-ID body-frame time-aware observation. |
| `src/uav_env/observations/global_state.py` | Critic state | Legacy pairwise state; V2 61D full-entity time-aware state. |
| `src/uav_env/opponents/pursuit.py` | Blue rule used by formal V2 | Project geometric pursuit, explicitly not the paper's predictive threat script. |
| `src/uav_env/opponents/straight.py`, `random.py` | Alternative rules | Baseline project rules. |
| `src/uav_env/algorithms/mappo/adapter.py` | Training interface | Converts Gym output to per-agent MAPPO tensors; team reward checked as mean of per-agent rewards. |
| `src/uav_env/algorithms/mappo/runner.py` | MAPPO runner | Uses per-agent rewards for rollout buffer; logs team mean and reward component diagnostics; time-aware timeout does not bootstrap. |

## 4. Final merged key parameters

### Formal V2, `scenario_3v3_v2.yaml`

| Key | Effective value |
|---|---:|
| physics_dt / decision_dt / hold steps | 0.1 / 0.5 / 5 |
| max_episode_seconds / max_decision_steps | 200.0 / 400 |
| speed min/max | 30.0 / 180.0 m/s |
| altitude min/max | 0.0 / 5000.0 m |
| flight path angle min/max | -0.7853981633974483 / 0.7853981633974483 rad |
| tangential overload min/max | -1.0 / 2.5 |
| normal overload min/max | -4.0 / 4.0 |
| attack distance min/max | 40.0 / 900.0 m |
| attack angle / escape angle / attack-area angle | pi/6 / pi/3 / pi/4 |
| advantage distance min/max / escape angle | 40.0 / 1300.0 m / pi/3 |
| damage thresholds / values | [0.1, 0.4, 0.8] / [51, 21, 11, 0] |
| initial health | 300.0 |
| reward profile / terminal profile | `project_3v3_v2` / `paper_2024_exact` |
| timeout reward | -4.0 per red slot |
| r_den0 / r_win0 / r_lose0 | 0.01 / 50.0 / -50.0 |
| scenario / profile | `head_on_mirrored_jitter_v2` / same |
| initial team distance / spacing / altitude / speed | 1800.0 / 500.0 / 1800.0 / 110.0 |
| jitter | longitudinal 50, lateral 75, altitude 75, speed 5, heading 0.05235987756 rad |
| opponent in MAPPO V2 config | `pursuit` |
| observation/global schema | `fixed_id_body_time_63d` / `full_entity_time_61d` |
| environment schema | `homogeneous_3v3_v2_timeaware` |

### Learnability V1, `scenario_3v3_learnability_v1.yaml`

Same physical, combat, reward, and schema values as formal V2, except:

| Key | Effective value |
|---|---:|
| scenario / profile | `head_on_learnability_v1` / same |
| initial team distance / spacing | 1200.0 / 250.0 |
| jitter | longitudinal 12.5, lateral 18.75, altitude 18.75, speed 1.25, heading 0.01308996939 rad |
| opponent in MAPPO learnability config | `straight` |

## 5. Provenance matrix

| Item | Paper basis | Code location | Classification | Semantic impact | Severity | Recommendation |
|---|---|---|---|---:|---|---|
| 3-DoF overload dynamics | 2024 p.3-4 Eq. (1)-(2); 2023 p.3-4 Eq. (1)-(2) | `dynamics/point_mass_3d.py`, `propagation.py` | paper_equivalent | yes | MINOR | Keep numerical guards; document as RK4/clipped implementation, not JSBSim. |
| 15 discrete actions | 2024 p.6 Table 1; 2023 p.12 Table 1 | `actions/discrete_15.py` | paper_exact except 0-based IDs | yes | pass | Keep; state that paper numbers 1-15 correspond to code IDs 0-14. |
| Attack condition | 2024 p.5 Eq. (3)-(4) | `combat/attack_geometry.py` | paper_exact | yes | pass | Keep. |
| Damage probabilities | 2024 p.5 Eq. (5) | `combat/damage.py` | paper_equivalent | yes | MINOR | Keep; simultaneous damage aggregation is project-defined. |
| Explicit missile model | No available Zheng/Duan paper support; Chen 2026 unavailable | no explicit module | missing_from_project only if claiming Chen 2026 missile environment | yes | insufficient_evidence | Do not claim missile/PN/JSBSim reproduction without the 2026 PDF. |
| Fixed homogeneous 3v3 as formal training env | 2024 trains 3v2; tests 3v3 with restriction on observation objects | `configs/scenario_3v3_v2.yaml`, `combat_multi_env.py` | paper_inspired_project_design | yes | MAJOR | Describe as project 3v3 adaptation inspired by 2024, not paper-exact scenario. |
| V2 63D actor observation | 2024 Eq. (9) has pairwise relative blocks; no time/own health/last action in local S | `observations/multi_observation.py` | project_original | yes | MAJOR | Keep if needed for Markov/fixed-slot 3v3; document as V2 project schema. |
| V2 61D global state | 2024 Eq. (10) pairwise red-blue state and red action/damage; not full entity absolute state | `observations/global_state.py` | project_original | yes | MODERATE | Keep as CTDE engineering; avoid calling it paper Eq. (10). |
| Time-aware episode progress | Not in available 2024/2023 PDFs | V2 observations/global state, MAPPO runner | project_original | yes | MODERATE | Keep as finite-horizon Markov engineering; document as project assumption. |
| Blue `pursuit` opponent | Paper Algorithm 1 evaluates predicted target/own actions through threat value | `opponents/pursuit.py` | project_original, correctly labeled | yes | MODERATE | Keep for experiments but do not describe as paper blue script. |
| Nearest target assignment for 3v3 | Paper Algorithm 1 uses target selection with `Ifree`; current independent nearest allows reuse | `combat/multi_combat.py`, `team_controller.py`, `combat_multi_env.py` | paper_inspired_project_design | yes | MODERATE | Consider implementing paper Algorithm 1 separately or label current as project rule. |
| V2 combat events bypass Algorithm 2 | Paper dense reward is `rden=rs+re` before Algorithm 2 | `combat_multi_env.py` | implementation_mismatch relative to paper; project design | yes | MAJOR | If paper consistency is required, either route all event reward through Algorithm 2 or rename V2 reward as project reward. |
| Timeout reward `-4` | Paper says timeout winner by survivor count; no fixed -4 reward in available PDF | `multi_reward.py`, V2 config | project_original | yes | DOCUMENTATION_ONLY/MODERATE | Keep if intentional; never label as paper formula. |
| `paper_2024_exact` terminal profile | 2024 Eq. (21)-(25) supports elimination terminal formula; weights/guards/draw/timeout are project choices | `rewards/multi_reward.py` | paper_equivalent with project assumptions | yes | DOCUMENTATION_ONLY | Rename or document "exact formula structure for elimination only". |
| Old 1v1/2v2 legacy branches | Supported by project history and tests, not fixed 3v3 objective | multiple env/obs/MAPPO files | unnecessary_engineering candidate | no direct V2 semantic change | MINOR | Isolate legacy envs if formal 3v3 is the only target. |
| Old 62D/60D V2 schema | No paper basis; now rejected at runtime | config/env validation | unnecessary_engineering residual | low | MINOR | Keep rejection tests; remove stale mentions if any appear. |

## 6. Consistent items

- Three-degree-of-freedom overload dynamics, RK4 integration, units of meters/seconds/m/s/radians.
- 15-action discrete maneuver library.
- Attack, attack-area, advantage-area distance/angle inequalities.
- Health/blood probabilistic damage values and thresholds.
- 400 decision steps over 200 seconds with 0.5s decision period and 0.1s physics step.
- Red agents trained with shared MAPPO-style CTDE interface.

## 7. Project autonomous designs

- Fixed homogeneous 3v3 training environment instead of 2024 main 3v2 training scenario.
- V2 63D fixed-ID body-frame actor observation.
- V2 61D full-entity critic state.
- Episode progress/time-aware finite-horizon feature.
- Mirrored head-on jitter initialization and learnability V1 curriculum-like scenario.
- Geometric `PursuitOpponent` and `StraightOpponent`.
- Simultaneous attack aggregation with effective/overkill damage accounting.
- Timeout terminal reward `-4` and simultaneous-elimination terminal reward `0`.
- MAPPO checkpoint schema validation and time-aware truncation bootstrap mask.

## 8. Unsupported or misleading attribution

| Claim / wording pattern | Evidence status | Severity | Suggested wording |
|---|---|---|---|
| "Chen/Luo/Guo 2026 environment" | 2026 PDF unavailable | MAJOR | "Current materials do not prove 2026 reproduction." |
| `paper_2024_exact` for whole V2 reward path | Only elimination terminal formulas match available paper structure; V2 timeout and combat-event split are project-defined | DOCUMENTATION_ONLY | "paper_2024_terminal_elimination_structure" or document exact scope. |
| "published 2023" for 2024 situation reward | 2024 Eq. (12)-(19) largely repeats 2023 components, but multi-agent `max_j` is 2024 | DOCUMENTATION_ONLY | Attribute pair components to 2023/2024, and multi-agent aggregation to 2024. |
| "Algorithm 2 semantics" after negative cap | Cap is a project assumption | DOCUMENTATION_ONLY | State "Algorithm 2 plus negative cap assumption". |

## 9. Potential unnecessary engineering

| Candidate | Judgment | Rationale | Recommendation |
|---|---|---|---|
| Legacy 1v1/2v2 compatibility inside shared adapter/env paths | temporarily retain | Useful tests/history, but clutters formal 3v3 path | Isolate under legacy modules after formal experiments. |
| Multiple terminal profiles (`project_balanced`, `paper_2024_exact`, V2 timeout labels) | suggest rename/isolate | Useful for ablation, but names can imply wrong provenance | Keep one formal profile; move ablation profiles to explicit namespace. |
| Legacy observation builders plus V2 builders in same files | temporarily retain | Tests cover both; mixed logic increases cognitive load | Keep now; later split `legacy_multi_observation.py` and `v2_observation.py`. |
| Multiple target assignment methods | suggest isolate | 2v2 distinct assignment and 3v3 independent nearest differ in semantics | Name them by semantics and bind in config explicitly. |
| Reward breakdown duplicate fields (`assigned_dense`, `assigned_shape`, `dense_reward`) | suggest rename | Helpful diagnostics but easy to confuse | Keep through training; document exact definitions in metrics dictionary. |
| Old 62D/60D schema gate | necessary | Prevents silent use of stale schema | Keep rejection, remove stale configs if any remain. |
| `scenario_name` and `scenario_profile` | suggest rename | Often identical; can diverge silently | Keep but document one as runtime selected reset layout, one as metadata profile. |

## 10. Test classification

| Test family | Classification | Notes |
|---|---|---|
| `test_action_table.py`, `test_rk4.py`, `test_dynamics.py`, `test_geometry.py` | mathematical boundary / paper formula support | Strong independent oracles. |
| `test_paper_distance_reward.py`, `test_paper_height_reward.py`, `test_terminal_reward_2024_exact.py` | paper formula regression | Mostly hand-computed formula checks; good coverage for available Zheng/Duan formulas. |
| `test_dense_reward_assignment.py`, `test_damage_semantics.py`, `test_multi_damage_resolution.py` | project semantics / regression | Validates current assumptions including simultaneous damage and negative cap. |
| `test_3v3_v2_environment.py`, `test_global_state.py`, `test_multi_observation.py` | project V2 semantics | Confirms schema shapes and time-aware behavior; not paper Eq. (9)/(10) exactness. |
| `test_mappo_*`, `test_vector_env.py`, `test_checkpoint_roundtrip.py` | training interface / regression | Useful engineering checks; not environment-paper oracle tests. |
| `test_reward_ordering_diagnostics.py` | diagnostic/regression | Should not be used to infer paper correctness. |

## 11. Overall environment conclusion

Current environment can accurately be described as:

> A project-specific fixed homogeneous 3v3 close-range UAV air-combat environment inspired by Zheng/Duan 2023 and Zheng/Wei/Duan 2024, using their 3-DoF dynamics, 15-action library, attack/damage geometry, and parts of their reward structure, with project-defined V2 observation/state, timeout semantics, opponent policy, and reward decomposition for MAPPO experiments.

It should not be described as:

- a verified reproduction of Chen/Luo/Guo 2026;
- a JSBSim or 6-DoF environment;
- a missile/proportional-navigation environment;
- a paper-exact 2024 3v2 swarm environment;
- a paper-exact 3v3 reward implementation in every branch;
- an implementation of the 2024 predictive blue Algorithm 1 when using `PursuitOpponent`.

## 12. Final categorical ratings

| Category | Rating |
|---|---|
| Paper reproduction correctness | pass_with_documentation for Zheng/Duan components; insufficient_evidence for Chen/Luo/Guo 2026 |
| Project autonomous design rationality | pass_with_documentation |
| Engineering implementation necessity | pass_with_minor_fixes |
| Reward mathematical correctness | pass_with_minor_fixes |
| Reward paper consistency | requires_major_fix if claiming paper-exact V2; pass_with_documentation if labeled project V2 |
| Training interface correctness | pass |
| Formal experiment usability | pass_with_documentation before 300k; requires clearer provenance labels before publication |
