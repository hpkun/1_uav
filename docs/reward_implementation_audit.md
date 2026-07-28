# Reward implementation and provenance audit

Scope: reward formulas, event timing, dense reward assignment, terminal reward, timeout semantics, and MAPPO training signal for the current fixed homogeneous 3v3 time-aware V2 project.

## 1. Available paper reward formulas

### Zheng, Wei, Duan 2024

| Formula | Location | Expression / semantics | Variables and trigger |
|---|---|---|---|
| Situation reward | p.9 Sec. 3.3.1 Eq. (12) | `rs,ij = w_phi*r_phi + w_d*r_d + w_v*r_v + w_h*r_h` | Computed after all UAVs perform actions at each decision step; weights sum to 1. |
| Angle reward | p.9 Eq. (13) | `r_phi=((pi-phi_att)/pi)*((pi-phi_esp)/pi)` | Larger when attacker points toward target and target is less able to escape. |
| Distance reward | p.9 Eq. (14)-(16) | `r_d=r_d1+r_d2`; approach term plus piecewise distance term | Uses previous-to-current distance change and current distance. |
| Height reward | p.9 Eq. (17) | Piecewise over `z_i-z_j` using `Hmax,Hadv,Hatt,Hmin` | Rewards favorable height advantage. |
| Speed reward | p.9 Eq. (18) | Piecewise over speed ratio `v_i/v_j` | Highest for 1.0-1.5 speed ratio. |
| Per-agent situation aggregation | p.9 Eq. (19) | `rs,i=max_j rs,ij` over surviving blue UAVs | One selected best enemy relation contributes situation reward. |
| Event rewards | p.10 Table 2, Eq. (20) | advantage, attack area, hit/destroy/attacked/destroyed/boundary/collision events | Accumulated when events trigger; advantage reward is distance/angle formula. |
| Dense reward | p.10 Sec. 3.3.3 | `rden,i = rs,i + re,i` | Situation and all event rewards are combined before assignment. |
| Dense assignment | p.11 Algorithm 2 | Damaged: `r'_den,i=-rden0*nr-min(Phi_r,den)`; positive branch scales by `rden0`, `0.003`, `0.007`; near-zero branch to 0 | Applies to red UAV dense rewards and damage flags. |
| Win terminal base | p.10-11 Eq. (21) | `rwin,all=rwin0*nr*(0.75+0.25*(Nstep-nstep)/Nstep)` | Triggered when red wins. |
| Win terminal allocation | p.11 Eq. (22) | Base share + `0.03*alive_count` + contribution + health terms | Uses event contribution score `beta_i` and surviving health. |
| Loss terminal base | p.11 Eq. (23) | `rlose,all=rlose0*nr*(0.80+0.20*(Nstep-nstep)/Nstep)` | Triggered when red loses. |
| Loss reshaping/allocation | p.11 Eq. (24)-(25) | Reverse contribution and reverse health terms | Uses negative base reward. |
| Timeout | p.5 Sec. 2.2; p.16 Eq. (28) | Ends at `Tmax`; winner by survivor count, tie by equal survivor count | No fixed per-agent timeout reward published in available PDF. |

Known PDF ambiguity: in the extracted 2024 Eq. (16) text, `a2` appears inconsistent with the 2023 formula. The implementation follows the 2023 piecewise distance coefficient and tests it independently.

### Zheng, Duan 2023

The 2023 paper defines 1v1 reward as dense/event/end-game. Its dense reward Eq. (22) includes `(weighted_terms - 1) * wdense`, unlike 2024 Eq. (12)-(19), which defines a nonnegative situation reward and then adds event rewards for Algorithm 2. Therefore, using 2023 component names for 2024 multi-agent reward should be documented carefully.

## 2. Current code reward data flow

Runtime path: `src/uav_env/envs/combat_multi_env.py::CombatMultiEnv.step`.

```text
previous state copy
→ red action parse and dead-agent hold action
→ blue target assignment
→ blue rule action selection
→ last_action written to all aircraft
→ previous_states captured
→ physics propagation over physics_steps_per_action
→ ground/ceiling boundary flags
→ pairwise collision resolution
→ simultaneous attack resolution and health update
→ red_damaged_this_step computed
→ decision_step/simulation_time incremented
→ timeout / elimination outcome computed
→ per-red situation reward
→ geometry event reward
→ combat event reward
→ V2 raw_shape = situation + geometry_event
→ assign_dense_rewards(raw_shape, damaged, fixed red_count)
→ V2 dense_reward = assigned_shape + combat_event
→ terminal allocation
→ per-agent total = dense_reward + terminal
→ team_reward = mean(per-agent totals)
→ MAPPO adapter passes per-agent reward vector to rollout buffer
```

Important timing facts:

- `previous_states` is captured after action IDs are written but before physics propagation; the kinematic state is still pre-action. This is suitable for distance-change reward.
- Attack resolution occurs before reward assembly, so hit/destroy/attacked/destroyed events are visible in the same step.
- In V2, an aircraft destroyed during the step gets zero situation after transition, but keeps current-step combat/boundary events and goes through the damaged dense branch once.
- Terminal reward appears only when `outcome.termination_reason != "ongoing"`.

## 3. Formula-by-formula implementation audit

| Reward part | Code location | Current implementation | Provenance classification | Correctness / concern | Severity |
|---|---|---|---|---|---|
| Angle reward | `rewards/components.py::angle_reward` | Clips both angles to [0,pi], multiplies `(pi-angle)/pi` terms | paper_exact | Formula direction and radians are correct. | pass |
| Distance approach | `paper_distance_approach_reward` | Rewards 0.25 when previous distance > current distance and current distance > midpoint | paper_exact | Uses true previous/current distance from pre/post step. | pass |
| Piecewise distance | `paper_piecewise_distance_reward` | 3 intervals and zero otherwise | paper_equivalent | Follows 2023 Eq. (25)-(26); 2024 extracted text has an apparent `a2` ambiguity. | DOCUMENTATION_ONLY |
| Height reward | `paper_height_reward` | Uses `z_red-z_blue` and `attack_distance_max` as upper branch bound | paper_exact to available text | Matches 2024/2023 extracted formula, including unusual `Hmax < dh <= Datt,max` branch. | pass |
| Speed reward | `speed_reward` | Piecewise over self/enemy speed ratio | paper_exact | Correct sign and thresholds. | pass |
| Pair situation | `multi_reward.py::pair_situation_reward` | 0.15/0.60/0.10/0.15 weighted sum | paper_exact | Matches 2024 Eq. (12) and p.13 weights. | pass |
| Multi-agent situation | `individual_situation_reward` | max over living blue UAVs | paper_exact | Matches 2024 Eq. (19). | pass |
| Advantage event | `combat_multi_env.py::_geometry_event_reward` + `advantage_reward` | Adds radv for each living blue in advantage area | paper_equivalent | Paper says event rewards accumulate after traversing events; per-enemy accumulation is plausible but scales with enemy count. | MODERATE |
| Attack-area event | `_geometry_event_reward` | +0.3 / -0.3 per enemy relation | paper_equivalent | Coefficients match Table 2; per-enemy accumulation changes scale in 3v3. | MODERATE |
| Hit/destroy/attacked/destroyed events | `_combat_event_reward` | +0.8, +1.5, -0.9, -1.6; destroyed penalty once per target step | paper_equivalent/project_engineered | Coefficients match Table 2; simultaneous-hit credit is project-defined and order-independent. | MODERATE |
| Boundary/collision event | `_combat_event_reward` | -0.5 for boundary or collision | paper_exact coefficient, project collision resolution | Coefficient matches Table 2; simultaneous collision mechanics are project-defined. | MINOR |
| Dense reward composition in legacy | `step`, non-V2 branch | `raw_dense=situation+all event` | paper_exact-ish | Matches 2024 `rden=rs+re` better than V2. | pass_with_documentation |
| Dense reward composition in V2 | `step`, V2 branch | `raw_shape=situation+geometry_event`; combat events bypass Algorithm 2 | project_original / implementation_mismatch if claimed paper | This is intentionally not paper Algorithm 2 because Table 2 combat events are no longer assigned. | MAJOR |
| Dense assignment positive branch | `assign_dense_rewards` | `factor=(rden0*nr + .003*|Iu|/nr + .007*alpha/nr)*value/alpha` | paper_exact | Formula matches Algorithm 2. | pass |
| Dense assignment near-zero branch | `assign_dense_rewards` | `(-0.01,0.01]` to 0; `<=-0.01` keeps original negative value | paper_equivalent/project assumption | Algorithm 2 omits explicit assignment for `rden<=-0.01`; keeping original is reasonable but not explicit. | DOCUMENTATION_ONLY |
| Dense assignment damaged branch | `assign_dense_rewards` | `min(-rden0*nr-minimum, -rden0*nr)` | project assumption modifying paper formula | Prevents positive damaged reward; not verbatim Algorithm 2. | DOCUMENTATION_ONLY/MODERATE |
| Terminal win/loss | `multi_terminal_reward_allocations` profile `paper_2024_exact` | Eq. (21)-(25)-style allocation | paper_equivalent | Formula structure matches available 2024 PDF; weights/edge guards are project values. | pass_with_documentation |
| Terminal draw | same | configured `draw_reward` | project_original | 2024 defines draw result, but not a clear draw reward formula in extracted text. | DOCUMENTATION_ONLY |
| V2 timeout terminal | same | every red slot gets `timeout_reward=-4`, profile `project_3v3_v2_timeout` | project_original | Not a paper formula. Correctly separated by profile string. | MODERATE if not documented |
| Simultaneous elimination | same | every red slot terminal 0 | project_original | Not clearly specified in available paper. | DOCUMENTATION_ONLY |

## 4. State timing and event lifecycle

| Question | Finding | Severity |
|---|---|---|
| Does distance reward use previous and current state correctly? | Yes. `previous_states` is captured before propagation and passed to `paper_distance_reward`. | pass |
| Are rewards computed before or after attack? | After attack resolution. Combat outcomes are available in same-step reward. | pass |
| Does a destroyed current-step UAV receive duplicated situation reward? | No. If it is not alive after propagation/attack, situation is zero; it keeps current event and damaged dense branch. | pass_with_documentation |
| Are post-destruction rewards zero? | Existing implementation and tests enforce zero later non-terminal rewards. | pass |
| Is terminal only terminal-step? | Yes, `multi_terminal_reward_allocations` returns zero for `ongoing`. | pass |

## 5. Angle definitions

`compute_combat_geometry(attacker,target)` uses displacement `target.position - attacker.position`.

- Attack angle: angle between attacker velocity and attacker-to-target LOS. This matches 2024 Eq. (4).
- Escape angle: angle between target velocity and the same attacker-to-target LOS. This matches the available extracted 2024 Eq. (4) text.
- All angles are in radians; `angle_between` clips arccos inputs.

No angle sign error was found from available Zheng/Duan sources.

## 6. Dense reward assignment

`assign_dense_rewards()` is close to Algorithm 2 but not fully verbatim:

```text
paper damaged value = -r_den0 * nr - min(Phi_r,den)
project final damaged value = min(paper damaged value, -r_den0 * nr)
```

This cap is mathematically intentional because the literal damaged branch can become positive when `minimum < -r_den0*nr`. It preserves negative damaged reward semantics, but it changes Algorithm 2. Therefore:

- It is not faithful verbatim Algorithm 2.
- It is a documented project assumption.
- It is preferable to an unbounded positive damaged reward for project scientific semantics.

The use of fixed `nr=3` in V2 is consistent with Algorithm 2's team-size parameter and with the user's latest fixed-3v3 requirement. It is not the same as surviving-agent count.

## 7. Terminal reward audit

### `paper_2024_exact`

Accurate scope:

- Good label for available-paper elimination terminal formula structure, Eq. (21)-(25).
- Not exact for numerical weights, because the available paper says weights sum to one but the current values are project config.
- Not exact for draw, zero-beta, zero-health guards, simultaneous elimination, or V2 timeout.

Recommended wording: "2024 terminal formula structure for elimination outcomes" rather than "paper exact" when writing results.

### `project_balanced`

Project-defined ablation profile. It should never be presented as a paper formula.

### V2 timeout

`project_3v3_v2_timeout` assigns a fixed per-red reward of `-4.0`. This is a project design to avoid timeout being routed through the elimination terminal formula. Available 2024 paper defines timeout outcome by survivor count but does not provide this fixed timeout reward.

## 8. Team reward and MAPPO training signal

| Item | Finding |
|---|---|
| Environment return scalar | `team_reward = mean(agent_rewards)` |
| Adapter check | `MAPPOEnvAdapter._pack` verifies scalar team reward equals per-agent mean |
| Rollout buffer reward | `result["rewards"]`, the per-agent reward matrix, is inserted into MAPPO buffer |
| Actor/advantage signal | Per-agent rewards drive returns/advantages |
| Critic | Centralized critic returns per-agent values with shared global-state input |
| Logging | `team_reward`, `agent_reward_sum`, and per-agent episode returns are diagnostics |

Conclusion: the scalar `team_reward` is mainly a Gym/logging convenience, not the sole MAPPO training signal. Training interface is correct for shared-parameter MAPPO as currently implemented.

## 9. Reward scale observations

- Situation reward is bounded around [0,1] per selected target.
- Geometry events can accumulate over up to three enemies in 3v3, so attack/advantage area shaping can scale larger than the one-target situation reward.
- Combat events can accumulate multiple hits per step; destroyed penalty is de-duplicated per target.
- Algorithm 2 positive assigned dense scale is small, around `r_den0*nr` plus small terms, while direct V2 combat events are much larger (`0.8`, `1.5`, `-0.9`, `-1.6`).
- Terminal elimination base can be large because `r_win0/r_lose0=±50` and is multiplied by `nr=3` plus allocation factors.
- Timeout reward `-4` is small relative to elimination terminal but large relative to one-step dense assignment.

This scale separation is project-defined. It may be reasonable for learning, but it is not a paper-exact reward scale.

## 10. Correct, suspicious, and erroneous items

### Correct implementation items

- 3-DoF/RK4 reward-related state timing is coherent.
- Angle, distance, height, and speed formulas match available Zheng/Duan equations closely.
- Attack, advantage, damage, and Table 2 coefficients are implemented with correct signs.
- Terminal elimination formulas are structurally aligned with 2024 Eq. (21)-(25).
- MAPPO receives per-agent rewards, not only team mean.
- Time-aware timeout no-bootstrap is correct for finite-horizon V2 training semantics.

### Suspicious or documentation-sensitive items

- `paper_2024_exact` name is broader than its true scope.
- V2 geometry events are accumulated per enemy; paper text permits accumulated events but does not separately analyze 3v3 scale amplification.
- 2024 distance coefficient extraction contains an ambiguity; tests follow 2023 formula.
- `scenario_name` and `scenario_profile` duplicate each other in current V2 configs.
- Reward breakdown fields are numerous and partially overlapping.

### Clear mismatches if claiming paper-exact V2

- Fixed homogeneous 3v3 formal training is not the 2024 paper's main 3v2 swarm training environment.
- V2 63D actor observation and 61D global state are not Eq. (9)/(10) reproductions.
- `PursuitOpponent` is not the paper Algorithm 1 predictive threat script.
- V2 combat events bypass Algorithm 2, while 2024 defines `rden=rs+re` before assignment.
- Timeout `-4` is not from the available paper.

## 11. Issue list and priority

| Severity | Count | Issues |
|---|---:|---|
| BLOCKER | 0 | No code-level reward/math blocker found under the stated project-V2 semantics. |
| MAJOR | 4 | Missing Chen/Luo/Guo 2026 evidence; 3v3 formal scenario vs 2024 3v2; V2 observation/state not paper Eq. (9)/(10); V2 combat events bypass Algorithm 2 if called paper-exact. |
| MODERATE | 7 | `PursuitOpponent` not Algorithm 1; nearest/reuse target assignment differs; per-enemy geometry scale amplification; simultaneous damage/credit project semantics; timeout reward project semantics; reward scale separation; time feature is project design. |
| MINOR | 4 | Legacy 1v1/2v2 clutter; duplicate scenario naming; many overlapping diagnostics; old schema rejection residue. |
| DOCUMENTATION_ONLY | 6 | `paper_2024_exact` scope; Algorithm 2 negative cap; 2023 vs 2024 component attribution; draw reward; coefficient ambiguity; project weights/edge guards. |

## 12. Recommended fix priority, no implementation in this audit

1. Add the missing Chen/Luo/Guo 2026 PDF if that paper is truly the target; otherwise rename the project claim away from the 2026 paper.
2. Decide whether formal experiments should be "Zheng/Wei/Duan-inspired fixed 3v3" or a strict 2024 3v2 reproduction. This is a scientific-design decision, not a training tweak.
3. Rename or document `paper_2024_exact` as elimination-terminal formula scope only.
4. Explicitly label V2 reward profile as project-defined because combat events bypass Algorithm 2 and timeout is project-defined.
5. If paper fidelity is required, implement a separate paper Algorithm 1 blue rule and paper 3v2 observation/state path rather than changing V2 silently.
6. After documentation cleanup, isolate legacy 1v1/2v2 code paths from the formal 3v3 MAPPO path.

## 13. Final categorical ratings

| Category | Rating |
|---|---|
| Paper reproduction correctness | insufficient_evidence for Chen/Luo/Guo 2026; pass_with_documentation for selected Zheng/Duan formulas |
| Project autonomous design rationality | pass_with_documentation |
| Engineering implementation necessity | pass_with_minor_fixes |
| Reward mathematical correctness | pass_with_minor_fixes |
| Reward paper consistency | requires_major_fix if described as paper-exact V2 |
| Training interface correctness | pass |
| Formal experiment usability | pass_with_documentation |
