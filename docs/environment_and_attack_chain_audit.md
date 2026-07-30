# Environment and attack-chain audit

Date: 2026-07-30

Scope: current fixed homogeneous 3v3 UAV air-combat project, with emphasis on the time-aware V2 environment, `greedy_combat` blue opponent, attack reachability, red/blue semantic consistency, MAPPO/HAPPO training interfaces, metrics, and termination semantics.

This audit intentionally does not modify production environment, reward, observation, action, dynamics, attack geometry, damage, timeout, MAPPO, HAPPO, `greedy_combat`, `pursuit`, or scenario parameters. It adds only audit tests and this report.

## 1. Executive summary

The implemented 3v3 attack chain is complete and executable: valid red actions and blue rule actions are written to `last_action`, propagated through the same point-mass dynamics for all aircraft, checked for boundary/collision failure, then resolved by one simultaneous multi-aircraft attack pass. Red and blue use the same `compute_combat_geometry()`, `DamageConfig`, RNG mechanism, health clamp, alive/damaged state update, and attack statistics path.

The strongest finding is not a broken hit/damage chain. New deterministic tests show that the geometry, damage, simultaneous multi-attacker resolution, reward event attribution, reset state, vector worker seeding, and short reachable attack micro-scenarios all work. The reported training symptom—red almost never entering `can_attack` against formal `greedy_combat`—is therefore more likely caused by task difficulty and asymmetry in policy information/control than by a basic attack-chain implementation failure.

Top likely causes of low red attack attempts in formal MAPPO vs `greedy_combat`:

1. `greedy_combat` has an explicit one-step geometry oracle and directly scores `can_attack`; the red shared actor must infer the same geometry from clipped normalized observations.
2. Formal `head_on_mirrored_jitter_v2` starts as a head-on engagement, while current `can_attack` semantics are tail-chase-like because both attacker angle and target escape angle must be small.
3. Shared MAPPO actor has no explicit own-agent ID feature; fixed slots plus near-symmetric initial geometry can encourage identical maneuvers, including later collective ceiling-violation behavior.

No BLOCKER implementation defect was found in the attack, hit, damage, death, terminal, or vector interface chain.

## 2. Complete environment step timing

Function-level sequence in `src/uav_env/envs/combat_multi_env.py::CombatMultiEnv.step()`:

```text
red action input
→ shape/action-space validation
→ dead red slots forced to LEVEL_HOLD
→ blue target assignment
→ blue rule action selection
→ action_map for every red/blue aircraft
→ write last_action for every aircraft before propagation
→ snapshot previous_states
→ propagate all living aircraft for physics_steps_per_action substeps
→ per-substep ground/ceiling checks
→ collision check after propagation
→ resolve_multi_attacks(all_aircraft)
→ apply updated health/alive/damaged states
→ increment decision_step and simulation_time
→ update statistics
→ compute attack_area_steps
→ terminal/truncated classification
→ red-only situation/event/dense/terminal rewards
→ team_reward = mean(red agent rewards)
→ events and info logging
```

Timing conclusions:

- Red and blue actions are chosen before propagation and applied over the same decision period.
- `last_action` is written before `previous_states` is snapshotted. Thus `previous_states` is pre-propagation kinematics but already contains the current action ID. This is relevant for observation/logging and for the next decision's `greedy_combat` target prediction.
- Attack geometry is evaluated after the whole decision period, not before movement and not at each physics substep.
- Boundary death is checked during physics substeps; collision is checked after propagation; combat attacks are resolved after boundary/collision.
- Aircraft killed by boundary/collision before combat are not living and therefore do not attack in `resolve_multi_attacks()`.
- Combat resolution is simultaneous over the living set at combat time: attempts are collected before any target health is updated, then target damage is aggregated.

## 3. Attack-chain call graph

Red side:

```text
MAPPO/HAPPO actor or test action
→ CombatMultiEnv.step(action)
→ action_space.contains()
→ red_actions list
→ action_map
→ state.last_action = red action
→ _propagate_all()
→ get_control(action)
→ propagate_state()
→ point_mass_3d_derivative()
→ rk4_step()
→ min/max speed and flight-path clipping
→ ground/ceiling death check
→ _resolve_collisions()
→ resolve_multi_attacks()
→ compute_combat_geometry(red, living blue)
→ can_attack
→ rng.random()
→ damage_for_random_value()
→ aggregate effective damage by target
→ health/alive/damaged update
→ _combat_event_reward() for red rewards
→ terminal allocation
→ info/statistics/metrics
```

Blue side:

```text
CombatMultiEnv.step()
→ assign_targets() when opponent_name == greedy_combat
   or assign_nearest_targets_independently() for 3v3 pursuit
→ _blue_actions(assignments)
→ GreedyCombatOpponent.select_action()
→ target last_action prediction
→ enumerate all 15 DiscreteAction15 candidates
→ get_control()
→ propagate_state() for full physics_steps_per_action
→ compute_combat_geometry(predicted_blue, predicted_red)
→ compute_combat_geometry(predicted_red, predicted_blue)
→ offense/defense/safety score
→ deterministic tie-break
→ action_map and shared propagation/attack chain above
```

Important asymmetry: blue `greedy_combat` directly calls `compute_combat_geometry()` and sees exact state objects for its assigned target. The red neural actor receives normalized observation features and must learn the equivalent condition.

## 4. Attack geometry formulas and semantics

Code: `src/uav_env/combat/attack_geometry.py::compute_combat_geometry()`.

Current formula:

```text
displacement = target.position - attacker.position
distance = ||displacement||_2
line_of_sight = displacement / distance
attacker_attack_angle = angle(attacker.velocity, displacement)
target_escape_angle = angle(target.velocity, displacement)
in_attack_area = attack_distance_min <= distance <= attack_distance_max
                 and attacker_attack_angle <= attack_area_angle_max
in_advantage_area = advantage_distance_min <= distance <= advantage_distance_max
                    and target_escape_angle <= advantage_escape_angle_max
can_attack = attack_distance_min <= distance <= attack_distance_max
             and attacker_attack_angle <= attack_angle_max
             and target_escape_angle <= escape_angle_max
```

Audit conclusions:

- Relative vector direction is attacker-to-target.
- Distance is full 3D Euclidean distance.
- Distance and angle thresholds are inclusive.
- Angles are radians.
- Zero-distance handling is numerically stable: line of sight becomes zero and protected vector-angle logic returns finite values.
- `in_attack_area` and `can_attack` are distinct: `in_attack_area` does not require the target escape-angle condition.
- Dead-target filtering happens in callers such as `resolve_multi_attacks()` and `_geometry_event_reward()`, not inside `compute_combat_geometry()` itself. Calling geometry on a dead state can still return geometric booleans; callers must filter living entities.

Direct tests:

- `tests/test_combat_geometry_oracles.py`
  - forward tail-chase can attack;
  - behind/too far/too close cannot attack;
  - upper/lower distance and angle thresholds are inclusive;
  - red/blue mirror preserves geometry;
  - attacker/target swap follows the directional definition;
  - vertical offset uses 3D distance and finite line-of-sight.

## 5. Red/blue symmetry

Attack rules are symmetric at the weapon-resolution level:

- `resolve_multi_attacks()` takes all living aircraft from both teams.
- For each living attacker, it scans all living targets from the opposite team.
- It calls the same `compute_combat_geometry()` and `damage_for_random_value()`.
- It aggregates target damage identically for red and blue.
- It writes updated states for both sides in one returned map.

Reward semantics are not symmetric:

- The environment is red-learning oriented.
- `agent_reward_breakdowns` and returned per-agent rewards are for red agents only.
- Blue events are logged in `statistics`, `attack_attempts`, `resolved_attacks`, and `events`, but blue does not receive training rewards in this environment path.

This is a design choice, not an attack-chain bug.

## 6. Attack resolution order

Code: `src/uav_env/combat/multi_combat.py::resolve_multi_attacks()`.

Resolution semantics:

1. Build `ordered = sorted(living aircraft by uav_id)`.
2. Optionally apply `sample_team_order`; current `CombatMultiEnv` leaves `damage_sample_team_order = None`.
3. For each living attacker, gather all living opposite-team targets with `geometry.can_attack`.
4. Select nearest attackable target, tie-broken by target UAV ID.
5. Draw one independent RNG sample per attack attempt.
6. Convert sample to nominal damage.
7. Group attempts by target.
8. For each target, compute:

```text
total_nominal = sum(nominal_damage)
total_effective = min(target_health_before, total_nominal)
per_attacker_effective = total_effective * nominal_damage_i / total_nominal
overkill = nominal_damage_i - effective_i
```

9. If the target is destroyed, assign one destroy credit to the attacker with largest effective allocation, then lower distance, then lower attacker ID.
10. Update health to `max(0, health - total_effective)`, `alive=False` only if destroyed.

Conclusions:

- Multiple simultaneous hits on one target are accumulated.
- Effective damage is clamped to remaining health.
- Overkill is logged but not counted as effective damage.
- Destroy credit is target-level and at most one per target per resolution pass.
- A target already dead before resolution is not in the living target set.
- There is no side-order advantage in health update because attempts are collected before any health mutation. RNG consumption order is deterministic by sorted living aircraft, so the random sample assigned to each attacker can depend on living set and optional sample order, but not on health mutation order.

Direct tests:

- `tests/test_combat_resolution_symmetry.py`
  - p=0 hit and p≈1 miss boundary;
  - both sides can attack in the same decision step;
  - three attackers hitting one target share effective damage and receive one destroy credit;
  - exact kill and overkill clamp correctly;
  - dead aircraft cannot attack or be selected as target;
  - mirrored red/blue geometry remains consistent.

## 7. RNG audit

Damage RNG:

- Environment-level `self.rng` is created in `CombatMultiEnv.__init__`.
- `reset(seed)` resets `self.rng = np.random.default_rng(seed)`.
- `resolve_multi_attacks()` consumes one `rng.random()` per attack attempt.
- One decision step with multiple attacks uses independent consecutive samples, not reused samples.

Blue rule RNG:

- `self.blue_rule_rng = np.random.default_rng(seed + 2_000_003)` at reset.
- `greedy_combat` ignores RNG; `random` uses it.

Vector env RNG:

- `SyncCombatVectorEnv.reset()` calls env reset with `base_seed + index`.
- `ParallelCombatVectorEnv` constructs worker envs with `base_seed + index` and also resets with `base_seed + index`.
- Worker exception propagation includes worker index and seed in the sync path, and process-worker errors are returned with traceback in the parallel path.

Evaluation:

- MAPPO and HAPPO evaluation use deterministic actor action selection when requested.
- Validation episodes use `seed_start + episode`; repeated validations intentionally reuse the same seed set for curve comparability.
- Test evaluation uses a separate configured seed range.

Direct tests:

- `tests/test_attack_chain_end_to_end.py::test_parallel_worker_seeds_are_distinct_and_finite`
- Existing checkpoint tests verify RNG state persistence for MAPPO/HAPPO.

## 8. Damage and death state audit

State invariant:

- `UAVState.__post_init__()` enforces `health <= 0`, `crashed`, or `alive=False` implies `alive=False` and `damaged=True`.
- `damaged` currently means combat failure/dead, not “has ever been hit”.
- “Was ever hit” is tracked separately as `ever_hit`.

Boundary and collision:

- Ground death: `_propagate_all()` sets `health=0`, `alive=False`, `damaged=True`, `crashed=True`, boundary reason `"ground"`.
- Ceiling death: sets `health=0`, `alive=False`, `damaged=True`, boundary reason `"ceiling"`, but `crashed` remains false.
- Collision death: `_resolve_collisions()` sets `health=0`, `alive=False`, `damaged=True`.
- Boundary/collision deaths do not create combat `destroy_credit`; they enter red rewards through `boundary_collision_penalty` when the affected aircraft is red.

Survivors:

- `_outcome()` counts survivors via `sum(u.is_alive)`.
- `red_alive` and `blue_alive` are side-level booleans.

Reset:

- `reset()` rebuilds all aircraft from scenario states, with full health, alive, not damaged, and `last_action=LEVEL_HOLD`.

## 9. Reward-chain audit

Red reward path in `CombatMultiEnv.step()`:

```text
previous_states
→ individual_situation_reward()
→ _geometry_event_reward()
→ _combat_event_reward()
→ raw_dense = situation + geometry_event
→ assign_dense_rewards()
→ dense_reward = assigned_shape + combat_event
→ terminal allocation
→ total per red agent
→ team_reward = mean(red totals)
```

Per-red event rewards:

- Red hit: `+0.8`.
- Red destroy credit: `+1.5`.
- Red is hit: `-0.9` per hit.
- Red destroyed by combat: `-1.6` once per target-level destroy credit.
- Red boundary/collision: `-0.5`.

Blue rewards:

- Blue attack and damage events are logged, but blue has no per-agent reward returned to MAPPO/HAPPO.

Typical magnitudes:

- A red hit event: `+0.8`.
- Red destroy event: additional `+1.5`.
- Red receives two hits and one destroy credit against it: `-0.9 * 2 - 1.6 = -3.4`.
- Terminal timeout in V2: `-4.0` per red agent via `project_3v3_v2_timeout`.
- Win/loss terminal under `paper_2024_exact`: base uses `r_win0=50`, `r_lose0=-50`, team size 3, and time/alive/contribution/health factors. It can dominate dense events.

Potential local optimum:

- A timeout with equal survivors still gives `-4` per red agent, but it can be better than early elimination with large negative terminal allocation and combat penalties.
- Therefore a survival/timeout policy can be locally attractive if the actor does not reliably discover attack geometry.

Direct tests:

- `tests/test_attack_chain_end_to_end.py::test_combat_event_reward_attribution_is_per_red_agent`.
- Existing dead-agent and dense-reward tests verify one-step damaged reward and post-death zeroing.

## 10. Observation and learnability audit

Current V2 local observation:

- Dimension: 63.
- Own block: altitude, speed, flight-path angle, heading sin/cos, health ratio, last action, episode progress.
- Ally blocks: fixed ID order excluding own red ID, with alive flag, body-frame relative position/velocity, relative z, health.
- Enemy blocks: fixed blue ID order, with alive flag, body-frame relative position/velocity, distance, bearing/elevation, attack angle, escape angle, health.

Conclusions:

- Actor can infer nearest enemy direction, body-frame relative direction, own attack angle, enemy escape angle, distance, own altitude, flight-path angle, and time remaining.
- Actor does not receive an explicit `can_attack` boolean.
- Actor does not receive an explicit own-agent ID one-hot or slot ID. It can infer some role information from ally/enemy geometry and fixed slot layout, but this is indirect.
- Fixed-ID slots are stable across time; enemy slots do not dynamically reorder by distance.
- Dead slots are masked to alive flag `-1` and zero remaining normalized fields.
- Observation clipping can saturate far relative positions and thereby reduce long-range directional resolution.

Current V2 global state:

- Dimension: 61.
- Fixed red then blue entity blocks with alive, health, absolute x/y/z, speed, flight-path, heading sin/cos, last action, and episode progress.
- Critic sees all entities and time.

Direct tests:

- `tests/test_combat_observation_consistency.py`
  - body-frame lateral/longitudinal sign;
  - rotation consistency;
  - attack geometry feature monotonicity;
  - fixed ID slots and dead masks;
  - global state blue entities and time feature.

## 11. Action and attack reachability audit

The 15-action table controls speed, climb/dive, and left/right turn through the same `propagate_state()` path used by the environment.

Direct tests confirm:

- left and right turns change heading in opposite directions modulo `[0, 2π)`;
- climb increases altitude relative to level hold;
- dive decreases altitude relative to level hold;
- accelerate/decelerate change speed as expected;
- level hold keeps near-horizontal flight in a short horizon.

Short reachable attack micro-scenario:

- A red greedy rule used only inside a test, against straight blue, enters attack attempts in a few decision steps when initialized in reachable tail-chase geometry.
- This proves that red attack, hit, and damage are reachable through the production environment when geometry is reachable.

Formal scenario inference:

- `head_on_mirrored_jitter_v2` starts head-on at large separation.
- Current `can_attack` is not a head-on missile shot; it requires target escape angle below threshold, meaning a tail/aspect advantage condition.
- A random or early untrained shared actor must first learn geometry-changing maneuvers before receiving frequent attack events.

## 12. `greedy_combat` audit

Code: `src/uav_env/opponents/greedy_combat.py`.

Confirmed behavior:

- It predicts the assigned red target using that target's `last_action`.
- It does not read the red actor's current unexecuted action or logits.
- At reset, red and blue `last_action` are `LEVEL_HOLD`.
- It enumerates all 15 blue actions.
- Each candidate uses `get_control()` and `propagate_state()` for the configured number of physics substeps.
- It scores one assigned target using offensive score and incoming threat.
- It directly uses exact `compute_combat_geometry()` outputs including `can_attack`.
- It applies deterministic tie-break: prefer non-turning action, then lower action ID.

Risks:

- Safety filtering checks only the final predicted candidate state, while the environment checks boundary death at every physics substep.
- Defense considers only the assigned target, not all red threats.
- `assign_targets()` guides maneuvering only; weapon resolution later chooses the nearest attackable target among all living opponents. Thus maneuver target and actual weapon target can differ.
- The rule has an explicit geometry oracle that the red neural actor does not have. This is acceptable for a fixed rule opponent but is a substantial learning-difficulty asymmetry.

Typical score scale from defaults:

- Offensive continuous angle/distance terms: up to `0.6 + 0.4 = 1.0`.
- Attack area bonus: `+0.5`.
- Advantage area bonus: `+0.25`.
- Can-attack bonus: `+2.0`.
- Incoming continuous terms: up to `1.0`.
- Incoming attack area penalty: `+0.75`.
- Incoming advantage area penalty: `+0.25`.
- Incoming can-attack penalty: `+2.0`, multiplied by `defense_weight=0.7`.

The `can_attack` bonus is therefore a dominant term, by design.

## 13. Shared MAPPO actor audit

MAPPO training path:

- `MAPPOEnvAdapter` returns local obs `(num_agents, obs_dim)`, global state, per-agent red rewards, alive masks, and action masks.
- `RolloutBuffer` preserves agent dimension `(t, env, agent)`.
- `MAPPOTrainer.update()` flattens `(t, env, agent)` and uses active masks so dead agents do not contribute actor loss.
- Critic state is expanded per agent and gathers the per-agent value head by agent ID.

No shape/index bug was found in this audit.

Design risks:

- Shared actor receives no explicit agent-ID feature.
- Fixed-ID slots are semantically different per row (`red_0` sees red_1/red_2 as allies; `red_1` sees red_0/red_2), but the generic feature names given to the network are slot-major rather than row-specific.
- Symmetric initial states plus shared parameters can yield correlated actions; this can plausibly explain synchronized climb/ceiling behavior if the policy drifts toward a common action preference.

## 14. Termination and win/loss classification

`CombatMultiEnv._outcome(timed_out)`:

- red and blue both zero survivors: draw, `simultaneous_elimination`;
- blue zero: red win, `blue_eliminated`;
- red zero: blue win, `red_eliminated`;
- timeout: winner by survivor count, or draw if equal;
- otherwise ongoing.

`combat_outcome_rates()`:

- `overall_red_win_rate` counts any red winner, including timeout survivor-count wins.
- `elimination_win_rate` counts only red winner with `termination_reason == "blue_eliminated"`.
- `timeout_survival_win_rate` separately counts red timeout survivor-count wins.

Direct test:

- `tests/test_attack_chain_end_to_end.py::test_timeout_survivor_win_is_not_elimination_win`.

Conclusion: timeout survivor-count win is not confused with elimination win in ranking metrics.

## 15. Parallel environment and metrics audit

Parallel workers:

- Each worker has seed `base_seed + index`.
- Reset sends `base_seed + index`.
- Worker errors propagate with traceback.
- Terminal states are retained in `terminal_steps`; completed workers are reset into `next_*`.

Metrics:

- Episode-level rollout metrics are appended only when an episode terminates/truncates during collection.
- Incomplete episodes remain in accumulators and are not counted as completed episodes for rollout means.
- Evaluation episodes are full episodes from fixed deterministic seed ranges.
- MAPPO/HAPPO final evaluation loads `initial`, `last`, `best` and evaluates with test seeds.

Potential metrics ambiguity:

- Some rollout keys such as `ground_crash_rate` historically aggregate any-side crashes, while evaluation now reports `red_crash_rate` and `blue_crash_rate`. Use side-specific metrics when interpreting formal experiments.

## 16. Test evidence added in this audit

New files:

- `tests/test_attack_chain_end_to_end.py`
- `tests/test_combat_geometry_oracles.py`
- `tests/test_combat_resolution_symmetry.py`
- `tests/test_combat_observation_consistency.py`

Coverage mapping:

- A geometry oracle: `test_combat_geometry_oracles.py`.
- B red/blue mirror symmetry: geometry/resolution tests.
- C simultaneous attack: both sides attack in one resolution pass.
- D multi-attacker: three red attackers hit one blue target.
- E effective damage: clamped to remaining health.
- F single destroy event: one destroy credit per target.
- G dead aircraft cannot attack: direct resolution test.
- H attack RNG reproducibility/boundary: damage boundary and deterministic sequence tests.
- I worker RNG independence: parallel worker seed test.
- J reward event attribution: direct `_combat_event_reward()` test.
- K timeout vs elimination: outcome-rate test.
- L reset cleanup: reset state test.
- M observation signs/frame: body-frame tests.
- N 15-action direction: action effects test.
- O short attack reachability: reachable micro-scenario test.
- P greedy red/blue rule probe: red greedy rule only in test.
- Q metrics accumulation consistency: vector terminal retention and side metric checks are covered by existing tests plus this audit's endpoint tests.

## 17. Issue list

### DESIGN CHOICE 1: `greedy_combat` has explicit geometry/can-attack oracle

- Evidence: `GreedyCombatOpponent._evaluate_action()` calls `compute_combat_geometry()` for offensive and incoming geometry and scores `can_attack` directly.
- Code location: `src/uav_env/opponents/greedy_combat.py`.
- Reproducible test: `tests/test_greedy_combat_opponent.py`, `tests/test_attack_chain_end_to_end.py`.
- Training impact: high. It likely explains why blue rapidly attacks while red rarely discovers attack geometry.
- Explains red almost no attacks: yes, likely a primary factor.
- Minimal future option: keep as hard fixed opponent but introduce curriculum, or add a weaker ablation opponent; do not call it paper-exact.

### MAJOR 1: Formal head-on scenario is difficult under tail/aspect `can_attack`

- Evidence: `can_attack` requires both attack angle and target escape angle below thresholds; head-on geometry does not immediately satisfy the escape/aspect condition.
- Code location: `src/uav_env/combat/attack_geometry.py`, `configs/scenario_3v3_v2.yaml`.
- Reproducible test: `test_forward_tail_chase_geometry_oracle()` and `test_geometry_rejects_behind_too_far_and_too_close()`.
- Training impact: high. The actor must learn to maneuver into tail/aspect geometry before receiving frequent hit events.
- Explains red almost no attacks: yes, likely.
- Minimal future option: curriculum from reachable tail/offset cases before formal jittered head-on.

### MAJOR 2: Shared actor lacks explicit own-agent identity

- Evidence: local own block contains kinematics/health/action/time but no one-hot red ID; row-specific feature names exist for debug, not as numeric actor input.
- Code location: `src/uav_env/observations/multi_observation.py::V2_OWN_FEATURES`.
- Reproducible test: `test_fixed_id_slots_and_dead_masks_are_stable()`.
- Training impact: high to moderate. It can produce correlated actions and weak role specialization.
- Explains red almost no attacks: partly; more directly explains synchronized maneuvers and ceiling violations.
- Minimal future option: add explicit normalized own slot/ID or use non-shared actors; this is a production semantic change and was not done here.

### MODERATE 1: Maneuver target assignment and weapon target selection can differ

- Evidence: `CombatMultiEnv.step()` passes assignments only to `_blue_actions()`. `resolve_multi_attacks()` independently selects nearest attackable target and receives no assignment.
- Code location: `src/uav_env/envs/combat_multi_env.py::step`, `src/uav_env/combat/multi_combat.py::resolve_multi_attacks`.
- Reproducible test: `test_attack_target_assignment_is_maneuver_guidance_not_weapon_constraint()`.
- Training impact: moderate. Blue can maneuver as if covering distinct targets but concentrate actual shots on one target if that target is nearest attackable.
- Explains red almost no attacks: no direct, but can amplify blue lethality.
- Minimal future option: either document this as intended or pass optional target constraints into attack resolution.

### MODERATE 2: `greedy_combat` safety checks only final predicted candidate

- Evidence: `_predict_state()` returns final state only; `_is_unsafe()` checks final state. Environment boundary death checks each physics substep.
- Code location: `src/uav_env/opponents/greedy_combat.py`.
- Reproducible test: current tests verify final unsafe filtering; no substep-crossing oracle was added because production was not changed.
- Training impact: low to moderate; could choose an action that briefly violates boundary if dynamics permits recovery inside one decision period.
- Explains red almost no attacks: unlikely.
- Minimal future option: track substep samples in `greedy_combat` candidate prediction.

### MODERATE 3: Some rollout crash metric names are less precise than evaluation metrics

- Evidence: MAPPO collect historically uses aggregate `ground_crash_rate`, while evaluation reports side-specific crash rates.
- Code location: `src/uav_env/algorithms/mappo/runner.py`.
- Training impact: moderate for interpretation, not for environment dynamics.
- Explains red almost no attacks: no.
- Minimal future option: prefer side-specific metrics in reports.

### DESIGN CHOICE 2: Blue has no reward, only stats/events

- Evidence: reward breakdowns are generated for red agents only; blue events go to statistics.
- Code location: `CombatMultiEnv.step()` reward loop.
- Training impact: expected for fixed-rule blue.
- Explains red almost no attacks: no.
- Minimal future option: none unless training blue agents.

### MINOR 1: `damaged` means dead/failure, not merely hit

- Evidence: `UAVState.validate_consistency()` enforces `damaged == not alive`; `ever_hit` tracks hit history.
- Code location: `src/uav_env/core/state.py`.
- Training impact: low if documented.
- Explains red almost no attacks: no.
- Minimal future option: document clearly; avoid using `damaged` to mean "has taken damage".

## 18. Root-cause judgment

Questions from the request:

1. Does the attack chain have an implementation error? No BLOCKER found. The chain is complete and directly tested.
2. Are red and blue attack rules symmetric? Yes for attack geometry, hit sampling, damage, death, and statistics. No for rewards/training signal, by design.
3. Top three likely causes of rare red attacks:
   - formal head-on scenario is hard under tail/aspect `can_attack`;
   - `greedy_combat` has exact one-step geometry/can-attack scoring while red must infer from observations;
   - shared actor lacks explicit own-ID and can fall into correlated formation-level actions.
4. Are hit and damage normal? Yes, direct tests confirm probability boundary, RNG samples, effective damage clamp, overkill separation, and destroy credit.
5. Are observations sufficient to learn attack? They contain distance, body-frame relative position/velocity, attack angle, escape angle, health, own kinematics, action, and time. They are sufficient in principle, but not as direct as `greedy_combat`'s oracle and may be weakened by clipping and no own-ID.
6. Are 15 actions sufficient to enter attack area? Yes in reachable micro-scenarios and action-direction tests. Formal scenario still requires learning a maneuver sequence.
7. Does `greedy_combat` have unreasonable information advantage? It has a strong rule-opponent advantage: exact current state and explicit `can_attack` oracle. This is reasonable for a hard fixed opponent only if described as such; it is not symmetric with learned red actor information.
8. Does reward create survival/timeout local optimum? Plausibly yes. Timeout can be less bad than early elimination, and attack events are sparse until geometry is discovered.
9. Where does ceiling-violation regression likely come from? Most likely correlated shared-actor action preference plus weak role identity, not attack-chain damage failure.
10. Can formal environment continue training now? Technically yes, but scientifically it should be treated as a hard-opponent setting. For learning diagnosis, run curriculum/ablation before committing long formal training.
11. Should next step be code fix, opponent change, curriculum, or no change? Recommended: do not patch attack/damage code first. Run curriculum/ablation and consider own-ID or opponent-strength changes only as explicit semantic experiments.
12. Directly proven conclusions: geometry semantics, resolution simultaneity, damage clamp, reward event attribution, observation frame signs, reset cleanup, worker seed difference, micro-scenario attack reachability.
13. Inferred conclusions: causes of 940,032-step training behavior, survival local optimum, shared actor causing ceiling drift.

## 19. Verification commands

Commands used:

```bash
python -m py_compile related audit/test files
pytest -q
```

No MAPPO/HAPPO training, long experiment, multi-seed training, checkpoint evaluation, or long rule rollout was run.
