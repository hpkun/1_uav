# Functional Heterogeneous 3v3 UAV Cooperative Combat Environment

This environment is named:

**Functionally Heterogeneous UAV Cooperative Combat Environment with Homogeneous Maneuver Dynamics**

中文定义：同机动性能双类型功能异构无人机3v3协同对抗环境。

## 1. Research question

The experiment asks whether a red team with two armed combat UAVs and one unarmed support UAV can benefit from role-asymmetric sensing and ideal information relay in a fixed 3v3 close-range air-combat task.

The implementation isolates functional heterogeneity from physical heterogeneity. All UAVs keep the same point-mass dynamics, action table, speed limits, overload limits, health, attack geometry and damage model.

## 2. Relation to TAM-HAPPO-style work

The project borrows the ideas of role asymmetry, an unarmed support platform and differentiated task objectives. It does not reproduce missile dynamics, communication constraints, sensing noise, control delay, GRU, attention, Transformer, TAM-HAPPO or HAPPO-specific network updates.

## 3. Why homogeneous maneuver dynamics

All red and blue vehicles use the same `UAVTypeProfile` and the same 15 discrete maneuvers. This makes the comparison about weapons, sensing, relay and reward role rather than about superior speed, climb, turn-rate or survivability.

## 4. Roles

Functional roles are independent of physical type:

- `combat`: armed combat UAV, short-range local sensing.
- `support`: unarmed support UAV, long-range sensing and support-oriented reward.

Fixed red IDs:

- `red_0`: combat
- `red_1`: combat
- `red_2`: support in heterogeneous modes

Blue UAVs remain homogeneous armed rule-controlled aircraft.

## 5. Three control modes

All modes use the same schema, observation/state dimensions, initial formation, dynamics, reward assignment algorithm, blue opponent and MAPPO hyperparameters.

| Mode | Roles | Weapons | Relay |
|---|---|---|---|
| `homogeneous_control` | combat, combat, combat | all red armed | off |
| `heterogeneous_no_relay` | combat, combat, support | support unarmed | off |
| `heterogeneous_relay` | combat, combat, support | support unarmed | on while support alive |

## 6. Formation

Scenario: `head_on_functional_heterogeneous_v1`.

Red uses an inverted triangle:

- `red_0`: `red_base_x + combat_forward_offset`, `-combat_lateral_spacing/2`
- `red_1`: `red_base_x + combat_forward_offset`, `+combat_lateral_spacing/2`
- `red_2`: `red_base_x - support_rear_offset`, `0`

Default values:

- `combat_forward_offset = 150 m`
- `support_rear_offset = 300 m`
- `combat_lateral_spacing = 500 m`

Blue keeps the 500 m line formation at `+initial_team_distance/2`.

## 7. Weapon gating

`resolve_multi_attacks()` accepts optional `armed_ids`.

- `None`: legacy behavior, all living aircraft may attack.
- set of IDs: only those living IDs can be attackers.

Unarmed support aircraft still move, observe, receive damage, collide, crash and count for termination. They can be attacked normally.

## 8. Sensing and relay

Distances use 3D Euclidean range only.

- `combat_detection_range = 1350 m`
- `support_detection_range = 2700 m`

For combat UAV `i`:

`V_local_i = {alive blue j | distance(i,j) <= combat_detection_range}`

For support:

`V_support = {alive blue j | distance(support,j) <= support_detection_range}`

In relay mode, while support is alive:

`V_i = V_local_i ∪ V_support`

Relay is ideal, delay-free and loss-free. No communication distance, queue, bandwidth, packet loss or topology is modeled.

## 9. 69D local observation

Schema: `fixed_id_role_visibility_time_69d`.

It extends the 63D fixed-ID body-frame time-aware observation:

- own block: 8 old fields + `own_role_support_flag`
- ally blocks: two 8-field old blocks + `role_support_flag`
- enemy blocks: three 13-field old blocks + `visible_flag`

Role flag:

- support: `+1`
- combat: `-1`

Enemy visibility semantics:

- alive and visible: `alive_flag=+1`, `visible_flag=+1`, continuous state real.
- alive and invisible: `alive_flag=0`, `visible_flag=-1`, continuous state zero.
- dead: `alive_flag=-1`, `visible_flag=-1`, continuous state zero.

Slots remain fixed by ID.

## 10. 64D global state

Schema: `full_entity_role_time_64d`.

It extends the old 61D full-entity state by adding one role flag to each red entity block:

- `red_0_role_support_flag`
- `red_1_role_support_flag`
- `red_2_role_support_flag`

The critic receives complete entity state and is not visibility-gated.

## 11. Support reward

Support raw shape:

`R_support_shape = 0.40 R_position + 0.35 R_coverage + 0.25 R_safety`

Position:

`p_ref = mean(alive combat positions) - support_rear_distance * mean_heading`

`R_position = clip(1 - ||p_support_xy - p_ref_xy|| / support_position_tolerance, -1, 1)`

Coverage:

`R_coverage = detected_alive_blue / alive_blue`

Safety:

For each alive blue against support:

- `can_attack`: threat `1.0`
- attack or advantage area: threat `0.5`
- otherwise `0.0`

`R_safety = -max(threat)`

Support raw shape enters the existing dense assignment function with combat raw shapes.

Support event changes:

- no self hit/destroy reward
- attacked penalty unchanged
- destroyed penalty multiplied by `support_loss_multiplier=1.5`
- boundary/collision penalty multiplied by `support_loss_multiplier=1.5`
- team event share: `clip(0.25 * combat positive hit/destroy events, 0, 1.0)`

The team event is computed after all red agents' own combat events have been
computed, so it does not depend on red-agent loop order.  Only positive combat
hit/destroy components from armed combat UAVs are shared; attacked, destroyed,
boundary and collision penalties are excluded.

For functional heterogeneous and time-aware homogeneous V2 environments the
per-agent reward assembly is explicitly split as:

`total = assigned_shape + combat_event + terminal_base_reward + mission_success_bonus`

where:

- `assigned_shape` is the Algorithm-2 assignment of the current raw shape
  reward;
- `combat_event` contains hit/destroy/attacked/destroyed/boundary events, and
  for support also includes the support team event and support loss adjustment;
- `terminal_base_reward` is the selected multi-agent terminal allocation;
- `mission_success_bonus` is separate from terminal allocation and is zero
  outside heterogeneous mission success.

Diagnostic fields keep both `assigned_shape` and `assigned_dense`; in this
schema `assigned_dense` is an alias of the assigned shape term, while
`dense_reward = assigned_shape + combat_event`.

## 12. Mission success

`mission_success = blue_eliminated AND support_alive`

In heterogeneous modes only, mission success adds `+1.0` to each red agent on the terminal blue-eliminated step.

Support death does not terminate the episode.

## 13. Metrics

Functional metrics:

- `has_support_agent`
- `support_metrics_applicable`
- `support_survival_rate`
- `mission_success_rate`
- `support_detection_coverage_mean`
- `relay_visible_enemy_count_mean`
- `support_incoming_threat_mean`
- `support_position_error_mean`
- `combat_attack_attempts_mean`
- `combat_hits_mean`
- `combat_effective_damage_mean`

Environment `info["functional_metrics"]` stores episode cumulative combat
quantities with `_total` suffix:

- `combat_attack_attempts_total`
- `combat_hits_total`
- `combat_effective_damage_total`

The MAPPO evaluation and rollout summaries report episode means with `_mean`
suffix.  Homogeneous-control mode has no support aircraft; it reports
`has_support_agent=0` and `support_metrics_applicable=0` so the support metrics
are not mistaken for failed support behavior.

Existing win, timeout, reward, hit, damage, survivor, crash, ceiling and collision metrics remain.

## 14. Fair comparison

The three MAPPO configs differ only in functional mode, roles and relay. They keep the same:

- scenario
- blue opponent
- initial stochastic process
- dynamics
- attack and damage
- observation/state dimensions
- MAPPO network and hyperparameters
- rollout/evaluation/checkpoint cadence

## 15. Explicitly not modeled

This environment does not add different flight performance, different health, ammunition, missile entities, radar field of view, sensor noise, communication range, communication delay, packet loss, command actions, target-assignment actions, GRU, Attention, Transformer, HAPPO, TAM-HAPPO, curriculum learning, self-play or opponent mixtures.
