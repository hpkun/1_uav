# Persistent-Wave Multi-UAV Combat Environment Design

Status: minimal Phase-A implementation available  
Implemented environment variant: `persistent_wave_v1` on the frozen V2.3 contract  
Baseline preserved unchanged: `2.3`

## 0. Current implementation boundary

The first implementation is intentionally smaller than the research roadmap in
the later sections. This section is authoritative wherever a later section
describes a possible extension.

- Blue uses only the existing `NearestTargetPursuitPolicy`.
- The 52-dimensional observation is unchanged.
- The V2.3 R1-R4 reward functions and coefficients are unchanged.
- The V2.3 probabilistic entry-triggered weapon and unlimited attempts are
  unchanged; no ammunition or cooldown state is present.
- There is no inter-wave waiting phase. A fresh four-aircraft Blue wave is
  spawned immediately after the previous wave is cleared.
- Red physical state, alive mask, and existing fire-trigger state persist.
- A non-final wave clear is not a terminal or truncation.
- The default mission has three waves and a 3,000-step global horizon.

Finite ammunition, alternative reward modes, additional Blue policies, action
repeat, inter-wave maneuvering, and observation extensions are deferred and are
not part of `persistent_wave_v1`.

## 1. Research purpose

The new environment models one persistent Red formation facing a sequence of
fresh Blue waves in one mission. Red physical states, losses, and weapon
resources persist across wave boundaries. Blue entities are replaced between
waves. The environment is intended to expose three problems that are absent or
weakly tested in a direct one-round engagement:

1. long-horizon credit assignment across wave boundaries;
2. force and ammunition preservation versus immediate combat gain;
3. generalization to uncertain future wave sizes, arrival geometries, and rule
   tactics.

The environment itself must not hard-code a hierarchical policy, graph policy,
or a particular cross-round learning algorithm. It should expose a neutral,
versioned benchmark on which flat, recurrent, hierarchical, and boundary-aware
algorithms can be compared fairly.

## 2. Literature-derived scope

The environment design uses the following papers as boundaries rather than as
templates to copy:

- Jiao et al. (2025), *Collaborative decision-making for UAV swarm
  confrontation based on reinforcement learning*, introduces a 3-vs-3
  continuous-wave demonstration: surviving Red UAVs face a newly spawned group
  of three Blue UAVs. It does not provide systematic multi-wave baselines,
  ablations, or cross-round credit assignment.
- Yan et al. (2025), *A sample selection mechanism for multi-UCAV air combat
  policy training using multi-agent reinforcement learning*, explicitly models
  two missiles per UCAV and separate missile entities. This supports finite
  ammunition as a meaningful persistent resource, but a complete BVR missile
  guidance simulation is beyond the first version of this WVR-like benchmark.
- Zheng et al. (2026), *Asynchronous hierarchical deep reinforcement learning
  with learnable reward shaping for distributed multi-UCAV air combat
  decision*, demonstrates variable-scale high-level target representation and a
  graph-based low-level critic. The environment should therefore expose masks
  and variable active counts without prescribing either attention or graphs.
- Pang et al. (2026), *Dynamic alliance operations of UAV swarms: A
  hierarchical reinforcement learning-based adaptive grouping and unitized
  decision-making framework*, studies multi-round engagement through dynamic
  grouping and hierarchical resource management. Dynamic grouping is not the
  proposed benchmark contribution; cross-wave persistence and credit are.
- Jiang et al. (2026), *UAV formation beyond-visual-range air combat decision
  based on multi-agent reinforcement learning*, returns a delayed hit reward to
  the launch event. The first environment version therefore logs attempt,
  launch-equivalent, hit, kill, and wave-boundary event identifiers so later
  algorithms can perform event-based credit assignment without changing the
  simulator.

## 3. Non-goals for version 3.0

Version 3.0 will not add the following:

- six-degree-of-freedom rigid-body dynamics;
- explicit radar search, track, and guidance handover;
- in-flight missile kinematics;
- fuel-burn or aerodynamic energy models;
- partial damage or component failures;
- learned Blue agents inside the environment implementation;
- dynamic Red sub-team grouping as an environment rule.

These features would change too many causal factors at once. Version 3.0 keeps
the V2.3 point-mass dynamics and probabilistic attack equation, while adding
only the mechanisms needed for persistent-wave research.

## 4. Compatibility and code organization

V2.3 must remain reproducible and checkpoint-compatible. Do not mutate its
observation contract, termination semantics, or configuration.

Recommended additions:

```text
configs/persistent_wave_environment.yaml
src/uav_combat/environment/persistent_env.py
src/uav_combat/environment/wave_manager.py
src/uav_combat/environment/opponent_pool.py
src/uav_combat/environment/resource.py
tests/test_persistent_wave_environment.py
docs/persistent_wave_environment_spec.md   # generated after implementation
```

The new class should be `PersistentWaveCombatEnv`. Shared V2.3 dynamics,
geometry, control, integration, and attack-probability code should be reused.
The new environment version and observation dimension must be stored in every
checkpoint so a V2.3 policy cannot be loaded silently into V3.0.

## 5. Mission and wave state machine

### 5.1 States

```text
RESET
  -> ACTIVE_WAVE
       -> INTER_WAVE          if all active Blue UAVs are removed,
                               Red survives, and waves remain
       -> MISSION_SUCCESS     if final wave is cleared and Red survives
       -> RED_ELIMINATED      if no Red UAV survives
       -> MISSION_TIMEOUT     if global mission horizon is reached
       -> WAVE_TIMEOUT        if a wave exceeds its local horizon
  INTER_WAVE
       -> ACTIVE_WAVE         after the configured countdown and a valid spawn
       -> RED_ELIMINATED      if a Red loss occurs during transition logic
       -> MISSION_TIMEOUT
```

`wave_cleared` is an event, not a Gymnasium terminal. In particular:

- `terminated=False` at every non-final wave boundary;
- Red recurrent hidden state is not reset by the environment;
- the next observation is a genuine transition observation;
- value functions may bootstrap through the boundary;
- only mission-level outcomes terminate an episode.

### 5.2 Default mission

- Red initial slots: 4.
- Maximum Blue slots per wave: 4.
- Training waves: sampled from 2 or 3.
- Evaluation can use 2-5 waves.
- Red dead slots remain dead for the rest of the mission.
- Red position, velocity, heading, pitch, last executed bank command, weapon
  state, and cumulative statistics persist.
- Blue slots are reused and fully reset at each new wave.

### 5.3 Inter-wave interval

The default inter-wave interval is 10 decision steps. Red remains controllable
during this interval, making regrouping and positioning meaningful. The actor
receives a normalized arrival countdown but not the next arrival bearing,
formation, size, or tactic.

An `instant` ablation sets the interval to zero. This separates benefits caused
by active regrouping from benefits caused only by wave-aware value learning.

## 6. Simulation and decision timing

The physics integrator remains at `dt=0.1 s`. To keep multi-wave training
tractable, V3.0 introduces action repeat:

- physics step: 0.1 s;
- default decision repeat: 5 physics steps;
- decision interval: 0.5 s;
- dynamics, boundary checks, attack-window checks, weapon cooldown, and hits are
  evaluated at every physics step;
- one policy action is held for all five physics steps;
- rewards and event counts are accumulated and returned once per decision step.

Recommended default horizons:

- maximum 300 decision steps per active wave (150 s);
- 10 decision steps between waves (5 s);
- maximum mission horizon derived from the sampled number of waves plus
  inter-wave intervals, with an explicit hard cap.

The action-repeat value is part of the environment version and must be reported
in papers and checkpoints.

## 7. Persistent weapon resource

### 7.1 First-version weapon state

Replace the single Boolean fire state with:

```python
WeaponResourceState(
    armed: bool,
    ammo_remaining: int,
    cooldown_physics_steps: int,
)
```

Default Red and Blue magazine capacity is 4 attempts per aircraft. The capacity
must be configurable; evaluation includes capacities 2, 4, and 6.

An attack attempt is generated only if:

1. attacker and target are alive;
2. a target is in the V2.3 fire window;
3. `armed` is true;
4. `ammo_remaining > 0`;
5. cooldown is zero.

On an attempt:

- decrement ammunition exactly once, regardless of hit or miss;
- start the cooldown;
- disarm the entry trigger;
- store an immutable event id with attacker, target, wave, physics step,
  geometry, hit outcome, and later kill credit.

The entry trigger is re-armed only after the attacker has left every valid fire
window and the cooldown has expired. Red ammunition is never replenished between
waves by default. Every new Blue wave receives a fresh magazine.

### 7.2 Why not add explicit missiles yet

The active environment is a short-range, probabilistic-entry benchmark. Adding
radar detection, launch control, missile motion, mid-course guidance, terminal
guidance, and countermeasures in one revision would create a different BVR
environment and make causal attribution difficult. V3.0 uses an
`attempt-as-munition` abstraction. A later V3.1 can add in-flight missiles using
the same event ids and resource interface.

## 8. Wave generation

### 8.1 Schedule sampled at reset

At reset, a `WaveSchedule` is sampled from an environment-owned random stream:

```text
WaveSpec:
  active_blue_count
  tactic_id
  formation_id
  arrival_bearing
  spawn_altitude_center
  spawn_speed_center
  inter_wave_steps
  ammo_capacity
```

The complete schedule is available in debug/evaluation info but is not included
in actor observations. The actor knows only the maximum mission waves, current
wave index, and arrival countdown.

### 8.2 Spawn constraints

For every wave after the first, sample Blue states by rejection under all of the
following constraints:

- every Blue UAV is inside the arena with a configurable margin;
- every Blue-Red distance exceeds `spawn_min_pair_distance`;
- no Blue-Red pair starts in either side's fire window;
- Blue formations do not overlap;
- altitude is above ground with a safety margin;
- initial speeds and attitudes satisfy aircraft limits;
- the heading is generally toward the current Red centroid, with tactic-specific
  perturbations;
- a bounded number of attempts is used, followed by a deterministic fallback
  layout so reset can never hang.

Recommended initial values are a 3.5-4.5 km global annulus, at least 3 km pair
separation, and the existing altitude/speed perturbation ranges. These values
must be calibrated by Monte Carlo because the current arena radius is only 5 km.

### 8.3 Blue rule-policy pool

The environment supplies rule policies, not learned opponents:

1. `nearest_pursuit`: current V2.3 policy;
2. `hungarian_assignment`: minimum-total-distance one-to-one assignment with
   periodic reassignment;
3. `focus_fire`: all available Blue UAVs prioritize one Red target selected by
   threat and distance;
4. `split_flank`: two subgroups approach the Red centroid from different
   bearings before switching to pursuit.

The first publishable implementation should contain at least the first three.
Policies must share the same flight and weapon limits. Tactic ids are logged but
not directly observed by Red.

## 9. Observation and centralized state

### 9.1 Actor observation

The current V2.3 observation is fixed at 52 dimensions:

```text
self:       7
3 allies:  3 x 7
4 enemies: 4 x 6
total:     52
```

V3.0 retains the four Red and four Blue slots and extends the observation to 67
dimensions:

```text
existing V2.3 observation                                      52
own ammo fraction, cooldown fraction, armed flag                3
three ally ammo fractions and cooldown fractions              3x2
mission context                                                 6
-----------------------------------------------------------------
total                                                          67
```

Mission context:

1. normalized current wave index;
2. normalized waves remaining;
3. normalized decision steps elapsed in the current phase;
4. normalized inter-wave arrival countdown;
5. Red survivor fraction;
6. current active Blue fraction.

Enemy ammunition and future wave properties are not observed. Dead Red agents
still receive an all-zero observation and are masked exactly as in V2.3.

### 9.2 Entity status

An empty Blue slot and a destroyed Blue slot are physically identical for the
current decision, but the mission context provides current active Blue fraction.
The environment must keep internal `present`, `alive`, and `wave_spawned`
concepts distinct for metrics and debugging.

### 9.3 Centralized state API

Add a separate `state()` method for CTDE algorithms. It should return all Red and
Blue normalized physical states, present/alive masks, weapon resource states,
and mission context. Algorithms may continue stacking local observations, but a
true global state prevents the environment from being tied to the current
attention critic implementation.

## 10. Reward and cost interface

### 10.1 Principles

- no reward may depend on an arbitrary aircraft index;
- no kill reward is multiplied by the wave number;
- wave spawning must not create a potential-reward discontinuity;
- reward components and operational costs are logged separately;
- mission metrics remain valid even if an algorithm replaces the training
  reward.

### 10.2 Default reward

The default `persistent_balanced` reward is:

```text
enemy destruction                     +10.0, divided among credited attackers
own destruction                        -10.0 to the destroyed Red slot
own boundary exit                      -10.0 to the exiting Red slot
weapon attempt                          -0.05 to the attacker
non-final wave clear                    +2.0 team reward
final mission completion              +20.0 team reward
V2.3 geometry shaping                 potential-difference form
```

Team rewards are divided equally among Red slots that were alive immediately
before the event. The exact alive-mask convention must be tested and documented.

Geometry shaping is evaluated only during `ACTIVE_WAVE`. At spawn and wave-clear
transitions, its potential baseline is reset so a newly appearing Blue formation
cannot create an artificial positive or negative reward.

### 10.3 Cost channel

Return an additional cost vector through `info` and rollout storage:

```text
friendly_loss_cost    1 per Red destruction
munition_cost         1 per Red attempt
high_threat_cost      optional diagnostic only in V3.0
```

The default MAPPO baseline ignores this channel. Constrained or risk-sensitive
algorithms can use it without changing simulator semantics.

### 10.4 Reward modes required for experiments

1. `paper_compatible`: V2.3 R1-R4 plus wave/mission events;
2. `persistent_sparse`: only kill, loss, attempt, wave-clear, mission events;
3. `persistent_balanced`: sparse events plus potential-based geometry shaping.

The Jiao-style index-dependent and wave-multiplied reward may be implemented only
as an explicitly named comparison mode, never as the default.

## 11. Termination and outcome semantics

Priority order after every physics step:

1. resolve dynamics;
2. resolve boundary and ground losses;
3. snapshot post-motion states;
4. calculate fire-window diagnostics;
5. resolve simultaneous Red and Blue attempts;
6. apply simultaneous kills and resource updates;
7. calculate reward and cost components;
8. determine mission outcome;
9. if a non-final wave is clear, enter `INTER_WAVE` without termination;
10. if the inter-wave countdown expires, spawn the next wave;
11. build the returned observation and info.

Outcomes:

- simultaneous final mutual destruction: draw;
- Red eliminated before final clearance: mission failure;
- final Blue wave cleared with at least one Red survivor: mission success;
- global timeout: mission failure;
- local wave timeout: mission failure by default, configurable for curriculum
  experiments;
- Blue boundary or ground loss removes the Blue entity but gives no Red kill
  reward. It still counts toward wave clearance, while diagnostics distinguish
  combat kills from noncombat removals.

## 12. Determinism and random streams

Use `numpy.random.SeedSequence` to create independent child streams for:

- initial Red/Blue geometry;
- wave schedule;
- per-wave spawn geometry;
- weapon noise;
- Blue-policy randomization.

Changing a diagnostic, adding a reward component, or evaluating a different Blue
policy must not silently change weapon-hit randomness. The sampled schedule and
child seeds are recorded in evaluation output.

## 13. Logging contract

### 13.1 Per-decision-step info

Add at least:

```text
mission_phase
wave_index
waves_total
waves_remaining
wave_step
wave_boundary
wave_cleared_this_step
inter_wave_steps_remaining
current_wave_blue_initial_count
current_wave_tactic_id
red_ammo_remaining[4]
blue_ammo_remaining[4]
red_step_attempts / hits / kills / losses
reward component vectors
cost component vectors
```

### 13.2 Per-wave record

Each completed wave produces one immutable record:

```text
start/end physics and decision step
initial Blue count, tactic, formation, bearing
Red start/end survivors
Red start/end ammunition
Red and Blue attempts, hits, combat kills, exits, and ground losses
clearance time
wave return and wave cost
end geometry summary
```

### 13.3 Mission metrics

Required evaluation metrics:

- mission completion rate;
- waves cleared;
- total combat kills;
- Red survivors after each wave;
- survival area under the wave curve;
- kill/loss ratio;
- attempts per kill and ammunition remaining;
- wave clearance time;
- conditional probability of clearing wave `r` given arrival at wave `r`;
- worst-decile and CVaR mission return;
- mean and percentile policy inference latency.

## 14. Training and evaluation distributions

### 14.1 Training distribution

- waves: 2 or 3;
- active Blue per wave: 2-4;
- tactics: nearest pursuit, Hungarian assignment, focus fire;
- Red magazine: 4;
- spawn bearing: full circle with rejection constraints;
- randomized formations, altitude, speed, and headings.

### 14.2 In-distribution evaluation

Use new seeds with the same ranges and at least 300 missions per trained seed.

### 14.3 Out-of-distribution suites

- `OOD-WAVE`: train on 2-3 waves, test on 4-5;
- `OOD-TACTIC`: hold `split_flank` out of training;
- `OOD-AMMO`: test Red capacities 2 and 6;
- `OOD-GEOMETRY`: hold arrival-bearing sectors or formation types out;
- `OOD-DIFFICULTY`: increase Blue speed or hit-probability scale within declared
  bounds;
- `OOD-MIX`: combine an unseen tactic with an unseen wave count.

Every suite uses identical schedules across algorithms.

## 15. Validation and tests

### 15.1 Unit tests

1. clearing a non-final Blue wave does not terminate or truncate;
2. clearing the final wave terminates with mission success;
3. Red elimination has priority over spawning another wave;
4. Red physical and weapon states persist exactly across a spawn;
5. dead Red slots never revive;
6. Blue slots and magazines reset to the new `WaveSpec`;
7. one attempt consumes exactly one munition;
8. cooldown and re-arming require the documented conditions;
9. a spawn never starts in a fire window or outside the arena;
10. bounded rejection sampling always returns or uses its fallback;
11. enemy-index permutations leave team event rewards invariant;
12. potential shaping is zeroed at spawn discontinuities;
13. non-final wave boundaries produce bootstrap-eligible transitions;
14. all observations, states, rewards, costs, and masks are finite;
15. a fixed seed reproduces the schedule, spawns, and hit outcomes.

### 15.2 Statistical validation

Run at least 1,000 randomized mission resets and report:

- spawn-constraint violations;
- distributions of pair distances, bearings, altitudes, and active counts;
- immediate fire-window incidence, which must be zero;
- initial geometric advantage balance;
- schedule/tactic frequencies;
- mission lengths and event rates under rule-policy baselines.

### 15.3 Difficulty calibration

Before formal RL experiments, evaluate simple Red baselines against every Blue
tactic:

- straight flight;
- nearest-target pursuit;
- random actions;
- a simple defensive turn rule.

Avoid benchmark settings where one rule wins either below 5% or above 95% across
all suites. The main training setting should admit measurable improvement while
remaining nontrivial.

## 16. Implementation phases and acceptance gates

### Phase A: wave semantics

- introduce the new class, version, config, state machine, schedule, spawn, and
  logging;
- reuse the current weapon model without finite magazines;
- preserve V2.3 tests and checkpoints.

Gate: all boundary/termination tests pass and 1,000 reset validation has zero
invalid spawns.

### Phase B: persistent weapon resource

- add magazines, cooldown, resource observations, costs, and event ids;
- validate simultaneous attacks and exact ammunition accounting.

Gate: conservation tests pass: initial Red ammunition equals attempts plus final
remaining ammunition for every slot.

### Phase C: opponent and evaluation suites

- add Hungarian and focus-fire policies;
- define train/ID/OOD schedule manifests;
- add per-wave and mission reports.

Gate: fixed manifests reproduce identical missions across MAPPO and MADSAC.

### Phase D: baseline training readiness

- adapt vector environments and evaluators to non-terminal wave boundaries;
- increase observation dimensions under the explicit V3.0 contract;
- run short MAPPO/MADSAC smoke tests and verify that gradients, returns, masks,
  and recurrent states remain finite.

Gate: at least one baseline improves over random without saturating the task.

Only after all four gates should a cross-round learning algorithm be added.

## 17. Paper-facing experiment protocol

For a credible journal submission:

- use at least five independent training seeds;
- use equal environment decision steps and equal evaluation manifests;
- evaluate 300-1,000 missions per final seed;
- report mean, standard deviation, and 95% bootstrap confidence intervals;
- include flat MAPPO, recurrent MAPPO, MADSAC, a wave-conditioned MAPPO, and the
  proposed algorithm;
- perform environment/reward ablations separately from algorithm ablations;
- publish exact configuration files and environment version hashes.

The environment contribution should be described as a persistent-wave benchmark
with explicit continuity and resource semantics. The algorithm contribution can
then focus on cross-round credit assignment. This separation prevents reviewers
from attributing performance improvements to hidden simulator or reward changes.
