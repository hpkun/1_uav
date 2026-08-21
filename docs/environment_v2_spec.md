# Enhanced Combat Environment V2 Specification

## 1. Scope

V2 is a homogeneous 4v4 multi-UAV combat decision environment. Four Red agents are learned with the existing MADSAC implementation. Four Blue aircraft use one deterministic rule policy. The environment is an academic, low-fidelity combat model, not an engineering flight simulator.

The design objective is a learnable combat chain under necessary aircraft constraints:

`approach -> maneuver -> fire opportunity -> stable solution -> kill`

## 2. Simulation contract

- Coordinates: north-east-down (NED).
- Aircraft state: `s=[x,y,z,v,theta,psi]` plus alive flag.
- Integration interval: `dt=0.1 s`.
- Dynamics and RK4 integration remain unchanged.
- Team size: 4 Red and 4 Blue.
- Actor action dimension: 3.
- Per-agent observation dimension: 52.
- Horizontal position does not create reward, recovery control, or termination.
- An alive aircraft outside the altitude envelope is destroyed once.

## 3. High-level action

For normalized actor output `a=[a_h,a_p,a_v] in [-1,1]^3`, define:

```
psi_d   = wrap(psi + Delta_psi_max * a_h)
theta_d = theta_cmd_max * a_p
v_d     = v_cmd_min + (a_v + 1) / 2 * (v_cmd_max - v_cmd_min)
```

The three channels mean:

- `a_h`: relative desired heading change. It is rotation invariant and can command either turn direction.
- `a_p`: desired flight-path pitch. Zero explicitly means level flight.
- `a_v`: desired speed. It expresses accelerate/decelerate intent without exposing tangential load.

V2 defaults are `Delta_psi_max=180 deg`, `theta_cmd_max=30 deg`, and `v_d in [170,280] m/s`.

## 4. Simple aircraft response layer

The controller is memoryless proportional response with physical rate limits, not a PID autopilot.

```
psi_dot_c   = clip(k_psi * wrap(psi_d-psi), -psi_rate_max, psi_rate_max)
theta_dot_c = clip(k_theta * (theta_d-theta), -theta_rate_max, theta_rate_max)
v_dot_c     = clip(k_v * (v_d-v), -accel_max, accel_max)
```

Using the existing 3DOF equations, define:

```
A   = cos(theta) + v/g * theta_dot_c
B   = v*cos(theta)/g * psi_dot_c
B   = clip(B, -|A|tan(phi_max), |A|tan(phi_max))
phi = atan2(B,A)
nz  = clip(sqrt(A^2+B^2), nz_min, nz_max)
nx  = sin(theta) + v_dot_c/g
```

This is an algebraic response layer. It has no integrator state, derivative state, gain scheduling, path planner, or hidden stabilization objective. It maps a tactical desired motion into the same paper-consistent point-mass inputs. The actor learns where to turn, climb, descend, accelerate, or slow; it does not learn the load-factor combination required to hold altitude during a bank.

## 5. Observation

The observation remains 52-dimensional and contains only state information required to infer combat geometry.

### 5.1 Self features, 3

1. Normalized speed: needed for closure, turn response and time-to-contact.
2. Normalized flight-path pitch: needed to command vertical maneuvers and interpret the local frame.
3. Normalized altitude: needed to respect the only physical battlefield envelope.

### 5.2 Relative slots, 7 each

There are three fixed teammate slots followed by four fixed opponent slots. Each slot contains:

1. Relative position in own flight-path frame, 3 values.
2. Relative velocity in own flight-path frame, 3 values.
3. Alive flag, 1 value.

Normalized continuous values are clipped to `[-1,1]`. A dead slot is zero. A dead observing agent receives an all-zero vector.

Relative position and velocity are sufficient to infer range, line of sight, closure, off-boresight relation and target aspect. V2 deliberately does not expose precomputed distance, attack angle, target aspect, attackable state or lock state. This keeps the policy problem as combat decision-making from kinematic state rather than imitation of internal rule flags.

## 6. Fire model

For attacker `i` and target `j`:

- `d_ij`: 3D Euclidean range.
- `alpha_ij`: angle between attacker velocity and attacker-to-target LOS.
- `beta_ij`: target aspect, angle between target velocity and attacker-to-target LOS. `beta=0` is rear/tail pursuit; `beta=pi` is head-on.

The firing window is:

```
I_fire(i,j) = 1[
    R_min <= d_ij <= R_max
    and alpha_ij <= alpha_max
    and beta_ij <= beta_max
]
```

Defaults:

- `R_min=300 m`
- `R_max=2000 m`
- `alpha_max=35 deg`
- `beta_max=120 deg`
- continuous dwell `N_lock=5` steps (`0.5 s`)

The environment automatically selects the best current firing-window target. A target is deterministically destroyed after the same attacker-target pair remains in the window for `N_lock` consecutive steps. Broken geometry or target change resets dwell. Same-step Red and Blue proposals are resolved simultaneously.

Why this is suitable for RL:

- It preserves range, pointing and target-aspect constraints central to air combat.
- A 2 km envelope and 0.5 s dwell create a visible opportunity without simulating missile kinematics.
- The target-aspect threshold excludes immediate head-on kills while allowing side/rear solutions.
- Deterministic resolution removes unnecessary weapon-noise variance from policy learning.
- Smooth reward features lead toward the hard window; the hard window itself remains an interpretable success event.

## 7. Stage-based combat reward

For each alive Red agent `i`, V2 computes four combat-only components. Let `j*` be its nearest alive target at the pre-transition state for approach progress. Let `best()` select the highest relation score over alive opponents.

### 7.1 Engagement progress

```
r_progress_i = w_p * 1[d_t > R_max]
               * clip((d_t-d_t+1)/(v_close_ref*dt), -1, 1)
```

The same target `j*` is measured before and after the step. This rewards actual closure only before the firing region and stops rewarding collision-like pursuit inside it.

### 7.2 Tactical advantage

For one directed relation:

```
G(d)     = clip((R_tactical-d)/(R_tactical-R_min), 0, 1)
A(alpha) = (1+cos(alpha))/2
B(beta)  = (1+cos(beta))/2
T(i,j)   = G(d_ij) * (0.6*A(alpha_ij) + 0.4*B(beta_ij))
```

Then:

```
r_tactical_i = w_t * (best_j T(i,j) - best_j T(j,i))
```

This arithmetic angular combination avoids the V1.4 triple-product collapse. The signed attack-minus-threat term rewards a sustained positional advantage and penalizes being in an opponent's corresponding advantage.

### 7.3 Fire opportunity

```
r_fire_i = w_f * max_j I_fire(i,j) - w_threat * max_j I_fire(j,i)
```

This term is deliberately small relative to combat events. It identifies the transition from maneuvering to an actionable weapon solution and makes holding the solution for the short dwell observable to the critic.

### 7.4 Combat event

```
r_event_i = K / n_attackers,  if i shares a Blue kill
            +D,               if i is destroyed
            0,                otherwise
```

Defaults are `K=+10` and `D=-10`. Altitude destruction uses the same death penalty because it removes the aircraft from combat; no separate boundary shaping is added.

Total reward:

```
r_i = r_progress_i + r_tactical_i + r_fire_i + r_event_i
```

Default dense weights are `w_p=0.03`, `w_t=0.02`, `w_f=w_threat=0.05`, `R_tactical=5000 m`, and `v_close_ref=600 m/s`.

No energy, formation, boundary, center-return, survival-time or artificial mission reward is permitted.

## 8. Initialization distribution

Each reset samples one mode independently. This is a stationary mixture, not curriculum learning.

- `head_on`: opposing headings approximately follow the center line.
- `offset`: both teams enter with crossing/heading offsets, producing nonzero lateral geometry.
- `flank`: one team is approximately crossing relative to the other; which team has the initial flank relation is randomized.

Default probabilities are `0.4/0.4/0.2`. A common random horizontal rotation prevents absolute-map memorization. Team-center separation is sampled from `[6000,8000] m`, speed from `[200,250] m/s`, altitude from `[2500,4000] m`, with bounded within-team perturbations. Every initial pair is outside the maximum firing range.

This distribution represents plausible pre-merge air-combat states without presenting kills at reset or progressively lowering difficulty.

## 9. Blue baseline

Blue selects its nearest alive Red target every step, commands pure pursuit heading and elevation, and commands a fixed cruise speed through the same V2 response layer. It has no look-ahead planner, missile evasion, team assignment, communication or learned behavior.

The fixed policy provides a deterministic, stable baseline while preserving Red's research problem: multi-agent tactical maneuvering against a consistently engaging opponent.

## 10. Episode and instrumentation

- Episode terminates when either team has no survivors.
- Episode truncates at `max_steps` otherwise.
- A simultaneous full-team loss is a draw.
- Required separated diagnostics remain: first attackable step, first completed-lock step, first kill step, kill counts, altitude losses and termination reason.
- V2 adds per-step reward component vectors and current fire-window/lock counts for validation.
- `executed_red_actions` remains the normalized high-level actor action stored in replay. MADSAC interfaces and update logic are unchanged.

## 11. Verification gates

### Unit tests

- Dynamics: trim/command stability and bounded finite integration.
- Action mapping: command endpoints, response direction and control bounds.
- Attack model: head-on exclusion, valid rear/side window, range limits and dwell reset.
- Reward: all components finite, approach sign, tactical sign and event accounting.
- Observation: shape `(4,52)`, finite bounded values, dead masks, rotation/translation invariance.

### Scripted baselines

- Straight/head-on: initial state is not attackable and no immediate kill occurs.
- Maneuver/combat: at least one deterministic trajectory records approach, nontrivial heading maneuver, fire opportunity, completed dwell and kill.

### 24k smoke

The smoke run is diagnostic, not a performance claim. Report:

- reward and component distributions over time;
- Red/Blue attackable episode rate;
- Red/Blue completed-lock episode rate;
- Red/Blue kill episode rate;
- representative trajectory statistics and altitude losses.

If attackable, lock and kill interaction remain identically zero, V2 fails the smoke gate and must not proceed to 500k.
