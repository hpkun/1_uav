# Public 4v4 low-fidelity combat environment

This repository uses an independent, deliberately low-fidelity academic
benchmark. It is not an exact simulator reconstruction and must not be used as
a flight-dynamics, weapon-effectiveness, or operational model. Every active
environment parameter is public in `configs/combat_environment.yaml`.

## Coordinates, state, and integration

The world uses NED coordinates: `x` north, `y` east, and `z` down. Each UAV has
state `[x, y, z, v, theta, psi]`. Controls are tangential load `nx`, normal load
`nz`, and roll angle `phi`. The continuous equations are

```text
x_dot     = v cos(theta) cos(psi)
y_dot     = v cos(theta) sin(psi)
z_dot     = -v sin(theta)
v_dot     = g (nx - sin(theta))
theta_dot = g/v (nz cos(phi) - cos(theta))
psi_dot   = g nz sin(phi) / (v cos(theta))
```

They are integrated by fixed-step RK4 with `dt=0.1 s`. Speed is clipped to
`[150,300] m/s`; pitch is clipped to `[-60,+60] deg`. The V1.2 normalized
action is clipped to `[-1,1]^3` and represents tangential acceleration, a
trim-relative vertical maneuver command, and bank:

```text
phi     = phi_max a2                  phi_max = pi/3
nx      = nx_scale a0                 nx_scale = 2
nz_trim = cos(theta)/cos(phi)
nz      = nz_trim + k_n a1            k_n = 2
```

The cosine division has a small numerical epsilon, although the active
`|phi| <= 60 deg` range ensures `cos(phi) >= 0.5`. Substitution gives

```text
theta_dot = g/v ([cos(theta)/cos(phi) + 2 a1] cos(phi) - cos(theta))
          = 2g/v a1 cos(phi).
```

Thus `a1=0` is instantaneous flight-path trim at every legal pitch and bank;
positive `a1` creates a climb tendency and negative `a1` a dive tendency. This
is a transparent state-dependent action reparameterization, not pitch,
heading, or altitude tracking. It contains no PID, integral/derivative term,
hidden autopilot, or automatic Red safety controller. The actor still supplies
a continuous 3D normalized action whose resulting `nx,nz,phi` enter the
unchanged dynamics. The design avoids artificial downward exploration bias
from independently perturbing bank and absolute normal load.

## Arena and initialization

The V1.2 arena is a vertical cylinder of horizontal radius `15,000 m`, altitude
`500..8,000 m`, and horizon `1,000` steps. Leaving either spatial bound destroys
the UAV. Timeout with survivors is a draw.

At reset, a common horizontal radial angle is uniform on `[-pi,pi]`. Red and
Blue formation centers lie at `-3,000 m` and `+3,000 m` on that radial axis.
Both teams use line-abreast offsets `[-450,-150,150,450] m`, altitude `3,000 m`,
speed `225 m/s`, and headings toward the opposing center. Each UAV independently
receives only heading noise `U(-5,+5) deg`, speed noise `U(-10,+10) m/s`, and
altitude noise `U(-100,+100) m`.

## Engagement and lock

For attacker `i`, target `j`, let `D=p_j-p_i`. Distance is `||D||`; attack angle
is the angle between attacker velocity and `D`; escape angle is the angle
between target velocity and `D`. A target is attackable iff distance is at most
`1,500 m`, attack angle at most `45 deg`, and escape angle at most `90 deg`.

Each attacker maintains one target and a consecutive lock counter. An existing
alive, attackable lock is retained. Otherwise the counter resets and the target
with highest engagement score is selected; ties prefer shorter distance and
then lower target ID. A kill proposal is produced on the third consecutive lock
step (`0.3 s` at `dt=0.1 s`). Both teams' proposals use one pre-hit snapshot and are applied
simultaneously. Duplicate proposals kill a target once; its `+10` kill reward is
split equally among all proposing attackers.

## Observation

Each Red UAV receives 54 values: five self features followed by seven features
for each of three fixed-ID allies and four fixed-ID enemies. Self features are
normalized speed, pitch, altitude, and the two horizontal components of the
vector to arena center in the own-heading frame. Each other-aircraft slot is
relative position (forward/right/up), relative velocity (forward/right/up), and
an alive mask. Horizontal position uses scale `20,000 m`, vertical position
`7,500 m`, relative velocity `600 m/s`, and center vector `10,000 m`. Dead slots
have zero kinematics and mask zero. No sensor, detection, weapon angle, or range
feature is present.

## Reward, policy, and outcomes

For one ordered attacker-target pair,

```text
range  = clip(1 - distance/8000, 0, 1)
attack = (1 + cos(attack_angle))/2
escape = (1 + cos(escape_angle))/2
score  = range * attack * escape
```

Each Red UAV's potential is its best Red-to-Blue score minus the best
Blue-to-that-Red score. Dense shaping is exactly `potential(next)-potential(now)`.
The only events are `+10` per destroyed Blue target (split across contributors)
and `-10` for own destruction, including boundary destruction, once.

The tactical distance scale is a fixed `8,000 m` and is deliberately independent
of the arena radius. Shaping weight remains `1.0`.

Blue uses boundary-aware nearest-alive-target pursuit with desired speed
`260 m/s`, heading gain `1.5`, and physical pitch-load gain `4.0`, through the
same public action mapping as Red. Desired extra load is
`4(desired_elevation-theta)` and is normalized by `k_n=2`. Inside `0.65` of the
horizontal arena radius it is pure pursuit.
Outside that threshold it blends normalized target and center directions, with
the center direction dominating at the boundary. Within `500 m` of the lower or
upper altitude bound, desired elevation is prevented from pointing farther out
of bounds. This recovery is a benchmark-opponent safety rule, not an expert
combat tactic, altitude controller, or maneuver FSM. Learned Red receives no
automatic safety correction. Outcomes are `red_win`, `blue_win`,
`draw_mutual_destruction`, or `draw_timeout`.

## Validation

Run `pytest -q` for deterministic unit tests and
`python scripts/validate_combat_environment.py` for 1,000 reset seeds, five-seed
fresh-actor and uniform-random stability regressions, boundary-aware
rule-vs-rule, a validation-only 50-step symmetric flank-then-pursuit baseline,
and a deterministic merge-then-turn tail-acquisition diagnostic. The validator
reports separate horizontal/low-altitude/high-altitude losses,
attackable/lock/kill timing, combat and boundary death fractions, and Red
shaping/event reward statistics.
