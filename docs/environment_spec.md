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
`[150,300] m/s`; pitch is clipped to `[-60,+60] deg`. The normalized action is
clipped to `[-1,1]^3` and maps directly to
`nx=2 a0`, `nz=1+4 a1`, `phi=60 deg a2`. Thus zero action is level-flight trim
at zero pitch.

## Arena and initialization

The arena is a vertical cylinder of horizontal radius `10,000 m`, altitude
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
`1,500 m`, attack angle at most `30 deg`, and escape angle at most `60 deg`.

Each attacker maintains one target and a consecutive lock counter. An existing
alive, attackable lock is retained. Otherwise the counter resets and the target
with highest engagement score is selected; ties prefer shorter distance and
then lower target ID. A kill proposal is produced on the third consecutive lock
step. Both teams' proposals use one pre-hit snapshot and are applied
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
range  = clip(1 - distance/(2*10000), 0, 1)
attack = (1 + cos(attack_angle))/2
escape = (1 + cos(escape_angle))/2
score  = range * attack * escape
```

Each Red UAV's potential is its best Red-to-Blue score minus the best
Blue-to-that-Red score. Dense shaping is exactly `potential(next)-potential(now)`.
The only events are `+10` per destroyed Blue target (split across contributors)
and `-10` for own destruction, including boundary destruction, once.

Blue uses nearest-alive-target pure pursuit with desired speed `225 m/s`,
heading gain `1.5`, and elevation gain `4.0`, through the same public action
mapping. Outcomes are `red_win`, `blue_win`, `draw_mutual_destruction`, or
`draw_timeout`.

## Validation

Run `pytest -q` for deterministic unit tests and
`python scripts/validate_combat_environment.py` for reset statistics plus
straight-vs-straight and pursuit-vs-pursuit scenario diagnostics.
