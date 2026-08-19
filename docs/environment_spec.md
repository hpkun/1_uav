# Lightweight translation-invariant 3D multi-UAV WVR benchmark

This repository implements an independent, deliberately lightweight academic
benchmark. It is not an operational flight or weapon model. The intended
process is canonical head-on initialization, free 3D maneuver, rear/side
geometry acquisition, weapon-envelope entry, three consecutive lock steps,
and simultaneous combat resolution.

## Dynamics and maneuver action

The NED state is `[x,y,z,v,theta,psi]`; altitude is `-z`. Fixed-step RK4 uses
`dt=0.1 s` and the unchanged point-mass equations:

```text
x_dot     = v cos(theta) cos(psi)
y_dot     = v cos(theta) sin(psi)
z_dot     = -v sin(theta)
v_dot     = g (nx - sin(theta))
theta_dot = g/v (nz cos(phi) - cos(theta))
psi_dot   = g nz sin(phi)/(v cos(theta))
```

Speed is limited to `150..300 m/s` and pitch to `±60 deg`. A normalized action
`[a0,a1,a2]` maps to:

```text
phi     = (pi/3) a2
nx      = 2 a0
nz_trim = cos(theta)/cos(phi)
nz      = nz_trim + 2 a1
```

Hence `theta_dot=2g/v a1 cos(phi)` and `a1=0` is instantaneous flight-path
trim at every legal bank. This is a state-dependent maneuver parameterization,
not a PID, autopilot, or hidden altitude/heading controller.

## Scenario and horizontal plane

Both homogeneous teams contain four UAVs in line-abreast formation. Their
centers start 6,000 m apart at altitude 3,000 m and nominal speed 225 m/s.
Independent perturbations are heading `±5 deg`, speed `±10 m/s`, and altitude
`±100 m`.

The horizontal plane is unbounded: `x,y ∈ R`. There is no active horizontal
radius, horizontal death, return-to-center rule, wrapping, reflection,
teleportation, or position clipping. Episodes remain finite through the
1,000-step (`100 s`) horizon. The global horizontal origin is only a convenient
initialization coordinate reference and has no gameplay meaning.

The only spatial flight envelope is altitude `500..8,000 m`. Crossing either
altitude limit destroys that UAV once and applies the existing `-10` own-death
event. There is no vertical soft zone, altitude potential, or automatic safety
controller.

## 52-dimensional observation

Each learned Red UAV observes three self features followed by seven features
for each of three fixed-ID allies and four fixed-ID enemies:

```text
3 + 7*(3+4) = 52.
```

Self features are normalized speed, pitch, and altitude. Each other-aircraft
slot contains relative position `(forward,right,up)`, relative velocity in the
same local flight-path frame, and an alive mask. Dead slots are zero. Position
uses a numerical scale of 10,000 m and velocity 600 m/s; values are not clipped.
The 10 km scale is neither a sensor range nor a map boundary. Because all
horizontal features are relative, observations are invariant to a common
horizontal translation and rotation.

## Engagement geometry and lock

For displacement from attacker to target, attack angle is measured against the
attacker velocity and escape angle against the target velocity. A target is
attackable iff distance is at most 1,500 m, attack angle at most 45 degrees,
and escape angle at most 90 degrees. Lock must remain valid for three
consecutive steps. Both teams resolve proposals from one pre-hit snapshot, so
mutual destruction is possible. Multiple attackers split one `+10` kill event.

## Tactical progress shaping

For an ordered attacker-target pair:

```text
range  = clip(1-distance/8000, 0, 1)
attack = (1+cos(attack_angle))/2
escape = (1+cos(escape_angle))/2
score  = range*attack*escape
```

For Red UAV `i`, `Phi_i` is its best attack score minus the strongest opponent
threat score. Dense progress shaping is exactly:

```text
r_shape = Phi_i(next)-Phi_i(current)
r_i     = r_event+r_shape
```

This is described as tactical potential-difference progress shaping, not as a
claim of classical policy-invariant potential shaping: the algorithm uses
`gamma != 1`, while the implemented difference contains no gamma multiplier.
Events are only split `+10` Blue kills and `-10` own destruction.

## Fixed opponent, termination, and metrics

Blue is a deterministic nearest-alive-target pure-pursuit opponent. It commands
LOS heading/elevation and desired speed 260 m/s through the same public
`action_toward()` helper and canonical action mapping as Red/scripted baselines.
It has no target assignment optimizer, formation logic, map management,
vertical safety clamp, or tactical FSM.

Termination is Red all dead, Blue all dead, mutual destruction, or the
1,000-step timeout. Validation reports Red/Blue-separated attackable/lock/kill reachability,
altitude loss causes, outcomes, episode length, nearest-enemy distance, and
horizontal pair spread. Large horizontal separation is diagnostic information,
not a failure against an implicit map radius.
