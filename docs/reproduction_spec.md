# Li et al. (2023) reproduction specification

This project is a paper-specification reproduction with explicitly documented assumptions for unpublished implementation details. The source paper is *Multi-UAV Cooperative Air Combat Decision-Making Based on Multi-Agent Double-Soft Actor-Critic* (Li et al., 2023).

## PAPER-SPECIFIED

- Point-mass dynamics are Equations (1)-(2), in NED coordinates.
- `psi` is in `[-pi, pi]`, `theta` in `[-pi/3, pi/3]`, `phi` in `[-pi/2, pi/2]`, and speed in `[150, 300] m/s` (Table 1).
- Four homogeneous red UAVs face four homogeneous blue UAVs in a circular 10 km diameter engagement area. Opposing formations initialize on opposite sides of a randomly selected diameter.
- Blue uses Section 2.5 nearest-alive-target pursuit, reselecting every simulation step.
- Sensor observations implement Equations (3)-(5), including a shared clipped Gaussian draw for all three position components and another shared draw for all three attitude components.
- Geometry implements Figure 2 / Equation (6): horizontal ATA and AA, vertical HA, and horizontal crossing angle HCA.
- Weapon launch implements Equation (7), with ATA and HA limits `pi/6` and maximum distance 4 km. Hit sampling implements both Equation (8) inequalities using the same `epsilon_fire ~ N(0,1)`.
- Normalized actor outputs map to physical `[delta_psi, delta_theta, delta_v]` ranges `[-pi,pi]`, `[-pi/3,pi/3]`, and `[-50,50]`, followed by Equation (23).
- Reward terms and all 30/15/5 degree and 4000 m boundaries implement Equation (25), without additional shaping.
- MADSAC uses a shared stochastic actor, two independent centralized attention critics, target actor, two target critics, replay, clipped double-Q, maximum entropy, delayed policy updates, soft target updates, CTDE, and decentralized execution.
- Actor: two 256-unit hidden layers. Critic: two attention heads and two 256-unit hidden layers. Adam LR `1e-4`; replay `1e6`; batch `1024`; `tau=0.001`; `gamma=0.99`; `alpha=0.1`.
- Formal protocol: 24 distinct training seeds, more than 8M sampled steps, 20 disjoint test seeds, five independent training runs, and a 95% confidence interval.

## PAPER-DERIVED

- Equation (24) naturally expands to 45 values per red UAV: self `7`, three friendly slots `3*6=18`, and four enemy slots `4*5=20`.
- A 10 km diameter gives a 5 km horizontal boundary radius.
- Homogeneous parameter sharing means one actor instance is applied to all four red observations.
- Equation (8)'s typesetting and text imply one `epsilon_fire` draw is shared by its ATA and HA inequalities.

## PAPER-UNSPECIFIED / reproduction assumptions

All values live under `reproduction_assumptions` in the YAML files.

- Integration step `dt=0.1 s`; episode horizon 2000 steps; controller gains `k_yaw=k_pitch=k_speed=1`; rate/acceleration and `nx/nz` limits are listed in `paper_environment.yaml`.
- Formation centers are 4000 m from the origin, with four symmetric 150 m tangential offsets, altitude 3000 m, and speed 225 m/s.
- Sensor noise: `c1=10 m`, `c2=0.01 rad`, `c3=1 m/s`, `b1=b2=b3=3`.
- Weapon: `D_firemin=0 m`, `D_hit=2000 m`, `c4=c5=0.05 rad`. There is no ammunition, cooldown, missile entity, or guidance simulation.
- The nearest alive enemy is the automatic-fire and local reward geometry target. Simultaneous fire is sampled first and then applied in stable team/ID order.
- Own absolute position is retained for boundary observability. Positions/distances divide by 5000 and speed by 300; relative friendly positions rotate into the observing UAV's body-horizontal frame. Dead slots are zero.
- Four local Equation (25) rewards are summed and the cooperative team value is broadcast to all red agents. When attack and threat conditions both hold, their R41 and R42 contributions are summed.
- Delayed policy frequency `d=2`, learning starts at 1024 transitions, and one gradient update is performed per environment step.

## Outcome accounting

A win requires four red attack kills. Boundary losses are deaths but never attack kills. `environment_outcome`, attack kills, and boundary losses are logged separately. Timeout or any outcome without four red attack kills is mission failure.
