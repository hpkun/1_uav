# Li et al. (2023) strict reproduction specification

This is a paper-specification reproduction with explicit assumptions for unpublished implementation details. Configuration, audit output, and runtime construction use `configs/paper_environment.yaml` and `configs/madsac.yaml` as the sources of truth.

## PAPER-SPECIFIED

- Equations (1)-(2) point-mass dynamics use NED coordinates. All models use the paper's `0.1 s` solution interval.
- Table 1 limits: `psi in [-pi,pi]`, `theta in [-pi/3,pi/3]`, `phi in [-pi/2,pi/2]`, and `v in [150,300] m/s`.
- Four homogeneous red UAVs and four homogeneous blue UAVs operate in a 10 km diameter area and initialize on opposite sides of a random diameter.
- Equations (3)-(5) add Gaussian noise to formal sensor observations. Noise being enabled is a paper model feature; only its coefficients are unpublished. `sensor_noise=False` is restricted to deterministic tests.
- Figure 2 / Equation (6) defines horizontal ATA/AA, vertical HA, and HCA. Equations (7)-(8) define automatic launch and probabilistic hit; maximum range is 4 km and ATA/HA limits are `pi/6`.
- Section 2.5 blue control reselects the nearest alive red UAV every solution step.
- Table 2 / Equation (23) actions are `[delta_psi,delta_theta,delta_v]` with ranges `[-pi,pi]`, `[-pi/3,pi/3]`, and `[-50,50]`.
- Equation (24) contains own position/speed/attitude, three friendly states, and four enemy geometry states. Equation (25) is the only reward.
- Aircraft leaving the engagement area are judged dead. Attack deaths and boundary deaths remain distinct. Red wins when every blue UAV is dead and at least one red UAV remains; blue wins symmetrically. Same-step mutual elimination is a draw. Timeout is not a red win. This interpretation is required by Section 4.2 and Table 4's success-rate semantics.
- MADSAC uses a shared stochastic actor, two independent centralized attention critics, target actor and critics, replay, minimum double-Q, maximum entropy, delayed policy update, soft targets, CTDE, and decentralized execution.
- Published hyperparameters: actor `2x256`, critic two heads and two 256-unit layers, Adam `1e-4`, replay `1e6`, batch `1024`, `tau=.001`, `gamma=.99`, `alpha=.1`.
- Formal protocol: 24 distinct parallel training seeds, more than 8M sampled transitions, 20 disjoint testing seeds, five independent runs, and 95% confidence intervals.

## PAPER-DERIVED

- Equation (24) naturally expands to `7 + 3*6 + 4*5 = 45` scalars per UAV.
- A 10 km diameter gives a 5 km horizontal boundary radius.
- Homogeneous parameter sharing means one actor object is applied to all four agents.
- Equation (8) uses the same printed `epsilon_fire` in its ATA and HA inequalities.

## PAPER-UNSPECIFIED reproduction assumptions

### Environment and controller

- Episode horizon 2000 steps. Formation centers are 4000 m from the origin with 150 m tangential spacing, altitude 3000 m, and speed 225 m/s.
- Sensor coefficients: `c1=10`, `c2=.01`, `c3=1`, `b1=b2=b3=3`.
- Weapon coefficients: `D_firemin=0`, `D_hit=2000`, `c4=c5=.05`. There is no ammunition, cooldown, missile entity, or guidance simulation.
- Controller gains, rates, acceleration, `nx`, and `nz` limits are under `reproduction_assumptions.controller`; `AircraftSpec` merges them with the paper flight-state limits at runtime. Action scaling is read from the PAPER-SPECIFIED `action` section.
- Automatic fire and local geometric reward use nearest-alive-enemy. All successful hit proposals are determined from one pre-attack snapshot and apply simultaneously, including mutual hits. Multiple hits on one target create one death and one credited kill.

### Observation scalar encoding

The paper specifies the physical quantities but not their exact scalar encoding. The implementation retains own absolute NED position, uses body-horizontal relative friendly positions, retains friendly global signed `psi/theta`, represents enemy distance plus signed AA/ATA/HA, divides position/distance by 5000 and speed by 300, fixes slots by aircraft ID, and encodes dead slots as zero. These choices are assumptions, not PAPER-SPECIFIED claims.

### Fixed-slot death handling

Each aircraft has an irreversible `NONE -> ATTACK|BOUNDARY` death ledger. Fixed tensor slots carry `alive_masks` and `next_alive_masks`. Dead executed actions and replay actions are zero; dead entities are excluded from attention key/value sets; actor loss, critic loss, entropy, and Q metrics use masked means; next-state entropy/Q bootstrap is multiplied by `next_alive_masks`. A query with no other live entity receives zero attention context without NaN. This is an engineering assumption required because the paper does not publish dead-slot handling.

### Reward timing and aggregation

After integration and boundary resolution, the environment freezes a pre-attack snapshot. R3/R41/R42 target selection and geometry plus all weapon proposals use that snapshot. Simultaneous deaths are then applied; R1 attack events and R2 boundary events are added. Thus a target killed this step remains the current-step reward target, and target switching occurs next step. Four local Eq.(25) rewards are summed and broadcast as the cooperative reward. The nearest target and broadcast-sum rules are unpublished assumptions.

### Network details and scheduling

- Actor activation ReLU, `log_std in [-5,2]`; critic activation LeakyReLU, two-layer embeddings, equal head dimension split, and final MLP layout are unpublished assumptions configured in `madsac.yaml`.
- `policy_delay=2`, `learning_starts=1024`, and `updates_per_transition=1.0` are assumptions. One 24-env vector step creates 24 sampled environment transitions and therefore earns 24 gradient-update credits at ratio 1.0; changing `num_envs` does not change the update/data ratio.
- `sampled_env_steps` counts individual environment transitions. `vector_steps` counts synchronous batches, so under uninterrupted 24-env training `sampled_env_steps=24*vector_steps`.
- Training seeds follow `base_seed + env_id + episode_index*seed_stride`. Twenty fixed evaluation seeds are disjoint and evaluation does not touch training RNG, environments, or replay.
- Independent runs use separate `run_id/seed` directories and initialization/training seeds; the evaluation seed set remains fixed for fair comparison.
- Full resume checkpoints persist networks, optimizers, replay, counters, evaluation history, configuration signature, and NumPy/Torch CPU/CUDA RNG. Physical vector environments are deterministically reset on resume rather than bitwise-restored; this is an explicit engineering assumption.
