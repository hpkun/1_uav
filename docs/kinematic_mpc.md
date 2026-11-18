# G4 direct-control MPC experiment

G4 is an isolated A/B controller for the existing kinematic interception
environment. At the user's direction, its reference is the predicted target
**center**, so it is a contact-seeking study. This differs from the attached
proposal's non-contact tracking objective. The original G0–G3 controllers,
assignment, target paths, event logic, and contact threshold are unchanged.

Model A uses CVXPY/Clarabel to optimize a time-varying acceleration sequence
over 2 s. The hard constraints are the 10 m/s² Euclidean acceleration limit
and the 20 m/s speed limit. Model B uses CasADi/IPOPT to optimize a sequence
of tangential accelerations and yaw/flight-path rates over 4 s. Its forward
rollout uses the direct-control execution model at 0.01 s intervals. Both
solvers use causal three-point CV target observations. The first control is
held until the next 0.2 s planning update; the optimizer is reinitialized
from the actual current UAV state. A target-association change invalidates
the held control. Solver failure triggers a bounded model-specific fallback.

The model parameters and objective weights are exploratory. The terminal
tracking objective is the target center, but this is not a terminal collision
optimizer or a proof of reachability. CPA-kNN has not been incorporated into
G4's direct-control optimization; `--avoidance` must remain `none` for G4.

Install optional solver dependencies in the Ubuntu `uav` Conda environment:

```bash
python -m pip install -e '.[mpc]'
```

Run from the repository root after `conda activate uav`:

```bash
python -m kinematic_interception.run --seed 17 --scale 1 --model A --method G4
python -m kinematic_interception.run --seed 17 --scale 20 --model B --method G4
```

The summary records assigned geometric contact rate, breach rate, solver
calls, fallbacks, and mean/P95/P99 solve times. A solve that reaches the IPOPT
iteration limit may supply a finite, input-bounded candidate; it is labeled
`iteration_limit`, never `optimal`. The first candidate action is checked
against the execution model before use. Offline solve times are reported so
that a controller slower than its 0.2 s planning period is not described as
real-time capable.
