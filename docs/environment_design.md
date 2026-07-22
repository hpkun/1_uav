# Environment Design

`core` contains unit-aware state, control, constants, and geometry primitives. `actions` maps the fixed discrete action IDs to controls. `dynamics` consumes only a six-dimensional kinematic vector and a control command, keeping rewards, combat, and agent policy out of the physics kernel.

`entities` holds state/profile associations. Future `combat`, `observations`, `rewards`, and `opponents` modules depend on core/entities but do not change the dynamics contract. `envs` will later compose those modules behind Gymnasium-style `reset()` and `step()` methods. `utils` provides configuration and validation helpers.

The current dependency direction is intentionally one-way: low-level numerical modules do not import environment classes or learning frameworks.
