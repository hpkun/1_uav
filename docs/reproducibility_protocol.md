# Reproducibility protocol

Each run fixes Python, NumPy, Torch, CUDA, and environment RNGs. Vector environment seeds are `training_seed+env_index`. The Trainer owns an independent NumPy minibatch generator so separate Runners cannot perturb one another through global NumPy shuffling; its bit-generator state is checkpointed.

Evaluation uses a documented independent contiguous seed range shared by every policy/checkpoint. Multi-seed claims report every seed plus overall mean, sample standard deviation, and 95% confidence interval. Binary baseline rates use Wilson intervals; continuous baseline metrics use normal-approximation intervals. No single episode or best seed establishes learnability or convergence.

Comparisons must match scenario, opponent, seed, maximum steps, normalization, terminal reward profile, and return semantics. Initial weights are evaluated alongside last and best. Resume must reproduce the uninterrupted rollout/update branch, including ValueNormalizer and minibatch ordering.
