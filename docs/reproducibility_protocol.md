# Reproducibility protocol

Each run fixes Python, NumPy, Torch, CUDA, and environment RNGs. Vector environment seeds are `training_seed+env_index`. The Trainer owns an independent NumPy minibatch generator so separate Runners cannot perturb one another through global NumPy shuffling; its bit-generator state is checkpointed.

Validation and final test use documented, non-overlapping contiguous seed ranges. Validation selects best checkpoints; final test reports initial/last/best. Multi-training-seed claims report every seed plus mean, sample standard deviation, and an untrimmed Student-t 95% interval. Binary per-episode baseline rates use Wilson intervals; continuous per-episode baseline metrics use normal-approximation intervals. No single episode or best seed establishes learnability or convergence.

Comparisons must match scenario, opponent, seed, maximum steps, normalization, terminal reward profile, and return semantics. Initial weights are evaluated alongside last and best. Resume must reproduce the uninterrupted rollout/update branch, including ValueNormalizer and minibatch ordering.
