#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_CONFIG="configs/persistent_wave_v2_environment.yaml"
DIRECT_BEST="outputs/d999_seed2023/best_eval.pt"

run_one() {
  local name="$1"
  shift
  mkdir -p "outputs/${name}"
  "$PYTHON_BIN" algorithm/train_modular_mappo.py \
    --device cuda --seed 2023 --num-envs 24 --total-sampled-steps 1500000 \
    --env-config "$ENV_CONFIG" --output-dir "outputs/${name}" "$@" \
    > "outputs/${name}/train.log" 2>&1
}

run_one pw_m6_warm_start_1p5m_seed2023 \
  --algorithm-config configs/pw_m6_warm_start.yaml \
  --warm-start-checkpoint "$DIRECT_BEST"
run_one pw_m5_wave_balance_1p5m_seed2023 \
  --algorithm-config configs/pw_m5_wave_balance.yaml
run_one pw_m1_wave_context_1p5m_seed2023 \
  --algorithm-config configs/pw_m1_wave_context.yaml
run_one pw_m3_popart_1p5m_seed2023 \
  --algorithm-config configs/pw_m3_popart.yaml

# Enable only after selecting the intended regression reference checkpoint.
# run_one pw_m8_policy_anchor_1p5m_seed2023 \
#   --algorithm-config configs/pw_m8_policy_anchor.yaml \
#   --reference-checkpoint "$DIRECT_BEST"
