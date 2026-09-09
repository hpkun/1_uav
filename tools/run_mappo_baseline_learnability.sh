#!/usr/bin/env bash
set -euo pipefail

[[ "${CONDA_DEFAULT_ENV:-}" == "uav" ]] || { echo "Activate conda environment uav first" >&2; exit 1; }
python -c 'import torch; assert torch.cuda.is_available(), "CUDA required"'
python -u tools/preflight_mappo_baseline_learnability.py --launch-check

mkdir -p outputs/diag_mappo_learnability

run_one() {
  local condition="$1" seed="$2" env_config="$3"
  local output="outputs/diag_mappo_learnability/${condition}_seed${seed}"
  [[ ! -e "$output" ]] || { echo "Refusing non-fresh $output" >&2; exit 1; }
  python -u algorithm/train_modular_mappo.py \
    --env-config "$env_config" \
    --algorithm-config configs/diag_mappo_learnability_common_3m.yaml \
    --output-dir "$output" \
    --device cuda --seed "$seed" --num-envs 24 --total-sampled-steps 3000000 \
    > "${output}_nohup.log" 2>&1
}

for seed in 5301 5302 5303; do
  run_one l1 "$seed" configs/diag_learnability_1wave_environment.yaml
  run_one l2 "$seed" configs/diag_learnability_2wave_environment.yaml
  run_one l3 "$seed" configs/persistent_wave_v2_environment.yaml
done
