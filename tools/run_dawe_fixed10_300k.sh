#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == "uav" ]] || { echo "ERROR: activate conda environment uav first" >&2; exit 2; }
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
python -u tools/preflight_dawe_fixed10_300k.py
runtime_sha(){ python -c 'from algorithm.common.protocol import runtime_source_manifest;from pathlib import Path;print(runtime_source_manifest(Path.cwd())["runtime_source_manifest_sha256"])'; }
LOCK="$(runtime_sha)"
for seed in 5301 5302 5303; do
  source="outputs/diag_mappo_learnability/l3_seed${seed}/checkpoint_1505280.pt"
  source_sha="$(sha256sum "$source" | awk '{print $1}')"
  for variant in control v1; do
    [[ "$(runtime_sha)" == "$LOCK" ]] || { echo "ERROR: runtime source changed" >&2; exit 2; }
    if [[ "$variant" == "control" ]]; then
      config="configs/dev_dawe_fixed10_control_300k.yaml"; output="outputs/dev_dawe_control_seed${seed}_300k"
    else
      config="configs/dev_dawe_fixed10_v1_300k.yaml"; output="outputs/dev_dawe_v1_seed${seed}_300k"
    fi
    [[ ! -e "$output" ]] || { echo "ERROR: refusing existing output $output" >&2; exit 2; }
    [[ "$(sha256sum "$source" | awk '{print $1}')" == "$source_sha" ]] || { echo "ERROR: source checkpoint changed" >&2; exit 2; }
    echo "===== DAWE SCREEN START seed=${seed} variant=${variant} output=${output} ====="
    python -u algorithm/train_modular_mappo.py --env-config configs/persistent_wave_v2_environment.yaml \
      --algorithm-config "$config" --branch-from "$source" --output-dir "$output" --device cuda \
      --seed "$seed" --num-envs 24 --total-sampled-steps 1805280
    [[ "$(sha256sum "$source" | awk '{print $1}')" == "$source_sha" ]] || { echo "ERROR: source checkpoint changed after run" >&2; exit 2; }
    echo "===== DAWE SCREEN DONE seed=${seed} variant=${variant} output=${output} ====="
  done
done
