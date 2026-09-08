#!/usr/bin/env bash
set -euo pipefail

[[ "${CONDA_DEFAULT_ENV:-}" == "uav" ]] || { echo "Activate conda environment uav first" >&2; exit 1; }
python -c 'import torch; assert torch.cuda.is_available(), "CUDA required"'
python -u tools/preflight_critic_mission_context_factorial.py

mkdir -p outputs/dev_critic_mission_context
MAX_PARALLEL_SEEDS=2

run_one() {
  local cell
  local seed
  local config
  local dir
  cell="$1"
  seed="$2"
  config="$3"
  dir="outputs/dev_critic_mission_context/${cell}_seed${seed}"

  [[ ! -e "$dir" ]] || {
    echo "Refusing non-fresh $dir" >&2
    exit 1
  }

  python -u algorithm/train_modular_mappo.py \
    --env-config configs/persistent_wave_v2_environment.yaml \
    --algorithm-config "$config" \
    --output-dir "$dir" \
    --device cuda \
    --seed "$seed" \
    --num-envs 24 \
    --total-sampled-steps 900000 \
    > "${dir}_nohup.log" 2>&1
}

run_seed() {
  local seed
  seed="$1"
  run_one c0r0 "$seed" configs/dev_c0r0_plain_mappo_900k.yaml
  run_one c0r1 "$seed" configs/dev_c0r1_pbrs_900k.yaml
  run_one c1r0 "$seed" configs/dev_c1r0_critic_context_900k.yaml
  run_one c1r1 "$seed" configs/dev_c1r1_critic_context_pbrs_900k.yaml
}

run_seed_pair() {
  local first_seed
  local second_seed
  local first_pid
  local second_pid
  local first_status
  local second_status
  first_seed="$1"
  second_seed="$2"
  first_status=0
  second_status=0

  run_seed "$first_seed" &
  first_pid=$!
  run_seed "$second_seed" &
  second_pid=$!
  echo "Started seed pipelines ${first_seed} (PID ${first_pid}) and ${second_seed} (PID ${second_pid}); max parallel seeds=${MAX_PARALLEL_SEEDS}"

  wait "$first_pid" || first_status=$?
  wait "$second_pid" || second_status=$?
  if (( first_status != 0 || second_status != 0 )); then
    echo "Parallel seed batch failed: seed ${first_seed} status=${first_status}, seed ${second_seed} status=${second_status}" >&2
    return 1
  fi
}

run_seed_pair 5201 5202
run_seed 5203
