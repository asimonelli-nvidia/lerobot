#!/usr/bin/env bash
set -euo pipefail

shared_root=${SHARED_ROOT:-/home/scratch.asimonelli_wwfo}
repo=${REPO_PATH:-${shared_root}/experiments/lerobot-batched-benchmark}
results=${RESULTS_ROOT:-${shared_root}/experiments/lerobot-batched-benchmark-results}
partition=${PARTITION:-h100-80gb-hbm3@ts6/mg62g4100/1gpu-32cpu-256gb}
account=${ACCOUNT:-wwfo-emea_h100-80gb-hbm3}
qos=${QOS:-batch}
time_limit=${TIME_LIMIT:-1-00:00:00}
cpus_per_task=${CPUS_PER_TASK:-32}
memory=${MEMORY:-200G}
exclusive=${EXCLUSIVE:-1}
system_label=${SYSTEM_LABEL:-h100-pcie}
dataset_profile=${DATASET_PROFILE:-libero}

exclusive_args=()
if [[ ${exclusive} == 1 ]]; then
  exclusive_args+=(--exclusive)
fi

mkdir -p "${results}/slurm"
sbatch --parsable \
  --account="${account}" \
  --partition="${partition}" \
  --qos="${qos}" \
  --job-name="groot-${dataset_profile}-${system_label}" \
  --time="${time_limit}" \
  --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task="${cpus_per_task}" --mem="${memory}" "${exclusive_args[@]}" \
  --output="${results}/slurm/slurm-%j.out" \
  --export="ALL,SHARED_ROOT=${shared_root},REPO_PATH=${repo},RESULTS_ROOT=${results},SYSTEM_LABEL=${system_label},DATASET_PROFILE=${dataset_profile},STEPS=${STEPS:-600},WARMUP_STEPS=${WARMUP_STEPS:-100},REPEATS=${REPEATS:-3},BATCH_SIZE=${BATCH_SIZE:-},NUM_WORKERS=${NUM_WORKERS:-15}" \
  "${repo}/benchmarks/system_profile/run_groot_libero_pair.sh"
