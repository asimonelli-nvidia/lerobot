#!/usr/bin/env bash
set -euo pipefail

: "${SHARED_ROOT:?Set SHARED_ROOT}"
: "${PARTITION:?Set PARTITION}"
: "${ACCOUNT:?Set ACCOUNT}"
: "${TARGET:?Set TARGET to dataloading or training}"
repo=${REPO_PATH:-${SHARED_ROOT}/experiments/lerobot-batched-benchmark}
results=${RESULTS_ROOT:-${SHARED_ROOT}/experiments/lerobot-batched-benchmark-results-v2}
target=${TARGET}
case ${target} in
  dataloading) entrypoint=${repo}/benchmarks/system_profile/run_dataloader_pair.sh ;;
  training) entrypoint=${repo}/benchmarks/system_profile/run_groot_libero_pair.sh ;;
  *) echo "unsupported TARGET: ${target}" >&2; exit 2 ;;
esac

mkdir -p "${results}/slurm"
sbatch --parsable \
  --account="${ACCOUNT}" --partition="${PARTITION}" --qos="${QOS:-batch}" \
  --job-name="${target}-${MODEL_PROFILE:-groot}-${DATASET_PROFILE:-libero}-${SYSTEM_LABEL:-system}" \
  --time="${TIME_LIMIT:-1-00:00:00}" --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task="${CPUS_PER_TASK:-32}" --mem="${MEMORY:-120G}" --exclusive \
  --output="${results}/slurm/slurm-%j.out" \
  --export="ALL,SHARED_ROOT=${SHARED_ROOT},REPO_PATH=${repo},RESULTS_ROOT=${results},SYSTEM_LABEL=${SYSTEM_LABEL:-system},DATASET_PROFILE=${DATASET_PROFILE:-libero},MODEL_PROFILE=${MODEL_PROFILE:-groot},STEPS=${STEPS:-300},WARMUP_STEPS=${WARMUP_STEPS:-50},REPEATS=${REPEATS:-3},BATCH_SIZE=${BATCH_SIZE:-128},NUM_WORKERS=${NUM_WORKERS:-15},WORKER_COUNTS=${WORKER_COUNTS:-0:4:8:15}" \
  "${entrypoint}"
