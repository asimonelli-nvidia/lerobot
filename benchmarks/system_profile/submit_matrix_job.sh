#!/usr/bin/env bash
set -euo pipefail

: "${SHARED_ROOT:?Set SHARED_ROOT}"
: "${PARTITION:?Set PARTITION}"
: "${ACCOUNT:?Set ACCOUNT}"
: "${TARGET:?Set TARGET to dataloading or training}"
repo=${REPO_PATH:-${SHARED_ROOT}/experiments/lerobot-batched-benchmark}
results=${RESULTS_ROOT:-${SHARED_ROOT}/experiments/lerobot-batched-benchmark-results-v2}
target=${TARGET}
exclusive_args=()
if [[ ${EXCLUSIVE_NODE:-false} == true ]]; then
  exclusive_args+=(--exclusive)
fi
dependency_args=()
if [[ -n ${DEPENDENCY:-} ]]; then
  dependency_args+=(--dependency="${DEPENDENCY}")
fi
resource_args=(--gres="${GPU_GRES:-gpu:1}")
if [[ -n ${CONSTRAINT:-} ]]; then
  resource_args+=(--constraint="${CONSTRAINT}")
fi
case ${target} in
  dataloading) entrypoint=${repo}/benchmarks/system_profile/run_dataloader_pair.sh ;;
  training) entrypoint=${repo}/benchmarks/system_profile/run_groot_libero_pair.sh ;;
  *) echo "unsupported TARGET: ${target}" >&2; exit 2 ;;
esac

mkdir -p "${results}/slurm"
steps=${STEPS:-${LOADER_STEPS:-300}}
warmup_steps=${WARMUP_STEPS:-${LOADER_WARMUP_STEPS:-50}}
repeats=${REPEATS:-${LOADER_REPEATS:-3}}
worker_counts=${WORKER_COUNTS:-${LOADER_WORKERS:-4}}

sbatch --parsable \
  --account="${ACCOUNT}" --partition="${PARTITION}" --qos="${QOS:-batch}" \
  --job-name="${target}-${MODEL_PROFILE:-groot}-${DATASET_PROFILE:-libero}-${SYSTEM_LABEL:-system}" \
  --time="${TIME_LIMIT:-1-00:00:00}" --nodes=1 --ntasks=1 "${resource_args[@]}" \
  --cpus-per-task="${CPUS_PER_TASK:-32}" --mem="${MEMORY:-120G}" "${exclusive_args[@]}" "${dependency_args[@]}" \
  --output="${results}/slurm/slurm-%j.out" \
  --export="ALL,SHARED_ROOT=${SHARED_ROOT},REPO_PATH=${repo},RESULTS_ROOT=${results},SYSTEM_LABEL=${SYSTEM_LABEL:-system},DATASET_PROFILE=${DATASET_PROFILE:-libero},MODEL_PROFILE=${MODEL_PROFILE:-groot},STEPS=${steps},WARMUP_STEPS=${warmup_steps},REPEATS=${repeats},BATCH_SIZE=${BATCH_SIZE:-},NUM_WORKERS=${NUM_WORKERS:-4},PREFETCH_FACTOR=${PREFETCH_FACTOR:-4},WORKER_COUNTS=${worker_counts}" \
  "${entrypoint}"
