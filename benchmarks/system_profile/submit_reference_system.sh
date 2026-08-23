#!/usr/bin/env bash
set -euo pipefail

: "${SHARED_ROOT:?Set SHARED_ROOT}"
: "${PARTITION:?Set PARTITION}"
: "${ACCOUNT:?Set ACCOUNT}"
: "${SYSTEM_LABEL:?Set SYSTEM_LABEL}"

repo=${REPO_PATH:-${SHARED_ROOT}/experiments/lerobot-batched-benchmark}
results=${RESULTS_ROOT:-${SHARED_ROOT}/experiments/lerobot-batched-benchmark-results-reference-4w}
targets=${TARGETS:-"dataloading training"}
datasets=${DATASETS:-"libero droid"}
models=${MODELS:-"groot diffusion smolvla"}

for target in ${targets}; do
  for dataset in ${datasets}; do
    for model in ${models}; do
      if [[ ${target} == dataloading ]]; then
        steps=${LOADER_STEPS:-300}
        warmup=${LOADER_WARMUP_STEPS:-50}
      else
        steps=${TRAINING_STEPS:-600}
        warmup=${TRAINING_WARMUP_STEPS:-100}
      fi
      job_id=$(
        SHARED_ROOT="${SHARED_ROOT}" REPO_PATH="${repo}" RESULTS_ROOT="${results}" \
          TARGET="${target}" PARTITION="${PARTITION}" ACCOUNT="${ACCOUNT}" QOS="${QOS:-batch}" \
          GPU_GRES="${GPU_GRES:-gpu:1}" SYSTEM_LABEL="${SYSTEM_LABEL}" \
          DATASET_PROFILE="${dataset}" MODEL_PROFILE="${model}" \
          STEPS="${steps}" WARMUP_STEPS="${warmup}" REPEATS=3 \
          BATCH_SIZE=32 NUM_WORKERS=4 WORKER_COUNTS=4 PREFETCH_FACTOR=4 \
          CPUS_PER_TASK="${CPUS_PER_TASK:-32}" MEMORY="${MEMORY:-120G}" \
          TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}" \
          "${repo}/benchmarks/system_profile/submit_matrix_job.sh"
      )
      printf '%s,%s,%s,%s,%s\n' "${job_id}" "${SYSTEM_LABEL}" "${target}" "${dataset}" "${model}"
    done
  done
done
