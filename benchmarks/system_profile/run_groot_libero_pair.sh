#!/usr/bin/env bash
set -euo pipefail

shared_root=${SHARED_ROOT:-/home/scratch.asimonelli_wwfo}
repo=${REPO_PATH:-${shared_root}/experiments/lerobot-batched-benchmark}
dataset_source=${DATASET_ROOT:-${shared_root}/dataloading/data/libero_spatial_v21}
environment=${PYTHON_ENV:-${shared_root}/dataloading/lerobot-batched-pr/.venv}
hf_source=${HF_SOURCE:-${shared_root}/dataloading/cache/huggingface}
results_base=${RESULTS_ROOT:-${shared_root}/experiments/lerobot-batched-benchmark-results}

baseline_sha=${BASELINE_SHA:-223a8ad16c52dad961cc1104477ffc3369c5189a}
proposal_sha=${PROPOSAL_SHA:-0aa734f39ccdec8e08f7dd070ca21a90284d47b5}
steps=${STEPS:-600}
warmup_steps=${WARMUP_STEPS:-100}
repeats=${REPEATS:-3}
batch_size=${BATCH_SIZE:-128}
num_workers=${NUM_WORKERS:-15}
telemetry_interval_ms=${TELEMETRY_INTERVAL_MS:-250}

job_id=${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}
results_root=${results_base}/h100-libero/${job_id}
runtime=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/lerobot-batched-${job_id}.XXXXXX")
mkdir -p "${results_root}" "${runtime}/sources" "${runtime}/data" "${runtime}/hf/hub" "${runtime}/outputs"

cleanup() {
  if [[ -n ${gpu_monitor_pid:-} ]]; then
    kill "${gpu_monitor_pid}" 2>/dev/null || true
    wait "${gpu_monitor_pid}" 2>/dev/null || true
  fi
  rm -rf "${runtime}"
}
trap cleanup EXIT

python=${environment}/bin/python
monitor=${repo}/benchmarks/system_profile/monitor_process.py
summarizer=${repo}/benchmarks/system_profile/summarize_run.py

for required in "${repo}/.git" "${dataset_source}" "${python}" "${monitor}" "${summarizer}"; do
  if [[ ! -e ${required} ]]; then
    echo "missing required benchmark input: ${required}" >&2
    exit 2
  fi
done

# Stage immutable inputs once per allocation. Setup time is kept outside every measured run.
cp -a "${dataset_source}" "${runtime}/data/libero_spatial_v21"
for model in models--nvidia--GR00T-N1.7-3B models--nvidia--Cosmos-Reason2-2B; do
  if [[ -d ${hf_source}/hub/${model} ]]; then
    cp -a "${hf_source}/hub/${model}" "${runtime}/hf/hub/${model}"
  fi
done

for entry in "baseline:${baseline_sha}" "proposal:${proposal_sha}"; do
  label=${entry%%:*}
  sha=${entry#*:}
  source_dir=${runtime}/sources/${label}
  mkdir -p "${source_dir}"
  git -C "${repo}" archive "${sha}" | tar -xf - -C "${source_dir}"
done

export HF_HOME=${runtime}/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=offline
export WANDB_SILENT=true
export WANDB_CONSOLE=off
export LD_LIBRARY_PATH=${shared_root}/dataloading/ffmpeg7-x86/lib:/home/tools/cuda/cudatoolkit_12.2.1/lib64:${LD_LIBRARY_PATH:-}
export TRITON_CACHE_DIR=${runtime}/triton
export TORCHINDUCTOR_CACHE_DIR=${runtime}/torchinductor

metadata=${results_root}/system
mkdir -p "${metadata}"
date -u +%FT%TZ >"${metadata}/started_at_utc.txt"
uname -a >"${metadata}/uname.txt"
lscpu >"${metadata}/lscpu.txt"
lsblk -O -J >"${metadata}/lsblk.json" 2>/dev/null || true
mount >"${metadata}/mounts.txt"
df -hT >"${metadata}/filesystems.txt"
nvidia-smi -q >"${metadata}/nvidia-smi-q.txt"
nvidia-smi topo -m >"${metadata}/nvidia-smi-topology.txt" 2>/dev/null || true
"${python}" -m pip freeze >"${metadata}/pip-freeze.txt" 2>/dev/null || true
git -C "${repo}" show --no-patch --format=fuller "${baseline_sha}" >"${metadata}/baseline-commit.txt"
git -C "${repo}" show --no-patch --format=fuller "${proposal_sha}" >"${metadata}/proposal-commit.txt"
cp "${runtime}/data/libero_spatial_v21/meta/info.json" "${metadata}/dataset-info.json"

image_transforms='{
  "brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.7, 1.3]}},
  "contrast":   {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast":   [0.6, 1.4]}},
  "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}},
  "hue":        {"weight": 1.0, "type": "ColorJitter", "kwargs": {"hue":        [-0.08, 0.08]}}
}'

run_one() {
  local implementation=$1
  local repeat=$2
  local sha source_dir label output_dir local_output console gpu_csv
  sha=${baseline_sha}
  if [[ ${implementation} == proposal ]]; then
    sha=${proposal_sha}
  fi
  source_dir=${runtime}/sources/${implementation}
  label=r${repeat}-${implementation}
  output_dir=${results_root}/${label}
  local_output=${runtime}/outputs/${label}
  console=${output_dir}/console.log
  gpu_csv=${output_dir}/gpu.csv
  mkdir -p "${output_dir}" "${local_output}"

  printf '%s\n' "${sha}" >"${output_dir}/git-sha.txt"
  printf '%s\n' "${implementation}" >"${output_dir}/implementation.txt"
  printf '%s\n' "${repeat}" >"${output_dir}/repeat.txt"

  nvidia-smi \
    --query-gpu=timestamp,index,name,uuid,utilization.gpu,utilization.memory,utilization.decoder,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem,pstate \
    --format=csv,noheader,nounits --loop-ms="${telemetry_interval_ms}" >"${gpu_csv}" &
  gpu_monitor_pid=$!

  set +e
  PYTHONPATH="${source_dir}/src" "${python}" "${monitor}" \
    --samples "${output_dir}/system.jsonl" \
    --summary "${output_dir}/process-summary.json" \
    --label "${label}" --interval 0.5 -- \
    "${python}" -m lerobot.scripts.lerobot_train \
      --dataset.repo_id=IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
      --dataset.root="${runtime}/data/libero_spatial_v21" \
      --dataset.revision=main --dataset.video_backend=torchcodec \
      --dataset.image_transforms.enable=true --dataset.image_transforms.max_num_transforms=4 \
      --dataset.image_transforms.tfs="${image_transforms}" \
      --policy.type=groot --policy.device=cuda \
      --policy.base_model_path=nvidia/GR00T-N1.7-3B --policy.embodiment_tag=libero_sim \
      --policy.push_to_hub=false --policy.use_relative_actions=false --policy.max_steps=20000 \
      --batch_size="${batch_size}" --steps="${steps}" --save_checkpoint=false \
      --env_eval_freq=0 --eval_steps=0 --log_freq=1 \
      --num_workers="${num_workers}" --prefetch_factor=1 --persistent_workers=true \
      --dataloader_multiprocessing_context=spawn \
      --wandb.enable=true --wandb.project=lerobot-system-benchmarks \
      --wandb.mode=offline --wandb.disable_artifact=true \
      --output_dir="${local_output}" --job_name="${label}" --seed=42 \
      > >(tee "${console}") 2>&1
  status=$?
  set -e

  kill "${gpu_monitor_pid}" 2>/dev/null || true
  wait "${gpu_monitor_pid}" 2>/dev/null || true
  unset gpu_monitor_pid
  printf '%s\n' "${status}" >"${output_dir}/exit-code.txt"
  if [[ -d ${local_output}/wandb ]]; then
    cp -a "${local_output}/wandb" "${output_dir}/wandb"
  fi

  if [[ -s ${output_dir}/process-summary.json && -s ${output_dir}/system.jsonl && -s ${gpu_csv} ]]; then
    "${python}" "${summarizer}" \
      --console "${console}" --gpu "${gpu_csv}" --system "${output_dir}/system.jsonl" \
      --process-summary "${output_dir}/process-summary.json" \
      --warmup-steps "${warmup_steps}" --output "${output_dir}/summary.json" || true
  fi
  if [[ ${status} -ne 0 ]]; then
    return "${status}"
  fi
}

# AB/BA/AB balances cache and order effects across the default three repeats.
orders=("baseline proposal" "proposal baseline" "baseline proposal")
for ((repeat = 0; repeat < repeats; repeat++)); do
  order=${orders[repeat % ${#orders[@]}]}
  for implementation in ${order}; do
    run_one "${implementation}" "${repeat}"
  done
done

date -u +%FT%TZ >"${metadata}/finished_at_utc.txt"
