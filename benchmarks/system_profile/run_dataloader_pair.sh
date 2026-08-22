#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

: "${SHARED_ROOT:?Set SHARED_ROOT to the shared benchmark workspace}"
shared_root=${SHARED_ROOT}
repo=${REPO_PATH:-${shared_root}/experiments/lerobot-batched-benchmark}
environment=${PYTHON_ENV:-${shared_root}/dataloading/lerobot-batched-pr/.venv}
model_source=${MODEL_SOURCE:-${shared_root}/v2d/artifacts/models/GR00T-N1.7-3B}
results_base=${RESULTS_ROOT:-${shared_root}/experiments/lerobot-batched-benchmark-results-v2}
system_label=${SYSTEM_LABEL:-unknown-system}
dataset_profile=${DATASET_PROFILE:-libero}
model_profile=${MODEL_PROFILE:-groot}

case ${dataset_profile} in
  libero)
    dataset_source=${DATASET_ROOT:-${shared_root}/dataloading/data/libero_spatial_v21}
    dataset_staged_name=libero_spatial_v21
    dataset_repo_id=IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot
    dataset_label=libero-spatial
    dataset_subset=all-432-episodes
    ;;
  droid)
    dataset_source=${DATASET_ROOT:-${shared_root}/dataloading/data/droid_1.0.1_first100}
    dataset_staged_name=droid_1.0.1_first100
    dataset_repo_id=lerobot/droid_1.0.1
    dataset_label=droid
    dataset_subset=episodes-0-99
    ;;
  *) echo "unsupported DATASET_PROFILE: ${dataset_profile}" >&2; exit 2 ;;
esac

case ${model_profile} in
  groot) model_id=nvidia/GR00T-N1.7-3B; metadata_video_layout=source ;;
  diffusion)
    model_id=lerobot/diffusion-resnet18
    metadata_video_layout=hwc-with-channel-axis
    diffusion_horizon=16
    [[ ${dataset_profile} == libero ]] && diffusion_horizon=32
    ;;
  *) echo "unsupported MODEL_PROFILE: ${model_profile}" >&2; exit 2 ;;
esac

baseline_sha=${BASELINE_SHA:-223a8ad16c52dad961cc1104477ffc3369c5189a}
proposal_sha=${PROPOSAL_SHA:-0aa734f39ccdec8e08f7dd070ca21a90284d47b5}
harness_sha=$(git -C "${repo}" rev-parse HEAD)
proposal_patch_id=$(git -C "${repo}" show --pretty=email --no-ext-diff "${proposal_sha}" | git patch-id --stable | awk '{print $1}')
steps=${STEPS:-300}
warmup_steps=${WARMUP_STEPS:-50}
repeats=${REPEATS:-3}
batch_size=${BATCH_SIZE:-128}
worker_counts=${WORKER_COUNTS:-"0 4 8 15"}
worker_counts=${worker_counts//:/ }
telemetry_interval_ms=${TELEMETRY_INTERVAL_MS:-250}

job_id=${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ ! ${system_label} =~ ^[a-z0-9][a-z0-9._-]*$ ]]; then
  echo "invalid SYSTEM_LABEL: ${system_label}" >&2
  exit 2
fi
results_root=${results_base}/dataloading-${system_label}-${dataset_label}-${model_profile}/${job_id}
runtime=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/lerobot-loader-${job_id}.XXXXXX")
mkdir -p "${results_root}" "${runtime}/sources" "${runtime}/data" "${runtime}/models" "${runtime}/hf"

cleanup() {
  if [[ -n ${gpu_monitor_pid:-} ]]; then
    kill "${gpu_monitor_pid}" 2>/dev/null || true
    wait "${gpu_monitor_pid}" 2>/dev/null || true
  fi
  rm -rf "${runtime}"
}
trap cleanup EXIT

python=${PYTHON_RUNTIME:-${shared_root}/dataloading/managed-python/cpython-3.12.3-linux-x86_64-gnu/bin/python}
site_packages=${environment}/lib/python3.12/site-packages
monitor=${repo}/benchmarks/system_profile/monitor_process.py
benchmark=${repo}/benchmarks/system_profile/benchmark_dataloader.py
run_summarizer=${repo}/benchmarks/system_profile/summarize_dataloader_run.py
experiment_summarizer=${repo}/benchmarks/system_profile/summarize_dataloader_experiment.py

for required in "${repo}/.git" "${dataset_source}" "${python}" "${monitor}" "${benchmark}"; do
  [[ -e ${required} ]] || { echo "missing benchmark input: ${required}" >&2; exit 2; }
done
if [[ ${model_profile} == groot && ! -d ${model_source} ]]; then
  echo "missing GR00T model input: ${model_source}" >&2
  exit 2
fi

cp -a "${dataset_source}" "${runtime}/data/${dataset_staged_name}"
if [[ ${model_profile} == diffusion ]]; then
  "${python}" - <<PY
import json
from pathlib import Path

path = Path("${runtime}/data/${dataset_staged_name}/meta/info.json")
info = json.loads(path.read_text())
for feature in info["features"].values():
    shape = feature.get("shape")
    if feature.get("dtype") == "video" and len(shape or []) == 3 and shape[-1] in (1, 3, 4):
        names = list(feature.get("names") or ["height", "width", "channel"])
        names[-1] = "channel"
        feature["names"] = names
path.write_text(json.dumps(info, indent=4) + "\n")
stats_path = path.with_name("stats.json")
stats = json.loads(stats_path.read_text())
for name, values in stats.get("action", {}).items():
    if isinstance(values, list) and len(values) > ${diffusion_horizon}:
        stats["action"][name] = values[:${diffusion_horizon}]
stats_path.write_text(json.dumps(stats, indent=4) + "\n")
PY
fi
if [[ ${model_profile} == groot ]]; then
  cp -a "${model_source}" "${runtime}/models/GR00T-N1.7-3B"
fi
for entry in "baseline:${baseline_sha}" "proposal:${proposal_sha}"; do
  label=${entry%%:*}; sha=${entry#*:}; source_dir=${runtime}/sources/${label}
  mkdir -p "${source_dir}"
  git -C "${repo}" archive "${sha}" | tar -xf - -C "${source_dir}"
done

export HF_HOME=${runtime}/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH=${shared_root}/dataloading/ffmpeg7-x86/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${site_packages}:${PYTHONPATH:-}

metadata=${results_root}/system
mkdir -p "${metadata}"
date -u +%FT%TZ >"${metadata}/started_at_utc.txt"
uname -a >"${metadata}/uname.txt"
lscpu >"${metadata}/lscpu.txt"
df -hT >"${metadata}/filesystems.txt" 2>&1 || true
nvidia-smi -q >"${metadata}/nvidia-smi-q.txt"
nvidia-smi topo -m >"${metadata}/nvidia-smi-topology.txt" 2>/dev/null || true
scontrol show job "${SLURM_JOB_ID}" >"${metadata}/slurm-job.txt" 2>/dev/null || true
cp "${runtime}/data/${dataset_staged_name}/meta/info.json" "${metadata}/dataset-info.json"
git -C "${repo}" show --no-patch --format=fuller "${baseline_sha}" >"${metadata}/baseline-commit.txt"
git -C "${repo}" show --no-patch --format=fuller "${proposal_sha}" >"${metadata}/proposal-commit.txt"
git -C "${repo}" show --no-patch --format=fuller "${harness_sha}" >"${metadata}/harness-commit.txt"
"${python}" - <<PY >"${metadata}/run-manifest.json"
import json
print(json.dumps({
  "schema_version": "1.0", "target": "dataloading", "system_label": "${system_label}",
  "dataset": {"repo_id": "${dataset_repo_id}", "profile": "${dataset_profile}", "subset": "${dataset_subset}",
              "metadata_video_layout": "${metadata_video_layout}"},
  "model": {"id": "${model_id}", "profile": "${model_profile}"},
  "comparison": {"baseline_sha": "${baseline_sha}", "proposal_sha": "${proposal_sha}",
                 "proposal_patch_id": "${proposal_patch_id}", "order": "AB/BA/AB"},
  "harness": {"revision": "${harness_sha}", "entrypoint": "benchmarks/system_profile/run_dataloader_pair.sh"},
  "benchmark": {"steps": ${steps}, "warmup_steps": ${warmup_steps}, "repeats": ${repeats},
                "batch_size": ${batch_size}, "worker_counts": [${worker_counts// /,}], "seed": 42,
                "prefetch_factor": 1, "persistent_workers": True, "storage": "node-local"},
}, indent=2, sort_keys=True))
PY

run_one() {
  local workers=$1 implementation=$2 repeat=$3 sha source_dir label output_dir console gpu_csv loader_summary
  sha=${baseline_sha}; [[ ${implementation} == proposal ]] && sha=${proposal_sha}
  source_dir=${runtime}/sources/${implementation}
  label=r${repeat}-${implementation}
  output_dir=${results_root}/workers-${workers}/${label}
  console=${output_dir}/console.log
  gpu_csv=${output_dir}/gpu.csv
  loader_summary=${output_dir}/loader-summary.json
  mkdir -p "${output_dir}"
  printf '%s\n' "${sha}" >"${output_dir}/git-sha.txt"
  printf '%s\n' "${implementation}" >"${output_dir}/implementation.txt"
  printf '%s\n' "${repeat}" >"${output_dir}/repeat.txt"

  nvidia-smi --query-gpu=timestamp,index,name,uuid,utilization.gpu,utilization.memory,utilization.decoder,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem,pstate --format=csv,noheader,nounits --loop-ms="${telemetry_interval_ms}" >"${gpu_csv}" &
  gpu_monitor_pid=$!
  local model_args=()
  if [[ ${model_profile} == groot ]]; then
    model_args+=(--model-path "${runtime}/models/GR00T-N1.7-3B")
  fi
  set +e
  PYTHONPATH="${source_dir}/src:${site_packages}" "${python}" "${monitor}" \
    --samples "${output_dir}/system.jsonl" --summary "${output_dir}/process-summary.json" \
    --label "workers-${workers}-${label}" --interval 0.1 -- \
    "${python}" "${benchmark}" \
      --dataset-profile "${dataset_profile}" --dataset-root "${runtime}/data/${dataset_staged_name}" \
      --dataset-repo-id "${dataset_repo_id}" --model-profile "${model_profile}" "${model_args[@]}" \
      --batch-size "${batch_size}" --num-workers "${workers}" --steps "${steps}" \
      --warmup-steps "${warmup_steps}" --prefetch-factor 1 --seed 42 \
      --samples "${output_dir}/batches.jsonl" --summary "${loader_summary}" \
      > >(tee "${console}") 2>&1
  status=$?
  set -e
  kill "${gpu_monitor_pid}" 2>/dev/null || true; wait "${gpu_monitor_pid}" 2>/dev/null || true; unset gpu_monitor_pid
  printf '%s\n' "${status}" >"${output_dir}/raw-exit-code.txt"
  if [[ ${status} -ne 0 && -s ${loader_summary} ]]; then
    if "${python}" - "${loader_summary}" "$((steps - warmup_steps))" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if int(summary.get("measured_batches", -1)) == int(sys.argv[2]) else 1)
PY
    then
      printf 'warning: normalizing post-measurement cleanup exit %s; raw status retained\n' "${status}" | tee -a "${console}"
      status=0
    fi
  fi
  printf '%s\n' "${status}" >"${output_dir}/exit-code.txt"
  if [[ ${status} -eq 0 ]]; then
    "${python}" "${run_summarizer}" --loader-summary "${loader_summary}" \
      --system "${output_dir}/system.jsonl" --process-summary "${output_dir}/process-summary.json" \
      --gpu "${gpu_csv}" --output "${output_dir}/summary.json"
  fi
  return "${status}"
}

orders=("baseline proposal" "proposal baseline" "baseline proposal")
for workers in ${worker_counts}; do
  for ((repeat = 0; repeat < repeats; repeat++)); do
    for implementation in ${orders[repeat % ${#orders[@]}]}; do
      run_one "${workers}" "${implementation}" "${repeat}"
      # Let Python's multiprocessing resource tracker finish before another
      # spawn-based DataLoader is created in the same allocation.
      sleep "${RUN_SETTLE_SECONDS:-2}"
    done
  done
done

date -u +%FT%TZ >"${metadata}/finished_at_utc.txt"
"${python}" "${experiment_summarizer}" "${results_root}"
