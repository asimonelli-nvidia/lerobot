#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

: "${SHARED_ROOT:?Set SHARED_ROOT to the shared benchmark workspace}"
shared_root=${SHARED_ROOT}
repo=${REPO_PATH:-${shared_root}/experiments/lerobot-batched-benchmark}
environment=${PYTHON_ENV:-${shared_root}/dataloading/lerobot-batched-pr/.venv}
model_source=${MODEL_SOURCE:-${shared_root}/v2d/artifacts/models/GR00T-N1.7-3B}
resnet_weights_source=${RESNET18_WEIGHTS_SOURCE:-${shared_root}/experiments/benchmark-assets/resnet18-f37072fd.pth}
hf_source=${HF_SOURCE:-${shared_root}/v2d/cache/huggingface}
results_base=${RESULTS_ROOT:-${shared_root}/experiments/lerobot-batched-benchmark-results}
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
    batch_size_default=32
    dataset_args=()
    ;;
  droid)
    dataset_source=${DATASET_ROOT:-${shared_root}/dataloading/data/droid_1.0.1_first100}
    dataset_staged_name=droid_1.0.1_first100
    dataset_repo_id=lerobot/droid_1.0.1
    dataset_label=droid
    dataset_subset=episodes-0-99
    batch_size_default=32
    droid_episodes=$(seq -s, 0 99)
    dataset_args=(--dataset.episodes="[${droid_episodes}]")
    ;;
  *)
    echo "unsupported DATASET_PROFILE: ${dataset_profile}" >&2
    exit 2
    ;;
esac

case ${model_profile} in
  groot)
    model_id=nvidia/GR00T-N1.7-3B
    model_label=groot-n1.7
    model_precision=BF16
    model_initialization='GR00T N1.7 pretrained checkpoint'
    image_transforms_enabled=True
    if [[ ${dataset_profile} == droid ]]; then
      embodiment_tag=new_embodiment
      horizon=16
      policy_args=(
        --policy.type=groot --policy.device=cuda
        --policy.base_model_path="${runtime_model_path:-__STAGED_GROOT_MODEL__}"
        --policy.embodiment_tag=new_embodiment
        --policy.chunk_size=16 --policy.n_action_steps=16
        --policy.use_relative_actions=true
        --policy.relative_exclude_joints='["gripper"]'
        --policy.use_bf16=true
        --policy.push_to_hub=false --policy.max_steps=20000
      )
    else
      embodiment_tag=libero_sim
      horizon=40
      policy_args=(
        --policy.type=groot --policy.device=cuda
        --policy.base_model_path="${runtime_model_path:-__STAGED_GROOT_MODEL__}"
        --policy.embodiment_tag=libero_sim
        --policy.use_relative_actions=false
        --policy.push_to_hub=false --policy.max_steps=20000
      )
    fi
    ;;
  diffusion)
    model_id=lerobot/diffusion-resnet18
    model_label=diffusion-resnet18
    embodiment_tag=dataset-native
    model_precision=FP32
    model_initialization='ImageNet-pretrained ResNet-18 backbone'
    image_transforms_enabled=False
    metadata_video_layout=hwc-with-channel-axis
    batch_size_default=32
    horizon=64
    action_steps=32
    drop_last=7
    policy_args=(
      --policy.type=diffusion --policy.device=cuda
      --policy.horizon="${horizon}" --policy.n_action_steps="${action_steps}"
      --policy.drop_n_last_frames="${drop_last}"
      --policy.push_to_hub=false
    )
    ;;
  smolvla)
    model_id=lerobot/smolvla_base
    model_label=smolvla-450m
    embodiment_tag=dataset-native
    model_precision=BF16
    model_initialization='Pretrained SmolVLM2-500M backbone + dataset-native action expert'
    image_transforms_enabled=False
    horizon=50
    batch_size_default=32
    metadata_video_layout=hwc-with-channel-axis
    policy_args=(
      --policy.type=smolvla --policy.device=cuda
      --policy.load_vlm_weights=true
      --policy.push_to_hub=false
    )
    ;;
  *)
    echo "unsupported MODEL_PROFILE: ${model_profile}" >&2
    exit 2
    ;;
esac
metadata_video_layout=${metadata_video_layout:-source}
stats_adjustment=none
if [[ ${dataset_profile} == droid && ${model_profile} != groot ]]; then
  stats_adjustment=recomputed-relative-action-stats-to-${horizon}
fi

baseline_sha=${BASELINE_SHA:-223a8ad16c52dad961cc1104477ffc3369c5189a}
proposal_sha=${PROPOSAL_SHA:-0aa734f39ccdec8e08f7dd070ca21a90284d47b5}
harness_sha=$(git -C "${repo}" rev-parse HEAD)
proposal_patch_id=$(git -C "${repo}" show --pretty=email --no-ext-diff "${proposal_sha}" | git patch-id --stable | awk '{print $1}')
steps=${STEPS:-600}
warmup_steps=${WARMUP_STEPS:-100}
repeats=${REPEATS:-3}
batch_size=${BATCH_SIZE:-${batch_size_default}}
num_workers=${NUM_WORKERS:-4}
prefetch_factor=${PREFETCH_FACTOR:-4}
telemetry_interval_ms=${TELEMETRY_INTERVAL_MS:-250}
gpu_selector=${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}
gpu_query_args=()
if [[ -n ${gpu_selector} && ${gpu_selector} != NoDevFiles ]]; then
  gpu_query_args+=(--id="${gpu_selector}")
fi

job_id=${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ ! ${system_label} =~ ^[a-z0-9][a-z0-9._-]*$ ]]; then
  echo "invalid SYSTEM_LABEL: ${system_label}" >&2
  exit 2
fi
results_root=${results_base}/training-${system_label}-${dataset_label}-${model_label}/${job_id}
runtime=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/lerobot-batched-${job_id}.XXXXXX")
mkdir -p \
  "${results_root}" "${runtime}/sources" "${runtime}/data" "${runtime}/hf/hub" \
  "${runtime}/models" "${runtime}/outputs" "${runtime}/home" "${runtime}/wandb-cache" \
  "${runtime}/wandb-config" "${runtime}/wandb-data" "${runtime}/cache/torch/kernels" \
  "${runtime}/cache/torch/hub/checkpoints"

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
dependency_paths=${site_packages}
if [[ ${model_profile} == smolvla ]]; then
  smolvla_python_overlay=${SMOLVLA_PYTHON_OVERLAY:-${shared_root}/experiments/smolvla-assets/python}
  if [[ ! -f ${smolvla_python_overlay}/num2words/__init__.py ]]; then
    echo "missing SmolVLA Python overlay: ${smolvla_python_overlay}" >&2
    exit 2
  fi
  dependency_paths=${smolvla_python_overlay}:${dependency_paths}
fi
monitor=${repo}/benchmarks/system_profile/monitor_process.py
summarizer=${repo}/benchmarks/system_profile/summarize_run.py
experiment_summarizer=${repo}/benchmarks/system_profile/summarize_experiment.py

for required in \
  "${repo}/.git" "${dataset_source}" "${python}" "${monitor}" "${summarizer}" \
  "${experiment_summarizer}"; do
  if [[ ! -e ${required} ]]; then
    echo "missing required benchmark input: ${required}" >&2
    exit 2
  fi
done
if [[ ${model_profile} == groot && ! -d ${model_source} ]]; then
  echo "missing GR00T model input: ${model_source}" >&2
  exit 2
fi
if [[ ${model_profile} == diffusion && ! -f ${resnet_weights_source} ]]; then
  echo "missing ImageNet ResNet-18 weights: ${resnet_weights_source}" >&2
  exit 2
fi

# Stage immutable inputs once per allocation. Setup time is kept outside every measured run.
cp -a "${dataset_source}" "${runtime}/data/${dataset_staged_name}"
if [[ ${dataset_profile} == droid && ${model_profile} != groot ]]; then
  PYTHONPATH="${repo}:${dependency_paths}" "${python}" - <<PY
from pathlib import Path

from benchmarks.system_profile.prepare_droid_subset import write_relative_stats

write_relative_stats(Path("${runtime}/data/${dataset_staged_name}"), ${horizon})
PY
fi
if [[ ${model_profile} == diffusion || ${model_profile} == smolvla ]]; then
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
PY
fi
if [[ ${model_profile} == groot ]]; then
  cp -a "${model_source}" "${runtime}/models/GR00T-N1.7-3B"
  for model in models--nvidia--Cosmos-Reason2-2B; do
    if [[ -d ${hf_source}/hub/${model} ]]; then
      cp -a "${hf_source}/hub/${model}" "${runtime}/hf/hub/${model}"
    fi
  done
  for ((i = 0; i < ${#policy_args[@]}; i++)); do
    if [[ ${policy_args[i]} == '--policy.base_model_path=__STAGED_GROOT_MODEL__' ]]; then
      policy_args[i]="--policy.base_model_path=${runtime}/models/GR00T-N1.7-3B"
    fi
  done
fi
if [[ ${model_profile} == diffusion ]]; then
  cp "${resnet_weights_source}" "${runtime}/cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
fi
if [[ ${model_profile} == smolvla ]]; then
  smolvla_hf_source=${SMOLVLA_HF_SOURCE:-${shared_root}/experiments/smolvla-assets/huggingface}
  smolvla_hub_source=${smolvla_hf_source}/hub
  if [[ -d ${smolvla_hf_source}/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct ]]; then
    smolvla_hub_source=${smolvla_hf_source}
  fi
  if [[ ! -d ${smolvla_hub_source}/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct ]]; then
    echo "missing staged SmolVLA backbone: ${smolvla_hf_source}" >&2
    exit 2
  fi
  cp -a "${smolvla_hub_source}/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct" "${runtime}/hf/hub/"
fi

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
export WANDB_MODE=offline
export WANDB_SILENT=true
export WANDB_CONSOLE=off
export WANDB_CACHE_DIR=${runtime}/wandb-cache
export WANDB_CONFIG_DIR=${runtime}/wandb-config
export WANDB_DATA_DIR=${runtime}/wandb-data
export XDG_CACHE_HOME=${runtime}/cache
export LD_LIBRARY_PATH=${shared_root}/dataloading/ffmpeg7-x86/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${dependency_paths}:${PYTHONPATH:-}
export TRITON_CACHE_DIR=${runtime}/triton
export TORCHINDUCTOR_CACHE_DIR=${runtime}/torchinductor

metadata=${results_root}/system
mkdir -p "${metadata}"
date -u +%FT%TZ >"${metadata}/started_at_utc.txt"
date +%Z%z >"${metadata}/local-timezone.txt"
uname -a >"${metadata}/uname.txt"
lscpu >"${metadata}/lscpu.txt"
lsblk -O -J >"${metadata}/lsblk.json" 2>/dev/null || true
mount >"${metadata}/mounts.txt"
df -hT >"${metadata}/filesystems.txt" 2>&1 || true
nvidia-smi -q >"${metadata}/nvidia-smi-q.txt"
nvidia-smi topo -m >"${metadata}/nvidia-smi-topology.txt" 2>/dev/null || true
printf '%s\n' "${gpu_selector:-unresolved}" >"${metadata}/gpu-selector.txt"
scontrol show job "${SLURM_JOB_ID}" >"${metadata}/slurm-job.txt" 2>/dev/null || true
"${python}" - <<'PY' >"${metadata}/python-packages.txt"
from importlib.metadata import distributions

packages = sorted(
    (name, dist.version)
    for dist in distributions()
    if (name := dist.metadata.get("Name")) is not None
)
for name, version in packages:
    print(f"{name}=={version}")
PY
git -C "${repo}" show --no-patch --format=fuller "${baseline_sha}" >"${metadata}/baseline-commit.txt"
git -C "${repo}" show --no-patch --format=fuller "${proposal_sha}" >"${metadata}/proposal-commit.txt"
git -C "${repo}" show --no-patch --format=fuller "${harness_sha}" >"${metadata}/harness-commit.txt"
cp "${runtime}/data/${dataset_staged_name}/meta/info.json" "${metadata}/dataset-info.json"
"${python}" - <<PY >"${metadata}/run-manifest.json"
import json

print(json.dumps({
    "schema_version": "1.0",
    "target": "training_throughput",
    "system_label": "${system_label}",
    "dataset": {
        "repo_id": "${dataset_repo_id}",
        "profile": "${dataset_profile}",
        "subset": "${dataset_subset}",
        "metadata_video_layout": "${metadata_video_layout}",
        "stats_adjustment": "${stats_adjustment}",
    },
    "model": {
        "id": "${model_id}",
        "profile": "${model_profile}",
        "embodiment_tag": "${embodiment_tag}",
        "initialization": "${model_initialization}",
        "precision": "${model_precision}",
        "action_horizon": ${horizon},
        "image_transforms_enabled": ${image_transforms_enabled},
    },
    "comparison": {
        "baseline_sha": "${baseline_sha}",
        "proposal_sha": "${proposal_sha}",
        "proposal_patch_id": "${proposal_patch_id}",
        "order": "AB/BA/AB",
    },
    "harness": {
        "revision": "${harness_sha}",
        "entrypoint": "benchmarks/system_profile/run_groot_libero_pair.sh",
    },
    "training": {
        "steps": ${steps},
        "warmup_steps": ${warmup_steps},
        "repeats": ${repeats},
        "batch_size_per_gpu": ${batch_size},
        "num_workers_per_gpu": ${num_workers},
        "prefetch_factor": ${prefetch_factor},
        "persistent_workers": True,
        "multiprocessing_context": "spawn",
        "pin_memory": True,
        "log_frequency": 10,
        "seed": 42,
        "checkpoints": False,
        "evaluation": False,
    },
    "telemetry": {
        "gpu_interval_ms": ${telemetry_interval_ms},
        "gpu_selector": "${gpu_selector:-unresolved}",
        "system_interval_ms": 500,
        "wandb_mode": "offline",
    },
}, indent=2, sort_keys=True))
PY

image_transforms='{
  "brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.7, 1.3]}},
  "contrast":   {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast":   [0.6, 1.4]}},
  "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}},
  "hue":        {"weight": 1.0, "type": "ColorJitter", "kwargs": {"hue":        [-0.08, 0.08]}}
}'
dataset_transform_args=()
if [[ ${model_profile} == groot ]]; then
  dataset_transform_args=(
    --dataset.image_transforms.enable=true
    --dataset.image_transforms.max_num_transforms=4
    "--dataset.image_transforms.tfs=${image_transforms}"
  )
fi

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
  mkdir -p "${output_dir}"

  printf '%s\n' "${sha}" >"${output_dir}/git-sha.txt"
  printf '%s\n' "${implementation}" >"${output_dir}/implementation.txt"
  printf '%s\n' "${repeat}" >"${output_dir}/repeat.txt"

  nvidia-smi "${gpu_query_args[@]}" \
    --query-gpu=timestamp,index,name,uuid,utilization.gpu,utilization.memory,utilization.decoder,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem,pstate \
    --format=csv,noheader,nounits --loop-ms="${telemetry_interval_ms}" >"${gpu_csv}" &
  gpu_monitor_pid=$!

  set +e
  PYTHONPATH="${source_dir}/src:${dependency_paths}" "${python}" "${monitor}" \
    --samples "${output_dir}/system.jsonl" \
    --summary "${output_dir}/process-summary.json" \
    --label "${label}" --interval 0.5 -- \
    "${python}" -m lerobot.scripts.lerobot_train \
      --dataset.repo_id="${dataset_repo_id}" \
      --dataset.root="${runtime}/data/${dataset_staged_name}" \
      --dataset.revision=main --dataset.video_backend=torchcodec \
      "${dataset_args[@]}" \
      "${dataset_transform_args[@]}" \
      "${policy_args[@]}" \
      --batch_size="${batch_size}" --steps="${steps}" --save_checkpoint=false \
      --env_eval_freq=0 --eval_steps=0 --log_freq=10 \
      --num_workers="${num_workers}" --prefetch_factor="${prefetch_factor}" --persistent_workers=true \
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
    sleep "${RUN_SETTLE_SECONDS:-2}"
  done
done

date -u +%FT%TZ >"${metadata}/finished_at_utc.txt"
"${python}" "${experiment_summarizer}" "${results_root}" || true
