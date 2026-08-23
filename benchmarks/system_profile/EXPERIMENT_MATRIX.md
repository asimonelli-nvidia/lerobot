# Experiment matrix

## Comparison identity

| Role | Revision |
| --- | --- |
| LeRobot main | `223a8ad16c52dad961cc1104477ffc3369c5189a` |
| Batched reader | `0aa734f39ccdec8e08f7dd070ca21a90284d47b5` |
| Source GitLab commit | `86da6a090952845c4312165ee3cd6c9d00489e14` |
| Stable patch ID | `0c7c0504f9c49eac1df1d3fedeea3b5c2028a81a` |

The proposal commit is a direct child of the baseline and has the same stable patch ID as the GitLab
commit. The paired source therefore changes only the five dataset-reader patch files.

## Reportable protocol

| Target | Measurement | Repeats | Warm-up | Primary card metric |
| --- | --- | ---: | ---: | --- |
| Dataloading | 300 batches | 3, AB/BA/AB | 50 batches | sustained samples/s |
| Training throughput | 600 steps | 3, AB/BA/AB | 100 steps | Model steps/s |

Both sides use the same dataset subset, model, precision, batch, workers, CPU allocation, node-local
data, transforms, sampler, backend, and seed. Every measured GPU receives 32 CPU cores. Every cell
uses 4 workers per GPU/rank, prefetch factor 4, persistent workers, the `spawn` start method, and
pinned memory. These values match the LeRobot reference loader configuration and are not tuned toward
either implementation.
GPU utilization, memory, and power telemetry is restricted to the GPU assigned by Slurm; the selector
is retained with the system metadata.

## Training recipes

| System | GR00T · LIBERO | GR00T · DROID | Diffusion · LIBERO | Diffusion · DROID | SmolVLA · LIBERO | SmolVLA · DROID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NVIDIA L40 | 32 | 32 | 32 | 32 | 32 | 32 |
| NVIDIA H100 SXM | 32 | 32 | 32 | 32 | 32 | 32 |
| NVIDIA H200 | 32 | 32 | 32 | 32 | 32 | 32 |

Values are per-GPU batch size; every cell uses the same batch. A non-reportable L40 GR00T DROID
capacity gate showed that batch 64 used 45.5 of 46.1 GiB and failed on an additional 20 MiB
allocation. Batch 32 is therefore the universal reference rather than a one-cell exception.

Diffusion Policy uses FP32, its default 64-step action horizon, and the default ImageNet-pretrained
ResNet-18 backbone. SmolVLA uses BF16, a 50-step horizon, the pretrained SmolVLM2-500M backbone, and a
dataset-native action expert. GR00T N1.7 uses BF16 and its workload-specific 40-step LIBERO or 16-step
DROID horizon. The GR00T augmentation recipe is enabled only for GR00T; Diffusion Policy and SmolVLA
use their representative transform-disabled recipes.

The source DROID subset stores 40-offset temporal action statistics. Node-local staged copies
recompute those statistics from the subset's action/state parquet at Diffusion Policy's 64-step or
SmolVLA's 50-step horizon. The shared source dataset remains immutable, and the adjustment is recorded
in each manifest.

The local DROID subset stores temporal action normalization statistics for 40 offsets. SmolVLA's
default 50-step horizon extends the staged copy by repeating the final recorded offset for positions
41–50. The source dataset is never modified, and the adjustment is explicit in every run manifest.
This matrix targets system efficiency rather than model-quality comparison.

Training-throughput runs log every 10 steps, disable evaluation and checkpoint writes, and retain a
600-step window with the first 100 excluded. This is a system benchmark rather than a convergence
recipe. Isolated-loader runs retain 300 batches with the first 50 excluded. Seed 42 is retained for
continuity with the GR00T example; it differs from LeRobot's generic seed 1000 but is not expected to
affect steady-state throughput.

## Interpretation

The Dataloading card attributes changes to the input path and retains sustained throughput,
batch-wait p50/p95, CPU seconds/sample, process memory, video-grouping opportunity, and temporal
deduplication. The Training card combines decode and preprocessing into one Dataloading stage and
shows Model steps/s separately from update time.

CUDA allocation failures are retained as calibration evidence and never presented as reportable
comparisons. DataLoader worker, IPC, shared-memory, or scheduler failures trigger an infrastructure
retry and are never labeled OOM. Multi-GPU scaling is intentionally deferred until the single-GPU
reference cards are stable.
