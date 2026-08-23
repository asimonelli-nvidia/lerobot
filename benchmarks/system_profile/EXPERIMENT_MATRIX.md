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
| NVIDIA L40 | 64 | 64* | 64 | 64 | 64 | 64 |
| NVIDIA H100 SXM | 64 | 64 | 64 | 64 | 64 | 64 |
| NVIDIA H200 | 64 | 64 | 64 | 64 | 64 | 64 |

Values are per-GPU batch size; every cell uses the same batch. The L40 GR00T DROID cell is the
capacity gate because its previous batch-32 run left about 5.5 GiB p95 memory headroom. Batch 64 is
reportable only if a short non-reportable fit test completes with safe headroom. If it does not, the
entire matrix moves to batch 32 rather than introducing a one-cell exception.

Diffusion Policy uses FP32, its default 64-step action horizon, and the default ImageNet-pretrained
ResNet-18 backbone. SmolVLA uses BF16, a 50-step horizon, the pretrained SmolVLM2-500M backbone, and a
dataset-native action expert. GR00T N1.7 uses BF16 and its workload-specific 40-step LIBERO or 16-step
DROID horizon. The GR00T augmentation recipe is enabled only for GR00T; Diffusion Policy and SmolVLA
use their representative transform-disabled recipes.

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

CUDA allocation failures in the capacity gate trigger the universal batch-32 fallback. DataLoader
worker, IPC, shared-memory, or scheduler failures trigger an infrastructure retry and are never
labeled OOM. Multi-GPU scaling is intentionally deferred until the single-GPU reference cards are
stable.
