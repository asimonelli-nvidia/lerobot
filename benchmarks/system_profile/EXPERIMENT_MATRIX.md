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
data, transforms, sampler, backend, and seed. Every measured GPU receives 32 CPU cores. Loader worker
counts are calibrated jointly and one shared count maximizes the slower side. Affordable recipes keep
the complete 4/8/15-worker sweep; large-batch recipes use a short sweep followed by full paired repeats
at the selected count.
GPU utilization, memory, and power telemetry is restricted to the GPU assigned by Slurm; the selector
is retained with the system metadata.

## Training recipes

| System | GR00T · LIBERO | GR00T · DROID | Diffusion · LIBERO | Diffusion · DROID |
| --- | ---: | ---: | ---: | ---: |
| NVIDIA L40 | 64 · 15 workers | 32 · 15 workers | 256 · 15 workers | 64 · 15 workers |
| NVIDIA H100 SXM | 128 · 15 workers | 64 · 15 workers | 384 · 15 workers | 32 · 15 workers |
| NVIDIA H200 | 320 · 15 workers | 64 · 8 workers | 512 · 15 workers | 64 · 15 workers |

Values are per-GPU batch followed by training worker count. Diffusion Policy uses a ResNet-18
backbone in FP32; GR00T N1.7 uses BF16. LIBERO Diffusion uses action horizon 32 and DROID uses 16.
The H200 GR00T LIBERO run uses 400 total steps with 100 warm-up steps to fit the cluster's eight-hour
job limit; its 300 measured steps still meet the reportable window. H200 GR00T on DROID uses eight
workers because repeated 15-worker attempts exposed shared-node semaphore pressure; both code
revisions use the same adjusted recipe.

SmolVLA is the third model family. Its recipe uses the pretrained SmolVLM2-500M backbone and a
dataset-native action expert, matching LeRobot's documented LIBERO training path. It uses BF16 and
an action horizon of 50 on both datasets. Calibration starts at batch 64, then probes lower or higher
batches to lock the highest-throughput stable shared recipe for baseline and proposal on each system
and dataset. This is deliberately not the largest batch that merely fits: the L40 probes showed lower
throughput near its memory ceiling. Reportable SmolVLA jobs use the same 600/100-step training window
and 300/50-batch loader window after the recipe is locked.

The local DROID subset stores temporal action normalization statistics for 40 offsets. SmolVLA's
default 50-step horizon extends the staged copy by repeating the final recorded offset for positions
41–50. The source dataset is never modified, and the adjustment is explicit in every run manifest.
This matrix targets system efficiency rather than model-quality comparison.

| SmolVLA cell | L40 | H100 SXM | H200 |
| --- | --- | --- | --- |
| LIBERO | 64 · 15 workers | 384 · 15 workers | calibrate after GR00T queue |
| DROID | 64 · 15 workers | 256 · 15 workers | calibrate after GR00T queue |

The L40 capacity probes fit batch 192 on LIBERO and 160 on DROID, but both were slower than batch 64;
batch 256 LIBERO failed with CUDA OOM. H100 throughput increased through batch 384 on LIBERO and 256
on DROID; batch 448 LIBERO failed with CUDA OOM. A joint 4/8/15-worker sweep selected 15 workers for
both implementations in all four locked cells.

## Interpretation

The Dataloading card attributes changes to the input path and retains sustained throughput,
batch-wait p50/p95, CPU seconds/sample, process memory, video-grouping opportunity, and temporal
deduplication. The Training card combines decode and preprocessing into one Dataloading stage and
shows Model steps/s separately from update time.

CUDA allocation failures trigger a smaller matched batch. DataLoader worker, IPC, shared-memory, or
scheduler failures trigger an infrastructure retry and are never labeled OOM. Cross-system results
are descriptive when the hardware-fit batch differs. Multi-GPU scaling is intentionally deferred
until the single-GPU reference cards are stable.
