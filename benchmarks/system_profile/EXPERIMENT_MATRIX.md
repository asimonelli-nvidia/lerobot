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
data, transforms, sampler, backend, and seed. Every measured GPU receives 32 CPU cores. Loader runs
sweep 4, 8, and 15 workers and select one shared worker count that maximizes the slower side.

## Training recipes

| System | GR00T · LIBERO | GR00T · DROID | Diffusion · LIBERO | Diffusion · DROID |
| --- | ---: | ---: | ---: | ---: |
| NVIDIA L40 | 64 · 15 workers | 32 · 15 workers | calibrating 192 | calibrating 32 / 64 / 128 |
| NVIDIA H100 SXM | 128 · 15 workers | 64 · 15 workers | 384 · 15 workers | 32 · 15 workers |
| NVIDIA H200 | 320 · 15 workers | 64 · 15 workers | 512 · 15 workers | 64 · 15 workers |

Values are per-GPU batch followed by training worker count. Diffusion Policy uses a ResNet-18
backbone in FP32; GR00T N1.7 uses BF16. LIBERO Diffusion uses action horizon 32 and DROID uses 16.

## Interpretation

The Dataloading card attributes changes to the input path and retains sustained throughput,
batch-wait p50/p95, CPU seconds/sample, process memory, video-grouping opportunity, and temporal
deduplication. The Training card combines decode and preprocessing into one Dataloading stage and
shows Model steps/s separately from update time.

CUDA allocation failures trigger a smaller matched batch. DataLoader worker, IPC, shared-memory, or
scheduler failures trigger an infrastructure retry and are never labeled OOM. Cross-system results
are descriptive when the hardware-fit batch differs. Multi-GPU scaling is intentionally deferred
until the single-GPU reference cards are stable.
