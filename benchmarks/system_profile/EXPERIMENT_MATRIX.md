# Initial experiment matrix

## Target

Measure steady-state GR00T N1.7 training-system throughput for LeRobot `main` versus the batched
dataset reader. This is a systems benchmark, not a model-quality evaluation.

## Matrix

| System tier | GPU | CPU / GPU | LIBERO Spatial batch | DROID batch | Paired runs |
| --- | --- | ---: | ---: | ---: | ---: |
| Low | NVIDIA L40S 48 GB | 32 | 64 | 32 | 3 |
| Mid | NVIDIA H100 PCIe 80 GB | 32 | 128 | 64 | 3 |
| High | NVIDIA H200 141 GB | 32 | 320 | 64 | 3 |

Each paired run executes baseline and proposal in AB/BA/AB order. A run uses 300 steps, excludes the
first 50 as warm-up, keeps 15 loader workers per GPU, stages data to node-local storage, disables
checkpoints and evaluation, and records offline W&B plus GPU and host telemetry.

LIBERO batch 320 is the documented GR00T recipe. Its H100 capacity check fails at the first update,
so lower tiers use hardware-fit batches. DROID batch 64 follows the documented new-embodiment
recipe. Any DROID batch reduction is recorded as a capacity result before throughput is measured.
On L40S, DROID batch 64 completes one step and then exhausts the 44.4 GiB usable memory; the paired
batch-32 validation completes on both revisions, so batch 32 is the measured low-tier recipe.

## Comparison rules

Within a system/dataset pair, only the dataset reader revision changes. Dataset subset, model,
precision, batch, worker count, seed, code environment, CPU allocation, and storage placement remain
fixed. Cross-system throughput is descriptive because the hardware-fit LIBERO batch changes by tier.

The primary result is samples per second derived from steady-state step time. Cards also retain
preprocessing/update time, data wait, GPU utilization and memory, power, CPU/process metrics,
startup-to-first-step time, run variance, failures, and exact revisions.

Multi-GPU scaling is intentionally deferred until these single-GPU reference cards are stable.
