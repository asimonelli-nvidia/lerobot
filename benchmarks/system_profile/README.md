# LeRobot system benchmark

This harness turns matched LeRobot runs into reproducible system evidence. It compares:

- baseline: LeRobot `main` at `223a8ad16c52dad961cc1104477ffc3369c5189a`;
- proposal: `0aa734f39ccdec8e08f7dd070ca21a90284d47b5`, the GitHub copy of NVIDIA GitLab commit `86da6a090952845c4312165ee3cd6c9d00489e14`.

The two proposal commits have the same stable patch ID, `0c7c0504f9c49eac1df1d3fedeea3b5c2028a81a`, across the same five files. Only the base repository differs.

## Targets

| Target | Primary metric | Supporting evidence |
| --- | --- | --- |
| Dataloading | loader samples/s | batch-wait p50/p95, CPU seconds/sample, process RSS, grouping opportunity |
| Training throughput | steps/s | samples/s, combined dataloading time, model-update time, GPU busy, memory, power |

The isolated dataloading target is where the batched reader can be attributed directly. End-to-end training may show a smaller difference when model work is the bottleneck.

## Matrix

The current matrix covers LIBERO Spatial and a deterministic 100-episode DROID 1.0.1 subset;
GR00T N1.7, LeRobot Diffusion Policy with a ResNet-18 backbone, and SmolVLA; and one NVIDIA L40,
H100 SXM, or H200 with 32 allocated CPU cores per measured GPU. SmolVLA follows LeRobot's
documented recipe: pretrained SmolVLM2-500M backbone weights with a dataset-native action expert.
That initialization is recorded on every card.
The staged DROID copy records any model-horizon metadata adjustment; source data remains immutable.

Each benchmark uses three paired repeats in AB/BA/AB order. Training runs 600 steps and excludes the first 100; dataloading runs 300 batches per repeat and excludes the first 50. The reference matrix uses batch 64 across every GPU and workload, gated by a short L40 GR00T DROID fit test. If that cell cannot fit safely, the complete matrix uses batch 32. Every run uses 4 workers per GPU/rank, prefetch factor 4, persistent workers, `spawn`, and pinned memory. Data and model assets are staged to node-local storage before measurement.

Exact throughput-optimized recipes and run status are recorded in
[EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md).

## Evidence retained

Every training run keeps per-step logs, offline W&B history, 250 ms telemetry bound to the
Slurm-assigned GPU, 500 ms process and host telemetry, system and filesystem metadata, exact
revisions, failures, and compact summaries. The resolved GPU selector is saved in the manifest.
Loader runs retain the same system evidence plus per-batch timing and grouping metrics. The
experiment-card importer keeps all raw artifacts and chooses only target-critical metrics for the
card.

Use `submit_matrix_job.sh` as the cluster entrypoint. It accepts `TARGET=training` or `TARGET=dataloading`; `BATCH_SIZE=64`, `NUM_WORKERS=4`, `WORKER_COUNTS=4`, and `PREFETCH_FACTOR=4` define the reference recipe. The agent-facing workflow and comparison guardrails live in `skills/physical-ai-experiments/SKILL.md`.
