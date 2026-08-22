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
The staged DROID copy also records the deterministic extension from its stored 40-offset action
statistics to SmolVLA's 50-step horizon; source data remains immutable.

Each benchmark uses three paired repeats in AB/BA/AB order. Training normally runs 600 steps and excludes the first 100; the long H200 GR00T LIBERO recipe uses 400 total and 300 measured steps. Dataloading runs 300 batches per repeat and excludes the first 50. Worker counts are calibrated jointly; large-batch recipes use a short sweep before full repeats at one shared worker count. Data and model assets are staged to node-local storage before measurement. Capacity and stability calibration is separate from reportable evidence; a failed fit is followed by a smaller batch or worker count rather than being treated as a completed benchmark.

Exact fit recipes and run status are recorded in [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md).

## Evidence retained

Every training run keeps per-step logs, offline W&B history, 250 ms GPU telemetry, 500 ms process and host telemetry, system and filesystem metadata, exact revisions, failures, and compact summaries. Loader runs retain the same system evidence plus per-batch timing and grouping metrics. The experiment-card importer keeps all raw artifacts and chooses only target-critical metrics for the card.

Use `submit_matrix_job.sh` as the cluster entrypoint. It accepts `TARGET=training` or `TARGET=dataloading`; loader-specific aliases such as `LOADER_WORKERS=4:8:15` are supported. The agent-facing workflow and comparison guardrails live in `skills/physical-ai-experiments/SKILL.md`.
