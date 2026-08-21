---
name: physical-ai-experiments
description: Define, launch, monitor, preserve, and compare system-centric Physical AI experiments as concise experiment cards. Use for GR00T or LeRobot training and inference work that needs an explicit target, reproducible compute evidence, bottleneck analysis, or a card generated from existing run artifacts.
---

# Physical AI Experiments

Make the human choose the decision; handle the low-level mechanics and retain the evidence.

## Define the card before the run

- Declare one measurable target. For system work, prefer throughput, latency, memory fit, GPU-hours,
  energy per sample, scaling efficiency, or time-to-target with the target stated explicitly.
- Record the exact dataset subset, model and processor configuration, compute, storage placement,
  precision, recipe, and immutable code revisions. Leave fields open when the user wants the agent
  to compare options; label assumptions rather than silently filling them.
- Treat model-quality metrics as reported outcomes unless quality is the declared target. Do not turn
  a system experiment into a general evaluation-platform project.

Read [references/evidence-contract.md](references/evidence-contract.md) when creating a card format,
ingesting artifacts, comparing runs, or deciding what may be shared.

## Calibrate, then benchmark

Use a short validation only to prove that the environment, data path, model, and batch fit. Label it
`Calibration`; never present it as a benchmark result. Preserve capacity failures as useful cards.

For an implementation comparison:

- Change only the implementation revision. Hold the target, dataset subset, model, precision, batch,
  workers, CPU allocation, storage, seed, steps, and warm-up policy fixed.
- Use a hardware-fit batch when the documented recipe does not fit. Keep the failed recipe as a
  capacity result and do not call cross-hardware throughput causal when batches differ.
- Prefer at least three paired repeats in balanced AB/BA/AB order. Report the paired change and
  variance. If the spread is comparable to run noise, say there is no clear lead.
- Stage immutable data and model inputs once on node-local storage before timing. Do not tune one
  implementation independently to make its result look better.

In this repository, use `benchmarks/system_profile/run_groot_libero_pair.sh` and the matching submit,
summary, and dataset-preparation helpers instead of reconstructing the commands. The run harness
keeps checkpoints and evaluation outside the throughput window and records offline W&B, GPU, host,
process, dataset, environment, Slurm, and Git evidence.

## Monitor and conclude

- Keep startup-to-first-step separate from steady-state step time. A slow first batch is not steady
  training throughput, and prefetched data wait does not prove decoding is free.
- Preserve complete console, tracker, GPU, process, hardware, and environment artifacts even when the
  card shows only decision-critical metrics.
- Diagnose the dominant measured stage before recommending another GPU, batch, worker count, decoder,
  or distributed strategy. Distinguish observed evidence from an agent inference.
- Do not hide errors, cancel inconvenient results, or replace measured values with estimates. A card
  may be incomplete; its raw evidence may not.

## Present the result

Default to a five-second read: what ran, whether the target was reached, the primary system metric,
GPU utilization and memory, the dominant pipeline stage, and the agent's one-sentence conclusion.
Keep configuration and classic graphs one level deeper and raw artifacts available to the agent.
