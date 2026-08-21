# Evidence contract

Use this contract for a portable experiment record. A UI card is a target-aware projection; it is
not the source of truth.

## Required identity

- Schema version, stable card ID, status, and card type (`Calibration`, `Benchmark`, or `Capacity`).
- Declared target and target unit.
- Dataset ID, revision, exact subset, frame and episode counts, camera keys, and relevant transforms.
- Model ID, revision, parameter scale, precision, processor, embodiment, and action horizon.
- GPU name and count, usable memory, CPU allocation and model, storage placement, and topology when
  relevant.
- Per-GPU batch, workers, steps, excluded warm-up, seed, distributed strategy, and exact baseline,
  proposal, and harness revisions.

## Results for a training-throughput target

The default card should show steady-state samples/s, step time, GPU utilization, GPU memory, and the
dominant measured stage. Retain data-wait, preprocessing, update, power, energy/sample, GPU-hours per
million samples, CPU/process memory, startup-to-first-step, wall time, distributions, repeat values,
and failures in the evidence bundle.

Do not average initialization into steady state. Derive throughput from the measured step-time window
and effective batch. State whether the comparison is paired and whether all expected repeats finished.

## Raw evidence

Keep relative pointers to the console log, offline tracker directory, GPU samples, host/process samples,
hardware inventory, filesystem and scheduler metadata, package inventory, dataset metadata, manifest,
and summaries. A nonzero exit without a summary must still appear in the bundle.

## Comparison guardrail

Call two implementation cards directly comparable only when target, dataset, model, system, and full
recipe match. Exclude implementation name and revision from that equality check. Cross-system or
hardware-fit-batch comparisons are descriptive and must be labeled accordingly.

## Sharing boundary

The evidence bundle may contain internal hostnames, mount paths, scheduler accounts, usernames, and
other environment metadata. Keep raw bundles internal by default. Public cards should retain product,
model, recipe, and measured system facts while removing infrastructure identifiers that are not needed
to reproduce the public claim.
