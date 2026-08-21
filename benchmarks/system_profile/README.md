# GR00T system-throughput benchmark

This harness compares LeRobot `main` with the batched-dataset proposal while holding the model,
dataset, seed, training recipe, software environment, GPU allocation, CPU allocation, and local
data placement constant.

The default experiment performs three paired repeats. Each run trains GR00T N1.7 for 600 steps;
the first 100 steps are retained as raw data but excluded from the steady-state summary. The repeat
order is AB/BA/AB to reduce cache and ordering bias. Inputs are staged to node-local storage once,
outside the measured runs.

Two recipe profiles are supported:

| Profile | Dataset | Deterministic scope | GR00T setup | Recipe batch |
| --- | --- | --- | --- | ---: |
| `libero` | LIBERO Spatial | all 432 episodes | `libero_sim`, 40-step action horizon | 320 |
| `droid` | DROID 1.0.1 | episodes 0–99 from the current full-schema dataset | `new_embodiment`, 16-step action horizon | 64 |

The DROID subset is intended for repeatable system-throughput measurement, not quality claims. It
uses the current DROID 1.0.1 schema rather than the older `droid_100` conversion because the latter
does not contain the state/action fields expected by the current GR00T integration.

Every run retains:

- per-step LeRobot metrics and the complete console log;
- an offline W&B run containing the configuration and training history;
- 250 ms GPU utilization, decoder utilization, memory, power, temperature, and clock samples;
- 500 ms process-tree CPU, memory, I/O, context-switch, host memory, disk, and network samples;
- hardware, filesystem, package, dataset, and exact Git commit metadata;
- a compact JSON summary with distribution statistics after the warm-up window.

Submit a LIBERO experiment from the cluster frontend (override the site-specific Slurm fields):

```bash
SYSTEM_LABEL=h200 DATASET_PROFILE=libero \
PARTITION=<partition> ACCOUNT=<account> CPUS_PER_TASK=32 MEMORY=220G \
  benchmarks/system_profile/submit_h100.sh
```

Run the DROID profile with the same system allocation:

```bash
SYSTEM_LABEL=h200 DATASET_PROFILE=droid \
PARTITION=<partition> ACCOUNT=<account> CPUS_PER_TASK=32 MEMORY=220G \
  benchmarks/system_profile/submit_h100.sh
```

For a validation run before the full experiment:

```bash
STEPS=20 WARMUP_STEPS=5 REPEATS=1 TIME_LIMIT=00:30:00 QOS=batch-short \
  benchmarks/system_profile/submit_h100.sh
```

The validation run is not a benchmark result. Its only purpose is to verify the environment,
dataset, model cache, and command before reserving the GPU for the full paired experiment.
