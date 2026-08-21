# GR00T system-throughput benchmark

This harness compares LeRobot `main` with the batched-dataset proposal while holding the model,
dataset, seed, training recipe, software environment, GPU allocation, CPU allocation, and local
data placement constant.

The default H100 experiment performs three paired repeats. Each run trains GR00T N1.7 on LIBERO
for 600 steps at batch size 128. The first 100 steps are retained as raw data but excluded from the
steady-state summary. The repeat order is AB/BA/AB to reduce cache and ordering bias.

Every run retains:

- per-step LeRobot metrics and the complete console log;
- an offline W&B run containing the configuration and training history;
- 250 ms GPU utilization, decoder utilization, memory, power, temperature, and clock samples;
- 500 ms process-tree CPU, memory, I/O, context-switch, host memory, disk, and network samples;
- hardware, filesystem, package, dataset, and exact Git commit metadata;
- a compact JSON summary with distribution statistics after the warm-up window.

Submit the default experiment from the cluster frontend:

```bash
benchmarks/system_profile/submit_h100.sh
```

For a validation run before the full experiment:

```bash
STEPS=5 WARMUP_STEPS=1 REPEATS=1 TIME_LIMIT=00:30:00 \
  benchmarks/system_profile/submit_h100.sh
```

The validation run is not a benchmark result. Its only purpose is to verify the environment,
dataset, model cache, and command before reserving the GPU for the full paired experiment.
