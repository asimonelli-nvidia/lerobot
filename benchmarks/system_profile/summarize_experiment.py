#!/usr/bin/env python

"""Aggregate balanced baseline/proposal repeats into one experiment-level result."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values: list[float]) -> dict[str, float | int] | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "stddev": statistics.pstdev(clean),
        "min": min(clean),
        "p50": statistics.median(clean),
        "max": max(clean),
    }


def metric(summary: dict[str, Any], group: str, name: str, statistic: str = "mean") -> float:
    if group == "training":
        value = summary["training"]["metrics"][name][statistic]
    elif group == "gpu":
        value = summary["gpu"]["steady_state"]["metrics"][name][statistic]
    else:
        value = summary["system"]["steady_state"][name][statistic]
    return float(value)


def summarize_run(summary: dict[str, Any], batch_size: int) -> dict[str, float]:
    step_time = metric(summary, "training", "step_s")
    return {
        "step_time_s": step_time,
        "throughput_samples_s": batch_size / step_time,
        "data_time_s": metric(summary, "training", "data_s"),
        "preprocess_time_s": metric(summary, "training", "prep_s"),
        "update_time_s": metric(summary, "training", "updt_s"),
        "gpu_utilization_percent": metric(summary, "gpu", "utilization_gpu_percent"),
        "gpu_memory_used_mib_p95": metric(summary, "gpu", "memory_used_mib", "p95"),
        "gpu_power_w": metric(summary, "gpu", "power_draw_w"),
        "host_cpu_percent": metric(summary, "system", "cpu_percent"),
        "process_rss_gib": metric(summary, "system", "process_tree_rss_gib"),
        "startup_to_first_step_s": float(summary["training"]["startup_to_first_step_s"]),
        "wall_time_s": float(summary["process"]["wall_s"]),
    }


def main() -> None:
    args = parse_args()
    run_directory = args.run_directory.resolve()
    manifest = read_json(run_directory / "system/run-manifest.json")
    batch_size = int(manifest["training"]["batch_size_per_gpu"])
    expected_repeats = int(manifest["training"]["repeats"])
    by_implementation: dict[str, dict[int, dict[str, float]]] = {"baseline": {}, "proposal": {}}

    for path in sorted(run_directory.glob("r*-*/summary.json")):
        run_dir = path.parent
        implementation = (run_dir / "implementation.txt").read_text().strip()
        repeat = int((run_dir / "repeat.txt").read_text().strip())
        exit_code_path = run_dir / "exit-code.txt"
        if not exit_code_path.is_file() or int(exit_code_path.read_text().strip()) != 0:
            continue
        by_implementation.setdefault(implementation, {})[repeat] = summarize_run(
            read_json(path), batch_size
        )

    aggregate: dict[str, Any] = {}
    for implementation, repeats in by_implementation.items():
        aggregate[implementation] = {
            "completed_repeats": sorted(repeats),
            "metrics": {
                name: stats([run[name] for run in repeats.values()])
                for name in sorted({name for run in repeats.values() for name in run})
            },
            "runs": {str(repeat): run for repeat, run in sorted(repeats.items())},
        }

    common_repeats = sorted(set(by_implementation["baseline"]) & set(by_implementation["proposal"]))
    paired_speedups = [
        (
            by_implementation["proposal"][repeat]["throughput_samples_s"]
            / by_implementation["baseline"][repeat]["throughput_samples_s"]
            - 1
        )
        * 100
        for repeat in common_repeats
    ]
    result = {
        "schema_version": "1.0",
        "status": "complete" if len(common_repeats) == expected_repeats else "incomplete",
        "expected_repeats": expected_repeats,
        "paired_repeats": common_repeats,
        "manifest": manifest,
        "implementations": aggregate,
        "comparison": {
            "proposal_throughput_change_percent": stats(paired_speedups),
            "interpretation": "paired within repeat; positive values favor the proposal",
        },
    }
    output = args.output or run_directory / "experiment-summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
