#!/usr/bin/env python

"""Aggregate balanced dataloader runs across a shared worker sweep."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def stats(values: list[float]) -> dict[str, float | int] | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
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


def run_metrics(run: dict[str, Any]) -> dict[str, float]:
    result = {
        "loader_samples_s": float(run["samples_per_s"]),
        "batch_wait_p50_s": float(run["batch_wait_s"]["p50"]),
        "batch_wait_p95_s": float(run["batch_wait_s"]["p95"]),
        "video_requests_per_file": float(run["grouping_opportunity"]["video_requests_per_file"]),
        "video_groupable_fraction": float(run["grouping_opportunity"]["video_groupable_fraction"]),
        "temporal_dedup_fraction": float(run["grouping_opportunity"]["temporal_dedup_fraction"]),
        "wall_s": float(run["wall_s"]),
    }
    if run["system"].get("cpu_seconds_per_sample") is not None:
        result["cpu_seconds_per_sample"] = float(run["system"]["cpu_seconds_per_sample"])
    if run["system"].get("host_cpu_percent"):
        result["host_cpu_percent"] = float(run["system"]["host_cpu_percent"]["mean"])
    if run["system"].get("process_rss_gib"):
        result["process_rss_gib"] = float(run["system"]["process_rss_gib"]["p95"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_directory.resolve()
    manifest = json.loads((root / "system/run-manifest.json").read_text(encoding="utf-8"))
    expected = int(manifest["benchmark"]["repeats"])
    profiles: dict[str, Any] = {}

    for worker_dir in sorted(root.glob("workers-*")):
        workers = int(worker_dir.name.split("-", 1)[1])
        by_impl: dict[str, dict[int, dict[str, float]]] = {"baseline": {}, "proposal": {}}
        for path in sorted(worker_dir.glob("r*-*/summary.json")):
            run_dir = path.parent
            if int((run_dir / "exit-code.txt").read_text().strip()) != 0:
                continue
            implementation = (run_dir / "implementation.txt").read_text().strip()
            repeat = int((run_dir / "repeat.txt").read_text().strip())
            by_impl[implementation][repeat] = run_metrics(json.loads(path.read_text(encoding="utf-8")))
        common = sorted(set(by_impl["baseline"]) & set(by_impl["proposal"]))
        implementations = {}
        for implementation, repeats in by_impl.items():
            names = sorted({name for metrics in repeats.values() for name in metrics})
            implementations[implementation] = {
                "completed_repeats": sorted(repeats),
                "runs": {str(index): value for index, value in sorted(repeats.items())},
                "metrics": {
                    name: stats([metrics[name] for metrics in repeats.values() if name in metrics])
                    for name in names
                },
            }
        paired = [
            (
                by_impl["proposal"][index]["loader_samples_s"]
                / by_impl["baseline"][index]["loader_samples_s"]
                - 1
            )
            * 100
            for index in common
        ]
        profiles[str(workers)] = {
            "status": "complete" if len(common) == expected else "incomplete",
            "paired_repeats": common,
            "implementations": implementations,
            "comparison": {"proposal_loader_change_percent": stats(paired)},
        }

    result = {
        "schema_version": "1.0",
        "target": "dataloading",
        "status": "complete"
        if profiles and all(profile["status"] == "complete" for profile in profiles.values())
        else "incomplete",
        "manifest": manifest,
        "worker_profiles": profiles,
    }
    output = args.output or root / "experiment-summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
