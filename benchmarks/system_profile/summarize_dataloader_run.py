#!/usr/bin/env python

"""Attach host and GPU evidence to one isolated dataloader run."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

GPU_FIELDS = [
    "timestamp",
    "index",
    "name",
    "uuid",
    "utilization_gpu_percent",
    "utilization_memory_percent",
    "utilization_decoder_percent",
    "memory_used_mib",
    "memory_total_mib",
    "power_draw_w",
    "power_limit_w",
    "temperature_gpu_c",
    "clocks_sm_mhz",
    "clocks_memory_mhz",
    "pstate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loader-summary", type=Path, required=True)
    parser.add_argument("--system", type=Path, required=True)
    parser.add_argument("--process-summary", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float | int] | None:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "min": min(clean),
        "p50": statistics.median(clean),
        "p95": sorted(clean)[max(0, math.ceil(len(clean) * 0.95) - 1)],
        "max": max(clean),
    }


def in_window(timestamp: str, start: float, end: float) -> bool:
    return start <= datetime.fromisoformat(timestamp).timestamp() <= end


def summarize_system(path: Path, start: float, end: float, measured_samples: int) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if in_window(row["timestamp_utc"], start, end)]
    cpu_percent = [float(row["host"]["cpu_percent"]) for row in selected]
    rss_gib = [float(row["process_tree"]["rss_bytes"]) / 1024**3 for row in selected]
    cpu_seconds = None
    if len(selected) >= 2:
        first, last = selected[0]["process_tree"], selected[-1]["process_tree"]
        cpu_seconds = max(
            0.0,
            float(last["cpu_user_s"] + last["cpu_system_s"])
            - float(first["cpu_user_s"] + first["cpu_system_s"]),
        )
    return {
        "samples": len(selected),
        "host_cpu_percent": stats(cpu_percent),
        "process_rss_gib": stats(rss_gib),
        "cpu_seconds": cpu_seconds,
        "cpu_seconds_per_sample": cpu_seconds / measured_samples if cpu_seconds is not None else None,
    }


def summarize_gpu(path: Path, start: float, end: float) -> dict[str, Any]:
    numeric: dict[str, list[float]] = {field: [] for field in GPU_FIELDS[4:-1]}
    names: set[str] = set()
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        for values in csv.reader(stream):
            if len(values) != len(GPU_FIELDS):
                continue
            row = dict(zip(GPU_FIELDS, (value.strip() for value in values), strict=True))
            try:
                timestamp = (
                    datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").astimezone().timestamp()
                )
            except ValueError:
                continue
            if not start <= timestamp <= end:
                continue
            names.add(row["name"])
            for field in numeric:
                with contextlib.suppress(ValueError):
                    numeric[field].append(float(row[field]))
    return {
        "gpu_names": sorted(names),
        "metrics": {name: value for name, values in numeric.items() if (value := stats(values))},
    }


def main() -> None:
    args = parse_args()
    loader = json.loads(args.loader_summary.read_text(encoding="utf-8"))
    process = json.loads(args.process_summary.read_text(encoding="utf-8"))
    start = datetime.fromisoformat(loader["measurement_started_utc"]).timestamp()
    end = datetime.fromisoformat(loader["measurement_ended_utc"]).timestamp()
    result = {
        **loader,
        "system": summarize_system(args.system, start, end, int(loader["measured_samples"])),
        "gpu": summarize_gpu(args.gpu, start, end),
        "process": process,
        "raw_files": {
            "gpu": str(args.gpu),
            "system": str(args.system),
            "process_summary": str(args.process_summary),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
