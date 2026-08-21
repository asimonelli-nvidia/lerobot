#!/usr/bin/env python

"""Create a compact statistical summary while retaining the raw benchmark files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TOKEN_RE = re.compile(r"(?P<key>[A-Za-z0-9_./-]+):(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")
STEP_RE = re.compile(r"(?:^|\s)step:(?P<step>\d+)(?:\s|$)")
LOG_TIME_RE = re.compile(r"(?:^|\s)(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:\s|$)")
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
    parser.add_argument("--console", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--system", type=Path, required=True)
    parser.add_argument("--process-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=100)
    return parser.parse_args()


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _stats(values: list[float]) -> dict[str, float | int] | None:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None
    ordered = sorted(clean)
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "stddev": statistics.pstdev(clean),
        "min": ordered[0],
        "p50": _quantile(ordered, 0.50),
        "p90": _quantile(ordered, 0.90),
        "p95": _quantile(ordered, 0.95),
        "p99": _quantile(ordered, 0.99),
        "max": ordered[-1],
    }


def _parse_training(
    console: Path, warmup_steps: int
) -> tuple[list[dict[str, float]], dict[str, Any], tuple[float, float] | None]:
    rows: list[dict[str, float]] = []
    for raw_line in console.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI_RE.sub("", raw_line)
        step_match = STEP_RE.search(line)
        if step_match is None:
            continue
        row: dict[str, float] = {"step": float(step_match.group("step"))}
        time_match = LOG_TIME_RE.search(line)
        if time_match is not None:
            local_time = datetime.strptime(time_match.group("timestamp"), "%Y-%m-%d %H:%M:%S").astimezone()
            row["_timestamp_epoch_s"] = local_time.timestamp()
        for match in TOKEN_RE.finditer(line):
            key = match.group("key")
            if key == "step":
                continue
            row[key] = float(match.group("value"))
        rows.append(row)

    measured = [row for row in rows if row["step"] > warmup_steps]
    keys = sorted({key for row in measured for key in row if key != "step" and not key.startswith("_")})
    warmup_boundary = [
        row["_timestamp_epoch_s"]
        for row in rows
        if row["step"] <= warmup_steps and "_timestamp_epoch_s" in row
    ]
    measured_timestamps = [row["_timestamp_epoch_s"] for row in measured if "_timestamp_epoch_s" in row]
    measurement_window = (
        (max(warmup_boundary), max(measured_timestamps))
        if warmup_boundary and measured_timestamps
        else None
    )
    summary = {
        "logged_steps": len(rows),
        "measured_steps": len(measured),
        "warmup_steps_excluded": warmup_steps,
        "first_logged_step": int(rows[0]["step"]) if rows else None,
        "last_logged_step": int(rows[-1]["step"]) if rows else None,
        "first_step_epoch_s": rows[0].get("_timestamp_epoch_s") if rows else None,
        "measurement_window_epoch_s": list(measurement_window) if measurement_window else None,
        "metrics": {key: _stats([row[key] for row in measured if key in row]) for key in keys},
    }
    return rows, summary, measurement_window


def _number(value: str) -> float | None:
    normalized = value.strip()
    if not normalized or normalized.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def _parse_gpu(path: Path, window: tuple[float, float] | None = None) -> dict[str, Any]:
    numeric: dict[str, list[float]] = {field: [] for field in GPU_FIELDS}
    rows_seen = 0
    rows_selected = 0
    names: set[str] = set()
    uuids: set[str] = set()
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        for values in csv.reader(stream):
            if len(values) != len(GPU_FIELDS):
                continue
            row = dict(zip(GPU_FIELDS, (value.strip() for value in values), strict=True))
            rows_seen += 1
            try:
                timestamp = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").astimezone().timestamp()
            except ValueError:
                continue
            if window is not None and not (window[0] < timestamp <= window[1]):
                continue
            rows_selected += 1
            names.add(row["name"])
            uuids.add(row["uuid"])
            for field, value in row.items():
                parsed = _number(value)
                if parsed is not None:
                    numeric[field].append(parsed)
    return {
        "samples": rows_selected,
        "samples_seen": rows_seen,
        "gpu_names": sorted(names),
        "gpu_uuids": sorted(uuids),
        "metrics": {field: stats for field, values in numeric.items() if (stats := _stats(values))},
    }


def _parse_system(path: Path, window: tuple[float, float] | None = None) -> dict[str, Any]:
    cpu: list[float] = []
    rss: list[float] = []
    processes: list[float] = []
    load_1m: list[float] = []
    rows_seen = 0
    rows_selected = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        rows_seen += 1
        timestamp = datetime.fromisoformat(sample["timestamp_utc"]).timestamp()
        if window is not None and not (window[0] < timestamp <= window[1]):
            continue
        rows_selected += 1
        host = sample["host"]
        tree = sample["process_tree"]
        cpu.append(float(host["cpu_percent"]))
        rss.append(float(tree["rss_bytes"]) / 1024**3)
        processes.append(float(tree["process_count"]))
        load_1m.append(float(host["load_average"][0]))
    return {
        "samples": rows_selected,
        "samples_seen": rows_seen,
        "cpu_percent": _stats(cpu),
        "process_tree_rss_gib": _stats(rss),
        "process_count": _stats(processes),
        "load_average_1m": _stats(load_1m),
    }


def main() -> None:
    args = parse_args()
    _, training, measurement_window = _parse_training(args.console, args.warmup_steps)
    process = json.loads(args.process_summary.read_text(encoding="utf-8"))
    first_step_epoch_s = training.get("first_step_epoch_s")
    if first_step_epoch_s is not None:
        process_started_epoch_s = datetime.fromisoformat(process["started_at_utc"]).timestamp()
        training["startup_to_first_step_s"] = max(0.0, first_step_epoch_s - process_started_epoch_s)
    else:
        training["startup_to_first_step_s"] = None
    result = {
        "schema_version": "1.0",
        "training": training,
        "gpu": {
            "steady_state": _parse_gpu(args.gpu, measurement_window),
            "full_run": _parse_gpu(args.gpu),
        },
        "system": {
            "steady_state": _parse_system(args.system, measurement_window),
            "full_run": _parse_system(args.system),
        },
        "process": process,
        "raw_files": {
            "console": str(args.console),
            "gpu": str(args.gpu),
            "system": str(args.system),
            "process_summary": str(args.process_summary),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
