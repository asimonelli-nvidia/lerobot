#!/usr/bin/env python

"""Run a command while recording process-tree and host telemetry as JSON Lines."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    return args


def _process_tree(root: psutil.Process) -> list[psutil.Process]:
    processes = [root]
    with contextlib.suppress(psutil.AccessDenied, psutil.NoSuchProcess):
        processes.extend(root.children(recursive=True))
    return processes


def _sample_process_tree(root: psutil.Process) -> dict[str, int | float]:
    totals: dict[str, int | float] = {
        "process_count": 0,
        "rss_bytes": 0,
        "vms_bytes": 0,
        "cpu_user_s": 0.0,
        "cpu_system_s": 0.0,
        "read_bytes": 0,
        "write_bytes": 0,
        "threads": 0,
        "voluntary_context_switches": 0,
        "involuntary_context_switches": 0,
    }
    for process in _process_tree(root):
        try:
            memory = process.memory_info()
            cpu = process.cpu_times()
            io = process.io_counters()
            switches = process.num_ctx_switches()
            totals["process_count"] += 1
            totals["rss_bytes"] += memory.rss
            totals["vms_bytes"] += memory.vms
            totals["cpu_user_s"] += cpu.user
            totals["cpu_system_s"] += cpu.system
            totals["read_bytes"] += io.read_bytes
            totals["write_bytes"] += io.write_bytes
            totals["threads"] += process.num_threads()
            totals["voluntary_context_switches"] += switches.voluntary
            totals["involuntary_context_switches"] += switches.involuntary
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return totals


def _counter_dict(counter: object) -> dict[str, int | float]:
    return {field: getattr(counter, field) for field in counter._fields}


def _sample_host() -> dict[str, object]:
    disk = psutil.disk_io_counters()
    network = psutil.net_io_counters()
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "cpu_percent": psutil.cpu_percent(),
        "cpu_percent_per_core": psutil.cpu_percent(percpu=True),
        "load_average": list(os.getloadavg()),
        "memory": _counter_dict(memory),
        "swap": _counter_dict(swap),
        "disk_io": _counter_dict(disk) if disk is not None else None,
        "network_io": _counter_dict(network) if network is not None else None,
    }


def main() -> None:
    args = parse_args()
    args.samples.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(UTC)
    started_monotonic = time.monotonic()
    child = subprocess.Popen(args.command, start_new_session=True)
    root = psutil.Process(child.pid)
    sample_count = 0
    peak_rss_bytes = 0
    peak_process_count = 0
    interrupted_signal: int | None = None

    def forward_signal(signum: int, _frame: object) -> None:
        nonlocal interrupted_signal
        interrupted_signal = signum
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signum)

    previous_handlers = {
        signum: signal.signal(signum, forward_signal) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    psutil.cpu_percent()
    psutil.cpu_percent(percpu=True)

    try:
        with args.samples.open("w", encoding="utf-8", buffering=1) as stream:
            while child.poll() is None:
                process_sample = _sample_process_tree(root)
                peak_rss_bytes = max(peak_rss_bytes, int(process_sample["rss_bytes"]))
                peak_process_count = max(peak_process_count, int(process_sample["process_count"]))
                sample = {
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "elapsed_s": time.monotonic() - started_monotonic,
                    "label": args.label,
                    "process_tree": process_sample,
                    "host": _sample_host(),
                }
                stream.write(json.dumps(sample, separators=(",", ":")) + "\n")
                sample_count += 1
                time.sleep(args.interval)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    returncode = child.wait()
    finished_at = datetime.now(UTC)
    summary = {
        "label": args.label,
        "command": args.command,
        "returncode": returncode,
        "interrupted_signal": interrupted_signal,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "wall_s": time.monotonic() - started_monotonic,
        "sample_interval_s": args.interval,
        "sample_count": sample_count,
        "allocated_cpu_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "peak_process_tree_rss_bytes": peak_rss_bytes,
        "peak_process_count": peak_process_count,
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
