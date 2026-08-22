#!/usr/bin/env python

"""Rebuild derived training summaries while preserving the original summaries and raw evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--revision", required=True, help="Revision containing the summarizer used")
    parser.add_argument("--write", action="store_true", help="Write summaries; otherwise print the plan")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    summarizer = script_dir / "summarize_run.py"
    experiment_summarizer = script_dir / "summarize_experiment.py"
    jobs: list[tuple[Path, int]] = []
    for root in args.roots:
        for manifest_path in root.rglob("system/run-manifest.json"):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("target") != "training_throughput":
                continue
            jobs.append((manifest_path.parent.parent, int(manifest["training"]["warmup_steps"])))

    rebuilt = 0
    for job_dir, warmup_steps in sorted(set(jobs)):
        runs = sorted(path for path in job_dir.iterdir() if path.is_dir() and path.name.startswith("r"))
        complete_runs = [
            run
            for run in runs
            if all((run / name).is_file() for name in ("console.log", "gpu.csv", "system.jsonl", "process-summary.json"))
        ]
        print(f"{job_dir}: {len(complete_runs)} runs, warm-up {warmup_steps}")
        if not args.write:
            continue
        for run in complete_runs:
            output = run / "summary.json"
            backup = run / "summary.pre-gpu-selection.json"
            if output.is_file() and not backup.exists():
                shutil.copy2(output, backup)
            subprocess.run(
                [
                    sys.executable,
                    str(summarizer),
                    "--console",
                    str(run / "console.log"),
                    "--gpu",
                    str(run / "gpu.csv"),
                    "--system",
                    str(run / "system.jsonl"),
                    "--process-summary",
                    str(run / "process-summary.json"),
                    "--warmup-steps",
                    str(warmup_steps),
                    "--output",
                    str(output),
                ],
                check=True,
            )
            rebuilt += 1
        subprocess.run([sys.executable, str(experiment_summarizer), str(job_dir)], check=True)
        record = {
            "reprocessed_at_utc": datetime.now(UTC).isoformat(),
            "reason": "select the job-active GPU from shared-node telemetry",
            "summarizer_revision": args.revision,
            "runs_rebuilt": len(complete_runs),
            "original_summaries": "summary.pre-gpu-selection.json",
        }
        (job_dir / "system" / "summary-reprocessing.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"rebuilt {rebuilt} run summaries")


if __name__ == "__main__":
    main()
