#!/usr/bin/env python

"""Build a deterministic DROID subset with precomputed GR00T relative-action stats."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="lerobot/droid_1.0.1")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--action-horizon", type=int, default=40)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--refresh-stats-only",
        action="store_true",
        help="Replace relative-action stats in an already prepared output directory.",
    )
    return parser.parse_args()


def copy_file(source: Path, output: Path, *, hardlink: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        try:
            os.link(source, output)
            return
        except OSError:
            pass
    shutil.copy2(source, output)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def filter_parquet(source: Path, output: Path, episodes: int) -> int:
    table = pq.read_table(source)
    selected = table.filter(pc.less(table["episode_index"], episodes))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    pq.write_table(selected, temporary, compression="zstd")
    temporary.replace(output)
    return selected.num_rows


def compute_horizon_relative_stats(dataset_root: Path, action_horizon: int) -> dict[str, list]:
    """Compute the horizon-preserving relative stats consumed by GR00T N1.7."""
    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    table = pq.read_table(dataset_root / "data/chunk-000/file-000.parquet")
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    episode_indices = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    if actions.ndim != 2 or states.ndim != 2 or actions.shape[0] != states.shape[0]:
        raise ValueError(f"unexpected action/state shapes: {actions.shape}, {states.shape}")
    if action_horizon <= 0 or len(episode_indices) < action_horizon:
        raise ValueError(f"invalid action horizon {action_horizon} for {len(episode_indices)} frames")

    starts = np.arange(len(episode_indices) - action_horizon + 1)
    starts = starts[episode_indices[starts] == episode_indices[starts + action_horizon - 1]]
    if len(starts) < 2:
        raise ValueError("fewer than two complete action chunks are available")

    offsets = np.arange(action_horizon)
    chunks = actions[starts[:, None] + offsets[None, :]].copy()
    raw_names = info["features"]["action"].get("names")
    if isinstance(raw_names, dict):
        raw_names = next((value for value in raw_names.values() if isinstance(value, list)), None)
    names = list(raw_names or [str(index) for index in range(actions.shape[1])])
    relative_mask = np.asarray([0.0 if name == "gripper" else 1.0 for name in names], dtype=np.float32)
    chunks -= states[starts, None, : actions.shape[1]] * relative_mask[None, None, :]

    computed = {
        "min": np.min(chunks, axis=0),
        "max": np.max(chunks, axis=0),
        "mean": np.mean(chunks, axis=0),
        "std": np.std(chunks, axis=0),
        "q01": np.quantile(chunks, 0.01, axis=0).astype(np.float32),
        "q99": np.quantile(chunks, 0.99, axis=0).astype(np.float32),
        "count": np.full(action_horizon, len(starts), dtype=np.int64),
    }
    return {key: value.tolist() for key, value in computed.items()}


def write_relative_stats(dataset_root: Path, action_horizon: int) -> None:
    stats_path = dataset_root / "meta/stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["action"] = compute_horizon_relative_stats(dataset_root, action_horizon)
    write_json(stats_path, stats)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        if args.refresh_stats_only:
            write_relative_stats(args.output, args.action_horizon)
            return
        raise FileExistsError(f"output already exists: {args.output}")
    if args.refresh_stats_only:
        raise FileNotFoundError(f"output does not exist: {args.output}")

    info_path = args.source / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if args.episodes <= 0 or args.episodes > int(info["total_episodes"]):
        raise ValueError(f"episodes must be in [1, {info['total_episodes']}]")

    args.output.mkdir(parents=True)
    for relative in (".gitattributes", "README.md", "meta/stats.json", "meta/tasks.parquet"):
        source = args.source / relative
        if source.exists():
            copy_file(source, args.output / relative)

    data_rows = filter_parquet(
        args.source / "data/chunk-000/file-000.parquet",
        args.output / "data/chunk-000/file-000.parquet",
        args.episodes,
    )
    episode_rows = filter_parquet(
        args.source / "meta/episodes/chunk-000/file-000.parquet",
        args.output / "meta/episodes/chunk-000/file-000.parquet",
        args.episodes,
    )
    if episode_rows != args.episodes:
        raise RuntimeError(f"expected {args.episodes} episode rows, found {episode_rows}")

    for key, feature in info["features"].items():
        if feature.get("dtype") != "video":
            continue
        copy_file(
            args.source / f"videos/{key}/chunk-000/file-000.mp4",
            args.output / f"videos/{key}/chunk-000/file-000.mp4",
            hardlink=True,
        )

    info["total_episodes"] = args.episodes
    info["total_frames"] = data_rows
    info["splits"] = {"train": f"0:{args.episodes}"}
    write_json(args.output / "meta/info.json", info)

    write_relative_stats(args.output, args.action_horizon)
    write_json(
        args.output / "meta/benchmark-subset.json",
        {
            "schema_version": "1.0",
            "source_repo_id": args.repo_id,
            "selection": {"episode_indices": [0, args.episodes - 1], "count": args.episodes},
            "action_stats": {
                "representation": "relative_except_gripper",
                "horizon": args.action_horizon,
            },
            "purpose": "deterministic system-throughput benchmark; not a quality benchmark",
        },
    )


if __name__ == "__main__":
    main()
