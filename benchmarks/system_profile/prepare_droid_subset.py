#!/usr/bin/env python

"""Build a deterministic DROID subset with precomputed GR00T relative-action stats."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import compute_relative_action_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="lerobot/droid_1.0.1")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--action-horizon", type=int, default=40)
    parser.add_argument("--workers", type=int, default=16)
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


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

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

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.output,
        episodes=list(range(args.episodes)),
        download_videos=False,
    )
    relative_stats = compute_relative_action_stats(
        hf_dataset=dataset.hf_dataset,
        features=dataset.meta.features,
        chunk_size=args.action_horizon,
        exclude_joints=["gripper"],
        num_workers=args.workers,
    )
    stats_path = args.output / "meta/stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats[ACTION] = {key: value.tolist() for key, value in relative_stats.items()}
    write_json(stats_path, stats)
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
