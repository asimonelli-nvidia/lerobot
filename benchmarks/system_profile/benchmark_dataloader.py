#!/usr/bin/env python

"""Measure LeRobot's input pipeline independently from model execution.

The benchmark deliberately consumes batches back-to-back. This measures the loader's
service rate rather than the usually tiny wait observed when model compute gives workers
enough time to prefetch. It uses the same dataset construction, sampler, transforms, video
backend, multiprocessing context, and batching path as training.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.transforms import ImageTransformsConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-profile", choices=("droid", "libero"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--model-profile", choices=("groot", "diffusion"), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= args.warmup_steps:
        parser.error("steps must be greater than warmup-steps")
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers non-negative")
    if args.model_profile == "groot" and args.model_path is None:
        parser.error("--model-path is required for the GR00T loader recipe")
    return args


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "stddev": statistics.pstdev(values),
        "min": min(values),
        "p50": quantile(values, 0.50),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": max(values),
    }


def make_policy_config(args: argparse.Namespace):
    if args.model_profile == "groot":
        return GrootConfig(
            device="cuda",
            base_model_path=str(args.model_path),
            embodiment_tag="new_embodiment" if args.dataset_profile == "droid" else "libero_sim",
            chunk_size=16 if args.dataset_profile == "droid" else 40,
            n_action_steps=16 if args.dataset_profile == "droid" else 40,
            use_relative_actions=args.dataset_profile == "droid",
            relative_exclude_joints=["gripper"] if args.dataset_profile == "droid" else [],
            push_to_hub=False,
        )
    horizon = 16 if args.dataset_profile == "droid" else 32
    return DiffusionConfig(
        device="cuda",
        horizon=horizon,
        n_action_steps=horizon // 2,
        drop_n_last_frames=horizon - horizon // 2 - 2 + 1,
        pretrained_backbone_weights=None,
        push_to_hub=False,
    )


def make_dataset(args: argparse.Namespace):
    episodes = list(range(100)) if args.dataset_profile == "droid" else None
    dataset_cfg = DatasetConfig(
        repo_id=args.dataset_repo_id,
        root=str(args.dataset_root),
        episodes=episodes,
        revision="main",
        video_backend="torchcodec",
        image_transforms=ImageTransformsConfig(enable=True, max_num_transforms=4),
    )
    config = TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=make_policy_config(args),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.num_workers > 0,
        dataloader_multiprocessing_context="spawn" if args.num_workers > 0 else None,
        seed=args.seed,
        steps=args.steps,
        save_checkpoint=False,
        env_eval_freq=0,
        eval_steps=0,
    )
    dataset, _ = make_train_eval_datasets(config)
    return dataset


def grouping_opportunity(dataset: Any, batch_size: int, seed: int, batches: int = 12) -> dict[str, float]:
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=0,
        shuffle=True,
        seed=seed,
        absolute_to_relative_idx=dataset.absolute_to_relative_idx,
    )
    iterator = iter(sampler)
    total_video_requests = 0
    unique_video_groups = 0
    total_temporal_indices = 0
    unique_temporal_indices = 0
    reader = dataset.reader
    video_keys = list(getattr(reader, "_selected_video_keys", dataset.meta.video_keys))

    for _ in range(batches):
        indices = []
        for _ in range(batch_size):
            try:
                indices.append(next(iterator))
            except StopIteration:
                break
        if not indices:
            break
        rows = dataset.hf_dataset.select_columns(["index", "episode_index"])[indices]
        groups: set[tuple[str, str]] = set()
        temporal: set[int] = set()
        temporal_count = 0
        for absolute, episode in zip(rows["index"], rows["episode_index"], strict=True):
            absolute_idx = int(absolute.item() if hasattr(absolute, "item") else absolute)
            episode_idx = int(episode.item() if hasattr(episode, "item") else episode)
            for video_key in video_keys:
                groups.add((str(dataset.meta.get_video_file_path(episode_idx, video_key)), video_key))
                total_video_requests += 1
            if reader.delta_indices is not None:
                query_indices, _ = reader._get_query_indices(absolute_idx, episode_idx)
                for values in query_indices.values():
                    temporal_count += len(values)
                    temporal.update(values)
        unique_video_groups += len(groups)
        total_temporal_indices += temporal_count
        unique_temporal_indices += len(temporal)

    return {
        "video_requests_per_file": total_video_requests / max(unique_video_groups, 1),
        "video_groupable_fraction": 1 - unique_video_groups / max(total_video_requests, 1),
        "temporal_dedup_fraction": 1 - unique_temporal_indices / max(total_temporal_indices, 1),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    dataset = make_dataset(args)
    opportunity = grouping_opportunity(dataset, args.batch_size, args.seed)
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=getattr(make_policy_config(args), "drop_n_last_frames", 0),
        shuffle=True,
        seed=args.seed,
        absolute_to_relative_idx=dataset.absolute_to_relative_idx,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
    )
    iterator = iter(loader)
    rows: list[dict[str, float | int | str]] = []
    measurement_started: str | None = None
    measurement_ended: str | None = None
    args.samples.parent.mkdir(parents=True, exist_ok=True)
    with args.samples.open("w", encoding="utf-8", buffering=1) as stream:
        for step in range(1, args.steps + 1):
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            wait_s = time.perf_counter() - started
            observed_batch = len(next(iter(batch.values())))
            timestamp = datetime.now(timezone.utc).isoformat()  # noqa: UP017 (cluster uses Python 3.10)
            if step == args.warmup_steps + 1:
                measurement_started = timestamp
            if step > args.warmup_steps:
                row = {
                    "step": step,
                    "timestamp_utc": timestamp,
                    "batch_size": observed_batch,
                    "batch_wait_s": wait_s,
                    "samples_per_s": observed_batch / max(wait_s, 1e-9),
                }
                rows.append(row)
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                print(
                    f"step:{step} wait_s:{wait_s:.6f} smp/s:{row['samples_per_s']:.3f}",
                    flush=True,
                )
            measurement_ended = timestamp
            del batch

    waits = [float(row["batch_wait_s"]) for row in rows]
    total_samples = sum(int(row["batch_size"]) for row in rows)
    total_wait = sum(waits)
    result = {
        "schema_version": "1.0",
        "target": "dataloading",
        "dataset_profile": args.dataset_profile,
        "model_profile": args.model_profile,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "measurement_started_utc": measurement_started,
        "measurement_ended_utc": measurement_ended,
        "measured_batches": len(rows),
        "measured_samples": total_samples,
        "wall_s": total_wait,
        "samples_per_s": total_samples / max(total_wait, 1e-9),
        "batch_wait_s": stats(waits),
        "batch_samples_per_s": stats([float(row["samples_per_s"]) for row in rows]),
        "grouping_opportunity": opportunity,
    }
    # Persist the complete measurement window before worker teardown. TorchCodec
    # occasionally aborts a worker while releasing decoder state after the final
    # batch; that cleanup event must not erase otherwise complete evidence.
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Explicitly stop persistent workers before this short benchmark process exits.
    # On shared Slurm nodes, leaving shutdown to interpreter finalization can race
    # with the next paired run and invalidate multiprocessing semaphores.
    shutdown_workers = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown_workers):
        try:
            shutdown_workers()
        except Exception as error:  # noqa: BLE001 - retain completed evidence on cleanup failure
            print(f"warning: DataLoader cleanup after completed measurement: {error}", flush=True)
    del iterator, loader
    gc.collect()


if __name__ == "__main__":
    main()
