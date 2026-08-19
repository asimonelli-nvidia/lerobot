#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Private reader component for LeRobotDataset. Handles random-access reading (HF dataset, delta indices, video decoding)."""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import datasets
import torch

from lerobot.configs import (
    DEFAULT_DEPTH_UNIT,
    DEPTH_METER_UNIT,
    DepthEncoderConfig,
)
from lerobot.utils.constants import DEFAULT_FEATURES

from .dataset_metadata import LeRobotDatasetMetadata
from .depth_utils import MM_PER_METRE, dequantize_depth
from .feature_utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_hf_features_from_features,
)
from .io_utils import (
    hf_transform_to_torch,
    load_nested_dataset,
)
from .utils import resolve_episode_indices
from .video_utils import decode_video_frames


class DatasetReader:
    """Encapsulates read-side state and methods for LeRobotDataset.

    Owns: hf_dataset, _absolute_to_relative_idx, delta_indices.
    """

    def __init__(
        self,
        meta: LeRobotDatasetMetadata,
        root: Path,
        episodes: list[int] | None,
        tolerance_s: float,
        video_backend: str,
        delta_timestamps: dict[str, list[float]] | None,
        image_transforms: Callable | None,
        return_uint8: bool = False,
        depth_output_unit: str = DEFAULT_DEPTH_UNIT,
        feature_keys: list[str] | None = None,
    ):
        """Initialize the reader with metadata, filtering, and transform config.

        The HF dataset is not loaded here — call :meth:`try_load` or
        :meth:`load_and_activate` afterward.

        Args:
            meta: Dataset metadata instance.
            root: Local dataset root directory.
            episodes: Optional list of episode indices to select. ``None``
                means all episodes.
            tolerance_s: Timestamp synchronization tolerance in seconds.
            video_backend: Video decoding backend identifier.
            delta_timestamps: Optional dict mapping feature keys to lists of
                relative timestamp offsets for temporal context windows.
            image_transforms: Optional torchvision v2 transform applied to
                visual features.
            return_uint8: If True, return RGB video frames as raw uint8 tensors
                instead of normalized float32.
            feature_keys: Optional dataset features to materialize. LeRobot's default
                metadata features are retained automatically.
            depth_output_unit: Physical unit depth maps are dequantized to
                (``"m"`` or ``"mm"``). Defaults to ``"mm"``.
        """
        self._meta = meta
        self.root = root
        self.episodes = resolve_episode_indices(episodes, meta.total_episodes)
        self._tolerance_s = tolerance_s
        self._video_backend = video_backend
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self._image_transforms = image_transforms
        self._return_uint8 = return_uint8
        self._depth_output_unit = depth_output_unit

        if feature_keys is None:
            self.feature_keys = None
            selected_feature_keys = set(meta.features)
        else:
            unknown_feature_keys = sorted(set(feature_keys) - set(meta.features))
            if unknown_feature_keys:
                raise ValueError(f"Unknown dataset feature key(s): {unknown_feature_keys}")
            selected_feature_keys = set(feature_keys)
            # Retain the metadata columns used to plan temporal and video reads.
            selected_feature_keys.update(DEFAULT_FEATURES)
            self.feature_keys = [key for key in meta.features if key in selected_feature_keys]
        self._selected_video_keys = [key for key in meta.video_keys if key in selected_feature_keys]

        self.hf_dataset: datasets.Dataset | None = None
        self._absolute_to_relative_idx: dict[int, int] | None = None

        # Setup delta_indices (doesn't depend on hf_dataset)
        self.delta_indices = None
        if delta_timestamps is not None:
            delta_timestamps = {
                key: timestamps
                for key, timestamps in delta_timestamps.items()
                if key in selected_feature_keys
            }
            if delta_timestamps:
                check_delta_timestamps(delta_timestamps, meta.fps, tolerance_s)
                self.delta_indices = get_delta_indices(delta_timestamps, meta.fps)

        self._depth_encoder_configs: dict[str, DepthEncoderConfig] = {
            vid_key: DepthEncoderConfig.from_video_info(self._meta.features[vid_key].get("info"))
            for vid_key in self._meta.depth_keys
            if vid_key in selected_feature_keys
        }

        # Get the input unit of each depth feature stored as raw images.
        self._image_depth_units: dict[str, str | None] = {
            key: (self._meta.features[key].get("info") or {}).get("depth_unit")
            for key in self._meta.depth_keys
            if key in self._meta.image_keys and key in selected_feature_keys
        }

    def set_image_transforms(self, image_transforms: Callable | None) -> None:
        """Replace the transform applied to visual observations."""
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self._image_transforms = image_transforms

    def clear_image_transforms(self) -> None:
        """Remove the transform applied to visual observations."""
        self._image_transforms = None

    def try_load(self) -> bool:
        """Attempt to load from local cache. Returns True if data is sufficient."""
        try:
            self.hf_dataset = self._load_hf_dataset()
        except (FileNotFoundError, NotADirectoryError):
            self.hf_dataset = None
            return False
        if not self._check_cached_episodes_sufficient():
            self.hf_dataset = None
            return False
        self._build_index_mapping()
        return True

    def load_and_activate(self) -> None:
        """Load HF dataset from disk and build index mapping. Call after data is on disk."""
        self.hf_dataset = self._load_hf_dataset()
        self._build_index_mapping()

    def _build_index_mapping(self) -> None:
        """Build absolute-to-relative index mapping from loaded hf_dataset."""
        self._absolute_to_relative_idx = None
        if self.episodes is not None and self.hf_dataset is not None:
            indices = self.hf_dataset.data.column("index").to_numpy()
            self._absolute_to_relative_idx = dict(zip(indices.tolist(), range(len(indices)), strict=True))

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        if self.episodes is not None and self.hf_dataset is not None:
            return len(self.hf_dataset)
        return self._meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self._meta.total_episodes

    def _load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        declared_features = get_hf_features_from_features(self._meta.features)
        self._validate_language_columns_declared(declared_features)
        columns = None
        features = declared_features
        if self.feature_keys is not None:
            columns = [key for key in self.feature_keys if key in declared_features]
            features = datasets.Features({key: declared_features[key] for key in columns})
        hf_dataset = load_nested_dataset(
            self.root / "data", features=features, episodes=self.episodes, columns=columns
        )
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _validate_language_columns_declared(self, features: datasets.Features) -> None:
        """Require language columns stored in Parquet to be declared in metadata."""
        # Leave empty datasets to fail through the normal loading path.
        try:
            sample = next((self.root / "data").glob("*/*.parquet"))
        except StopIteration:
            return

        from pyarrow import parquet as _pq  # noqa: PLC0415

        # LeRobot shards are schema-uniform, so one schema represents the dataset.
        schema_names = set(_pq.read_schema(sample).names)
        from .language import LANGUAGE_COLUMNS  # noqa: PLC0415

        missing = sorted(set(LANGUAGE_COLUMNS) & schema_names - set(features))
        if missing:
            raise ValueError(
                f"Dataset Parquet files contain language feature(s) missing from metadata: {missing}. "
                "Metadata must describe the stored data; add the entries returned by "
                "lerobot.datasets.language.language_feature_info() to meta/info.json['features'] "
                "or rerun the annotation pipeline's metadata synchronization."
            )

    def _check_cached_episodes_sufficient(self) -> bool:
        """Check if the cached dataset contains all requested episodes and their video files."""
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False

        available_episodes = {
            ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx
            for ep_idx in self.hf_dataset.unique("episode_index")
        }

        if self.episodes is None:
            requested_episodes = set(range(self._meta.total_episodes))
        else:
            requested_episodes = set(self.episodes)

        if not requested_episodes.issubset(available_episodes):
            return False

        if self._selected_video_keys:
            for ep_idx in requested_episodes:
                for vid_key in self._selected_video_keys:
                    video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
                    if not video_path.exists():
                        return False

        return True

    def get_episodes_file_paths(self) -> list[Path]:
        """Return deduplicated file paths (data + video) for selected episodes.

        Used to build the ``allow_patterns`` list for ``snapshot_download``.
        """
        episodes = self.episodes if self.episodes is not None else list(range(self._meta.total_episodes))
        fpaths = [str(self._meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if self._selected_video_keys:
            video_files = [
                str(self._meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self._selected_video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files
        # episodes are stored in the same files, so we return unique paths only
        fpaths = list(set(fpaths))
        return fpaths

    def _get_query_indices(
        self, abs_idx: int, ep_idx: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        """Compute query indices for delta timestamps."""
        ep = self._meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self._selected_video_keys:
            if query_indices is not None and key in query_indices:
                if self._absolute_to_relative_idx is not None:
                    relative_indices = [self._absolute_to_relative_idx[idx] for idx in query_indices[key]]
                    timestamps = self.hf_dataset[relative_indices]["timestamp"]
                else:
                    timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """Query dataset for indices across keys, skipping video keys."""
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self._selected_video_keys:
                continue
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            try:
                result[key] = torch.stack(self.hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result

    def _decode_video(self, video_path: Path, timestamps: Sequence[float], video_key: str) -> torch.Tensor:
        frames = decode_video_frames(
            video_path,
            list(timestamps),
            self._tolerance_s,
            self._video_backend,
            return_uint8=self._return_uint8,
            is_depth=video_key in self._meta.depth_keys,
        )
        if video_key in self._meta.depth_keys:
            depth_encoder = self._depth_encoder_configs[video_key]
            frames = dequantize_depth(
                frames,
                depth_min=depth_encoder.depth_min,
                depth_max=depth_encoder.depth_max,
                shift=depth_encoder.shift,
                use_log=depth_encoder.use_log,
                output_unit=self._depth_output_unit,
            )
        return frames

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault.
        """
        ep = self._meta.episodes[ep_idx]

        def _decode_single(vid_key: str, query_ts: list[float]) -> tuple[str, torch.Tensor]:
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]
            video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
            frames = self._decode_video(video_path, shifted_query_ts, vid_key)
            return vid_key, frames.squeeze(0)

        items = list(query_timestamps.items())

        # Single camera: no threading overhead
        if len(items) <= 1:
            return {vid_key: _decode_single(vid_key, query_ts)[1] for vid_key, query_ts in items}

        # Multi-camera: decode in parallel (video decoding releases the GIL)
        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            futures = [pool.submit(_decode_single, k, ts) for k, ts in items]
            return dict(f.result() for f in futures)

    def _query_videos_batch(
        self,
        requests_by_video: dict[tuple[Path, str], list[tuple[int, tuple[float, ...]]]],
    ) -> dict[tuple[int, str], torch.Tensor]:
        """Decode each physical video once and scatter frames to their batch destinations."""

        def _decode_group(
            group: tuple[tuple[Path, str], list[tuple[int, tuple[float, ...]]]],
        ) -> list[tuple[tuple[int, str], torch.Tensor]]:
            (video_path, video_key), requests = group
            timestamps = [timestamp for _, request in requests for timestamp in request]
            frames = self._decode_video(video_path, timestamps, video_key)

            decoded = []
            offset = 0
            for item_index, request in requests:
                end = offset + len(request)
                decoded.append(((item_index, video_key), frames[offset:end].squeeze(0)))
                offset = end
            return decoded

        groups = list(requests_by_video.items())
        max_workers = min(len(groups), len(self._selected_video_keys))
        if max_workers <= 1:
            decoded_groups = [_decode_group(group) for group in groups]
        else:
            # Match the existing multi-camera path while bounding work by selected cameras.
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                decoded_groups = list(pool.map(_decode_group, groups))

        return {destination: frames for group in decoded_groups for destination, frames in group}

    @staticmethod
    def _unbatch_hf_rows(rows: dict, size: int) -> list[dict]:
        return [{key: values[index] for key, values in rows.items()} for index in range(size)]

    def _relative_index(self, absolute_index: int) -> int:
        if self._absolute_to_relative_idx is None:
            return absolute_index
        return self._absolute_to_relative_idx[absolute_index]

    def get_items(self, indices: Sequence[int]) -> list[dict]:
        """Read a batch while sharing metadata and video work across items.

        Indices are relative to the selected dataset. Returned items preserve their order and duplicates.
        """
        indices = [int(index) for index in indices]
        if not indices:
            return []

        # Load all base rows in one Arrow lookup.
        items = self._unbatch_hf_rows(self.hf_dataset[indices], len(indices))
        query_plans: list[tuple[dict[str, list[int]] | None, dict[str, torch.Tensor]]] = []
        temporal_indices: set[int] = set()
        temporal_columns: set[str] = set()

        # Plan temporal reads for the full batch and deduplicate overlapping indices.
        for item in items:
            if self.delta_indices is None:
                query_plans.append((None, {}))
                continue

            query_indices, padding = self._get_query_indices(
                int(item["index"].item()), int(item["episode_index"].item())
            )
            query_plans.append((query_indices, padding))
            for key, absolute_indices in query_indices.items():
                temporal_indices.update(self._relative_index(index) for index in absolute_indices)
                temporal_columns.add("timestamp" if key in self._selected_video_keys else key)

        # Reuse base rows and fetch every missing temporal row in one additional Arrow lookup.
        rows_by_index = dict(zip(indices, (dict(item) for item in items), strict=True))
        missing_indices = sorted(temporal_indices - rows_by_index.keys())
        if missing_indices:
            columns = [key for key in self.hf_dataset.column_names if key in temporal_columns]
            temporal_rows = self._unbatch_hf_rows(
                self.hf_dataset.select_columns(columns)[missing_indices], len(missing_indices)
            )
            rows_by_index.update(zip(missing_indices, temporal_rows, strict=True))

        # Scatter temporal values and padding flags back to each item.
        for item, (query_indices, padding) in zip(items, query_plans, strict=True):
            item.update(padding)
            if query_indices is None:
                continue
            for key, absolute_indices in query_indices.items():
                if key in self._selected_video_keys:
                    continue
                relative_indices = [self._relative_index(index) for index in absolute_indices]
                item[key] = torch.stack([rows_by_index[index][key] for index in relative_indices])

        # Group requests that share a physical video before decoding.
        requests_by_video: dict[tuple[Path, str], list[tuple[int, tuple[float, ...]]]] = {}
        for item_index, (item, (query_indices, _padding)) in enumerate(zip(items, query_plans, strict=True)):
            episode_index = int(item["episode_index"].item())
            episode = self._meta.episodes[episode_index]
            for video_key in self._selected_video_keys:
                if query_indices is not None and video_key in query_indices:
                    relative_indices = [self._relative_index(index) for index in query_indices[video_key]]
                    timestamps = tuple(
                        float(rows_by_index[index]["timestamp"].item()) for index in relative_indices
                    )
                else:
                    timestamps = (float(item["timestamp"].item()),)

                from_timestamp = episode[f"videos/{video_key}/from_timestamp"]
                video_path = self.root / self._meta.get_video_file_path(episode_index, video_key)
                requests_by_video.setdefault((video_path, video_key), []).append(
                    (item_index, tuple(from_timestamp + timestamp for timestamp in timestamps))
                )

        for (item_index, video_key), frames in self._query_videos_batch(requests_by_video).items():
            items[item_index][video_key] = frames

        return [self._finalize_item(item) for item in items]

    def get_item(self, idx) -> dict:
        """Core __getitem__ logic. Assumes hf_dataset is loaded.

        ``idx`` is a *relative* index into the (possibly episode-filtered)
        HF dataset, **not** the absolute frame index stored in the ``index``
        column.  The absolute index is retrieved from the row itself.
        """
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if self._selected_video_keys:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        return self._finalize_item(item)

    def _finalize_item(self, item: dict) -> dict:
        """Apply transforms and derived fields shared by scalar and batched reads."""
        if self._image_transforms is not None:
            for cam in self._meta.camera_keys:
                if cam not in item or cam in self._meta.depth_keys:
                    continue
                item[cam] = self._image_transforms(item[cam])

        # Convert depth features to the output unit.
        for key, stored_unit in self._image_depth_units.items():
            if key in item and stored_unit is not None and stored_unit != self._depth_output_unit:
                item[key] = (
                    item[key] * MM_PER_METRE if stored_unit == DEPTH_METER_UNIT else item[key] / MM_PER_METRE
                )

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self._meta.tasks.iloc[task_idx].name

        return item
