#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
This script will help you convert any LeRobot dataset from codebase version 3.0 back to 2.1.
It will:

- Split merged data parquet files back into per-episode parquet files.
- Split concatenated video files back into per-episode video files using timestamps.
- Convert meta/episodes parquet back to episodes.jsonl and episodes_stats.jsonl.
- Convert meta/tasks.parquet back to tasks.jsonl.
- Revert meta/info.json to v2.1 format.

Usage:

Convert a local dataset (works in place):
```bash
python src/lerobot/scripts/convert_dataset_v30_to_v21.py \
    --repo-id=lerobot/pusht \
    --root=/path/to/local/dataset/directory
```

Convert a dataset from the hub:
```bash
python src/lerobot/scripts/convert_dataset_v30_to_v21.py \
    --repo-id=lerobot/pusht
```
"""

import argparse
import logging
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from lerobot.utils.import_utils import require_package

require_package("jsonlines", extra="dataset")

import jsonlines
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tqdm
from datasets import Dataset
from huggingface_hub import snapshot_download

from lerobot.datasets.io_utils import (
    load_info,
    load_stats,
    load_tasks,
)
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_PATH,
    EPISODES_DIR,
    INFO_PATH,
    LEGACY_EPISODES_PATH,
    LEGACY_EPISODES_STATS_PATH,
    LEGACY_TASKS_PATH,
)
from lerobot.datasets.video_utils import get_video_duration_in_s
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.io_utils import load_json, write_json
from lerobot.utils.utils import init_logging, unflatten_dict

V21 = "v2.1"
V30 = "v3.0"

"""
This script reverses the v2.1 -> v3.0 conversion:

v3.0 -> v2.1 changes:
-------------------------
NEW (v3.0)
data/chunk-000/file-000.parquet  (multiple episodes merged)

OLD (v2.1)
data/chunk-000/episode_000000.parquet  (one per episode)
-------------------------
NEW (v3.0)
videos/CAMERA/chunk-000/file-000.mp4  (multiple episodes concatenated)

OLD (v2.1)
videos/chunk-000/CAMERA/episode_000000.mp4  (one per episode)
-------------------------
NEW (v3.0)
meta/episodes/chunk-000/file-000.parquet
  episode_index | stats/... | data/chunk_index | ...

OLD (v2.1)
meta/episodes.jsonl
  {"episode_index": 0, "tasks": [...], "length": 266}
meta/episodes_stats.jsonl
  {"episode_index": 0, "stats": {"feature_name": {"min": ..., ...}}}
-------------------------
NEW (v3.0)
meta/tasks.parquet

OLD (v2.1)
meta/tasks.jsonl
  {"task_index": 0, "task": "..."}
-------------------------
UPDATE
meta/info.json
-------------------------
"""


def _convert_numpy_to_native(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return [_convert_numpy_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_to_native(x) for x in obj]
    return obj


def write_jsonlines(fpath: Path, data: list[dict]) -> None:
    """Write a list of dicts to a jsonlines file."""
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(fpath, mode="w") as writer:
        for item in data:
            writer.write(_convert_numpy_to_native(item))


def load_episodes_full(local_dir: Path) -> Dataset:
    """Load episodes parquet including stats columns."""
    from lerobot.datasets.io_utils import load_nested_dataset

    return load_nested_dataset(local_dir / EPISODES_DIR)


def validate_local_dataset_version(local_path: Path) -> None:
    """Validate that the local dataset has the expected v3.0 version."""
    info = load_info(local_path)
    dataset_version = info.codebase_version or "unknown"
    if dataset_version != V30:
        raise ValueError(
            f"Local dataset has codebase version '{dataset_version}', expected '{V30}'. "
            f"This script is specifically for converting v3.0 datasets to v2.1."
        )


def convert_tasks_to_v21(root: Path, new_root: Path) -> None:
    """Convert meta/tasks.parquet -> meta/tasks.jsonl"""
    logging.info("Converting tasks to v2.1 format")
    tasks_df = load_tasks(root)
    # tasks_df has index named "task" and column "task_index"
    tasks_list = []
    for task_str, row in tasks_df.iterrows():
        tasks_list.append({"task_index": int(row["task_index"]), "task": str(task_str)})
    tasks_list.sort(key=lambda x: x["task_index"])
    write_jsonlines(new_root / LEGACY_TASKS_PATH, tasks_list)


def convert_data_to_v21(root: Path, new_root: Path, episodes_df: pd.DataFrame) -> None:
    """Split merged data parquet files back into per-episode parquet files.

    v3.0: data/chunk-000/file-000.parquet (multiple episodes)
    v2.1: data/chunk-000/episode_000000.parquet (one per episode)
    """
    logging.info("Converting data files to v2.1 format (per-episode parquet)")

    num_episodes = len(episodes_df)
    # Determine chunk size for v2.1 output
    chunk_size = DEFAULT_CHUNK_SIZE

    # Group episodes by their source data file (chunk_index, file_index)
    # Read merged parquet files and split by episode_index
    data_dir = root / "data"
    all_parquet_paths = sorted(data_dir.glob("*/*.parquet"))

    if len(all_parquet_paths) == 0:
        raise FileNotFoundError(f"No data parquet files found in {data_dir}")

    # Read all data into one dataframe (for simplicity; for very large datasets
    # this could be done per-file with episode index filtering)
    logging.info(f"Reading {len(all_parquet_paths)} data file(s)...")
    frames = []
    for path in all_parquet_paths:
        frames.append(pd.read_parquet(path))
    all_data = pd.concat(frames, ignore_index=True)

    # The data has an "episode_index" column that identifies which episode each row belongs to
    if "episode_index" not in all_data.columns:
        raise ValueError("Data parquet files do not contain 'episode_index' column")

    for ep_idx in tqdm.tqdm(range(num_episodes), desc="split data files"):
        ep_data = all_data[all_data["episode_index"] == ep_idx]
        if len(ep_data) == 0:
            logging.warning(f"No data found for episode {ep_idx}, skipping")
            continue

        # v2.1 path: data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet
        chunk_idx = ep_idx // chunk_size
        out_path = new_root / f"data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ep_data.to_parquet(out_path, index=False)


def split_video_by_timestamps(
    input_video_path: Path,
    output_video_path: Path,
    start_timestamp: float,
    end_timestamp: float,
) -> None:
    """Split a segment from a video file using ffmpeg.

    Uses stream copy (no re-encoding) for speed. Falls back to re-encoding
    if stream copy produces issues.
    """
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    duration = end_timestamp - start_timestamp

    # Use ffmpeg with stream copy for fast extraction
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_timestamp),
        "-i", str(input_video_path),
        "-t", str(duration),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(output_video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Fallback: re-encode if stream copy fails
        logging.warning(
            f"Stream copy failed for {output_video_path.name}, falling back to re-encode. "
            f"stderr: {result.stderr[:200]}"
        )
        cmd_reencode = [
            "ffmpeg",
            "-y",
            "-ss", str(start_timestamp),
            "-i", str(input_video_path),
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            str(output_video_path),
        ]
        result2 = subprocess.run(cmd_reencode, capture_output=True, text=True)
        if result2.returncode != 0:
            raise RuntimeError(
                f"Failed to split video segment [{start_timestamp:.3f}, {end_timestamp:.3f}] "
                f"from {input_video_path}: {result2.stderr[:500]}"
            )


def convert_videos_to_v21(root: Path, new_root: Path, episodes_df: pd.DataFrame) -> None:
    """Split concatenated video files back into per-episode video files.

    v3.0: videos/CAMERA/chunk-000/file-000.mp4 (concatenated)
    v2.1: videos/chunk-000/CAMERA/episode_000000.mp4 (one per episode)
    """
    info = load_info(root)
    features = info.features
    video_keys = sorted([key for key, ft in features.items() if ft["dtype"] == "video"])

    if len(video_keys) == 0:
        logging.info("No video features found, skipping video conversion")
        return

    num_episodes = len(episodes_df)
    chunk_size = DEFAULT_CHUNK_SIZE

    for video_key in video_keys:
        logging.info(f"Converting videos for camera: {video_key}")

        # Check which columns are available for this video_key
        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"
        from_ts_col = f"videos/{video_key}/from_timestamp"
        to_ts_col = f"videos/{video_key}/to_timestamp"

        if from_ts_col not in episodes_df.columns or to_ts_col not in episodes_df.columns:
            logging.warning(
                f"Timestamp columns for {video_key} not found in episodes metadata. "
                f"Available columns: {list(episodes_df.columns)}"
            )
            continue

        # Group episodes by source video file
        # Build a mapping: (chunk_index, file_index) -> list of (ep_idx, from_ts, to_ts)
        video_file_episodes: dict[tuple[int, int], list[tuple[int, float, float]]] = {}
        for _, row in episodes_df.iterrows():
            ep_idx = int(row["episode_index"])
            v_chunk = int(row[chunk_col])
            v_file = int(row[file_col])
            from_ts = float(row[from_ts_col])
            to_ts = float(row[to_ts_col])
            key = (v_chunk, v_file)
            if key not in video_file_episodes:
                video_file_episodes[key] = []
            video_file_episodes[key].append((ep_idx, from_ts, to_ts))

        for (v_chunk, v_file), ep_list in video_file_episodes.items():
            # Source video path in v3.0 format
            src_video_path = root / DEFAULT_VIDEO_PATH.format(
                video_key=video_key, chunk_index=v_chunk, file_index=v_file
            )
            if not src_video_path.exists():
                logging.warning(f"Source video not found: {src_video_path}")
                continue

            for ep_idx, from_ts, to_ts in tqdm.tqdm(
                ep_list, desc=f"split {video_key} chunk-{v_chunk:03d}/file-{v_file:03d}"
            ):
                # v2.1 path: videos/chunk-{chunk_idx:03d}/{video_key}/episode_{ep_idx:06d}.mp4
                out_chunk_idx = ep_idx // chunk_size
                out_path = (
                    new_root
                    / f"videos/chunk-{out_chunk_idx:03d}/{video_key}/episode_{ep_idx:06d}.mp4"
                )
                split_video_by_timestamps(src_video_path, out_path, from_ts, to_ts)


def convert_episodes_to_v21(root: Path, new_root: Path, episodes_df: pd.DataFrame) -> None:
    """Convert episodes parquet to episodes.jsonl and episodes_stats.jsonl.

    v3.0: meta/episodes/chunk-000/file-000.parquet (wide table with stats)
    v2.1: meta/episodes.jsonl + meta/episodes_stats.jsonl
    """
    logging.info("Converting episodes metadata to v2.1 format")

    # Load tasks to map task indices back to task strings
    tasks_df = load_tasks(root)
    # tasks_df index is "task" (string), column is "task_index"
    task_index_to_task = {int(row["task_index"]): str(task_str) for task_str, row in tasks_df.iterrows()}

    episodes_list = []
    episodes_stats_list = []

    # Identify stats columns
    stats_columns = [col for col in episodes_df.columns if col.startswith("stats/")]
    non_stats_columns = [col for col in episodes_df.columns if not col.startswith("stats/")]

    for _, row in episodes_df.iterrows():
        ep_idx = int(row["episode_index"])

        # Build episodes.jsonl entry
        ep_entry: dict[str, Any] = {"episode_index": ep_idx}

        # Get tasks for this episode
        if "tasks" in row and row["tasks"] is not None:
            tasks_val = row["tasks"]
            if isinstance(tasks_val, (list, np.ndarray)):
                ep_entry["tasks"] = list(tasks_val)
            else:
                ep_entry["tasks"] = [str(tasks_val)]
        else:
            ep_entry["tasks"] = []

        # Get length
        if "length" in row:
            ep_entry["length"] = int(row["length"])
        elif "dataset_from_index" in row and "dataset_to_index" in row:
            ep_entry["length"] = int(row["dataset_to_index"]) - int(row["dataset_from_index"])

        episodes_list.append(ep_entry)

        # Build episodes_stats.jsonl entry
        if len(stats_columns) > 0:
            stats_flat = {}
            for col in stats_columns:
                # col is like "stats/feature_name/min"
                # Remove the "stats/" prefix for the nested structure
                stat_key = col[len("stats/"):]
                value = row[col]
                if isinstance(value, (list, np.ndarray)):
                    stats_flat[stat_key] = list(value) if isinstance(value, np.ndarray) else value
                elif isinstance(value, (np.floating, float)):
                    stats_flat[stat_key] = float(value)
                elif isinstance(value, (np.integer, int)):
                    stats_flat[stat_key] = int(value)
                else:
                    stats_flat[stat_key] = value

            # Unflatten: "feature_name/min" -> {"feature_name": {"min": ...}}
            stats_nested = unflatten_dict(stats_flat)
            episodes_stats_list.append({"episode_index": ep_idx, "stats": stats_nested})

    # Sort by episode_index
    episodes_list.sort(key=lambda x: x["episode_index"])
    episodes_stats_list.sort(key=lambda x: x["episode_index"])

    write_jsonlines(new_root / LEGACY_EPISODES_PATH, episodes_list)
    if episodes_stats_list:
        write_jsonlines(new_root / LEGACY_EPISODES_STATS_PATH, episodes_stats_list)


def convert_info_to_v21(root: Path, new_root: Path, num_episodes: int) -> None:
    """Revert info.json from v3.0 to v2.1 format."""
    logging.info("Converting info.json to v2.1 format")

    info = load_json(root / INFO_PATH)

    # Change version
    info["codebase_version"] = V21

    # Add back legacy fields
    # total_chunks: number of chunks used for data
    chunk_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)
    total_chunks = math.ceil(num_episodes / chunk_size) if num_episodes > 0 else 1
    info["total_chunks"] = total_chunks

    # total_videos: same as total_episodes for video datasets
    video_keys = [key for key, ft in info["features"].items() if ft["dtype"] == "video"]
    info["total_videos"] = num_episodes * len(video_keys) if video_keys else 0

    # Remove v3.0-specific fields
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)

    # Revert path templates to v2.1 format
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    if video_keys:
        info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    else:
        info["video_path"] = None

    # Remove per-feature fps (v2.1 only has top-level fps)
    for key in info["features"]:
        if info["features"][key]["dtype"] != "video":
            info["features"][key].pop("fps", None)

    # Remove fields not present in v2.1
    info.pop("chunks_size", None)

    # Write
    out_path = new_root / INFO_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(info, out_path)


def convert_dataset(
    repo_id: str,
    root: str | Path | None = None,
    push_to_hub: bool = False,
):
    """Main entry point for converting a v3.0 dataset back to v2.1."""

    # Set root based on whether local dataset path is provided
    use_local_dataset = False
    root = HF_LEROBOT_HOME / repo_id if root is None else Path(root)
    if root.exists():
        validate_local_dataset_version(root)
        use_local_dataset = True
        print(f"Using local dataset at {root}")

    # if not use_local_dataset:
    #     print(f"Downloading v3.0 dataset from hub: {repo_id}")
    #     snapshot_download(
    #         repo_id,
    #         repo_type="dataset",
    #         revision=V30,
    #         local_dir=root,
    #     )
    #     validate_local_dataset_version(root)

    old_root = root.parent / f"{root.name}_v30_backup"
    new_root = root.parent / f"{root.name}_v21"

    # Cleanup
    if new_root.is_dir():
        shutil.rmtree(new_root)

    # Load episodes metadata (full, including stats columns)
    logging.info("Loading episodes metadata...")
    episodes_ds = load_episodes_full(root)
    episodes_df = episodes_ds.to_pandas()

    num_episodes = len(episodes_df)
    logging.info(f"Found {num_episodes} episodes")

    # Run conversions
    convert_info_to_v21(root, new_root, num_episodes)
    convert_tasks_to_v21(root, new_root)
    convert_data_to_v21(root, new_root, episodes_df)
    convert_videos_to_v21(root, new_root, episodes_df)
    convert_episodes_to_v21(root, new_root, episodes_df)

    # Copy stats.json if it exists (v2.1 may have used it)
    stats_src = root / "meta" / "stats.json"
    if stats_src.exists():
        stats_dst = new_root / "meta" / "stats.json"
        stats_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stats_src, stats_dst)

    # Swap directories
    shutil.move(str(root), str(old_root))
    shutil.move(str(new_root), str(root))

    print(f"Conversion complete! v2.1 dataset at: {root}")
    print(f"Original v3.0 backup at: {old_root}")

    if push_to_hub:
        from huggingface_hub import HfApi

        hub_api = HfApi()
        hub_api.upload_folder(
            folder_path=str(root),
            repo_id=repo_id,
            repo_type="dataset",
            revision="v2.1",
        )
        print(f"Pushed v2.1 dataset to hub: {repo_id} (branch: v2.1)")


if __name__ == "__main__":
    init_logging()
    parser = argparse.ArgumentParser(
        description="Convert a LeRobot dataset from v3.0 to v2.1 format."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=False,
        help="Repository identifier on Hugging Face: a community or a user name `/` the name of the dataset "
        "(e.g. `lerobot/pusht`, `<USER>/aloha_sim_insertion_human`).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local directory containing the v3.0 dataset. Defaults to $HF_LEROBOT_HOME/repo_id.",
    )
    parser.add_argument(
        "--push-to-hub",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Push the converted v2.1 dataset to the hub (default: false).",
    )

    args = parser.parse_args()
    convert_dataset(**vars(args))
