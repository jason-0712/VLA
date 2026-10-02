#!/usr/bin/env python3
"""Validate one deterministic RoboTwin LeRobot v2.1 conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


EXPECTED_LEROBOT_COMMIT = "a445d9c9da6bea99a8972daa4fe1fdd053d711d2"
EXPECTED_EPISODES = 50
EXPECTED_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} to an absolute path")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tree_manifest(root: Path) -> tuple[list[dict[str, object]], str]:
    entries = []
    combined = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest = sha256(path)
        size = path.stat().st_size
        entries.append({"path": relative, "bytes": size, "sha256": digest})
        combined.update(f"{digest} {size} {relative}\n".encode())
    return entries, combined.hexdigest()


def fixed_list_to_numpy(table, name: str) -> np.ndarray:
    return np.asarray(table[name].to_pylist(), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if "randomized" in args.task.lower():
        raise RuntimeError(f"Refusing randomized task: {args.task}")

    data_root = required_path("CLEAN_DATA_ROOT")
    lerobot_root = required_path("LEROBOT_ROOT")
    conversion_audit_path = required_path("LEROBOT_CONVERSION_AUDIT")
    actual_lerobot_commit = subprocess.check_output(
        ["git", "-C", str(lerobot_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_lerobot_commit != EXPECTED_LEROBOT_COMMIT:
        raise RuntimeError(
            f"LeRobot revision mismatch: {actual_lerobot_commit} != {EXPECTED_LEROBOT_COMMIT}"
        )

    conversion = json.loads(conversion_audit_path.read_text())
    expected_conversion = {
        "status": "ok",
        "task": args.task,
        "seed": args.seed,
        "processed_episode_count": EXPECTED_EPISODES,
        "lerobot_codebase_version": "v2.1",
    }
    for key, expected in expected_conversion.items():
        if conversion.get(key) != expected:
            raise RuntimeError(
                f"Conversion audit mismatch for {key}: {conversion.get(key)!r} != {expected!r}"
            )

    dataset_path = Path(conversion["lerobot_path"]).resolve()
    processed_path = Path(conversion["processed_hdf5_path"]).resolve()
    raw_path = Path(conversion["raw_path"]).resolve()
    for path in (dataset_path, processed_path, raw_path):
        if "randomized" in str(path).lower():
            raise RuntimeError(f"Forbidden randomized path: {path}")
        if not path.is_dir():
            raise FileNotFoundError(path)

    info = json.loads((dataset_path / "meta/info.json").read_text())
    expected_info = {
        "codebase_version": "v2.1",
        "robot_type": "aloha",
        "total_episodes": 50,
        "total_frames": 5682,
        "total_videos": 0,
        "total_chunks": 1,
        "fps": 50,
    }
    for key, expected in expected_info.items():
        if info.get(key) != expected:
            raise RuntimeError(f"Metadata mismatch for {key}: {info.get(key)!r} != {expected!r}")

    expected_features = {
        "observation.state",
        "action",
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    if set(info["features"]) != expected_features:
        raise RuntimeError(f"Feature mismatch: {sorted(info['features'])}")
    for name in ("observation.state", "action"):
        feature = info["features"][name]
        if feature["dtype"] != "float32" or feature["shape"] != [14]:
            raise RuntimeError(f"Unexpected {name} feature: {feature}")
    for camera in EXPECTED_CAMERAS:
        feature = info["features"][f"observation.images.{camera}"]
        if feature["dtype"] != "image" or feature["shape"] != [3, 480, 640]:
            raise RuntimeError(f"Unexpected {camera} feature: {feature}")

    episodes = json_lines(dataset_path / "meta/episodes.jsonl")
    tasks = json_lines(dataset_path / "meta/tasks.jsonl")
    episode_stats = json_lines(dataset_path / "meta/episodes_stats.jsonl")
    if [row["episode_index"] for row in episodes] != list(range(EXPECTED_EPISODES)):
        raise RuntimeError("Episode metadata is not ordered episode_0..episode_49")
    if len(episode_stats) != EXPECTED_EPISODES:
        raise RuntimeError(f"Expected 50 episode stats rows, found {len(episode_stats)}")
    if len(tasks) != info["total_tasks"]:
        raise RuntimeError(f"Task table mismatch: {len(tasks)} != {info['total_tasks']}")
    task_by_index = {row["task_index"]: row["task"] for row in tasks}

    np.random.seed(args.seed)
    expected_prompts = []
    for index in range(EXPECTED_EPISODES):
        instructions = json.loads(
            (raw_path / "instructions" / f"episode{index}.json").read_text()
        )["seen"]
        expected_prompts.append(str(np.random.choice(instructions)))
    actual_prompts = [row["tasks"][0] for row in episodes]
    if actual_prompts != expected_prompts:
        mismatch = [
            index
            for index, (actual, expected) in enumerate(zip(actual_prompts, expected_prompts))
            if actual != expected
        ]
        raise RuntimeError(f"Deterministic instruction mismatch in episodes: {mismatch}")

    parquet_files = sorted((dataset_path / "data/chunk-000").glob("episode_*.parquet"))
    if len(parquet_files) != EXPECTED_EPISODES:
        raise RuntimeError(f"Expected 50 Parquet files, found {len(parquet_files)}")
    video_files = list(dataset_path.rglob("*.mp4"))
    if video_files:
        raise RuntimeError(f"Image-mode dataset unexpectedly contains videos: {video_files}")

    expected_columns = [
        "observation.state",
        "action",
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    total_rows = 0
    global_index = 0
    decoded_images = 0
    exact_image_matches = 0
    state_min = np.full(14, np.inf, dtype=np.float64)
    state_max = np.full(14, -np.inf, dtype=np.float64)
    action_min = np.full(14, np.inf, dtype=np.float64)
    action_max = np.full(14, -np.inf, dtype=np.float64)
    per_episode = []

    for index, parquet_path in enumerate(parquet_files):
        expected_name = f"episode_{index:06d}.parquet"
        if parquet_path.name != expected_name:
            raise RuntimeError(f"Parquet order mismatch: {parquet_path.name} != {expected_name}")
        table = pq.read_table(parquet_path)
        if table.column_names != expected_columns:
            raise RuntimeError(f"Parquet columns mismatch in {parquet_path}: {table.column_names}")
        rows = table.num_rows
        if rows != episodes[index]["length"]:
            raise RuntimeError(f"Episode length mismatch for {index}: {rows}")

        states = fixed_list_to_numpy(table, "observation.state")
        actions = fixed_list_to_numpy(table, "action")
        if states.shape != (rows, 14) or actions.shape != (rows, 14):
            raise RuntimeError(f"State/action shape mismatch in episode {index}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise RuntimeError(f"NaN/Inf in episode {index} state/action")

        processed_hdf5 = processed_path / f"episode_{index}" / f"episode_{index}.hdf5"
        with h5py.File(processed_hdf5, "r") as processed:
            expected_states = processed["observations/qpos"][:]
            expected_actions = processed["action"][:]
            if not np.array_equal(states, expected_states):
                raise RuntimeError(f"State changed during LeRobot conversion: episode {index}")
            if not np.array_equal(actions, expected_actions):
                raise RuntimeError(f"Action changed during LeRobot conversion: episode {index}")

            for camera in EXPECTED_CAMERAS:
                records = table[f"observation.images.{camera}"].to_pylist()
                source_images = processed[f"observations/images/{camera}"]
                if len(records) != rows or source_images.shape != (rows,):
                    raise RuntimeError(f"Image length mismatch: episode {index} {camera}")
                for frame_index, record in enumerate(records):
                    payload = record["bytes"]
                    if not payload:
                        raise RuntimeError(
                            f"Empty embedded image: episode {index} {camera} {frame_index}"
                        )
                    converted_image = np.asarray(Image.open(BytesIO(payload)).convert("RGB"))
                    if converted_image.shape != (480, 640, 3):
                        raise RuntimeError(
                            f"Converted image shape mismatch: {converted_image.shape}"
                        )
                    source_image = cv2.imdecode(
                        np.frombuffer(source_images[frame_index], np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if source_image is None:
                        raise RuntimeError(
                            f"Cannot decode processed image: episode {index} {camera} {frame_index}"
                        )
                    decoded_images += 1
                    if not np.array_equal(converted_image, source_image):
                        raise RuntimeError(
                            f"Image changed during LeRobot conversion: "
                            f"episode {index} {camera} {frame_index}"
                        )
                    exact_image_matches += 1

        frame_indices = table["frame_index"].to_numpy()
        episode_indices = table["episode_index"].to_numpy()
        indices = table["index"].to_numpy()
        timestamps = table["timestamp"].to_numpy()
        task_indices = table["task_index"].to_numpy()
        if not np.array_equal(frame_indices, np.arange(rows)):
            raise RuntimeError(f"Frame indices mismatch in episode {index}")
        if not np.array_equal(episode_indices, np.full(rows, index)):
            raise RuntimeError(f"Episode indices mismatch in episode {index}")
        if not np.array_equal(indices, np.arange(global_index, global_index + rows)):
            raise RuntimeError(f"Global indices mismatch in episode {index}")
        if not np.allclose(timestamps, np.arange(rows) / info["fps"], atol=1e-6):
            raise RuntimeError(f"Timestamps mismatch in episode {index}")
        unique_task_indices = np.unique(task_indices)
        if len(unique_task_indices) != 1:
            raise RuntimeError(f"Multiple task indices in episode {index}")
        task_index = int(unique_task_indices[0])
        if task_by_index[task_index] != actual_prompts[index]:
            raise RuntimeError(f"Task index/text mismatch in episode {index}")

        state_min = np.minimum(state_min, states.min(axis=0))
        state_max = np.maximum(state_max, states.max(axis=0))
        action_min = np.minimum(action_min, actions.min(axis=0))
        action_max = np.maximum(action_max, actions.max(axis=0))
        per_episode.append(
            {
                "episode": index,
                "frames": rows,
                "global_start": global_index,
                "global_end_exclusive": global_index + rows,
                "task_index": task_index,
                "task": actual_prompts[index],
            }
        )
        total_rows += rows
        global_index += rows

    if total_rows != info["total_frames"]:
        raise RuntimeError(f"Total frame mismatch: {total_rows} != {info['total_frames']}")
    if decoded_images != total_rows * len(EXPECTED_CAMERAS):
        raise RuntimeError(f"Converted image count mismatch: {decoded_images}")

    # Verify the exact pinned LeRobot reader, not just the Parquet representation.
    sys.path.insert(0, str(lerobot_root))
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    dataset = LeRobotDataset(conversion["lerobot_repo_id"], root=dataset_path)
    if len(dataset) != total_rows or dataset.num_episodes != EXPECTED_EPISODES:
        raise RuntimeError("Pinned LeRobot reader length/episode mismatch")
    loader_indices = [0, episodes[0]["length"] - 1, episodes[0]["length"], total_rows - 1]
    for index in loader_indices:
        sample = dataset[index]
        if tuple(sample["observation.state"].shape) != (14,):
            raise RuntimeError(f"Loader state shape mismatch at index {index}")
        if tuple(sample["action"].shape) != (14,):
            raise RuntimeError(f"Loader action shape mismatch at index {index}")
        for camera in EXPECTED_CAMERAS:
            if tuple(sample[f"observation.images.{camera}"].shape) != (3, 480, 640):
                raise RuntimeError(f"Loader camera shape mismatch at index {index}: {camera}")

    dataset_files, dataset_tree_sha256 = tree_manifest(dataset_path)
    processed_files, processed_tree_sha256 = tree_manifest(processed_path)
    result = {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "seed": args.seed,
        "clean_only_guard": "passed",
        "conversion_audit": str(conversion_audit_path),
        "conversion_audit_sha256": sha256(conversion_audit_path),
        "lerobot_commit": actual_lerobot_commit,
        "lerobot_codebase_version": info["codebase_version"],
        "dataset_path": str(dataset_path),
        "repo_id": conversion["lerobot_repo_id"],
        "episode_count": len(episodes),
        "frame_count": total_rows,
        "fps": info["fps"],
        "task_text_count": len(tasks),
        "parquet_count": len(parquet_files),
        "video_count": len(video_files),
        "image_storage": "embedded image bytes in Parquet",
        "decoded_image_count": decoded_images,
        "exact_processed_to_lerobot_image_matches": exact_image_matches,
        "state_dim": 14,
        "action_dim": 14,
        "state_min": state_min.tolist(),
        "state_max": state_max.tolist(),
        "action_min": action_min.tolist(),
        "action_max": action_max.tolist(),
        "timestamp_rule": "frame_index / 50",
        "deterministic_prompt_matches": EXPECTED_EPISODES,
        "loader_sample_indices": loader_indices,
        "dataset_bytes": sum(int(row["bytes"]) for row in dataset_files),
        "dataset_file_count": len(dataset_files),
        "dataset_tree_sha256": dataset_tree_sha256,
        "processed_bytes": sum(int(row["bytes"]) for row in processed_files),
        "processed_file_count": len(processed_files),
        "processed_tree_sha256": processed_tree_sha256,
        "dataset_files": dataset_files,
        "per_episode": per_episode,
        "scope": "one_task_lerobot_v2_1_full_integrity_audit",
    }
    audit_dir = data_root / "audits"
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    audit_path = audit_dir / f"lerobot_audit_{args.task}_seed{args.seed}_{timestamp}.json"
    with audit_path.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")

    summary = {key: value for key, value in result.items() if key not in {"dataset_files", "per_episode"}}
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"audit_path={audit_path}")
    print("COMPETITION_LEROBOT_AUDIT_OK")


if __name__ == "__main__":
    main()
