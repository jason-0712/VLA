#!/usr/bin/env python3
"""Audit one extracted RoboTwin clean task before LeRobot conversion."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np


EXPECTED_EPISODES = 50
EXPECTED_CAMERAS = (
    "front_camera",
    "head_camera",
    "left_camera",
    "right_camera",
)
TRAINING_CAMERAS = ("head_camera", "left_camera", "right_camera")
EXPECTED_DATASETS = {
    "endpose/left_endpose": (7,),
    "endpose/left_gripper": (),
    "endpose/right_endpose": (7,),
    "endpose/right_gripper": (),
    "joint_action/left_arm": (6,),
    "joint_action/left_gripper": (),
    "joint_action/right_arm": (6,),
    "joint_action/right_gripper": (),
    "joint_action/vector": (14,),
    "pointcloud": (0,),
}
for camera in EXPECTED_CAMERAS:
    EXPECTED_DATASETS.update(
        {
            f"observation/{camera}/cam2world_gl": (4, 4),
            f"observation/{camera}/extrinsic_cv": (3, 4),
            f"observation/{camera}/intrinsic_cv": (3, 3),
            f"observation/{camera}/rgb": (),
        }
    )


def indexed_files(directory: Path, suffix: str) -> dict[int, Path]:
    result = {}
    for path in directory.glob(f"episode*{suffix}"):
        token = path.name.removeprefix("episode").removesuffix(suffix)
        if token.isdigit():
            result[int(token)] = path
    return result


def dataset_names(group: h5py.Group) -> list[str]:
    names = []

    def visitor(name: str, value: object) -> None:
        if isinstance(value, h5py.Dataset):
            names.append(name)

    group.visititems(visitor)
    return sorted(names)


def finite_stats(data: np.ndarray) -> dict[str, float | int | None]:
    if data.size == 0:
        return {"count": 0, "min": None, "max": None}
    return {
        "count": int(data.size),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


def add_issue(issues: list[dict[str, object]], episode: int | None, message: str) -> None:
    issues.append({"episode": episode, "message": message})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    if not args.task.replace("_", "").isalnum() or args.task.lower() != args.task:
        raise ValueError(f"Invalid task name: {args.task}")
    clean_data_root = os.environ.get("CLEAN_DATA_ROOT")
    if not clean_data_root:
        raise RuntimeError("Set CLEAN_DATA_ROOT to the competition data root")
    root = Path(clean_data_root).expanduser().resolve()
    raw_path = root / "raw" / args.task / "aloha-agilex_clean_50"
    if "randomized" in str(raw_path).lower():
        raise RuntimeError(f"Refusing randomized data path: {raw_path}")
    if not raw_path.is_dir():
        raise FileNotFoundError(raw_path)

    issues: list[dict[str, object]] = []
    hdf5_files = indexed_files(raw_path / "data", ".hdf5")
    instruction_files = indexed_files(raw_path / "instructions", ".json")
    video_files = indexed_files(raw_path / "video", ".mp4")
    trajectory_files = indexed_files(raw_path / "_traj_data", ".pkl")
    expected_indices = set(range(EXPECTED_EPISODES))
    artifact_sets = {
        "hdf5": set(hdf5_files),
        "instructions": set(instruction_files),
        "videos": set(video_files),
        "trajectories": set(trajectory_files),
    }
    for name, indices in artifact_sets.items():
        if indices != expected_indices:
            add_issue(
                issues,
                None,
                f"{name} indices mismatch: missing={sorted(expected_indices - indices)} "
                f"extra={sorted(indices - expected_indices)}",
            )

    seeds = []
    seed_path = raw_path / "seed.txt"
    try:
        seeds = [int(value) for value in seed_path.read_text().split()]
    except Exception as exc:  # noqa: BLE001
        add_issue(issues, None, f"Cannot parse seed.txt: {exc}")
    if len(seeds) != EXPECTED_EPISODES or len(set(seeds)) != EXPECTED_EPISODES:
        add_issue(issues, None, f"Expected 50 unique seeds, got {seeds}")

    scene_info = {}
    try:
        scene_info = json.loads((raw_path / "scene_info.json").read_text())
    except Exception as exc:  # noqa: BLE001
        add_issue(issues, None, f"Cannot parse scene_info.json: {exc}")
    expected_scene_keys = {f"episode_{index}" for index in expected_indices}
    if set(scene_info) != expected_scene_keys:
        add_issue(issues, None, "scene_info.json does not contain exactly episode_0..49")

    clean_scene_count = 0
    for index in range(EXPECTED_EPISODES):
        scene = scene_info.get(f"episode_{index}")
        if not isinstance(scene, dict):
            continue
        texture = scene.get("texture_info", {})
        is_clean = (
            scene.get("cluttered_table_info") == []
            and texture.get("wall_texture") is None
            and texture.get("table_texture") is None
        )
        if is_clean:
            clean_scene_count += 1
        else:
            add_issue(issues, index, f"Scene metadata is not clean: {scene}")

    lengths = []
    converted_lengths = []
    decoded_images = 0
    numeric_value_count = 0
    explicit_timestamp_fields = set()
    video_fps_values = set()
    state_action_vector_max_abs_error = 0.0
    global_ranges: dict[str, dict[str, float | int | None]] = {}
    per_episode = []

    for index in range(EXPECTED_EPISODES):
        hdf5_path = hdf5_files.get(index)
        instruction_path = instruction_files.get(index)
        video_path = video_files.get(index)
        trajectory_path = trajectory_files.get(index)
        if None in (hdf5_path, instruction_path, video_path, trajectory_path):
            continue
        if trajectory_path.stat().st_size == 0:
            add_issue(issues, index, "Trajectory pickle is empty")

        episode_summary: dict[str, object] = {"episode": index}
        try:
            with h5py.File(hdf5_path, "r") as hdf5:
                names = dataset_names(hdf5)
                if set(names) != set(EXPECTED_DATASETS):
                    add_issue(
                        issues,
                        index,
                        f"HDF5 dataset names mismatch: {names}",
                    )
                for name in names:
                    if "time" in name.lower():
                        explicit_timestamp_fields.add(name)

                vector = hdf5["joint_action/vector"][:]
                length = int(vector.shape[0])
                lengths.append(length)
                converted_lengths.append(max(0, length - 1))
                episode_summary["raw_frames"] = length
                episode_summary["converted_frames"] = max(0, length - 1)
                if length < 2:
                    add_issue(issues, index, f"Episode is too short: {length}")

                for name, tail_shape in EXPECTED_DATASETS.items():
                    dataset = hdf5.get(name)
                    if dataset is None:
                        continue
                    if dataset.shape != (length, *tail_shape):
                        add_issue(
                            issues,
                            index,
                            f"Shape mismatch for {name}: {dataset.shape} != {(length, *tail_shape)}",
                        )
                        continue
                    if name.endswith("/rgb"):
                        if dataset.dtype.kind not in {"S", "O", "V"}:
                            add_issue(issues, index, f"Unexpected RGB dtype for {name}: {dataset.dtype}")
                        continue
                    data = dataset[:]
                    numeric_value_count += int(data.size)
                    if not np.issubdtype(data.dtype, np.number):
                        add_issue(issues, index, f"Non-numeric dataset {name}: {data.dtype}")
                    elif not np.isfinite(data).all():
                        add_issue(issues, index, f"NaN or Inf found in {name}")
                    else:
                        stats = finite_stats(data)
                        if name not in global_ranges:
                            global_ranges[name] = stats
                        elif stats["min"] is not None:
                            current = global_ranges[name]
                            current["min"] = min(float(current["min"]), float(stats["min"]))
                            current["max"] = max(float(current["max"]), float(stats["max"]))
                            current["count"] = int(current["count"]) + int(stats["count"])

                left_gripper = hdf5["joint_action/left_gripper"][:][:, None]
                right_gripper = hdf5["joint_action/right_gripper"][:][:, None]
                reconstructed = np.concatenate(
                    [
                        hdf5["joint_action/left_arm"][:],
                        left_gripper,
                        hdf5["joint_action/right_arm"][:],
                        right_gripper,
                    ],
                    axis=1,
                )
                vector_error = float(np.max(np.abs(reconstructed - vector)))
                state_action_vector_max_abs_error = max(
                    state_action_vector_max_abs_error, vector_error
                )
                if vector_error != 0.0:
                    add_issue(issues, index, f"14-D joint vector mismatch: {vector_error}")

                for camera in EXPECTED_CAMERAS:
                    rgb = hdf5[f"observation/{camera}/rgb"]
                    if rgb.shape != (length,):
                        add_issue(issues, index, f"RGB length mismatch for {camera}: {rgb.shape}")
                        continue
                    for frame_index in range(length):
                        encoded = rgb[frame_index]
                        decoded = cv2.imdecode(
                            np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR
                        )
                        if decoded is None:
                            add_issue(
                                issues,
                                index,
                                f"Cannot decode {camera} frame {frame_index}",
                            )
                            continue
                        if decoded.shape != (240, 320, 3):
                            add_issue(
                                issues,
                                index,
                                f"Unexpected {camera} frame shape {decoded.shape}",
                            )
                        decoded_images += 1
        except Exception as exc:  # noqa: BLE001
            add_issue(issues, index, f"HDF5 audit failed: {exc}")
            continue

        try:
            instruction_data = json.loads(instruction_path.read_text())
            if set(instruction_data) != {"seen", "unseen"}:
                add_issue(
                    issues,
                    index,
                    f"Instruction keys mismatch: {sorted(instruction_data)}",
                )
            for split in ("seen", "unseen"):
                prompts = instruction_data.get(split)
                if not isinstance(prompts, list) or len(prompts) != 100:
                    add_issue(
                        issues,
                        index,
                        f"Expected 100 {split} instructions, got {type(prompts).__name__}",
                    )
                    continue
                if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
                    add_issue(issues, index, f"Invalid text in {split} instructions")
            episode_summary["seen_instructions"] = len(instruction_data.get("seen", []))
            episode_summary["unseen_instructions"] = len(instruction_data.get("unseen", []))
        except Exception as exc:  # noqa: BLE001
            add_issue(issues, index, f"Instruction audit failed: {exc}")

        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                add_issue(issues, index, "Cannot open diagnostic MP4")
            else:
                video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                video_fps = float(capture.get(cv2.CAP_PROP_FPS))
                video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_fps_values.add(video_fps)
                episode_summary["video_frames"] = video_frames
                episode_summary["video_fps"] = video_fps
                if video_frames != episode_summary.get("raw_frames"):
                    add_issue(
                        issues,
                        index,
                        f"MP4/HDF5 frame mismatch: {video_frames} != {episode_summary.get('raw_frames')}",
                    )
                if (video_width, video_height) != (320, 240):
                    add_issue(
                        issues,
                        index,
                        f"Unexpected MP4 size: {(video_width, video_height)}",
                    )
                readable, frame = capture.read()
                if not readable or frame is None:
                    add_issue(issues, index, "Cannot decode first MP4 frame")
        finally:
            capture.release()
        per_episode.append(episode_summary)

    if len(lengths) != EXPECTED_EPISODES:
        add_issue(issues, None, f"Audited only {len(lengths)} readable HDF5 episodes")
    expected_decoded_images = sum(lengths) * len(EXPECTED_CAMERAS)
    if decoded_images != expected_decoded_images:
        add_issue(
            issues,
            None,
            f"Decoded image count mismatch: {decoded_images} != {expected_decoded_images}",
        )

    result = {
        "status": "ok" if not issues else "failed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "raw_path": str(raw_path),
        "clean_only_guard": "passed" if "randomized" not in str(raw_path).lower() else "failed",
        "episode_count": len(lengths),
        "seeds": seeds,
        "unique_seed_count": len(set(seeds)),
        "clean_scene_metadata_count": clean_scene_count,
        "raw_frame_count": sum(lengths),
        "converted_frame_count_expected": sum(converted_lengths),
        "episode_length": {
            "min": min(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
            "mean": float(np.mean(lengths)) if lengths else None,
        },
        "raw_joint_vector_dim": 14,
        "state_action_vector_max_abs_error": state_action_vector_max_abs_error,
        "numeric_value_count": numeric_value_count,
        "nan_or_inf_count": 0 if not any("NaN or Inf" in issue["message"] for issue in issues) else None,
        "camera_count": len(EXPECTED_CAMERAS),
        "decoded_image_count": decoded_images,
        "decoded_image_shape": [240, 320, 3],
        "training_camera_mapping": {
            "head_camera": "observation.images.cam_high",
            "left_camera": "observation.images.cam_left_wrist",
            "right_camera": "observation.images.cam_right_wrist",
            "front_camera": "not used by the official LingBot RoboTwin converter",
        },
        "explicit_timestamp_fields": sorted(explicit_timestamp_fields),
        "alignment": {
            "type": "implicit_frame_index",
            "official_raw_to_aloha_rule": "state[t] -> action[t+1]",
            "dropped_frames_per_episode": 1,
            "source_mp4_fps_values": sorted(video_fps_values),
            "official_lerobot_metadata_fps": 50,
            "note": "MP4 files are diagnostics; the official converter reads HDF5 frames by index.",
        },
        "instruction_policy": {
            "seen_per_episode": 100,
            "unseen_per_episode": 100,
            "official_training_split": "seen",
        },
        "pickle_policy": "presence/size checked only; pickle was not deserialized",
        "global_numeric_ranges": global_ranges,
        "per_episode": per_episode,
        "issues": issues,
    }

    audit_dir = root / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    audit_path = audit_dir / f"raw_episode_audit_{args.task}_{timestamp}.json"
    with audit_path.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(json.dumps({key: value for key, value in result.items() if key not in {"per_episode", "global_numeric_ranges"}}, indent=2, sort_keys=True))
    print(f"audit_path={audit_path}")
    if issues:
        print("COMPETITION_RAW_DATA_AUDIT_FAILED", file=sys.stderr)
        return 1
    print("COMPETITION_RAW_DATA_AUDIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
