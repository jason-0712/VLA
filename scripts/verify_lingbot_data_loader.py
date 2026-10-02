#!/usr/bin/env python3
"""Verify the pinned LingBot loader against one clean RoboTwin dataset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch


EXPECTED_LINGBOT_COMMIT = "be969b8fd117fb70550c5d4bf4bc328211b5b1b6"
EXPECTED_LEROBOT_V2_COMMIT = "a445d9c9da6bea99a8972daa4fe1fdd053d711d2"
EXPECTED_CODEBASE_VERSION = "v2.1"
EXPECTED_EPISODES = 50
EXPECTED_FRAMES = 5_682
EXPECTED_STATE_ACTION_DIM = 14
EXPECTED_CHUNK_SIZE = 50


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
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def tensor_record(value: torch.Tensor) -> dict:
    result = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }
    if value.numel() and (value.is_floating_point() or value.is_complex()):
        result.update(
            {
                "finite": bool(torch.isfinite(value).all()),
                "min": float(value.min()),
                "max": float(value.max()),
            }
        )
    return result


def assert_exact(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"{label} shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    if not torch.equal(actual, expected):
        maximum_error = float((actual - expected).abs().max())
        raise RuntimeError(f"{label} value mismatch: max_abs_error={maximum_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="beat_block_hammer")
    args = parser.parse_args()

    lingbot_root = required_path("LINGBOT_VLA_ROOT")
    lerobot_v2_root = required_path("LEROBOT_V2_ROOT")
    dataset_path = required_path("TRAIN_DATASET_PATH")
    output_base = required_path("OUTPUT_BASE")

    if "randomized" in str(dataset_path).lower():
        raise RuntimeError(f"Forbidden randomized dataset path: {dataset_path}")

    actual_commit = git_output(lingbot_root, "rev-parse", "HEAD")
    if actual_commit != EXPECTED_LINGBOT_COMMIT:
        raise RuntimeError(
            f"LingBot source pin mismatch: {actual_commit} != {EXPECTED_LINGBOT_COMMIT}"
        )
    actual_lerobot_commit = git_output(lerobot_v2_root, "rev-parse", "HEAD")
    if actual_lerobot_commit != EXPECTED_LEROBOT_V2_COMMIT:
        raise RuntimeError(
            "LeRobot v2 source pin mismatch: "
            f"{actual_lerobot_commit} != {EXPECTED_LEROBOT_V2_COMMIT}"
        )

    robot_config_root = lingbot_root / "configs/robot_configs"
    robot_config_path = robot_config_root / "robotwin.yaml"
    if not robot_config_path.is_file():
        raise FileNotFoundError(robot_config_path)

    info_path = dataset_path / "meta/info.json"
    with info_path.open() as stream:
        dataset_info = json.load(stream)
    expected_metadata = {
        "codebase_version": EXPECTED_CODEBASE_VERSION,
        "total_episodes": EXPECTED_EPISODES,
        "total_frames": EXPECTED_FRAMES,
    }
    for key, expected in expected_metadata.items():
        actual = dataset_info.get(key)
        if actual != expected:
            raise RuntimeError(
                f"Dataset metadata mismatch for {key}: {actual!r} != {expected!r}"
            )
    for feature in ("observation.state", "action"):
        actual_shape = dataset_info["features"][feature]["shape"]
        if actual_shape != [EXPECTED_STATE_ACTION_DIM]:
            raise RuntimeError(
                f"Dataset feature mismatch for {feature}: {actual_shape}"
            )

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = output_base / "data_loader_smoke" / f"raw_{args.task}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "train_manifest.txt"
    manifest_line = f"robotwin {dataset_path}\n"
    manifest_path.write_text(manifest_line, encoding="utf-8")
    if "randomized" in manifest_path.read_text(encoding="utf-8").lower():
        raise RuntimeError("Generated training manifest contains randomized data")

    sys.path.insert(0, str(lingbot_root))
    sys.path.insert(0, str(lerobot_v2_root))
    import lerobot  # noqa: PLC0415
    from lingbotvla.data import build_vla_dataset  # noqa: PLC0415
    from lingbotvla.data.vla_data.base_dataset import (  # noqa: PLC0415
        LEROBOT_DATASET_API,
    )

    data_config = SimpleNamespace(
        data_name="multi",
        train_path=str(manifest_path),
        robot_config_root=str(robot_config_root),
        chunk_size=EXPECTED_CHUNK_SIZE,
        prompt_type="global",
        img_size=256,
        image_augment=False,
        use_future_image=False,
        joints=None,
        cameras=None,
        norm_type=None,
    )
    dataset = build_vla_dataset(
        dataset_config=data_config,
        model_config=None,
        config=None,
        processor=None,
        do_nomalize=False,
        return_item=True,
        disabled_image_features=True,
        use_depth_align=False,
    )

    if len(dataset) != EXPECTED_FRAMES:
        raise RuntimeError(f"Loader frame count mismatch: {len(dataset)}")
    if dataset.num_episodes != EXPECTED_EPISODES:
        raise RuntimeError(
            f"Loader episode count mismatch: {dataset.num_episodes}"
        )
    if len(dataset._datasets) != 1:  # noqa: SLF001
        raise RuntimeError("Expected exactly one underlying dataset")

    inner = dataset._datasets[0]  # noqa: SLF001
    sample_indices = [0, 124, 125, len(dataset) - 1]
    sample_records = []
    for index in sample_indices:
        raw = inner.dataset[index]
        transformed = dataset.getdata(index)

        raw_state = raw["observation.state"]
        raw_action = raw["action"]
        if raw_state.shape != (EXPECTED_STATE_ACTION_DIM,):
            raise RuntimeError(
                f"Raw state shape mismatch at {index}: {tuple(raw_state.shape)}"
            )
        if raw_action.shape != (EXPECTED_CHUNK_SIZE, EXPECTED_STATE_ACTION_DIM):
            raise RuntimeError(
                f"Raw action shape mismatch at {index}: {tuple(raw_action.shape)}"
            )

        expected_state_arm = torch.cat((raw_state[0:6], raw_state[7:13]))
        expected_state_effector = torch.cat((raw_state[6:7], raw_state[13:14]))
        expected_action_arm = torch.cat(
            (raw_action[..., 0:6], raw_action[..., 7:13]), dim=-1
        )
        expected_action_effector = torch.cat(
            (raw_action[..., 6:7], raw_action[..., 13:14]), dim=-1
        )

        assert_exact(
            transformed["observation.state.arm.position"],
            expected_state_arm,
            f"state arm at {index}",
        )
        assert_exact(
            transformed["observation.state.effector.position"],
            expected_state_effector,
            f"state effector at {index}",
        )
        assert_exact(
            transformed["action.arm.position"],
            expected_action_arm,
            f"action arm at {index}",
        )
        assert_exact(
            transformed["action.effector.position"],
            expected_action_effector,
            f"action effector at {index}",
        )
        assert_exact(
            transformed["action_is_pad"],
            raw["action_is_pad"],
            f"action padding at {index}",
        )
        if transformed["rep_id"] != "robotwin":
            raise RuntimeError(
                f"Unexpected robot config id at {index}: {transformed['rep_id']!r}"
            )
        if not transformed["task"]:
            raise RuntimeError(f"Empty language task at {index}")

        sample_records.append(
            {
                "index": index,
                "episode_index": int(transformed["episode_index"]),
                "frame_index": int(transformed["frame_index"]),
                "task": transformed["task"],
                "state_arm": tensor_record(
                    transformed["observation.state.arm.position"]
                ),
                "state_effector": tensor_record(
                    transformed["observation.state.effector.position"]
                ),
                "action_arm": tensor_record(transformed["action.arm.position"]),
                "action_effector": tensor_record(
                    transformed["action.effector.position"]
                ),
                "padded_action_steps": int(
                    transformed["action_is_pad"].to(torch.int64).sum()
                ),
            }
        )

    result = {
        "status": "ok",
        "check": "lingbot_raw_training_loader_one_task",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "lingbot_root": str(lingbot_root),
            "lingbot_commit": actual_commit,
            "robot_config": str(robot_config_path),
            "robot_config_sha256": sha256(robot_config_path),
            "dataset_path": str(dataset_path),
            "dataset_info_sha256": sha256(info_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
        },
        "loader": {
            "python_executable": sys.executable,
            "installed_lerobot_distribution_version": importlib.metadata.version(
                "lerobot"
            ),
            "imported_lerobot_source": inspect.getfile(lerobot),
            "imported_lerobot_commit": actual_lerobot_commit,
            "lingbot_lerobot_api_branch": LEROBOT_DATASET_API,
            "frames": len(dataset),
            "episodes": dataset.num_episodes,
            "chunk_size": EXPECTED_CHUNK_SIZE,
            "normalization_enabled": False,
            "image_loading_enabled": False,
        },
        "mapping": {
            "raw_state_action_dim": EXPECTED_STATE_ACTION_DIM,
            "arm_indices": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
            "effector_indices": [6, 13],
            "exact_value_checks": len(sample_records) * 5,
            "samples": sample_records,
        },
        "scope": "raw_loader_and_exact_mapping_only_no_normalization_no_images_no_model",
    }
    result_path = run_dir / "audit.json"
    with result_path.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"result_path={result_path}", flush=True)
    print("LINGBOT_RAW_DATA_LOADER_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
