#!/usr/bin/env python3
"""Deterministically wrap the pinned RoboTwin-to-LeRobot v2.1 conversion."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


EXPECTED_ROBOTWIN_COMMIT = "13c3c47ff4312dd62484bcd51be034af55c062d1"
EXPECTED_LEROBOT_COMMIT = "a445d9c9da6bea99a8972daa4fe1fdd053d711d2"
EXPECTED_EPISODES = 50


def required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} to an absolute path")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def episode_index(path: Path) -> int:
    token = path.stem.removeprefix("episode_")
    if not token.isdigit():
        raise ValueError(f"Unexpected episode filename: {path}")
    return int(token)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.task.replace("_", "").isalnum() or args.task.lower() != args.task:
        raise ValueError(f"Invalid task name: {args.task}")
    if "randomized" in args.task:
        raise RuntimeError(f"Refusing randomized task: {args.task}")

    data_root = required_path("CLEAN_DATA_ROOT")
    robotwin_root = required_path("ROBOTWIN_ROOT")
    lerobot_root = required_path("LEROBOT_ROOT")
    raw_audit_path = required_path("RAW_DATA_AUDIT")
    if git_head(robotwin_root) != EXPECTED_ROBOTWIN_COMMIT:
        raise RuntimeError("RoboTwin source revision does not match the competition pin")
    if git_head(lerobot_root) != EXPECTED_LEROBOT_COMMIT:
        raise RuntimeError("LeRobot source revision does not match the official RoboTwin pin")

    raw_path = data_root / "raw" / args.task / "aloha-agilex_clean_50"
    if not raw_path.is_dir() or "randomized" in str(raw_path).lower():
        raise RuntimeError(f"Invalid clean raw path: {raw_path}")
    raw_audit = json.loads(raw_audit_path.read_text())
    expected_audit = {
        "status": "ok",
        "task": args.task,
        "episode_count": EXPECTED_EPISODES,
        "raw_path": str(raw_path),
    }
    for key, expected in expected_audit.items():
        if raw_audit.get(key) != expected:
            raise RuntimeError(
                f"Raw audit mismatch for {key}: {raw_audit.get(key)!r} != {expected!r}"
            )
    if raw_audit.get("issues"):
        raise RuntimeError(f"Raw audit contains issues: {raw_audit['issues']}")

    processed_name = f"{args.task}-aloha-agilex_clean_50-50"
    processed_target = data_root / "processed_hdf5" / processed_name
    repo_id = f"local/{args.task}_clean_50_seed{args.seed}"
    lerobot_target = data_root / "lerobot_v2_1" / repo_id
    for target in (processed_target, lerobot_target):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite conversion output: {target}")

    staging_root = data_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"convert-{args.task}-", dir=staging_root))
    processing_stage = staging / "processed_hdf5"
    hf_stage = staging / "hf_lerobot_home"
    processing_stage.mkdir()
    hf_stage.mkdir()

    process_script = robotwin_root / "policy/pi0/scripts/process_data.py"
    converter_script = (
        robotwin_root
        / "policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py"
    )
    try:
        process_module = load_module("pinned_robotwin_process_data", process_script)
        processed_episode_count = process_module.data_transform(
            str(raw_path), EXPECTED_EPISODES, str(processing_stage)
        )
        if processed_episode_count != EXPECTED_EPISODES:
            raise RuntimeError(
                f"RoboTwin processed {processed_episode_count} episodes, expected 50"
            )

        hdf5_files = sorted(
            processing_stage.glob("episode_*/episode_*.hdf5"), key=episode_index
        )
        if [episode_index(path) for path in hdf5_files] != list(range(EXPECTED_EPISODES)):
            raise RuntimeError("Processed HDF5 files are not exactly episode_0..episode_49")

        processed_frames = 0
        for path in hdf5_files:
            with h5py.File(path, "r") as episode:
                state_shape = episode["observations/qpos"].shape
                action_shape = episode["action"].shape
                if state_shape != action_shape or state_shape[1:] != (14,):
                    raise RuntimeError(
                        f"Processed state/action mismatch in {path}: "
                        f"{state_shape} vs {action_shape}"
                    )
                frame_count = state_shape[0]
                processed_frames += frame_count
                for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                    if episode[f"observations/images/{camera}"].shape != (frame_count,):
                        raise RuntimeError(f"Processed camera length mismatch: {path} {camera}")
                if not np.isfinite(episode["observations/qpos"][:]).all():
                    raise RuntimeError(f"Non-finite processed state in {path}")
                if not np.isfinite(episode["action"][:]).all():
                    raise RuntimeError(f"Non-finite processed action in {path}")
        expected_frames = raw_audit["converted_frame_count_expected"]
        if processed_frames != expected_frames:
            raise RuntimeError(
                f"Processed frame mismatch: {processed_frames} != {expected_frames}"
            )

        # LeRobot reads HF_LEROBOT_HOME during module import.
        os.environ["HF_LEROBOT_HOME"] = str(hf_stage)
        sys.path.insert(0, str(lerobot_root))
        converter_module = load_module("pinned_robotwin_lerobot_converter", converter_script)
        from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION  # noqa: PLC0415

        if CODEBASE_VERSION != "v2.1":
            raise RuntimeError(f"Expected LeRobot v2.1, imported {CODEBASE_VERSION}")

        # Upstream port_aloha uses unsorted os.walk and unseeded np.random.choice.
        # Call the same two conversion functions with a fixed episode order and seed.
        dataset = converter_module.create_empty_dataset(
            repo_id,
            robot_type="aloha",
            mode="image",
            has_effort=converter_module.has_effort(hdf5_files),
            has_velocity=converter_module.has_velocity(hdf5_files),
            dataset_config=converter_module.DEFAULT_DATASET_CONFIG,
        )
        np.random.seed(args.seed)
        dataset = converter_module.populate_dataset(
            dataset,
            hdf5_files,
            task=args.task,
            episodes=list(range(EXPECTED_EPISODES)),
        )
        dataset.stop_image_writer()

        generated = hf_stage / repo_id
        if not generated.is_dir():
            raise RuntimeError(f"LeRobot output was not generated: {generated}")
        processed_target.parent.mkdir(parents=True, exist_ok=True)
        lerobot_target.parent.mkdir(parents=True, exist_ok=True)
        processing_stage.rename(processed_target)
        generated.rename(lerobot_target)
    except Exception:
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    result = {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "seed": args.seed,
        "raw_audit": str(raw_audit_path),
        "raw_audit_sha256": sha256(raw_audit_path),
        "raw_path": str(raw_path),
        "robotwin_commit": EXPECTED_ROBOTWIN_COMMIT,
        "lerobot_commit": EXPECTED_LEROBOT_COMMIT,
        "lerobot_codebase_version": "v2.1",
        "processed_hdf5_path": str(processed_target),
        "processed_episode_count": EXPECTED_EPISODES,
        "processed_frame_count": processed_frames,
        "lerobot_repo_id": repo_id,
        "lerobot_path": str(lerobot_target),
        "determinism": {
            "episode_order": "numeric episode_0..episode_49",
            "instruction_split": "seen",
            "numpy_instruction_seed": args.seed,
        },
        "upstream_functions": [
            "RoboTwin policy/pi0/scripts/process_data.py:data_transform",
            "RoboTwin policy/pi0 converter:create_empty_dataset",
            "RoboTwin policy/pi0 converter:populate_dataset",
        ],
        "scope": "one_task_raw_to_lerobot_v2_1_conversion",
    }
    audit_dir = data_root / "audits"
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    audit_path = audit_dir / f"lerobot_conversion_{args.task}_seed{args.seed}_{timestamp}.json"
    with audit_path.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"audit_path={audit_path}")
    print("COMPETITION_LEROBOT_CONVERSION_OK")


if __name__ == "__main__":
    main()
