#!/usr/bin/env python3
"""Prepare an immutable one-step LingBot training run from the pinned recipe."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


EXPECTED_LINGBOT_COMMIT = "be969b8fd117fb70550c5d4bf4bc328211b5b1b6"
EXPECTED_LEROBOT_COMMIT = "a445d9c9da6bea99a8972daa4fe1fdd053d711d2"
EXPECTED_UPSTREAM_CONFIG_SHA256 = (
    "ee915e4ad34589a6def35c9e3c12bb8ff7d9f768dfc7ad44ce789fa7aaea57e2"
)
EXPECTED_EPISODES = 50
EXPECTED_FRAMES = 5682
EXPECTED_WORLD_SIZE = 4
EXPERIMENT_ID = "E000-training-step-smoke"

ALLOWED_OVERRIDE_LEAVES = {
    ("data", "num_workers"),
    ("train", "async_save_hf_weights"),
    ("train", "enable_gradient_checkpointing"),
    ("train", "enable_resume"),
    ("train", "global_batch_size"),
    ("train", "gradient_accumulation_steps"),
    ("train", "max_steps"),
    ("train", "micro_batch_size"),
    ("train", "save_epochs"),
    ("train", "save_steps"),
    ("train", "use_compile"),
    ("train", "use_wandb"),
}


def required_path(name: str, *, directory: bool | None = None) -> Path:
    raw = os.environ.get(name)
    if not raw:
        raise RuntimeError(f"Set {name} to an absolute path")
    path = Path(raw).expanduser().resolve()
    if not path.is_absolute() or not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if directory is True and not path.is_dir():
        raise NotADirectoryError(path)
    if directory is False and not path.is_file():
        raise FileNotFoundError(path)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def assert_pinned_checkout(root: Path, expected: str, label: str) -> dict[str, Any]:
    actual = git(root, "rev-parse", "HEAD")
    if actual != expected:
        raise RuntimeError(f"{label} pin mismatch: {actual} != {expected}")
    tracked_diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet"], check=False
    ).returncode
    staged_diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--cached", "--quiet"], check=False
    ).returncode
    if tracked_diff or staged_diff:
        raise RuntimeError(f"{label} has tracked or staged modifications")
    return {
        "root": str(root),
        "commit": actual,
        "status_porcelain": git(root, "status", "--short"),
    }


def leaf_items(value: dict[str, Any], prefix: tuple[str, ...] = ()):
    for key, child in value.items():
        path = (*prefix, str(key))
        if isinstance(child, dict):
            yield from leaf_items(child, path)
        else:
            yield path, child


def set_nested(mapping: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = mapping
    for key in path[:-1]:
        if key not in current or not isinstance(current[key], dict):
            raise KeyError(f"Upstream config has no mapping at {'.'.join(path)}")
        current = current[key]
    # Some TrainingArguments fields are supplied by dataclass defaults and are
    # intentionally absent from the upstream YAML. The exact leaf whitelist is
    # enforced before this function, so materializing such a field is safe.
    current[path[-1]] = value


def immutable_write_text(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
    path.chmod(0o444)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overrides", required=True, type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    lingbot_root = required_path("LINGBOT_VLA_ROOT", directory=True)
    lerobot_root = required_path("LEROBOT_V2_ROOT", directory=True)
    train_manifest = required_path("TRAIN_MANIFEST", directory=False)
    norm_stats_path = required_path("NORM_STATS_PATH", directory=False)
    model_path = required_path("MODEL_PATH", directory=True)
    qwen_path = required_path("QWEN3VL_PATH", directory=True)
    moge_path = required_path("MOGE_PATH", directory=False)
    output_dir = args.output_dir.expanduser().resolve()
    overrides_path = args.overrides.expanduser().resolve()

    if not overrides_path.is_file():
        raise FileNotFoundError(overrides_path)
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {output_dir}")
    if repo_root == output_dir or repo_root in output_dir.parents:
        raise RuntimeError("Training output must be outside the Git repository")

    lingbot_source = assert_pinned_checkout(
        lingbot_root, EXPECTED_LINGBOT_COMMIT, "LingBot"
    )
    lerobot_source = assert_pinned_checkout(
        lerobot_root, EXPECTED_LEROBOT_COMMIT, "LeRobot v2"
    )
    competition_source = {
        "root": str(repo_root),
        "commit": git(repo_root, "rev-parse", "HEAD"),
        "status_porcelain": git(repo_root, "status", "--short"),
    }

    manifest_text = train_manifest.read_text(encoding="utf-8")
    if "randomized" in manifest_text.lower() or "randomized" in str(train_manifest).lower():
        raise RuntimeError("Forbidden randomized path in training manifest")
    manifest_rows = [line.split() for line in manifest_text.splitlines() if line.strip()]
    if len(manifest_rows) != 1 or len(manifest_rows[0]) != 2:
        raise RuntimeError("E000 must use exactly one two-column manifest row")
    robot_name, dataset_raw = manifest_rows[0]
    if robot_name != "robotwin":
        raise RuntimeError(f"Unexpected robot config key: {robot_name}")
    dataset_path = Path(dataset_raw).expanduser().resolve()
    if "randomized" in str(dataset_path).lower() or not dataset_path.is_dir():
        raise RuntimeError(f"Invalid clean dataset path: {dataset_path}")

    info_path = dataset_path / "meta/info.json"
    with info_path.open(encoding="utf-8") as stream:
        dataset_info = json.load(stream)
    if dataset_info.get("total_episodes") != EXPECTED_EPISODES:
        raise RuntimeError("One-step dataset must contain exactly 50 clean episodes")
    if dataset_info.get("total_frames") != EXPECTED_FRAMES:
        raise RuntimeError("One-step dataset frame count changed")
    with norm_stats_path.open(encoding="utf-8") as stream:
        norm_stats = json.load(stream)
    if norm_stats.get("count") != EXPECTED_FRAMES:
        raise RuntimeError("Normalization-stat count does not match the clean dataset")

    upstream_config_path = lingbot_root / "configs/vla/robotwin/robotwin.yaml"
    upstream_robot_config_path = lingbot_root / "configs/robot_configs/robotwin.yaml"
    if sha256(upstream_config_path) != EXPECTED_UPSTREAM_CONFIG_SHA256:
        raise RuntimeError("Pinned upstream training config content changed")
    for path in (
        upstream_robot_config_path,
        model_path / "model.safetensors.index.json",
        qwen_path / "config.json",
        model_path / "depth/model.pt",
        model_path / "dino_video/teacher_step_10000.pth",
        model_path / "dino_video/config.yaml",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if moge_path.name != "model.pt":
        raise RuntimeError("MOGE_PATH must point to the pinned model.pt")

    with upstream_config_path.open(encoding="utf-8") as stream:
        runtime_config = yaml.safe_load(stream)
    with overrides_path.open(encoding="utf-8") as stream:
        overrides = yaml.safe_load(stream)
    override_leaves = dict(leaf_items(overrides))
    unknown = set(override_leaves) - ALLOWED_OVERRIDE_LEAVES
    missing = ALLOWED_OVERRIDE_LEAVES - set(override_leaves)
    if unknown or missing:
        raise RuntimeError(
            f"Unexpected override leaf set; unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    for path, value in override_leaves.items():
        set_nested(runtime_config, path, value)

    expected_smoke_values = {
        ("train", "enable_gradient_checkpointing"): True,
        ("train", "use_compile"): False,
        ("train", "use_wandb"): False,
        ("train", "enable_resume"): False,
        ("train", "micro_batch_size"): 1,
        ("train", "global_batch_size"): EXPECTED_WORLD_SIZE,
        ("train", "gradient_accumulation_steps"): 1,
        ("train", "max_steps"): 1,
        ("train", "save_steps"): 1,
        ("train", "save_epochs"): 0,
        ("train", "async_save_hf_weights"): True,
    }
    for path, expected in expected_smoke_values.items():
        actual = runtime_config[path[0]][path[1]]
        if actual != expected:
            raise RuntimeError(f"Smoke override mismatch for {'.'.join(path)}")

    output_dir.mkdir(parents=True, exist_ok=False)
    runtime_manifest_path = output_dir / "train_manifest.txt"
    immutable_write_text(runtime_manifest_path, manifest_text)

    runtime_robot_root = output_dir / "robot_configs"
    runtime_robot_root.mkdir()
    with upstream_robot_config_path.open(encoding="utf-8") as stream:
        source_robot_config = yaml.safe_load(stream)
    runtime_robot_config = copy.deepcopy(source_robot_config)
    runtime_robot_config["norm_stats"] = str(norm_stats_path)
    source_without_stats = copy.deepcopy(source_robot_config)
    runtime_without_stats = copy.deepcopy(runtime_robot_config)
    source_without_stats.pop("norm_stats", None)
    runtime_without_stats.pop("norm_stats", None)
    if source_without_stats != runtime_without_stats:
        raise RuntimeError("Runtime robot config changed beyond norm_stats")
    runtime_robot_config_path = runtime_robot_root / "robotwin.yaml"
    immutable_write_text(
        runtime_robot_config_path,
        yaml.safe_dump(runtime_robot_config, sort_keys=False),
    )

    runtime_config["model"]["model_path"] = str(model_path)
    runtime_config["model"]["tokenizer_path"] = str(qwen_path)
    runtime_config["data"]["train_path"] = str(runtime_manifest_path)
    runtime_config["data"]["robot_config_root"] = str(runtime_robot_root)
    runtime_config["train"]["output_dir"] = str(output_dir)
    runtime_config["train"]["align_params"]["depth"]["moge_path"] = str(moge_path)
    runtime_config["train"]["align_params"]["depth"]["morgbd_path"] = str(
        model_path / "depth/model.pt"
    )
    runtime_config["train"]["align_params"]["video"]["ckpt_path"] = str(
        model_path / "dino_video/teacher_step_10000.pth"
    )
    runtime_config["train"]["align_params"]["video"]["config_path"] = str(
        model_path / "dino_video/config.yaml"
    )
    runtime_config["train"]["align_params"]["visual_dir"] = str(
        output_dir / "images"
    )

    runtime_config_path = output_dir / "config.yaml"
    immutable_write_text(
        runtime_config_path, yaml.safe_dump(runtime_config, sort_keys=False)
    )

    preflight = {
        "status": "prepared",
        "experiment_id": EXPERIMENT_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "one_clean_batch_four_gpu_fp32_fsdp2_backward_muon_step_dcp_hf_export_reload",
        "source": {
            "competition": competition_source,
            "lingbot": lingbot_source,
            "lerobot_v2": lerobot_source,
            "upstream_config": str(upstream_config_path),
            "upstream_config_sha256": sha256(upstream_config_path),
            "overrides": str(overrides_path),
            "overrides_sha256": sha256(overrides_path),
        },
        "data": {
            "source_manifest": str(train_manifest),
            "runtime_manifest": str(runtime_manifest_path),
            "manifest_sha256": sha256(runtime_manifest_path),
            "dataset_path": str(dataset_path),
            "episodes": dataset_info["total_episodes"],
            "frames": dataset_info["total_frames"],
            "norm_stats": str(norm_stats_path),
            "norm_stats_sha256": sha256(norm_stats_path),
        },
        "assets": {
            "model_path": str(model_path),
            "qwen3vl_path": str(qwen_path),
            "moge_path": str(moge_path),
        },
        "runtime": {
            "output_dir": str(output_dir),
            "config": str(runtime_config_path),
            "config_sha256": sha256(runtime_config_path),
            "robot_config": str(runtime_robot_config_path),
            "robot_config_sha256": sha256(runtime_robot_config_path),
            "expected_world_size": EXPECTED_WORLD_SIZE,
        },
    }
    preflight_path = output_dir / "preflight.json"
    immutable_write_text(
        preflight_path, json.dumps(preflight, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(preflight, indent=2, sort_keys=True))
    print(f"output_dir={output_dir}")
    print("LINGBOT_TRAINING_STEP_SMOKE_PREPARED")


if __name__ == "__main__":
    main()
