#!/usr/bin/env python3
"""Build and audit one fully processed LingBot RoboTwin training batch."""

from __future__ import annotations

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
import yaml


EXPECTED_LINGBOT_COMMIT = "be969b8fd117fb70550c5d4bf4bc328211b5b1b6"
EXPECTED_LEROBOT_V2_COMMIT = "a445d9c9da6bea99a8972daa4fe1fdd053d711d2"
EXPECTED_EPISODES = 50
EXPECTED_FRAMES = 5_682
EXPECTED_CHUNK_SIZE = 50
EXPECTED_UNIFIED_DIM = 55
EXPECTED_ACTIVE_INDICES = list(range(12)) + [28, 29]


def required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} to an absolute path")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_record(value: torch.Tensor) -> dict:
    result = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(torch.isfinite(value).all())
        if value.is_floating_point()
        else None,
    }
    if value.numel() and value.is_floating_point():
        result["min"] = float(value.min())
        result["max"] = float(value.max())
    return result


def value_record(value) -> dict:
    if torch.is_tensor(value):
        return tensor_record(value)
    if isinstance(value, list):
        return {"type": "list", "length": len(value), "value": value}
    return {"type": type(value).__name__, "value": value}


def assert_shape(value: torch.Tensor, shape: tuple[int, ...], label: str) -> None:
    if value.shape != shape:
        raise RuntimeError(f"{label} shape mismatch: {tuple(value.shape)} != {shape}")


def assert_finite_tensors(values: dict, label: str) -> None:
    for key, value in values.items():
        if torch.is_tensor(value) and value.is_floating_point():
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Non-finite tensor in {label}.{key}")


def main() -> None:
    lingbot_root = required_path("LINGBOT_VLA_ROOT")
    lerobot_v2_root = required_path("LEROBOT_V2_ROOT")
    dataset_path = required_path("TRAIN_DATASET_PATH")
    norm_stats_path = required_path("NORM_STATS_PATH")
    model_path = required_path("MODEL_PATH")
    qwen_path = required_path("QWEN3VL_PATH")
    output_base = required_path("OUTPUT_BASE")

    for label, path in (
        ("dataset", dataset_path),
        ("normalization statistics", norm_stats_path),
    ):
        if "randomized" in str(path).lower():
            raise RuntimeError(f"Forbidden randomized {label} path: {path}")

    lingbot_commit = git_output(lingbot_root, "rev-parse", "HEAD")
    lerobot_v2_commit = git_output(lerobot_v2_root, "rev-parse", "HEAD")
    if lingbot_commit != EXPECTED_LINGBOT_COMMIT:
        raise RuntimeError(f"LingBot source pin mismatch: {lingbot_commit}")
    if lerobot_v2_commit != EXPECTED_LEROBOT_V2_COMMIT:
        raise RuntimeError(f"LeRobot v2 source pin mismatch: {lerobot_v2_commit}")

    training_config_path = lingbot_root / "configs/vla/robotwin/robotwin.yaml"
    source_robot_config_path = lingbot_root / "configs/robot_configs/robotwin.yaml"
    info_path = dataset_path / "meta/info.json"
    for path in (training_config_path, source_robot_config_path, info_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (model_path / "model.safetensors.index.json").is_file():
        raise FileNotFoundError("Missing base-model shard index")
    if not (qwen_path / "config.json").is_file():
        raise FileNotFoundError("Missing Qwen3-VL config")

    with info_path.open() as stream:
        dataset_info = json.load(stream)
    if dataset_info.get("total_episodes") != EXPECTED_EPISODES:
        raise RuntimeError("Dataset episode count changed")
    if dataset_info.get("total_frames") != EXPECTED_FRAMES:
        raise RuntimeError("Dataset frame count changed")

    with norm_stats_path.open() as stream:
        norm_stats = json.load(stream)
    expected_norm_dims = {
        "observation.state.arm.position": 12,
        "observation.state.effector.position": 2,
        "action.arm.position": 12,
        "action.effector.position": 2,
    }
    if set(norm_stats.get("norm_stats", {})) != set(expected_norm_dims):
        raise RuntimeError(
            f"Unexpected normalization keys: {sorted(norm_stats.get('norm_stats', {}))}"
        )
    if norm_stats.get("count") != EXPECTED_FRAMES:
        raise RuntimeError(
            f"Normalization state count mismatch: {norm_stats.get('count')}"
        )
    for feature, dimension in expected_norm_dims.items():
        feature_stats = norm_stats["norm_stats"][feature]
        for statistic in ("mean", "std", "q01", "q99", "q02", "q98"):
            values = feature_stats[statistic]
            if len(values) != dimension:
                raise RuntimeError(
                    f"Normalization dimension mismatch for {feature}.{statistic}"
                )

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = output_base / "data_loader_smoke" / f"full_beat_block_hammer_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "train_manifest.txt"
    manifest_path.write_text(f"robotwin {dataset_path}\n", encoding="utf-8")

    with source_robot_config_path.open() as stream:
        source_robot_config = yaml.safe_load(stream)
    runtime_robot_config = dict(source_robot_config)
    runtime_robot_config["norm_stats"] = str(norm_stats_path)
    runtime_robot_config_root = run_dir / "robot_configs"
    runtime_robot_config_root.mkdir()
    runtime_robot_config_path = runtime_robot_config_root / "robotwin.yaml"
    with runtime_robot_config_path.open("x") as stream:
        yaml.safe_dump(runtime_robot_config, stream, sort_keys=False)

    source_mapping = dict(source_robot_config)
    runtime_mapping = dict(runtime_robot_config)
    source_mapping.pop("norm_stats")
    runtime_mapping.pop("norm_stats")
    if source_mapping != runtime_mapping:
        raise RuntimeError("Runtime robot config changed more than norm_stats")

    sys.path.insert(0, str(lingbot_root))
    sys.path.insert(0, str(lerobot_v2_root))
    import lerobot  # noqa: PLC0415
    from lingbotvla.data import (  # noqa: PLC0415
        VLADataCollatorWithPacking,
        build_vla_dataset,
    )
    from lingbotvla.data.vla_data.base_dataset import (  # noqa: PLC0415
        LEROBOT_DATASET_API,
    )
    from lingbotvla.models import build_processor  # noqa: PLC0415
    from lingbotvla.models.config_registry import (  # noqa: PLC0415
        get_config_registry,
    )

    with training_config_path.open() as stream:
        training_config = yaml.safe_load(stream)
    config_kwargs = {**training_config["model"], **training_config["train"]}
    config_kwargs["model_path"] = str(model_path)
    config_kwargs["tokenizer_path"] = str(qwen_path)
    config_registry = get_config_registry()
    config_class = config_registry.get_config_cls_from_config_key(
        config_kwargs["config_key"]
    )
    model_config = config_class(**config_kwargs)
    for name, expected in (
        ("action_dim", EXPECTED_UNIFIED_DIM),
        ("max_action_dim", EXPECTED_UNIFIED_DIM),
        ("max_state_dim", EXPECTED_UNIFIED_DIM),
        ("chunk_size", EXPECTED_CHUNK_SIZE),
    ):
        actual = getattr(model_config, name)
        if actual != expected:
            raise RuntimeError(f"Model config mismatch for {name}: {actual}")

    processor = build_processor(str(qwen_path))
    data_config = SimpleNamespace(
        data_name="multi",
        train_path=str(manifest_path),
        robot_config_root=str(runtime_robot_config_root),
        chunk_size=EXPECTED_CHUNK_SIZE,
        prompt_type="global",
        img_size=training_config["data"].get("img_size", 256),
        image_augment=False,
        use_future_image=True,
        joints=[
            str({"arm.position": 14}),
            str({"end.position": 14}),
            str({"effector.position": 2}),
        ],
        cameras=["camera_top", "camera_wrist_left", "camera_wrist_right"],
        norm_type=[
            str({"arm.position": "bounds_99_woclip"}),
            str({"end.position": "bounds_99_woclip"}),
            str({"effector.position": "bounds_99_woclip"}),
        ],
    )
    args_model = SimpleNamespace(tokenizer_path=str(qwen_path))
    dataset = build_vla_dataset(
        dataset_config=data_config,
        model_config=args_model,
        config=model_config,
        processor=processor,
        do_nomalize=True,
        return_item=False,
        disabled_image_features=False,
        use_depth_align=False,
    )
    if len(dataset) != EXPECTED_FRAMES or dataset.num_episodes != EXPECTED_EPISODES:
        raise RuntimeError("Full loader dataset size mismatch")

    sample = dataset.getdata(0)
    assert_finite_tensors(sample, "sample")
    assert_shape(sample["state"], (EXPECTED_UNIFIED_DIM,), "state")
    assert_shape(
        sample["actions"],
        (EXPECTED_CHUNK_SIZE, EXPECTED_UNIFIED_DIM),
        "actions",
    )
    assert_shape(sample["action_is_pad"], (EXPECTED_CHUNK_SIZE,), "action_is_pad")
    assert_shape(sample["state_joint_mask"], (EXPECTED_UNIFIED_DIM,), "state mask")
    assert_shape(sample["action_joint_mask"], (EXPECTED_UNIFIED_DIM,), "action mask")
    assert_shape(
        sample["joint_mask"],
        (EXPECTED_CHUNK_SIZE, EXPECTED_UNIFIED_DIM),
        "chunk joint mask",
    )

    state_active = torch.nonzero(sample["state_joint_mask"], as_tuple=False).flatten().tolist()
    action_active = torch.nonzero(sample["action_joint_mask"], as_tuple=False).flatten().tolist()
    if state_active != EXPECTED_ACTIVE_INDICES or action_active != EXPECTED_ACTIVE_INDICES:
        raise RuntimeError(
            f"Unexpected active joint indices: state={state_active}, action={action_active}"
        )
    if not torch.equal(
        sample["joint_mask"],
        sample["action_joint_mask"].unsqueeze(0).expand(EXPECTED_CHUNK_SIZE, -1),
    ):
        raise RuntimeError("Chunk joint mask is inconsistent with action joint mask")
    if torch.count_nonzero(sample["state"][~sample["state_joint_mask"]]):
        raise RuntimeError("Padded state dimensions are not zero")
    if torch.count_nonzero(sample["actions"][:, ~sample["action_joint_mask"]]):
        raise RuntimeError("Padded action dimensions are not zero")
    if int(sample["img_masks"].sum()) != 3:
        raise RuntimeError(f"Expected three active cameras: {sample['img_masks']}")

    collator = VLADataCollatorWithPacking()
    batch = collator([sample])
    assert_finite_tensors(batch, "batch")
    assert_shape(batch["state"], (1, EXPECTED_UNIFIED_DIM), "batch state")
    assert_shape(
        batch["actions"],
        (1, EXPECTED_CHUNK_SIZE, EXPECTED_UNIFIED_DIM),
        "batch actions",
    )

    sample_keys = (
        "images",
        "img_masks",
        "state",
        "lang_tokens",
        "lang_masks",
        "actions",
        "action_is_pad",
        "joint_mask",
        "state_joint_mask",
        "action_joint_mask",
        "image_grid_thw",
    )
    result = {
        "status": "ok",
        "check": "lingbot_full_training_batch_one_task",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "lingbot_commit": lingbot_commit,
            "lerobot_v2_commit": lerobot_v2_commit,
            "training_config": str(training_config_path),
            "training_config_sha256": sha256(training_config_path),
            "source_robot_config_sha256": sha256(source_robot_config_path),
            "runtime_robot_config": str(runtime_robot_config_path),
            "runtime_robot_config_sha256": sha256(runtime_robot_config_path),
            "dataset_path": str(dataset_path),
            "dataset_info_sha256": sha256(info_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "norm_stats_path": str(norm_stats_path),
            "norm_stats_sha256": sha256(norm_stats_path),
            "model_path": str(model_path),
            "qwen3vl_path": str(qwen_path),
        },
        "loader": {
            "python_executable": sys.executable,
            "installed_lerobot_distribution_version": importlib.metadata.version(
                "lerobot"
            ),
            "imported_lerobot_source": inspect.getfile(lerobot),
            "lingbot_lerobot_api_branch": LEROBOT_DATASET_API,
            "frames": len(dataset),
            "episodes": dataset.num_episodes,
            "normalization": "clean-only bounds_99_woclip",
            "image_augmentation": False,
            "future_image_query": True,
        },
        "mapping": {
            "raw_dimension": 14,
            "unified_dimension": EXPECTED_UNIFIED_DIM,
            "active_indices": EXPECTED_ACTIVE_INDICES,
            "active_dimension_count": len(EXPECTED_ACTIVE_INDICES),
        },
        "sample": {key: tensor_record(sample[key]) for key in sample_keys},
        "batch": {key: value_record(value) for key, value in batch.items()},
        "scope": "one_cpu_batch_no_model_forward_no_training",
    }
    result_path = run_dir / "audit.json"
    with result_path.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"result_path={result_path}", flush=True)
    print("LINGBOT_FULL_DATA_BATCH_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
