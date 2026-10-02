#!/usr/bin/env python3
"""Strict, single-GPU FP32 load smoke for the LingBot-VLA 2.0 base model."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml


EXPECTED_LINGBOT_COMMIT = "be969b8fd117fb70550c5d4bf4bc328211b5b1b6"
EXPECTED_BASE_REVISION = "11c703bf6a5c1f45b3b69168482da11fdbba53d7"
EXPECTED_QWEN_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


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


def main() -> None:
    lingbot_root = required_path("LINGBOT_VLA_ROOT")
    model_path = required_path("MODEL_PATH")
    qwen_path = required_path("QWEN3VL_PATH")
    output_base = required_path("OUTPUT_BASE")

    actual_commit = git_output(lingbot_root, "rev-parse", "HEAD")
    if actual_commit != EXPECTED_LINGBOT_COMMIT:
        raise RuntimeError(
            f"LingBot source pin mismatch: {actual_commit} != {EXPECTED_LINGBOT_COMMIT}"
        )

    training_config_path = lingbot_root / "configs/vla/robotwin/robotwin.yaml"
    if not training_config_path.is_file():
        raise FileNotFoundError(training_config_path)
    if not (model_path / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(
            f"Missing sharded base-model index in {model_path}"
        )
    if not (qwen_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing Qwen config in {qwen_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "This smoke must expose exactly one GPU; set CUDA_VISIBLE_DEVICES to one index"
        )

    # Imports happen after the source-pin assertion and use the checked-out upstream code.
    sys.path.insert(0, str(lingbot_root))
    from lingbotvla.models import build_foundation_model  # noqa: PLC0415
    from lingbotvla.models.config_registry import get_config_registry  # noqa: PLC0415

    with training_config_path.open() as stream:
        raw_config = yaml.safe_load(stream)
    config_kwargs = {**raw_config["model"], **raw_config["train"]}
    config_kwargs["model_path"] = str(model_path)
    config_kwargs["tokenizer_path"] = str(qwen_path)

    config_registry = get_config_registry()
    config_class = config_registry.get_config_cls_from_config_key(
        config_kwargs["config_key"]
    )
    config = config_class(**config_kwargs)

    required_config = {
        "post_training": True,
        "adanorm_time": True,
        "action_dim": 55,
        "max_action_dim": 55,
        "max_state_dim": 55,
        "use_moe": True,
        "token_num_experts": 32,
        "token_top_k": 4,
    }
    for key, expected in required_config.items():
        actual = getattr(config, key)
        if actual != expected:
            raise RuntimeError(f"Config mismatch for {key}: {actual!r} != {expected!r}")

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.monotonic()
    model = build_foundation_model(
        config_path=str(training_config_path),
        config_cls=config,
        weights_path=str(model_path),
        torch_dtype="float32",
        init_device="cuda",
        config_kwargs=config_kwargs,
        moe_implementation=config_kwargs.get("moe_implementation"),
    )
    model.eval()
    torch.cuda.synchronize(0)
    elapsed_seconds = time.monotonic() - started

    parameters = list(model.parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    trainable_parameter_count = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    dtype_counts = Counter(str(parameter.dtype) for parameter in parameters)
    device_counts = Counter(parameter.device.type for parameter in parameters)
    meta_parameters = [name for name, value in model.named_parameters() if value.is_meta]

    if meta_parameters:
        raise RuntimeError(f"Parameters remain on meta device: {meta_parameters[:10]}")
    non_fp32 = {
        dtype: count for dtype, count in dtype_counts.items() if dtype != "torch.float32"
    }
    if non_fp32:
        raise RuntimeError(f"Non-FP32 parameters found: {non_fp32}")
    if set(device_counts) != {"cuda"}:
        raise RuntimeError(f"Not all parameters are on CUDA: {dict(device_counts)}")

    gpu = torch.cuda.get_device_properties(0)
    result = {
        "status": "ok",
        "check": "single_gpu_fp32_strict_model_load",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "source": {
            "lingbot_root": str(lingbot_root),
            "lingbot_commit": actual_commit,
            "training_config": str(training_config_path),
            "training_config_sha256": sha256(training_config_path),
        },
        "assets": {
            "model_path": str(model_path),
            "base_model_revision": EXPECTED_BASE_REVISION,
            "qwen3vl_path": str(qwen_path),
            "qwen3vl_revision": EXPECTED_QWEN_REVISION,
        },
        "model": {
            "class": type(model).__name__,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "parameter_dtype_counts": dict(dtype_counts),
            "parameter_device_counts": dict(device_counts),
            "action_dim": config.action_dim,
            "max_action_dim": config.max_action_dim,
            "max_state_dim": config.max_state_dim,
            "token_moe_layers": len(config.token_moe_layers),
            "token_num_experts": config.token_num_experts,
            "token_top_k": config.token_top_k,
        },
        "gpu": {
            "visible_device": 0,
            "name": gpu.name,
            "total_memory_bytes": gpu.total_memory,
            "allocated_bytes": torch.cuda.memory_allocated(0),
            "reserved_bytes": torch.cuda.memory_reserved(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        },
        "scope": "load_only_no_forward_no_training",
    }

    run_dir = output_base / "model_load_smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    result_path = run_dir / f"fp32_gpu0_{timestamp}.json"
    with result_path.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"result_path={result_path}", flush=True)
    print("LINGBOT_MODEL_LOAD_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
