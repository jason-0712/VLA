#!/usr/bin/env python3
"""Run a fixed-input LingBot core-VLA FP32 forward/loss without backward."""

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
from types import SimpleNamespace

import torch
import yaml


EXPECTED_LINGBOT_COMMIT = "be969b8fd117fb70550c5d4bf4bc328211b5b1b6"
EXPECTED_LEROBOT_V2_COMMIT = "a445d9c9da6bea99a8972daa4fe1fdd053d711d2"
EXPECTED_FRAMES = 5_682
EXPECTED_EPISODES = 50
EXPECTED_CHUNK_SIZE = 50
EXPECTED_UNIFIED_DIM = 55
EXPECTED_PARAMETER_COUNT = 6_375_906_359
SEED = 42
FIXED_FLOW_TIME = 0.5


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


def json_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def scalar(value) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise RuntimeError(f"Expected scalar tensor, got {tuple(value.shape)}")
        return float(value.detach().cpu())
    return float(value)


def tensor_summary(value: torch.Tensor) -> dict:
    record = {"shape": list(value.shape), "dtype": str(value.dtype)}
    if value.numel() and value.is_floating_point():
        record.update(
            {
                "finite": bool(torch.isfinite(value).all()),
                "min": float(value.detach().min().cpu()),
                "max": float(value.detach().max().cpu()),
            }
        )
    return record


def loss_log_summary(loss_log: dict) -> dict:
    summary = {}
    for key, value in loss_log.items():
        if torch.is_tensor(value):
            detached = value.detach().cpu()
            if detached.numel() == 1:
                summary[key] = float(detached)
            else:
                summary[key] = {
                    "shape": list(detached.shape),
                    "dtype": str(detached.dtype),
                    "finite": bool(torch.isfinite(detached).all())
                    if detached.is_floating_point()
                    else None,
                    "mean": float(detached.float().mean()),
                }
        elif isinstance(value, (int, float, bool, str)) or value is None:
            summary[key] = value
        else:
            summary[key] = repr(value)
    return summary


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

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expose exactly one CUDA GPU for this smoke with CUDA_VISIBLE_DEVICES"
        )
    torch.cuda.set_device(0)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

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
    if dataset_info.get("total_frames") != EXPECTED_FRAMES:
        raise RuntimeError("Dataset frame count changed")
    if dataset_info.get("total_episodes") != EXPECTED_EPISODES:
        raise RuntimeError("Dataset episode count changed")
    with norm_stats_path.open() as stream:
        norm_stats = json.load(stream)
    if norm_stats.get("count") != EXPECTED_FRAMES:
        raise RuntimeError("Normalization statistics count changed")

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = output_base / "forward_loss_smoke" / f"core_fp32_{timestamp}"
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
    from lingbotvla.data import (  # noqa: PLC0415
        VLADataCollatorWithPacking,
        build_vla_dataset,
    )
    from lingbotvla.models import (  # noqa: PLC0415
        build_foundation_model,
        build_processor,
    )
    from lingbotvla.models.config_registry import (  # noqa: PLC0415
        get_config_registry,
    )
    from lingbotvla.models.vla.lingbot_vla import (  # noqa: PLC0415
        qwen2_action_expert,
    )

    with training_config_path.open() as stream:
        training_config = yaml.safe_load(stream)
    source_align_params = training_config["train"].get("align_params", {})
    config_kwargs = {**training_config["model"], **training_config["train"]}
    config_kwargs["model_path"] = str(model_path)
    config_kwargs["tokenizer_path"] = str(qwen_path)
    config_kwargs["use_compile"] = False
    for float_key in ("sequence_wise_loss_coeff", "router_z_loss_coeff"):
        config_kwargs[float_key] = float(config_kwargs[float_key])
    source_moe_implementation = config_kwargs.get("moe_implementation")
    if source_moe_implementation != "fused":
        raise RuntimeError(
            f"Pinned training config changed MoE implementation: {source_moe_implementation!r}"
        )
    runtime_moe_implementation = source_moe_implementation
    disable_robby_moe_kernel = os.environ.get("DISABLE_ROBBY_MOE_KERNEL") == "1"
    if disable_robby_moe_kernel:
        qwen2_action_expert.robby_moe_forward = None

    config_registry = get_config_registry()
    config_class = config_registry.get_config_cls_from_config_key(
        config_kwargs["config_key"]
    )
    model_config = config_class(**config_kwargs)
    if model_config.align_params != source_align_params:
        raise RuntimeError("Model config did not retain the official alignment topology")
    for name, expected in (
        ("action_dim", EXPECTED_UNIFIED_DIM),
        ("max_action_dim", EXPECTED_UNIFIED_DIM),
        ("max_state_dim", EXPECTED_UNIFIED_DIM),
        ("chunk_size", EXPECTED_CHUNK_SIZE),
        ("loss_type", "L1_fm"),
    ):
        actual = getattr(model_config, name)
        if actual != expected:
            raise RuntimeError(f"Model config mismatch for {name}: {actual!r}")

    processor = build_processor(str(qwen_path))
    data_config = SimpleNamespace(
        data_name="multi",
        train_path=str(manifest_path),
        robot_config_root=str(runtime_robot_config_root),
        chunk_size=EXPECTED_CHUNK_SIZE,
        prompt_type="global",
        img_size=training_config["data"].get("img_size", 256),
        image_augment=False,
        use_future_image=False,
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
    sample = dataset.getdata(0)
    batch = VLADataCollatorWithPacking()([sample])
    dataset_names = batch.pop("rep_id", None)
    if dataset_names != ["robotwin"]:
        raise RuntimeError(f"Unexpected dataset identity: {dataset_names}")
    if batch["state"].shape != (1, EXPECTED_UNIFIED_DIM):
        raise RuntimeError(f"Unexpected state shape: {batch['state'].shape}")
    if batch["actions"].shape != (
        1,
        EXPECTED_CHUNK_SIZE,
        EXPECTED_UNIFIED_DIM,
    ):
        raise RuntimeError(f"Unexpected action shape: {batch['actions'].shape}")

    batch = {
        key: value.cuda(non_blocking=False) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    noise = torch.randn(
        batch["actions"].shape,
        generator=generator,
        device="cuda",
        dtype=batch["state"].dtype,
    )
    flow_time = torch.full(
        (batch["state"].shape[0],),
        FIXED_FLOW_TIME,
        device="cuda",
        dtype=batch["state"].dtype,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    load_started = time.monotonic()
    model = build_foundation_model(
        config_path=str(training_config_path),
        config_cls=model_config,
        weights_path=str(model_path),
        torch_dtype="float32",
        init_device="cuda",
        force_use_huggingface=False,
        config_kwargs=config_kwargs,
        moe_implementation=config_kwargs.get("moe_implementation"),
    )
    if not getattr(model.model, "use_depth_align", False):
        raise RuntimeError("Loaded model is missing the official alignment query topology")
    model.config.align_params = {}
    model.model.config.align_params = {}
    model.eval()
    torch.cuda.synchronize(0)
    load_seconds = time.monotonic() - load_started
    allocated_after_load = torch.cuda.memory_allocated(0)
    reserved_after_load = torch.cuda.memory_reserved(0)
    torch.cuda.reset_peak_memory_stats(0)

    cache_owner = model.model.qwenvl_with_expert
    cache_names = (
        "pos_embeds",
        "position_embeddings",
        "cu_seqlens",
        "visual_split_sizes",
        "visual_max_seqlen",
    )

    def cache_state() -> dict[str, bool]:
        return {
            name: getattr(cache_owner, name, None) is not None
            for name in cache_names
        }

    cache_before_cold = cache_state()
    forward_started = time.monotonic()
    with torch.no_grad():
        cold_outputs = model(**batch, noise=noise, time=flow_time)
    torch.cuda.synchronize(0)
    forward_seconds = time.monotonic() - forward_started
    cache_after_cold = cache_state()

    repeat_a_started = time.monotonic()
    with torch.no_grad():
        repeat_a_outputs = model(**batch, noise=noise, time=flow_time)
    torch.cuda.synchronize(0)
    repeat_a_forward_seconds = time.monotonic() - repeat_a_started

    repeat_b_started = time.monotonic()
    with torch.no_grad():
        repeat_b_outputs = model(**batch, noise=noise, time=flow_time)
    torch.cuda.synchronize(0)
    repeat_b_forward_seconds = time.monotonic() - repeat_b_started

    output_lengths = tuple(
        len(value)
        for value in (cold_outputs, repeat_a_outputs, repeat_b_outputs)
    )
    if output_lengths != (11, 11, 11):
        raise RuntimeError(
            f"Unexpected model output lengths: {output_lengths}"
        )
    (
        total_loss,
        vla_loss,
        depth_loss,
        future_depth_loss,
        future_video_loss,
        sequence_wise_loss,
        loss_log,
        depth_preds,
        future_depth_preds,
        future_video_preds,
        current_video_preds,
    ) = repeat_a_outputs

    def extract_loss_values(model_outputs) -> dict[str, float]:
        output_loss_log = model_outputs[6]
        return {
            "total_loss": scalar(model_outputs[0]),
            "vla_loss": scalar(model_outputs[1]),
            "depth_loss": scalar(model_outputs[2]),
            "future_depth_loss": scalar(model_outputs[3]),
            "future_video_loss": scalar(model_outputs[4]),
            "sequence_wise_loss": scalar(model_outputs[5]),
            "router_z_loss": scalar(output_loss_log.get("router_z_loss", 0.0)),
        }

    cold_loss_values = extract_loss_values(cold_outputs)
    loss_values = extract_loss_values(repeat_a_outputs)
    repeat_loss_values = extract_loss_values(repeat_b_outputs)
    cold_to_warm_deltas = {
        name: abs(cold_loss_values[name] - loss_values[name])
        for name in loss_values
    }
    warm_repeat_deltas = {
        name: abs(loss_values[name] - repeat_loss_values[name])
        for name in loss_values
    }
    post_warmup_repeat_bitwise_equal = all(
        delta == 0.0 for delta in warm_repeat_deltas.values()
    )
    for name, value in loss_values.items():
        if not torch.isfinite(torch.tensor(value)):
            raise RuntimeError(f"Non-finite {name}: {value}")
    reconstructed = (
        loss_values["vla_loss"]
        + loss_values["depth_loss"]
        + loss_values["future_depth_loss"]
        + loss_values["future_video_loss"]
        + loss_values["sequence_wise_loss"]
        + loss_values["router_z_loss"]
    )
    if abs(loss_values["total_loss"] - reconstructed) > 1e-5:
        raise RuntimeError(
            f"Loss decomposition mismatch: {loss_values['total_loss']} != {reconstructed}"
        )
    if any(
        value is not None
        for value in (
            depth_preds,
            future_depth_preds,
            future_video_preds,
            current_video_preds,
        )
    ):
        raise RuntimeError("Auxiliary predictions must be disabled in the core smoke")
    parameters_with_grad = sum(
        parameter.grad is not None for parameter in model.parameters()
    )
    if parameters_with_grad:
        raise RuntimeError("Gradients were created despite forward-only scope")

    parameters = list(model.parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    trainable_parameter_count = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    dtype_counts = Counter(str(parameter.dtype) for parameter in parameters)
    device_counts = Counter(parameter.device.type for parameter in parameters)
    if set(dtype_counts) != {"torch.float32"}:
        raise RuntimeError(f"Non-FP32 model parameters: {dict(dtype_counts)}")
    if set(device_counts) != {"cuda"}:
        raise RuntimeError(f"Non-CUDA model parameters: {dict(device_counts)}")
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"Model parameter count changed: {parameter_count}")

    gpu = torch.cuda.get_device_properties(0)
    result = {
        "status": "ok",
        "check": "lingbot_core_vla_fp32_forward_loss",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "lingbot_commit": lingbot_commit,
            "lerobot_v2_commit": lerobot_v2_commit,
            "verifier_script": str(Path(__file__).resolve()),
            "verifier_script_sha256": sha256(Path(__file__).resolve()),
            "training_config": str(training_config_path),
            "training_config_sha256": sha256(training_config_path),
            "source_align_params_sha256": json_sha256(source_align_params),
            "runtime_override": (
                "full alignment topology/weights loaded; auxiliary loss dispatch "
                "disabled after load with model.config.align_params={}; use_compile=false; "
                f"moe={runtime_moe_implementation}; "
                f"disable_robby_moe_kernel={disable_robby_moe_kernel}"
            ),
            "source_moe_implementation": source_moe_implementation,
            "runtime_moe_implementation": runtime_moe_implementation,
            "robby_moe_kernel_disabled_for_diagnostic": disable_robby_moe_kernel,
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "norm_stats_path": str(norm_stats_path),
            "norm_stats_sha256": sha256(norm_stats_path),
            "runtime_robot_config": str(runtime_robot_config_path),
            "runtime_robot_config_sha256": sha256(runtime_robot_config_path),
            "model_path": str(model_path),
            "qwen3vl_path": str(qwen_path),
        },
        "protocol": {
            "seed": SEED,
            "model_mode": "eval",
            "torch_grad_enabled": False,
            "optimizer_created": False,
            "backward_executed": False,
            "checkpoint_written": False,
            "precision": "FP32",
            "loss_type": model_config.loss_type,
            "fixed_flow_time": FIXED_FLOW_TIME,
            "explicit_noise": True,
            "cold_forward_is_cache_warmup": True,
            "post_warmup_repeat_bitwise_equal": post_warmup_repeat_bitwise_equal,
            "alignment_query_topology_loaded": True,
            "auxiliary_alignment_loss_enabled": False,
        },
        "batch": {key: tensor_summary(value) for key, value in batch.items()},
        "cold_loss": cold_loss_values,
        "loss": loss_values,
        "repeat_loss": repeat_loss_values,
        "cold_to_warm_loss_absolute_deltas": cold_to_warm_deltas,
        "post_warmup_repeat_loss_absolute_deltas": warm_repeat_deltas,
        "loss_log": loss_log_summary(loss_log),
        "visual_precompute_cache": {
            "before_cold_forward": cache_before_cold,
            "after_cold_forward": cache_after_cold,
        },
        "model": {
            "class": type(model).__name__,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "parameter_dtype_counts": dict(dtype_counts),
            "parameter_device_counts": dict(device_counts),
            "parameters_with_grad": parameters_with_grad,
        },
        "timing": {
            "model_load_seconds": round(load_seconds, 3),
            "cold_forward_seconds": round(forward_seconds, 3),
            "warm_forward_a_seconds": round(repeat_a_forward_seconds, 3),
            "warm_forward_b_seconds": round(repeat_b_forward_seconds, 3),
        },
        "gpu": {
            "visible_device": 0,
            "name": gpu.name,
            "total_memory_bytes": gpu.total_memory,
            "allocated_after_load_bytes": allocated_after_load,
            "reserved_after_load_bytes": reserved_after_load,
            "peak_allocated_forward_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_forward_bytes": torch.cuda.max_memory_reserved(0),
        },
        "scope": (
            "teacher_free_vla_forward_loss_with_full_alignment_topology_"
            "no_backward_no_optimizer"
        ),
    }
    result_path = run_dir / "audit.json"
    with result_path.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"result_path={result_path}", flush=True)
    if not post_warmup_repeat_bitwise_equal:
        print(
            "NOTICE: fixed-input repeated CUDA forward was not bitwise identical; "
            f"deltas={warm_repeat_deltas}",
            flush=True,
        )
    print("LINGBOT_CORE_FORWARD_LOSS_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
