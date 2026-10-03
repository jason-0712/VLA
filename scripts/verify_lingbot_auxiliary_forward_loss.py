#!/usr/bin/env python3
"""Run the complete LingBot auxiliary-teacher FP32 forward/loss without backward."""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from verify_lingbot_forward_loss import (
    EXPECTED_CHUNK_SIZE,
    EXPECTED_EPISODES,
    EXPECTED_FRAMES,
    EXPECTED_LEROBOT_V2_COMMIT,
    EXPECTED_LINGBOT_COMMIT,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_UNIFIED_DIM,
    FIXED_FLOW_TIME,
    SEED,
    git_output,
    json_sha256,
    loss_log_summary,
    required_path,
    scalar,
    sha256,
    tensor_summary,
)


def parameter_summary(module: torch.nn.Module) -> dict:
    parameters = list(module.parameters())
    return {
        "class": type(module).__name__,
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "parameter_dtype_counts": dict(
            Counter(str(parameter.dtype) for parameter in parameters)
        ),
        "parameter_device_counts": dict(
            Counter(parameter.device.type for parameter in parameters)
        ),
        "parameters_with_grad": sum(
            parameter.grad is not None for parameter in parameters
        ),
    }


def assert_finite(name: str, value: torch.Tensor) -> None:
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise RuntimeError(f"Non-finite or non-floating tensor: {name}")


def main() -> None:
    verifier_path = Path(__file__).resolve()
    core_verifier_path = verifier_path.with_name("verify_lingbot_forward_loss.py")
    lingbot_root = required_path("LINGBOT_VLA_ROOT")
    lerobot_v2_root = required_path("LEROBOT_V2_ROOT")
    dataset_path = required_path("TRAIN_DATASET_PATH")
    norm_stats_path = required_path("NORM_STATS_PATH")
    model_path = required_path("MODEL_PATH")
    qwen_path = required_path("QWEN3VL_PATH")
    moge_path = required_path("MOGE_PATH")
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
    morgbd_path = model_path / "depth/model.pt"
    video_checkpoint_path = model_path / "dino_video/teacher_step_10000.pth"
    video_config_path = model_path / "dino_video/config.yaml"
    for path in (
        training_config_path,
        source_robot_config_path,
        info_path,
        morgbd_path,
        video_checkpoint_path,
        video_config_path,
        core_verifier_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not moge_path.is_file():
        raise FileNotFoundError(f"MOGE_PATH must name model.pt: {moge_path}")
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
    dataset_fps = float(dataset_info.get("fps", 0))
    if dataset_fps != 50.0:
        raise RuntimeError(f"Dataset FPS changed: {dataset_fps}")
    with norm_stats_path.open() as stream:
        norm_stats = json.load(stream)
    if norm_stats.get("count") != EXPECTED_FRAMES:
        raise RuntimeError("Normalization statistics count changed")

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = output_base / "forward_loss_smoke" / f"auxiliary_fp32_{timestamp}"
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
    from lingbotvla.models.vla.vision_models.module_utils import (  # noqa: PLC0415
        build_depth_model,
        build_video_model,
        get_depth_target,
        get_video_target,
    )

    with training_config_path.open() as stream:
        training_config = yaml.safe_load(stream)
    source_align_params = copy.deepcopy(training_config["train"]["align_params"])
    runtime_align_params = copy.deepcopy(source_align_params)
    runtime_align_params["depth"]["moge_path"] = str(moge_path)
    runtime_align_params["depth"]["morgbd_path"] = str(morgbd_path)
    runtime_align_params["video"]["ckpt_path"] = str(video_checkpoint_path)
    runtime_align_params["video"]["config_path"] = str(video_config_path)
    runtime_align_params["visual_dir"] = str(run_dir / "images")

    comparison_align_params = copy.deepcopy(runtime_align_params)
    comparison_align_params["depth"]["moge_path"] = source_align_params["depth"][
        "moge_path"
    ]
    comparison_align_params["depth"]["morgbd_path"] = source_align_params["depth"][
        "morgbd_path"
    ]
    comparison_align_params["video"]["ckpt_path"] = source_align_params["video"][
        "ckpt_path"
    ]
    comparison_align_params["video"]["config_path"] = source_align_params["video"][
        "config_path"
    ]
    comparison_align_params.pop("visual_dir")
    if comparison_align_params != source_align_params:
        raise RuntimeError("Runtime alignment config changed more than asset paths")

    config_kwargs = {**training_config["model"], **training_config["train"]}
    config_kwargs["model_path"] = str(model_path)
    config_kwargs["tokenizer_path"] = str(qwen_path)
    config_kwargs["align_params"] = runtime_align_params
    config_kwargs["use_compile"] = False
    for float_key in ("sequence_wise_loss_coeff", "router_z_loss_coeff"):
        config_kwargs[float_key] = float(config_kwargs[float_key])
    if config_kwargs.get("moe_implementation") != "fused":
        raise RuntimeError("Pinned training config no longer uses fused MoE")

    config_registry = get_config_registry()
    config_class = config_registry.get_config_cls_from_config_key(
        config_kwargs["config_key"]
    )
    model_config = config_class(**config_kwargs)
    if model_config.align_params != runtime_align_params:
        raise RuntimeError("Model config did not retain runtime alignment parameters")
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
        use_depth_align=True,
    )
    if len(dataset) != EXPECTED_FRAMES or dataset.num_episodes != EXPECTED_EPISODES:
        raise RuntimeError("Auxiliary loader dataset size mismatch")
    sample = dataset.getdata(0)
    batch = VLADataCollatorWithPacking()([sample])
    dataset_names = batch.pop("rep_id", None)
    if dataset_names != ["robotwin"]:
        raise RuntimeError(f"Unexpected dataset identity: {dataset_names}")
    required_batch_keys = {
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
        "pil_images",
        "future_pil_images",
    }
    missing_keys = required_batch_keys.difference(batch)
    if missing_keys:
        raise RuntimeError(f"Missing auxiliary batch keys: {sorted(missing_keys)}")
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
    pil_images = batch.pop("pil_images")
    future_pil_images = batch.pop("future_pil_images")
    future_video_effective_fps = batch.pop("future_video_effective_fps", None)
    if pil_images.shape[:2] != (1, 3):
        raise RuntimeError(f"Unexpected current teacher images: {pil_images.shape}")
    if future_pil_images.shape[:2] != (1, 3):
        raise RuntimeError(f"Unexpected future teacher images: {future_pil_images.shape}")
    resolved_video_effective_fps = (
        float(future_video_effective_fps.detach().flatten()[0].cpu())
        if torch.is_tensor(future_video_effective_fps)
        else (
            float(future_video_effective_fps)
            if future_video_effective_fps is not None
            else float(runtime_align_params["video"]["effective_fps"])
        )
    )

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
    model_load_started = time.monotonic()
    model = build_foundation_model(
        config_path=str(training_config_path),
        config_cls=model_config,
        weights_path=str(model_path),
        torch_dtype="float32",
        init_device="cuda",
        force_use_huggingface=False,
        config_kwargs=config_kwargs,
        moe_implementation=config_kwargs["moe_implementation"],
    )
    model.train()
    torch.cuda.synchronize(0)
    model_load_seconds = time.monotonic() - model_load_started
    allocated_after_model = torch.cuda.memory_allocated(0)

    if not getattr(model.model, "use_depth_align", False):
        raise RuntimeError("Loaded model did not enable depth alignment")
    if not getattr(model.model, "use_future_video", False):
        raise RuntimeError("Loaded model did not enable future-video alignment")

    depth_teacher_started = time.monotonic()
    moge_model, morgbd_model = build_depth_model(runtime_align_params)
    torch.cuda.synchronize(0)
    depth_teacher_load_seconds = time.monotonic() - depth_teacher_started
    allocated_after_depth_teachers = torch.cuda.memory_allocated(0)

    video_teacher_started = time.monotonic()
    video_teacher = build_video_model(runtime_align_params["video"])
    torch.cuda.synchronize(0)
    video_teacher_load_seconds = time.monotonic() - video_teacher_started
    allocated_after_all_teachers = torch.cuda.memory_allocated(0)

    target_started = time.monotonic()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            depth_targets, _ = get_depth_target(
                "MoRGBD", (moge_model, morgbd_model), pil_images
            )
            future_depth_targets, _ = get_depth_target(
                "MoRGBD", (moge_model, morgbd_model), future_pil_images
            )
            video_target_bundle = get_video_target(
                video_teacher,
                pil_images,
                future_pil_images,
                runtime_align_params["video"],
                effective_fps=future_video_effective_fps,
            )
    torch.cuda.synchronize(0)
    target_seconds = time.monotonic() - target_started
    if not isinstance(video_target_bundle, dict):
        raise RuntimeError("Expected current/future DINO patch target dictionary")
    future_video_targets = video_target_bundle.get("patch")
    future_video_cls_targets = video_target_bundle.get("cls")
    current_video_targets = video_target_bundle.get("current_patch")
    if future_video_cls_targets is not None:
        raise RuntimeError("Pinned config unexpectedly enabled DINO CLS targets")

    target_tensors = {
        "depth_targets": depth_targets,
        "future_depth_targets": future_depth_targets,
        "future_video_targets": future_video_targets,
        "current_video_targets": current_video_targets,
    }
    for name, value in target_tensors.items():
        if value is None:
            raise RuntimeError(f"Missing auxiliary target: {name}")
        assert_finite(name, value)

    forward_started = time.monotonic()
    with torch.no_grad():
        outputs = model(
            **batch,
            noise=noise,
            time=flow_time,
            depth_targets=depth_targets,
            future_depth_targets=future_depth_targets,
            future_video_targets=future_video_targets,
            future_video_cls_targets=future_video_cls_targets,
            future_video_current_patch=current_video_targets,
        )
    torch.cuda.synchronize(0)
    forward_seconds = time.monotonic() - forward_started
    if len(outputs) != 11:
        raise RuntimeError(f"Unexpected model output length: {len(outputs)}")
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
    ) = outputs

    loss_values = {
        "total_loss": scalar(total_loss),
        "vla_loss": scalar(vla_loss),
        "depth_loss": scalar(depth_loss),
        "future_depth_loss": scalar(future_depth_loss),
        "future_video_loss": scalar(future_video_loss),
        "sequence_wise_loss": scalar(sequence_wise_loss),
        "router_z_loss": scalar(loss_log.get("router_z_loss", 0.0)),
    }
    for name, value in loss_values.items():
        if not torch.isfinite(torch.tensor(value)):
            raise RuntimeError(f"Non-finite {name}: {value}")
    for name in ("depth_loss", "future_depth_loss", "future_video_loss"):
        if loss_values[name] <= 0:
            raise RuntimeError(f"Auxiliary loss was not active: {name}={loss_values[name]}")
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

    prediction_tensors = {
        "depth_preds": depth_preds,
        "future_depth_preds": future_depth_preds,
        "future_video_preds": future_video_preds,
        "current_video_preds": current_video_preds,
    }
    prediction_target_pairs = (
        ("depth", depth_preds, depth_targets),
        ("future_depth", future_depth_preds, future_depth_targets),
        ("future_video", future_video_preds, future_video_targets),
        ("current_video", current_video_preds, current_video_targets),
    )
    for name, prediction, target in prediction_target_pairs:
        if prediction is None:
            raise RuntimeError(f"Missing auxiliary prediction: {name}")
        assert_finite(f"{name}_prediction", prediction)
        if prediction.shape != target.shape:
            raise RuntimeError(
                f"{name} prediction/target mismatch: {prediction.shape} != {target.shape}"
            )

    model_parameters = parameter_summary(model)
    if model_parameters["parameter_count"] != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Model parameter count changed: {model_parameters['parameter_count']}"
        )
    if set(model_parameters["parameter_dtype_counts"]) != {"torch.float32"}:
        raise RuntimeError(
            f"Non-FP32 model parameters: {model_parameters['parameter_dtype_counts']}"
        )
    if set(model_parameters["parameter_device_counts"]) != {"cuda"}:
        raise RuntimeError(
            f"Non-CUDA model parameters: {model_parameters['parameter_device_counts']}"
        )
    if model_parameters["parameters_with_grad"]:
        raise RuntimeError("Gradients were created despite forward-only scope")

    teacher_summaries = {
        "moge": parameter_summary(moge_model),
        "morgbd": parameter_summary(morgbd_model),
        "dino_video": parameter_summary(video_teacher),
    }
    for name, summary in teacher_summaries.items():
        if summary["trainable_parameter_count"] or summary["parameters_with_grad"]:
            raise RuntimeError(f"Teacher is unexpectedly trainable: {name}")

    gpu = torch.cuda.get_device_properties(0)
    result = {
        "status": "ok",
        "check": "lingbot_complete_auxiliary_fp32_forward_loss",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "lingbot_commit": lingbot_commit,
            "lerobot_v2_commit": lerobot_v2_commit,
            "verifier_script": str(verifier_path),
            "verifier_script_sha256": sha256(verifier_path),
            "core_verifier_dependency": str(core_verifier_path),
            "core_verifier_dependency_sha256": sha256(core_verifier_path),
            "training_config": str(training_config_path),
            "training_config_sha256": sha256(training_config_path),
            "dataset_info_sha256": sha256(info_path),
            "source_align_params_sha256": json_sha256(source_align_params),
            "runtime_align_params_sha256": json_sha256(runtime_align_params),
            "runtime_override": (
                "teacher asset paths and visual_dir resolved from environment; "
                "use_compile=false; image_augment=false; all other alignment settings unchanged"
            ),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "norm_stats_path": str(norm_stats_path),
            "norm_stats_sha256": sha256(norm_stats_path),
            "runtime_robot_config": str(runtime_robot_config_path),
            "runtime_robot_config_sha256": sha256(runtime_robot_config_path),
            "model_path": str(model_path),
            "qwen3vl_path": str(qwen_path),
            "moge_path": str(moge_path),
            "moge_sha256": sha256(moge_path),
            "morgbd_path": str(morgbd_path),
            "morgbd_sha256": sha256(morgbd_path),
            "video_checkpoint_path": str(video_checkpoint_path),
            "video_checkpoint_sha256": sha256(video_checkpoint_path),
            "video_config_path": str(video_config_path),
            "video_config_sha256": sha256(video_config_path),
        },
        "protocol": {
            "seed": SEED,
            "model_mode": "train",
            "model_forward_grad_enabled": False,
            "teacher_target_autocast": "BF16 (official training path)",
            "model_precision": "FP32",
            "optimizer_created": False,
            "backward_executed": False,
            "checkpoint_written": False,
            "loss_type": model_config.loss_type,
            "fixed_flow_time": FIXED_FLOW_TIME,
            "explicit_noise": True,
            "image_augmentation": False,
            "future_image_query": True,
            "future_frame_offset_steps": EXPECTED_CHUNK_SIZE - 1,
            "future_frame_offset_seconds": (EXPECTED_CHUNK_SIZE - 1) / dataset_fps,
            "dataset_fps": dataset_fps,
            "teacher_camera": "camera_top (camera index 0)",
            "resolved_video_effective_fps": resolved_video_effective_fps,
            "complete_alignment_loss_enabled": True,
        },
        "batch": {key: tensor_summary(value) for key, value in batch.items()},
        "teacher_inputs": {
            "pil_images": tensor_summary(pil_images),
            "future_pil_images": tensor_summary(future_pil_images),
            "future_video_effective_fps": (
                tensor_summary(future_video_effective_fps)
                if torch.is_tensor(future_video_effective_fps)
                else future_video_effective_fps
            ),
        },
        "targets": {
            key: tensor_summary(value) for key, value in target_tensors.items()
        },
        "predictions": {
            key: tensor_summary(value) for key, value in prediction_tensors.items()
        },
        "loss": loss_values,
        "loss_log": loss_log_summary(loss_log),
        "model": model_parameters,
        "teachers": teacher_summaries,
        "timing": {
            "model_load_seconds": round(model_load_seconds, 3),
            "depth_teacher_load_seconds": round(depth_teacher_load_seconds, 3),
            "video_teacher_load_seconds": round(video_teacher_load_seconds, 3),
            "target_generation_seconds": round(target_seconds, 3),
            "full_forward_seconds": round(forward_seconds, 3),
        },
        "gpu": {
            "visible_device": 0,
            "name": gpu.name,
            "total_memory_bytes": gpu.total_memory,
            "allocated_after_model_bytes": allocated_after_model,
            "allocated_after_depth_teachers_bytes": allocated_after_depth_teachers,
            "allocated_after_all_teachers_bytes": allocated_after_all_teachers,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        },
        "scope": (
            "complete_official_auxiliary_teacher_target_and_loss_forward_"
            "no_backward_no_optimizer"
        ),
    }
    result_path = run_dir / "audit.json"
    with result_path.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"result_path={result_path}", flush=True)
    print("LINGBOT_AUXILIARY_FORWARD_LOSS_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
