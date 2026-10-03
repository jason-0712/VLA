#!/usr/bin/env python3
"""Validate and summarize the one-step LingBot training transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_PARAMETER_COUNT = 6_375_906_359


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def latest_json(path: Path) -> Path:
    candidates = sorted(path.glob("*.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one reload result in {path}, got {candidates}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()

    preflight_path = run_dir / "preflight.json"
    log_path = run_dir / "train.log"
    checkpoint_path = run_dir / "checkpoints/global_step_1"
    model_dir = checkpoint_path / "model"
    optimizer_dir = checkpoint_path / "optimizer"
    extra_state_dir = checkpoint_path / "extra_state"
    hf_path = checkpoint_path / "hf_ckpt"
    hf_index_path = hf_path / "model.safetensors.index.json"
    reload_path = latest_json(run_dir / "reload/model_load_smoke")
    extra_state_files = sorted(extra_state_dir.glob("extra_state_rank_*.pt"))
    if len(extra_state_files) != 4:
        raise RuntimeError(
            f"Expected four rank-local extra-state files, got {extra_state_files}"
        )

    required = (
        preflight_path,
        log_path,
        model_dir / ".metadata",
        optimizer_dir / ".metadata",
        hf_index_path,
        reload_path,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    with preflight_path.open(encoding="utf-8") as stream:
        preflight = json.load(stream)
    with reload_path.open(encoding="utf-8") as stream:
        reload_result = json.load(stream)
    if reload_result.get("status") != "ok":
        raise RuntimeError("Strict FP32 HF checkpoint reload did not pass")
    if reload_result["model"]["parameter_count"] != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("Reloaded checkpoint parameter count changed")

    log = log_path.read_text(encoding="utf-8", errors="replace")
    required_markers = (
        "Starting training from scratch.",
        "Distributed checkpoint saved at",
        "Reached max_steps=1, stopping training.",
        "HF checkpoint finished for",
    )
    missing_markers = [marker for marker in required_markers if marker not in log]
    if missing_markers:
        raise RuntimeError(f"Training log is missing markers: {missing_markers}")

    metric_pattern = re.compile(
        r"Step 1/\d+.*?Loss (?P<loss>[0-9.eE+-]+), "
        r"VLA_Loss (?P<vla>[0-9.eE+-]+), "
        r"Depth_Loss (?P<depth>[0-9.eE+-]+), "
        r"Future_Depth_Loss (?P<future_depth>[0-9.eE+-]+), "
        r"FutureVideo_Loss (?P<future_video>[0-9.eE+-]+), "
        r"SeqWise_Loss (?P<seq>[0-9.eE+-]+), "
        r"RouterZ_Loss (?P<router>[0-9.eE+-]+), .*?"
        r"GradNorm (?P<grad_norm>[0-9.eE+-]+), "
        r"LR (?P<lr>[0-9.eE+-]+), .*?"
        r"StepTime (?P<step_time>[0-9.eE+-]+)s, "
        r"Depth_Forward_Time\s+(?P<teacher_time>[0-9.eE+-]+)s"
    )
    matches = list(metric_pattern.finditer(log))
    if not matches:
        raise RuntimeError("Could not parse step-1 training metrics")
    metrics = {key: float(value) for key, value in matches[-1].groupdict().items()}

    memory_matches = re.findall(r"max ([0-9.]+)GB", log)
    if not memory_matches:
        raise RuntimeError("Could not parse peak GPU allocation")
    peak_allocated_gib = max(float(value) for value in memory_matches)

    audit = {
        "status": "ok",
        "check": "four_gpu_fp32_fsdp2_one_step_checkpoint_export_reload",
        "experiment_id": preflight["experiment_id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": preflight["scope"],
        "source": preflight["source"],
        "data": preflight["data"],
        "runtime": {
            **preflight["runtime"],
            "precision": "FP32 policy; official BF16 teacher autocast",
            "world_size": 4,
            "optimizer": "muon_with_adamw_fallback",
            "data_parallel_mode": "fsdp2",
            "enable_full_shard": False,
            "gradient_checkpointing": True,
            "torch_compile": False,
            "steps": 1,
            "peak_allocated_gib_from_rank0_log": peak_allocated_gib,
        },
        "step_1": metrics,
        "checkpoint": {
            "path": str(checkpoint_path),
            "dcp_model_bytes": directory_size(model_dir),
            "dcp_optimizer_bytes": directory_size(optimizer_dir),
            "dcp_extra_state_bytes": directory_size(extra_state_dir),
            "hf_path": str(hf_path),
            "hf_bytes": directory_size(hf_path),
            "hf_index_sha256": sha256(hf_index_path),
            "strict_reload_result": str(reload_path),
            "strict_reload_result_sha256": sha256(reload_path),
            "reloaded_parameter_count": reload_result["model"]["parameter_count"],
            "reloaded_parameter_dtype_counts": reload_result["model"][
                "parameter_dtype_counts"
            ],
        },
        "protocol": {
            "backward_executed": True,
            "optimizer_step_executed": True,
            "dcp_model_optimizer_extra_state_written": True,
            "hf_checkpoint_exported": True,
            "hf_checkpoint_strictly_reloaded": True,
            "scored_model": False,
            "overfit_run": False,
            "throughput_run": False,
        },
        "log": {"path": str(log_path), "sha256": sha256(log_path)},
    }
    audit_path = run_dir / "audit.json"
    with audit_path.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True)
        stream.write("\n")
    audit_path.chmod(0o444)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"audit_path={audit_path}")
    print("LINGBOT_TRAINING_STEP_SMOKE_OK")


if __name__ == "__main__":
    main()
