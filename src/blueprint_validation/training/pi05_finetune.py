"""pi0.5 fine-tuning orchestration helpers."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from ..common import get_logger, write_json
from ..config import Pi05AdapterBackendConfig, PolicyFinetuneConfig

logger = get_logger("training.pi05_finetune")


def _resolve_script(repo_root: Path, script_value: str) -> Path:
    script = Path(script_value)
    if script.is_absolute():
        return script
    return (repo_root / script).resolve()


def build_pi05_norm_stats_command(
    openpi_repo: Path,
    norm_stats_script: str,
    dataset_root: Path,
    dataset_name: str,
    profile: str,
    extra_args: Optional[list[str]] = None,
) -> list[str]:
    script_path = _resolve_script(openpi_repo, norm_stats_script)
    cmd = [
        sys.executable,
        str(script_path),
        "--dataset_root",
        str(dataset_root),
        "--dataset_name",
        dataset_name,
        "--profile",
        profile,
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def build_pi05_train_command(
    openpi_repo: Path,
    train_script: str,
    dataset_root: Path,
    dataset_name: str,
    profile: str,
    exp_name: str,
    run_root_dir: Path,
    base_model_name: str,
    base_checkpoint: Optional[Path],
    finetune_config: PolicyFinetuneConfig,
    extra_args: Optional[list[str]] = None,
) -> list[str]:
    script_path = _resolve_script(openpi_repo, train_script)
    cmd = [
        sys.executable,
        str(script_path),
        profile,
        "--exp_name",
        exp_name,
        "--run_root_dir",
        str(run_root_dir),
        "--dataset_root",
        str(dataset_root),
        "--dataset_name",
        dataset_name,
        "--base_model",
        base_model_name,
        "--batch_size",
        str(finetune_config.batch_size),
        "--learning_rate",
        str(finetune_config.learning_rate),
        "--max_steps",
        str(finetune_config.max_steps),
    ]
    if base_checkpoint and base_checkpoint.exists():
        cmd.extend(["--base_checkpoint", str(base_checkpoint)])
    if finetune_config.wandb_project:
        cmd.extend(["--wandb_project", finetune_config.wandb_project])
    if finetune_config.wandb_entity:
        cmd.extend(["--wandb_entity", finetune_config.wandb_entity])
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def resolve_latest_pi05_checkpoint(run_root_dir: Path) -> Path | None:
    """Resolve the newest completed pi0.5 checkpoint path under a run root."""
    if not run_root_dir.exists():
        return None

    candidates: list[Path] = []
    for pattern in ("model.safetensors", "pytorch_model.bin", "checkpoint.pt"):
        for model_file in run_root_dir.rglob(pattern):
            candidates.append(model_file.parent)
    if not candidates:
        for directory in run_root_dir.rglob("*"):
            if directory.is_dir() and "checkpoint" in directory.name.lower():
                candidates.append(directory)
    if not candidates:
        return None

    candidates = sorted(
        set(candidates),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0]


def run_pi05_finetune(
    config: PolicyFinetuneConfig,
    backend: Pi05AdapterBackendConfig,
    base_model_name: str,
    base_checkpoint: Optional[Path],
    dataset_root: Path,
    dataset_name: str,
    facility_id: str,
    output_dir: Path,
) -> Dict:
    """Execute pi0.5 fine-tuning with norm-stats + train phases."""
    if backend.train_backend != "pytorch":
        raise RuntimeError(
            f"Unsupported pi05 train_backend={backend.train_backend}. Only 'pytorch' is supported."
        )
    if not backend.openpi_repo.exists():
        raise RuntimeError(
            f"openpi repo not found at {backend.openpi_repo}. "
            "Set policy_adapter.pi05.openpi_repo to a valid checkout."
        )
    if not dataset_root.exists():
        raise RuntimeError(f"Dataset root does not exist: {dataset_root}")
    if not (dataset_root / dataset_name).exists():
        raise RuntimeError(
            f"pi05 dataset directory missing. Expected {dataset_root / dataset_name}."
        )

    train_script = _resolve_script(backend.openpi_repo, backend.train_script)
    norm_stats_script = _resolve_script(backend.openpi_repo, backend.norm_stats_script)
    if not train_script.exists():
        raise RuntimeError(f"pi05 train script not found at {train_script}")
    if not norm_stats_script.exists():
        raise RuntimeError(f"pi05 norm-stats script not found at {norm_stats_script}")

    run_root_dir = output_dir / "runs"
    run_root_dir.mkdir(parents=True, exist_ok=True)
    exp_name = f"blueprint_{facility_id}_{int(time.time())}"

    norm_cmd = build_pi05_norm_stats_command(
        openpi_repo=backend.openpi_repo,
        norm_stats_script=backend.norm_stats_script,
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        profile=backend.profile,
    )
    train_cmd = build_pi05_train_command(
        openpi_repo=backend.openpi_repo,
        train_script=backend.train_script,
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        profile=backend.profile,
        exp_name=exp_name,
        run_root_dir=run_root_dir,
        base_model_name=base_model_name,
        base_checkpoint=base_checkpoint,
        finetune_config=config,
        extra_args=list(backend.extra_train_args),
    )

    start = time.monotonic()
    norm_proc = subprocess.run(
        norm_cmd,
        cwd=str(backend.openpi_repo),
        capture_output=True,
        text=True,
    )
    if norm_proc.returncode != 0:
        elapsed = time.monotonic() - start
        result = {
            "status": "failed",
            "facility_id": facility_id,
            "phase": "norm_stats",
            "elapsed_seconds": elapsed,
            "returncode": norm_proc.returncode,
            "dataset_name": dataset_name,
            "command_norm_stats": norm_cmd,
            "stdout": (norm_proc.stdout or "")[-3000:],
            "stderr": (norm_proc.stderr or "")[-3000:],
            "run_root_dir": str(run_root_dir),
        }
        write_json(result, output_dir / "policy_finetune_log.json")
        return result

    train_proc = subprocess.run(
        train_cmd,
        cwd=str(backend.openpi_repo),
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - start

    result = {
        "status": "success" if train_proc.returncode == 0 else "failed",
        "facility_id": facility_id,
        "elapsed_seconds": elapsed,
        "returncode": train_proc.returncode,
        "dataset_name": dataset_name,
        "base_model_name": base_model_name,
        "base_checkpoint": str(base_checkpoint) if base_checkpoint else "",
        "command_norm_stats": norm_cmd,
        "command_train": train_cmd,
        "run_root_dir": str(run_root_dir),
    }
    if train_proc.returncode != 0:
        result["stdout"] = (train_proc.stdout or "")[-3000:]
        result["stderr"] = (train_proc.stderr or "")[-3000:]
    else:
        checkpoint = resolve_latest_pi05_checkpoint(run_root_dir)
        if checkpoint is None:
            result["status"] = "failed"
            result["stderr"] = (
                "pi05 fine-tune command succeeded but no checkpoint artifacts were found under "
                f"{run_root_dir}"
            )
        else:
            result["adapted_checkpoint_path"] = str(checkpoint)

    write_json(result, output_dir / "policy_finetune_log.json")
    return result
