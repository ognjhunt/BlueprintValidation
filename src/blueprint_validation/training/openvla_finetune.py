"""OpenVLA fine-tuning orchestration (LoRA/OFT adapter stage)."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict

from ..common import get_logger, write_json
from ..config import PolicyFinetuneConfig

logger = get_logger("training.openvla_finetune")


def _resolve_finetune_script(openvla_repo: Path, script_value: str) -> Path:
    script_path = Path(script_value)
    if script_path.is_absolute():
        return script_path
    return (openvla_repo / script_path).resolve()


def build_openvla_finetune_command(
    script_path: Path,
    config: PolicyFinetuneConfig,
    vla_path: str,
    run_root_dir: Path,
    adapter_tmp_dir: Path,
) -> list[str]:
    """Build a torchrun command matching the OpenVLA finetune entrypoint."""
    if config.data_root_dir is None:
        raise RuntimeError(
            "policy_finetune.data_root_dir is required when policy_finetune.enabled=true."
        )

    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc-per-node",
        str(config.nproc_per_node),
        str(script_path),
        "--vla_path",
        vla_path,
        "--data_root_dir",
        str(config.data_root_dir),
        "--dataset_name",
        config.dataset_name,
        "--run_root_dir",
        str(run_root_dir),
        "--adapter_tmp_dir",
        str(adapter_tmp_dir),
        "--lora_rank",
        str(config.lora_rank),
        "--batch_size",
        str(config.batch_size),
        "--grad_accumulation_steps",
        str(config.grad_accumulation_steps),
        "--learning_rate",
        str(config.learning_rate),
        "--save_steps",
        str(config.save_steps),
        "--max_steps",
        str(config.max_steps),
        "--image_aug",
        "True" if config.image_aug else "False",
    ]
    if config.wandb_project:
        cmd.extend(["--wandb_project", config.wandb_project])
    if config.wandb_entity:
        cmd.extend(["--wandb_entity", config.wandb_entity])
    # OFT-oriented knobs are passed through as explicit flags expected by OFT forks.
    if config.recipe.lower() == "oft":
        cmd.extend(["--action_chunk_size", str(config.action_chunk_size)])
        cmd.extend(["--parallel_decoding", "True" if config.parallel_decoding else "False"])
        cmd.extend(
            ["--continuous_actions", "True" if config.use_continuous_actions else "False"]
        )
        cmd.extend(["--l1_regression", "True" if config.use_l1_regression else "False"])
    if config.extra_args:
        cmd.extend(config.extra_args)
    return cmd


def _resolve_artifact_path(adapter_tmp_dir: Path, run_root_dir: Path) -> Path | None:
    if adapter_tmp_dir.exists() and any(adapter_tmp_dir.iterdir()):
        return adapter_tmp_dir
    if not run_root_dir.exists():
        return None
    candidates = sorted(run_root_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def run_openvla_finetune(
    config: PolicyFinetuneConfig,
    vla_path: str,
    facility_id: str,
    output_dir: Path,
) -> Dict:
    """Execute OpenVLA fine-tuning and return run metadata."""
    if not config.openvla_repo.exists():
        raise RuntimeError(
            f"OpenVLA repo not found at {config.openvla_repo}. "
            "Set policy_finetune.openvla_repo to a valid checkout."
        )
    if config.data_root_dir is None:
        raise RuntimeError(
            "policy_finetune.data_root_dir must be set when policy_finetune.enabled=true."
        )
    if not config.data_root_dir.exists():
        raise RuntimeError(
            f"policy_finetune.data_root_dir does not exist: {config.data_root_dir}"
        )
    if not (config.data_root_dir / config.dataset_name).exists():
        raise RuntimeError(
            "OpenVLA dataset directory missing. Expected "
            f"{config.data_root_dir / config.dataset_name}. "
            "Provide policy_finetune.data_root_dir and policy_finetune.dataset_name."
        )
    if not shutil.which("torchrun"):
        raise RuntimeError("torchrun not found in PATH. Install PyTorch distributed tooling.")

    script_path = _resolve_finetune_script(config.openvla_repo, config.finetune_script)
    if not script_path.exists():
        raise RuntimeError(
            f"OpenVLA fine-tune script not found at {script_path}. "
            "Check policy_finetune.finetune_script."
        )

    run_root_dir = output_dir / "runs"
    adapter_tmp_dir = output_dir / "adapters"
    run_root_dir.mkdir(parents=True, exist_ok=True)
    adapter_tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_openvla_finetune_command(
        script_path=script_path,
        config=config,
        vla_path=vla_path,
        run_root_dir=run_root_dir,
        adapter_tmp_dir=adapter_tmp_dir,
    )
    logger.info(
        "Starting OpenVLA fine-tuning for facility=%s dataset=%s",
        facility_id,
        config.dataset_name,
    )
    logger.info("Command: %s", " ".join(cmd))

    start = time.monotonic()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(config.openvla_repo),
    )
    elapsed = time.monotonic() - start

    finetune_result = {
        "facility_id": facility_id,
        "status": "success" if result.returncode == 0 else "failed",
        "elapsed_seconds": elapsed,
        "returncode": result.returncode,
        "vla_path": vla_path,
        "dataset_name": config.dataset_name,
        "data_root_dir": str(config.data_root_dir),
        "recipe": config.recipe,
        "command": cmd,
        "run_root_dir": str(run_root_dir),
        "adapter_tmp_dir": str(adapter_tmp_dir),
    }

    artifact_path = _resolve_artifact_path(adapter_tmp_dir, run_root_dir)
    if result.returncode != 0:
        finetune_result["stdout"] = (result.stdout or "")[-3000:]
        finetune_result["stderr"] = (result.stderr or "")[-3000:]
    elif artifact_path is None:
        finetune_result["status"] = "failed"
        finetune_result["stderr"] = (
            "OpenVLA fine-tune command succeeded but no artifacts were found under "
            f"{adapter_tmp_dir} or {run_root_dir}."
        )
    else:
        finetune_result["adapted_checkpoint_path"] = str(artifact_path)

    write_json(finetune_result, output_dir / "policy_finetune_log.json")
    return finetune_result
