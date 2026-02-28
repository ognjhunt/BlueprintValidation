"""DreamDojo LoRA fine-tuning orchestration."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List

from ..common import get_logger, write_json
from ..config import FinetuneConfig

logger = get_logger("training.dreamdojo_finetune")


def list_dreamdojo_experiments(dreamdojo_root: Path) -> List[str]:
    """List dynamically registered experiment names from DreamDojo YAML configs."""
    configs_root = dreamdojo_root / "configs"
    if not configs_root.exists():
        return []
    experiments = []
    for yaml_file in sorted(configs_root.glob("*.yaml")):
        experiments.append(f"dreamdojo_{yaml_file.stem}".lower())
    return experiments


def resolve_dreamdojo_experiment_name(dreamdojo_root: Path, configured: str | None) -> str:
    """Resolve a configured experiment into the vendor's expected Hydra experiment name."""
    available = list_dreamdojo_experiments(dreamdojo_root)
    if not available:
        raise RuntimeError(
            f"No DreamDojo YAML configs found under {dreamdojo_root / 'configs'}. "
            "Ensure finetune.dreamdojo_repo points to a valid DreamDojo checkout."
        )

    if configured:
        token = configured.strip()
        maybe_path = Path(token)
        if maybe_path.is_absolute() or "/" in token or token.endswith(".yaml"):
            candidate = maybe_path if maybe_path.is_absolute() else (dreamdojo_root / "configs" / maybe_path)
            if candidate.suffix != ".yaml":
                candidate_yaml = candidate.with_suffix(".yaml")
                if candidate_yaml.exists():
                    candidate = candidate_yaml
            if not candidate.exists():
                raise RuntimeError(
                    f"DreamDojo experiment config not found: {configured}. "
                    f"Expected a YAML under {dreamdojo_root / 'configs'}."
                )
            experiment_name = f"dreamdojo_{candidate.stem}".lower()
            if experiment_name in available:
                return experiment_name
            raise RuntimeError(
                f"Experiment '{experiment_name}' is not registered from {candidate}. "
                "Check DreamDojo config loading."
            )

        normalized = token.lower()
        if normalized.startswith("dreamdojo_") and normalized in available:
            return normalized

        candidate = f"dreamdojo_{Path(token).stem}".lower()
        if candidate in available:
            return candidate

        raise RuntimeError(
            f"Unknown DreamDojo experiment '{configured}'. "
            f"Available examples: {', '.join(available[:5])}"
        )

    preferred = [
        "dreamdojo_2b_480_640_gr1",
        "dreamdojo_2b_480_640_g1",
        "dreamdojo_2b_480_640_agibot",
        "dreamdojo_2b_480_640_yam",
    ]
    for candidate in preferred:
        if candidate in available:
            logger.warning(
                "Auto-selected DreamDojo experiment: %s "
                "(set finetune.experiment_config to override)",
                candidate,
            )
            return candidate

    picked = available[0]
    logger.warning(
        "Auto-selected DreamDojo experiment: %s (set finetune.experiment_config to override)",
        picked,
    )
    return picked


def build_dreamdojo_launch_command(
    dreamdojo_root: Path,
    experiment_name: str,
    dataset_dir: Path,
    output_dir: Path,
    config: FinetuneConfig,
    facility_id: str,
) -> list[str]:
    """Build a portable DreamDojo training command without cluster-specific launch.sh wrappers."""
    train_script = dreamdojo_root / "scripts" / "train.py"
    if not train_script.exists():
        raise RuntimeError(
            f"DreamDojo train entrypoint not found at {train_script}. "
            "Ensure finetune.dreamdojo_repo points to a valid DreamDojo checkout."
        )

    steps_per_epoch = max(1, int(os.environ.get("DREAMDOJO_STEPS_PER_EPOCH", "1000")))
    max_iter = max(1, int(config.num_epochs) * steps_per_epoch)
    nproc = max(1, int(os.environ.get("DREAMDOJO_NPROC", "1")))

    run_dir = output_dir / "lora_weights"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc-per-node",
        str(nproc),
        "-m",
        "scripts.train",
        "--config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
        "--",
        f"experiment={experiment_name}",
        f"job.name=blueprint_{facility_id}",
        "job.wandb_mode=disabled",
        f"job.path_local={run_dir}",
        f"dataloader_train.dataset.dataset_path={dataset_dir}",
        f"dataloader_train.batch_size={config.batch_size}",
        f"trainer.grad_accum_iter={config.gradient_accumulation_steps}",
        f"trainer.max_iter={max_iter}",
        f"optimizer.lr={config.learning_rate}",
        f"checkpoint.load_path={config.dreamdojo_checkpoint}",
        f"model.config.use_lora={'true' if config.use_lora else 'false'}",
        f"model.config.lora_rank={config.lora_rank}",
        f"model.config.lora_alpha={config.lora_alpha}",
        f"model.config.lora_target_modules={config.lora_target_modules}",
    ]
    return cmd


def _resolve_latest_checkpoint(lora_dir: Path) -> Path | None:
    checkpoints_dir = lora_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return None

    checkpoint_dirs = [
        path
        for path in checkpoints_dir.iterdir()
        if path.is_dir() and path.name.startswith("iter_")
    ]
    if checkpoint_dirs:
        checkpoint_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoint_dirs[0]
    if any(checkpoints_dir.iterdir()):
        return checkpoints_dir
    return None


def run_dreamdojo_finetune(
    dataset_dir: Path,
    output_dir: Path,
    config: FinetuneConfig,
    facility_id: str,
) -> Dict:
    """Run DreamDojo LoRA fine-tuning on a prepared dataset."""
    dreamdojo_root = config.dreamdojo_repo
    lora_dir = output_dir / "lora_weights"
    lora_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "finetune_log.json"

    if not dreamdojo_root.exists():
        raise RuntimeError(
            f"DreamDojo repo not found at {dreamdojo_root}. "
            "Set finetune.dreamdojo_repo to a valid local checkout."
        )
    if not config.dreamdojo_checkpoint.exists():
        raise RuntimeError(
            f"DreamDojo checkpoint not found: {config.dreamdojo_checkpoint}. "
            "Run model download/setup first."
        )
    if not dataset_dir.exists():
        raise RuntimeError(f"DreamDojo dataset directory not found: {dataset_dir}")

    experiment_name = resolve_dreamdojo_experiment_name(
        dreamdojo_root=dreamdojo_root,
        configured=config.experiment_config,
    )
    cmd = build_dreamdojo_launch_command(
        dreamdojo_root=dreamdojo_root,
        experiment_name=experiment_name,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        config=config,
        facility_id=facility_id,
    )

    logger.info(
        "Starting DreamDojo LoRA fine-tuning: facility=%s experiment=%s lr=%s",
        facility_id,
        experiment_name,
        config.learning_rate,
    )
    logger.info("Command: %s", " ".join(cmd))

    start_time = time.monotonic()
    timeout_sec = int(config.max_training_hours * 3600)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(dreamdojo_root),
        timeout=timeout_sec,
    )
    elapsed = time.monotonic() - start_time
    status = "success" if result.returncode == 0 else "failed"

    checkpoint_path = _resolve_latest_checkpoint(lora_dir)
    train_result = {
        "facility_id": facility_id,
        "status": status,
        "elapsed_seconds": elapsed,
        "lora_weights_path": str(lora_dir),
        "adapted_checkpoint_path": str(checkpoint_path or ""),
        "returncode": result.returncode,
        "experiment_name": experiment_name,
    }

    if result.returncode != 0:
        train_result["stderr"] = (result.stderr or "")[-2000:]
        train_result["stdout"] = (result.stdout or "")[-2000:]
        logger.error("DreamDojo fine-tuning failed: %s", train_result["stderr"])
    else:
        logger.info("DreamDojo fine-tuning complete in %.1fs", elapsed)
        if checkpoint_path is None:
            train_result["status"] = "failed"
            train_result["stderr"] = (
                "DreamDojo command exited successfully but produced no checkpoints under "
                f"{lora_dir / 'checkpoints'}."
            )

    train_result["loss_history"] = _parse_loss_from_output(result.stdout or "")
    write_json(train_result, log_path)
    return train_result


def _parse_loss_from_output(stdout: str) -> list[dict]:
    """Try to parse training loss from DreamDojo stdout."""
    losses = []
    for line in stdout.split("\n"):
        if "loss" in line.lower() and "epoch" in line.lower():
            parts = line.strip().split()
            for i, part in enumerate(parts):
                if part.lower().startswith("loss") and i + 1 < len(parts):
                    val = parts[i + 1].strip(":,=")
                    try:
                        losses.append({"raw": line.strip(), "loss": float(val)})
                    except ValueError:
                        continue
    return losses
