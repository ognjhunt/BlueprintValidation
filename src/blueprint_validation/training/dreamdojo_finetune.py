"""DreamDojo LoRA fine-tuning orchestration."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

from ..common import get_logger, write_json
from ..config import FinetuneConfig

logger = get_logger("training.dreamdojo_finetune")


def run_dreamdojo_finetune(
    dataset_dir: Path,
    output_dir: Path,
    config: FinetuneConfig,
    facility_id: str,
) -> Dict:
    """Run DreamDojo LoRA fine-tuning on a prepared dataset.

    Returns a dict with training results and checkpoint paths.
    """
    lora_dir = output_dir / "lora_weights"
    lora_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "finetune_log.json"

    # Build training config for DreamDojo post-training
    train_config = {
        "model_path": str(config.dreamdojo_checkpoint),
        "model_size": config.model_size,
        "dataset_path": str(dataset_dir),
        "output_dir": str(lora_dir),
        "lora": {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "training": {
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "warmup_steps": config.warmup_steps,
            "save_every_n_epochs": config.save_every_n_epochs,
            "max_training_hours": config.max_training_hours,
        },
        "facility_id": facility_id,
    }

    config_path = output_dir / "train_config.json"
    write_json(train_config, config_path)

    # Try DreamDojo's built-in post-training script
    dreamdojo_root = config.dreamdojo_repo
    posttrain_script = dreamdojo_root / "scripts" / "posttrain.py"

    if not posttrain_script.exists():
        # Fallback: look for launch.sh
        launch_script = dreamdojo_root / "launch.sh"
        if launch_script.exists():
            posttrain_script = launch_script

    # Build command
    cmd = [
        "python", str(posttrain_script),
        "--config", str(config_path),
        "--model_path", str(config.dreamdojo_checkpoint),
        "--dataset_path", str(dataset_dir),
        "--output_dir", str(lora_dir),
        "--lora_rank", str(config.lora_rank),
        "--lora_alpha", str(config.lora_alpha),
        "--learning_rate", str(config.learning_rate),
        "--num_epochs", str(config.num_epochs),
        "--batch_size", str(config.batch_size),
        "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
    ]

    logger.info("Starting DreamDojo fine-tuning: facility=%s, epochs=%d, lr=%s",
                facility_id, config.num_epochs, config.learning_rate)
    logger.info("Command: %s", " ".join(cmd))

    start_time = time.monotonic()
    timeout_sec = int(config.max_training_hours * 3600)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(dreamdojo_root) if dreamdojo_root.exists() else None,
        timeout=timeout_sec,
    )

    elapsed = time.monotonic() - start_time

    train_result = {
        "facility_id": facility_id,
        "status": "success" if result.returncode == 0 else "failed",
        "elapsed_seconds": elapsed,
        "lora_weights_path": str(lora_dir),
        "config": train_config,
        "returncode": result.returncode,
    }

    if result.returncode != 0:
        train_result["stderr"] = result.stderr[-2000:]
        logger.error("DreamDojo fine-tuning failed:\n%s", result.stderr[-500:])
    else:
        logger.info("DreamDojo fine-tuning complete in %.1fs", elapsed)

    # Parse training loss from stdout if available
    train_result["loss_history"] = _parse_loss_from_output(result.stdout)

    write_json(train_result, log_path)
    return train_result


def _parse_loss_from_output(stdout: str) -> list[dict]:
    """Try to parse training loss from DreamDojo stdout."""
    losses = []
    for line in stdout.split("\n"):
        if "loss" in line.lower() and "epoch" in line.lower():
            try:
                # Try to find numeric loss values
                parts = line.strip().split()
                for i, part in enumerate(parts):
                    if part.lower().startswith("loss"):
                        if i + 1 < len(parts):
                            val = parts[i + 1].strip(":,=")
                            try:
                                losses.append({"raw": line.strip(), "loss": float(val)})
                            except ValueError:
                                pass
            except Exception:
                continue
    return losses
