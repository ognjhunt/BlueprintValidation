"""DreamDojo LoRA fine-tuning orchestration."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Dict

from ..common import get_logger, write_json
from ..config import FinetuneConfig

logger = get_logger("training.dreamdojo_finetune")


def _resolve_experiment_config(dreamdojo_root: Path, configured: str | None) -> Path:
    """Resolve the DreamDojo experiment config script."""
    configs_root = dreamdojo_root / "configs"

    if configured:
        raw = Path(configured)
        if raw.exists():
            return raw.resolve()

        if configs_root.exists():
            candidates = [
                configs_root / configured,
                configs_root / f"{configured}.sh",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate.resolve()

        raise RuntimeError(
            f"DreamDojo experiment config not found: {configured}. "
            f"Provide finetune.experiment_config as a file path or config name under {configs_root}."
        )

    if not configs_root.exists():
        raise RuntimeError(
            f"DreamDojo configs directory not found: {configs_root}. "
            "Set finetune.experiment_config to a valid config script path."
        )

    available = sorted(configs_root.rglob("*.sh"))
    if not available:
        raise RuntimeError(
            f"No DreamDojo config scripts found under {configs_root}. "
            "Set finetune.experiment_config explicitly."
        )

    # Prefer post-training/adapted config scripts when auto-selecting.
    preferred = [
        p for p in available
        if "post" in p.as_posix().lower() or "adapt" in p.as_posix().lower()
    ]
    picked = preferred[0] if preferred else available[0]
    logger.warning(
        "Auto-selected DreamDojo config script: %s (set finetune.experiment_config to override)",
        picked,
    )
    return picked.resolve()


def render_dreamdojo_config_script(
    base_config: Path,
    dataset_dir: Path,
    output_dir: Path,
    config: FinetuneConfig,
    facility_id: str,
) -> Path:
    """Render a DreamDojo launch config script that layers Blueprint overrides."""
    config_script = output_dir / "dreamdojo_launch_config.sh"
    lora_dir = output_dir / "lora_weights"
    exp_name = f"blueprint_{facility_id}_lora"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'source "{base_config}"',
        "",
        f'EXP_NAME="${{EXP_NAME:-{exp_name}}}"',
        f'CHECKPOINT="${{CHECKPOINT:-{config.dreamdojo_checkpoint}}}"',
        f'DATASET_PATH="${{DATASET_PATH:-{dataset_dir}}}"',
        f'TRAIN_DATASET_PATH="${{TRAIN_DATASET_PATH:-{dataset_dir}}}"',
        f'BASE_EXPERIMENT_PATH="${{BASE_EXPERIMENT_PATH:-{lora_dir}}}"',
        "",
        "# LoRA + training overrides for post-training configs.",
        'TRAIN_ARCHITECTURE="${TRAIN_ARCHITECTURE:-lora}"',
        'USE_LORA="${USE_LORA:-true}"',
        f'LORA_RANK="${{LORA_RANK:-{config.lora_rank}}}"',
        f'LORA_ALPHA="${{LORA_ALPHA:-{config.lora_alpha}}}"',
        f'LEARNING_RATE="${{LEARNING_RATE:-{config.learning_rate}}}"',
        f'LR="${{LR:-{config.learning_rate}}}"',
        f'NUM_EPOCHS="${{NUM_EPOCHS:-{config.num_epochs}}}"',
        f'EPOCHS="${{EPOCHS:-{config.num_epochs}}}"',
        f'BATCH_SIZE="${{BATCH_SIZE:-{config.batch_size}}}"',
        f'GRADIENT_ACCUMULATION_STEPS="${{GRADIENT_ACCUMULATION_STEPS:-{config.gradient_accumulation_steps}}}"',
        f'WARMUP_STEPS="${{WARMUP_STEPS:-{config.warmup_steps}}}"',
    ]
    config_script.write_text("\n".join(lines) + "\n")
    config_script.chmod(0o755)
    return config_script


def build_dreamdojo_launch_command(
    dreamdojo_root: Path,
    config_script: Path,
) -> list[str]:
    """Build the DreamDojo launch command."""
    launch_script = dreamdojo_root / "launch.sh"
    if not launch_script.exists():
        raise RuntimeError(
            f"DreamDojo launch script not found at {launch_script}. "
            "Ensure finetune.dreamdojo_repo points to a valid DreamDojo checkout."
        )
    return ["bash", str(launch_script), str(config_script)]


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

    base_config = _resolve_experiment_config(dreamdojo_root, config.experiment_config)
    config_script = render_dreamdojo_config_script(
        base_config=base_config,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        config=config,
        facility_id=facility_id,
    )
    cmd = build_dreamdojo_launch_command(dreamdojo_root, config_script)

    logger.info(
        "Starting DreamDojo LoRA fine-tuning: facility=%s epochs=%d lr=%s",
        facility_id,
        config.num_epochs,
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

    train_result = {
        "facility_id": facility_id,
        "status": status,
        "elapsed_seconds": elapsed,
        "lora_weights_path": str(lora_dir),
        "adapted_checkpoint_path": str(lora_dir),
        "returncode": result.returncode,
        "base_experiment_config": str(base_config),
        "rendered_config_script": str(config_script),
    }

    if result.returncode != 0:
        train_result["stderr"] = (result.stderr or "")[-2000:]
        train_result["stdout"] = (result.stdout or "")[-2000:]
        logger.error("DreamDojo fine-tuning failed: %s", train_result["stderr"])
    else:
        logger.info("DreamDojo fine-tuning complete in %.1fs", elapsed)
        # Strict mode: require artifact presence.
        if not any(lora_dir.rglob("*")):
            train_result["status"] = "failed"
            train_result["stderr"] = (
                "DreamDojo command exited successfully but produced no LoRA artifacts "
                f"under {lora_dir}."
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
