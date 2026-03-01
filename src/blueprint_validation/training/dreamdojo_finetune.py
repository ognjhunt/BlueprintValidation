"""DreamDojo LoRA fine-tuning orchestration."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from ..common import get_logger, write_json
from ..config import FinetuneConfig

logger = get_logger("training.dreamdojo_finetune")


def _quote_hydra_string(raw: str) -> str:
    """Quote a scalar override value so Hydra treats commas as part of a string."""
    token = (raw or "").strip()
    escaped = token.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


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
            candidate = (
                maybe_path
                if maybe_path.is_absolute()
                else (dreamdojo_root / "configs" / maybe_path)
            )
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
    python_executable: Path,
    video_dataset_backend: str,
    checkpoint_path: Path | None = None,
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

    resolved_checkpoint = checkpoint_path or config.dreamdojo_checkpoint
    disable_heavy_callbacks = (
        os.environ.get("DREAMDOJO_DISABLE_HEAVY_CALLBACKS", "1").strip().lower()
        not in {"0", "false", "no"}
    )
    cmd = [
        str(python_executable),
        "-m",
        "torch.distributed.run",
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
        "job.project=blueprint_validation",
        f"job.group={facility_id}",
        f"job.name=blueprint_{facility_id}",
        "job.wandb_mode=disabled",
        f"dataloader_train.dataset.dataset_path={dataset_dir}",
        f"dataloader_val.dataset.dataset_path={dataset_dir}",
        f"dataloader_train.batch_size={config.batch_size}",
        f"trainer.grad_accum_iter={config.gradient_accumulation_steps}",
        f"trainer.max_iter={max_iter}",
        f"optimizer.lr={config.learning_rate}",
        f"checkpoint.load_path={resolved_checkpoint}",
        f"model.config.use_lora={'true' if config.use_lora else 'false'}",
        f"model.config.lora_rank={config.lora_rank}",
        f"model.config.lora_alpha={config.lora_alpha}",
        f"model.config.lora_target_modules={_quote_hydra_string(config.lora_target_modules)}",
        "~dataloader_train.dataloaders",
    ]
    if video_dataset_backend == "opencv":
        cmd.extend(
            [
                "dataloader_train.dataset._target_="
                "blueprint_validation.training.dreamdojo_video_dataset.BlueprintVideoActionDataset",
                "dataloader_val.dataset._target_="
                "blueprint_validation.training.dreamdojo_video_dataset.BlueprintVideoActionDataset",
            ]
        )
    if disable_heavy_callbacks:
        # These callbacks are useful for long training runs, but they can dominate short debug runs.
        cmd.extend(
            [
                "~trainer.callbacks.every_n_sample_reg",
                "~trainer.callbacks.every_n_sample_ema",
                "~trainer.callbacks.wandb",
                "~trainer.callbacks.wandb_10x",
                "trainer.logging_iter=1",
            ]
        )
    return cmd


def _resolve_stage3_python(config: FinetuneConfig) -> Path:
    python_executable = Path(config.python_executable) if config.python_executable else Path(sys.executable)
    if not python_executable.exists():
        raise RuntimeError(
            f"Stage 3 python_executable not found: {python_executable}. "
            "Set finetune.python_executable to a valid runtime path."
        )
    return python_executable


def _build_stage3_env(lora_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["IMAGINAIRE_OUTPUT_ROOT"] = str(lora_dir)
    project_src = Path(__file__).resolve().parents[2]  # .../src
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_src}{os.pathsep}{existing}" if existing else str(project_src)
    return env


def _probe_stage3_runtime(
    *,
    python_executable: Path,
    dreamdojo_root: Path,
    dataset_dir: Path,
    video_dataset_backend: str,
    probe_dataloader_sample: bool,
    env: dict[str, str],
) -> None:
    probe_code = r"""
from pathlib import Path
import sys

dataset_dir = Path(sys.argv[1])
backend = sys.argv[2]
probe_loader = sys.argv[3] == "1"

import lightning  # noqa: F401

sample_video = next((dataset_dir / "videos").glob("*.mp4"), None)
if backend == "vendor":
    import torchcodec  # noqa: F401
    from torchcodec.decoders import VideoDecoder
    if sample_video is not None:
        decoder = VideoDecoder(str(sample_video), dimension_order="NHWC")
        if len(decoder) <= 0:
            raise RuntimeError(f"torchcodec opened {sample_video} but found zero frames")
        _ = decoder.get_frames_in_range(0, 1).data
elif backend == "opencv":
    from blueprint_validation.training.dreamdojo_video_dataset import BlueprintVideoActionDataset

    if probe_loader:
        dataset = BlueprintVideoActionDataset(
            dataset_path=str(dataset_dir),
            num_frames=13,
            data_split="train",
        )
        _ = dataset[0]
else:
    raise RuntimeError(f"Unsupported Stage 3 video dataset backend: {backend}")

print("stage3_runtime_probe_ok")
"""
    try:
        result = subprocess.run(
            [
                str(python_executable),
                "-c",
                probe_code,
                str(dataset_dir),
                video_dataset_backend,
                "1" if probe_dataloader_sample else "0",
            ],
            capture_output=True,
            text=True,
            cwd=str(dreamdojo_root),
            timeout=90,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - runtime dependent
        raise RuntimeError(
            "Stage 3 runtime probe timed out. Dataloader/decode backend did not produce "
            "a sample within 90s."
        ) from exc

    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-2000:]
        stdout_tail = (result.stdout or "")[-2000:]
        raise RuntimeError(
            "Stage 3 runtime probe failed for DreamDojo environment. "
            f"stdout tail: {stdout_tail}\n"
            f"stderr tail: {stderr_tail}"
        )


def _has_dcp_metadata(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "model" / ".metadata").exists()


def _resolve_checkpoint_load_path(configured_path: Path) -> Path:
    """Resolve checkpoint.load_path to an iter_* directory with DCP metadata when needed."""
    if _has_dcp_metadata(configured_path):
        return configured_path

    iter_candidates = [
        path
        for path in configured_path.rglob("iter_*")
        if path.is_dir() and _has_dcp_metadata(path)
    ]
    if iter_candidates:
        iter_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        picked = iter_candidates[0]
        logger.warning(
            "Resolved DreamDojo checkpoint path from %s to %s",
            configured_path,
            picked,
        )
        return picked
    return configured_path


def _resolve_latest_checkpoint(lora_dir: Path) -> Path | None:
    if not lora_dir.exists():
        return None

    # DreamDojo writes under IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/checkpoints.
    iter_dirs = [path for path in lora_dir.rglob("iter_*") if path.is_dir()]
    if iter_dirs:
        iter_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return iter_dirs[0]

    checkpoint_dirs = [path for path in lora_dir.rglob("checkpoints") if path.is_dir()]
    if checkpoint_dirs:
        checkpoint_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoint_dirs[0]
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
    python_executable = _resolve_stage3_python(config)
    video_dataset_backend = (config.video_dataset_backend or "opencv").strip().lower()
    if video_dataset_backend not in {"opencv", "vendor"}:
        raise RuntimeError(
            f"Unsupported finetune.video_dataset_backend={config.video_dataset_backend!r}. "
            "Expected one of: opencv, vendor."
        )
    stage3_env = _build_stage3_env(lora_dir)
    _probe_stage3_runtime(
        python_executable=python_executable,
        dreamdojo_root=dreamdojo_root,
        dataset_dir=dataset_dir,
        video_dataset_backend=video_dataset_backend,
        probe_dataloader_sample=config.probe_dataloader_sample,
        env=stage3_env,
    )

    experiment_name = resolve_dreamdojo_experiment_name(
        dreamdojo_root=dreamdojo_root,
        configured=config.experiment_config,
    )
    checkpoint_path = _resolve_checkpoint_load_path(config.dreamdojo_checkpoint)
    if not _has_dcp_metadata(checkpoint_path):
        raise RuntimeError(
            "DreamDojo checkpoint path is missing distributed-checkpoint metadata "
            f"(expected {checkpoint_path / 'model' / '.metadata'}). "
            "Point finetune.dreamdojo_checkpoint to an iter_* directory or rerun model setup."
        )
    cmd = build_dreamdojo_launch_command(
        dreamdojo_root=dreamdojo_root,
        experiment_name=experiment_name,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        config=config,
        facility_id=facility_id,
        python_executable=python_executable,
        video_dataset_backend=video_dataset_backend,
        checkpoint_path=checkpoint_path,
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
        env=stage3_env,
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
        train_result["stderr"] = (result.stderr or "")[-8000:]
        train_result["stdout"] = (result.stdout or "")[-8000:]
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
