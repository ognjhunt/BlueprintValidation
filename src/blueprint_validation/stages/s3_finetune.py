"""Stage 3: DreamDojo fine-tuning on enriched video."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..common import StageResult, get_logger
from ..config import FacilityConfig, ValidationConfig
from ..training.dataset_builder import build_dreamdojo_dataset
from ..training.dreamdojo_finetune import run_dreamdojo_finetune
from .base import PipelineStage

logger = get_logger("stages.s3_finetune")


class FinetuneStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s3_finetune"

    @property
    def description(self) -> str:
        return "LoRA fine-tune DreamDojo-2B on enriched facility video"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        finetune_dir = work_dir / "finetune"
        finetune_dir.mkdir(parents=True, exist_ok=True)

        # Check for enriched manifest
        enriched_manifest = work_dir / "enriched" / "enriched_manifest.json"
        if not enriched_manifest.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=f"Enriched manifest not found at {enriched_manifest}. Run Stage 2 first.",
            )

        # Build DreamDojo-compatible dataset
        logger.info("Building DreamDojo dataset from enriched videos")
        dataset_dir = build_dreamdojo_dataset(
            enriched_manifest_path=enriched_manifest,
            output_dir=finetune_dir,
            facility_name=facility.name,
        )

        # Run fine-tuning
        logger.info(
            "Starting DreamDojo fine-tuning (lr=%s, epochs=%d)",
            config.finetune.learning_rate,
            config.finetune.num_epochs,
        )

        # Derive facility_id from work_dir
        facility_id = work_dir.name

        train_result = run_dreamdojo_finetune(
            dataset_dir=dataset_dir,
            output_dir=finetune_dir,
            config=config.finetune,
            facility_id=facility_id,
        )

        status = train_result.get("status", "failed")
        adapted_checkpoint = (
            train_result.get("adapted_checkpoint_path")
            or train_result.get("checkpoint_dir")
            or train_result.get("lora_weights_path")
            or ""
        )
        lora_path = train_result.get("lora_weights_path", adapted_checkpoint)

        return StageResult(
            stage_name=self.name,
            status=status,
            elapsed_seconds=0,
            outputs={
                "finetune_dir": str(finetune_dir),
                "checkpoint_dir": adapted_checkpoint,
                "adapted_checkpoint_path": adapted_checkpoint,
                "lora_weights_path": lora_path,
                "dataset_dir": str(dataset_dir),
                "train_log": str(finetune_dir / "finetune_log.json"),
            },
            metrics={
                "num_epochs": config.finetune.num_epochs,
                "learning_rate": config.finetune.learning_rate,
                "training_seconds": train_result.get("elapsed_seconds", 0),
                "final_loss": (
                    train_result.get("loss_history", [{}])[-1].get("loss")
                    if train_result.get("loss_history")
                    else None
                ),
            },
        )
