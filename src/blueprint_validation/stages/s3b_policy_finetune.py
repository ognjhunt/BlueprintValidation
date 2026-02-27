"""Optional OpenVLA policy fine-tuning stage."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..common import StageResult, get_logger
from ..config import FacilityConfig, ValidationConfig
from ..training.openvla_finetune import run_openvla_finetune
from .base import PipelineStage

logger = get_logger("stages.s3b_policy_finetune")


class PolicyFinetuneStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s3b_policy_finetune"

    @property
    def description(self) -> str:
        return "Optional OpenVLA LoRA/OFT fine-tuning on manipulation trajectories"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        del facility, previous_results

        if not config.policy_finetune.enabled:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="policy_finetune.enabled=false",
            )

        stage_dir = work_dir / "policy_finetune"
        stage_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = config.eval_policy.openvla_checkpoint
        vla_path = (
            str(checkpoint_path)
            if checkpoint_path and checkpoint_path.exists()
            else config.eval_policy.openvla_model
        )

        train_result = run_openvla_finetune(
            config=config.policy_finetune,
            vla_path=vla_path,
            facility_id=work_dir.name,
            output_dir=stage_dir,
        )

        return StageResult(
            stage_name=self.name,
            status=train_result.get("status", "failed"),
            elapsed_seconds=0,
            outputs={
                "policy_finetune_dir": str(stage_dir),
                "adapted_openvla_checkpoint": train_result.get("adapted_checkpoint_path", ""),
                "train_log": str(stage_dir / "policy_finetune_log.json"),
            },
            metrics={
                "dataset_name": train_result.get("dataset_name"),
                "elapsed_seconds": train_result.get("elapsed_seconds"),
                "returncode": train_result.get("returncode"),
            },
            detail=train_result.get("stderr", ""),
        )
