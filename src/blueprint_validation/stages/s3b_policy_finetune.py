"""Optional OpenVLA-OFT policy fine-tuning stage."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..common import StageResult, get_logger
from ..config import FacilityConfig, PolicyFinetuneConfig, ValidationConfig
from ..training.openvla_finetune import run_openvla_finetune
from .base import PipelineStage

logger = get_logger("stages.s3b_policy_finetune")


class PolicyFinetuneStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s3b_policy_finetune"

    @property
    def description(self) -> str:
        return "Optional OpenVLA-OFT fine-tuning on manipulation trajectories"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        del facility

        if not config.policy_finetune.enabled:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="policy_finetune.enabled=false",
            )

        stage_dir = work_dir / "policy_finetune"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Auto-wire to S4a RLDS export output if available
        finetune_config = config.policy_finetune
        s4a = previous_results.get("s4a_rlds_export")
        if s4a and s4a.status == "success":
            rlds_dir = s4a.outputs.get("rlds_dataset_dir")
            rlds_name = s4a.outputs.get("dataset_name")
            if rlds_dir and rlds_name:
                logger.info(
                    "Using pipeline-generated RLDS dataset: %s from %s",
                    rlds_name, rlds_dir,
                )
                # Point data_root_dir to the parent so OpenVLA-OFT finds dataset_name/ inside
                from pathlib import Path as _Path

                rlds_dataset_path = _Path(rlds_dir)
                finetune_config = PolicyFinetuneConfig(
                    enabled=finetune_config.enabled,
                    openvla_repo=finetune_config.openvla_repo,
                    finetune_script=finetune_config.finetune_script,
                    data_root_dir=rlds_dataset_path.parent,
                    dataset_name=rlds_name,
                    run_root_dir=finetune_config.run_root_dir,
                    adapter_tmp_dir=finetune_config.adapter_tmp_dir,
                    lora_rank=finetune_config.lora_rank,
                    batch_size=finetune_config.batch_size,
                    grad_accumulation_steps=finetune_config.grad_accumulation_steps,
                    learning_rate=finetune_config.learning_rate,
                    save_steps=finetune_config.save_steps,
                    max_steps=finetune_config.max_steps,
                    image_aug=finetune_config.image_aug,
                    nproc_per_node=finetune_config.nproc_per_node,
                    wandb_project=finetune_config.wandb_project,
                    wandb_entity=finetune_config.wandb_entity,
                )
        elif not finetune_config.data_root_dir:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail=(
                    "No RLDS dataset from S4a and no data_root_dir configured. "
                    "Run Stage 4 + S4a first, or set policy_finetune.data_root_dir."
                ),
            )

        checkpoint_path = config.eval_policy.openvla_checkpoint
        vla_path = (
            str(checkpoint_path)
            if checkpoint_path and checkpoint_path.exists()
            else config.eval_policy.openvla_model
        )

        train_result = run_openvla_finetune(
            config=finetune_config,
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
