"""Optional policy fine-tuning stage via selected adapter."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict

from ..common import StageResult, get_logger
from ..config import FacilityConfig, ValidationConfig
from ..policy_adapters import get_policy_adapter
from .base import PipelineStage

logger = get_logger("stages.s3b_policy_finetune")


class PolicyFinetuneStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s3b_policy_finetune"

    @property
    def description(self) -> str:
        return "Optional adapter-based fine-tuning on manipulation trajectories"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        del facility
        if (
            (getattr(config.eval_policy, "headline_scope", "wm_only") or "wm_only")
            .strip()
            .lower()
            == "wm_only"
        ):
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail=(
                    "Skipped by policy: eval_policy.headline_scope=wm_only "
                    "(OpenVLA stages deferred)."
                ),
            )

        if not config.policy_finetune.enabled:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="policy_finetune.enabled=false",
            )

        stage_dir = work_dir / "policy_finetune"
        stage_dir.mkdir(parents=True, exist_ok=True)

        adapter = get_policy_adapter(config.policy_adapter)

        # Auto-wire to S4a RLDS export output if available.
        finetune_config = config.policy_finetune
        dataset_name = finetune_config.dataset_name
        source_dataset_dir: Path | None = None
        s4a = previous_results.get("s4a_rlds_export")
        if s4a and s4a.status == "success":
            train_jsonl = s4a.outputs.get("train_jsonl")
            rlds_name = s4a.outputs.get("dataset_name")
            if train_jsonl and rlds_name:
                logger.info(
                    "Using pipeline-generated rollout dataset: %s from %s",
                    rlds_name,
                    train_jsonl,
                )
                source_dataset_dir = Path(train_jsonl).parent
                dataset_name = str(rlds_name)
                finetune_config = replace(
                    finetune_config,
                    dataset_name=dataset_name,
                )
        elif finetune_config.data_root_dir is not None:
            source_dataset_dir = finetune_config.data_root_dir / dataset_name
        else:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail=(
                    "No rollout dataset from S4a and no data_root_dir configured. "
                    "Run Stage 4 + S4a first, or set policy_finetune.data_root_dir."
                ),
            )

        if source_dataset_dir is None:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Could not resolve source dataset directory for policy fine-tuning.",
            )

        adapter_dataset = adapter.dataset_transform(
            source_dataset_dir=source_dataset_dir,
            output_root=stage_dir / "dataset",
            dataset_name=dataset_name,
        )
        base_model_name, base_checkpoint = adapter.base_model_ref(config.eval_policy)

        train_result = adapter.train_policy(
            base_model_name=base_model_name,
            base_checkpoint=base_checkpoint,
            dataset_root=adapter_dataset.parent,
            dataset_name=adapter_dataset.name,
            output_dir=stage_dir / "train",
            finetune_config=finetune_config,
        )
        adapted_path = str(train_result.adapted_checkpoint_path or "")

        return StageResult(
            stage_name=self.name,
            status=train_result.status,
            elapsed_seconds=0,
            outputs={
                "policy_finetune_dir": str(stage_dir),
                "adapter_name": adapter.name,
                "adapted_policy_checkpoint": adapted_path,
                "adapted_openvla_checkpoint": adapted_path,  # legacy compatibility
                "train_log": str(stage_dir / "train" / "policy_finetune_log.json"),
            },
            metrics={
                "dataset_name": dataset_name,
                "elapsed_seconds": train_result.elapsed_seconds,
                "returncode": train_result.raw.get("returncode"),
            },
            detail=train_result.detail,
        )
