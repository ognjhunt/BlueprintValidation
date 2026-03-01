"""Stage 4c: Train paired policies from same init on baseline vs site datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..policy_adapters import get_policy_adapter
from .base import PipelineStage

logger = get_logger("stages.s4c_policy_pair_train")


class PolicyPairTrainStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s4c_policy_pair_train"

    @property
    def description(self) -> str:
        return "Train policy_base and policy_site from same initialization and budget"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        del facility, previous_results
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

        dataset_root = config.rollout_dataset.export_dir / work_dir.name
        summary_path = dataset_root / "dataset_export_summary.json"
        if not summary_path.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Dataset export summary missing. Run Stage 4b first.",
            )
        summary_payload = read_json(summary_path)
        if bool(summary_payload.get("claim_mode", False)):
            action_contract = summary_payload.get("action_contract", {})
            reliability_gate = summary_payload.get("reliability_gate", {})
            if not bool(action_contract.get("compliant", False)):
                return StageResult(
                    stage_name=self.name,
                    status="failed",
                    elapsed_seconds=0,
                    detail=f"Claim mode action contract failed: {action_contract}",
                )
            if not bool(reliability_gate.get("passed", False)):
                return StageResult(
                    stage_name=self.name,
                    status="failed",
                    elapsed_seconds=0,
                    detail=f"Claim mode reliability gate failed: {reliability_gate}",
                )
        adapter = get_policy_adapter(config.policy_adapter)

        train_root = work_dir / "policy_pair_train"
        base_root = train_root / "policy_base"
        site_root = train_root / "policy_site"
        base_root.mkdir(parents=True, exist_ok=True)
        site_root.mkdir(parents=True, exist_ok=True)

        baseline_train_dir = dataset_root / "baseline" / "train"
        adapted_train_dir = dataset_root / "adapted" / "train"
        base_dataset_root = adapter.dataset_transform(
            source_dataset_dir=baseline_train_dir,
            output_root=base_root / "dataset",
            dataset_name=config.rollout_dataset.baseline_dataset_name,
        )
        site_dataset_root = adapter.dataset_transform(
            source_dataset_dir=adapted_train_dir,
            output_root=site_root / "dataset",
            dataset_name=config.rollout_dataset.adapted_dataset_name,
        )

        common_model, common_checkpoint = adapter.base_model_ref(config.eval_policy)
        base_result = adapter.train_policy(
            base_model_name=common_model,
            base_checkpoint=common_checkpoint,
            dataset_root=base_dataset_root.parent,
            dataset_name=config.rollout_dataset.baseline_dataset_name,
            output_dir=base_root / "train",
            finetune_config=config.policy_finetune,
        )
        site_result = adapter.train_policy(
            base_model_name=common_model,
            base_checkpoint=common_checkpoint,
            dataset_root=site_dataset_root.parent,
            dataset_name=config.rollout_dataset.adapted_dataset_name,
            output_dir=site_root / "train",
            finetune_config=config.policy_finetune,
        )

        status = (
            "success"
            if base_result.status == "success" and site_result.status == "success"
            else "failed"
        )
        summary = {
            "policy_base": {
                "status": base_result.status,
                "adapted_checkpoint_path": str(base_result.adapted_checkpoint_path or ""),
                "elapsed_seconds": base_result.elapsed_seconds,
                "detail": base_result.detail,
            },
            "policy_site": {
                "status": site_result.status,
                "adapted_checkpoint_path": str(site_result.adapted_checkpoint_path or ""),
                "elapsed_seconds": site_result.elapsed_seconds,
                "detail": site_result.detail,
            },
        }
        write_json(summary, train_root / "policy_pair_train_summary.json")

        return StageResult(
            stage_name=self.name,
            status=status,
            elapsed_seconds=0,
            outputs={
                "train_root": str(train_root),
                "policy_base_checkpoint": str(base_result.adapted_checkpoint_path or ""),
                "policy_site_checkpoint": str(site_result.adapted_checkpoint_path or ""),
                "summary_path": str(train_root / "policy_pair_train_summary.json"),
            },
            metrics={
                "policy_base_status": base_result.status,
                "policy_site_status": site_result.status,
                "policy_base_seconds": round(base_result.elapsed_seconds, 2),
                "policy_site_seconds": round(site_result.elapsed_seconds, 2),
            },
            detail=f"base={base_result.detail} site={site_result.detail}",
        )
