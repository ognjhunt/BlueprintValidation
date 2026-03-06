"""Stage 4c: Train paired policies from same init on baseline vs site datasets."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Dict

from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..evaluation.claim_protocol import claim_protocol_enabled
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
        dataset_lineage = dict(summary_payload.get("dataset_lineage", {}) or {})
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
        if claim_protocol_enabled(config):
            lineage_error = _claim_dataset_lineage_error(
                summary_payload=summary_payload,
                work_dir=work_dir,
            )
            if lineage_error is not None:
                return StageResult(
                    stage_name=self.name,
                    status="failed",
                    elapsed_seconds=0,
                    detail=lineage_error,
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
        if claim_protocol_enabled(config):
            base_runs = []
            site_runs = []
            for seed in [int(v) for v in list(config.eval_policy.claim_replication.training_seeds)]:
                seed_cfg = replace(config.policy_finetune, seed=int(seed))
                base_result = adapter.train_policy(
                    base_model_name=common_model,
                    base_checkpoint=common_checkpoint,
                    dataset_root=base_dataset_root.parent,
                    dataset_name=config.rollout_dataset.baseline_dataset_name,
                    output_dir=base_root / f"seed_{seed:02d}",
                    finetune_config=seed_cfg,
                )
                site_result = adapter.train_policy(
                    base_model_name=common_model,
                    base_checkpoint=common_checkpoint,
                    dataset_root=site_dataset_root.parent,
                    dataset_name=config.rollout_dataset.adapted_dataset_name,
                    output_dir=site_root / f"seed_{seed:02d}",
                    finetune_config=seed_cfg,
                )
                base_runs.append(
                    {
                        "seed": int(seed),
                        "status": base_result.status,
                        "adapted_checkpoint_path": str(base_result.adapted_checkpoint_path or ""),
                        "elapsed_seconds": base_result.elapsed_seconds,
                        "detail": base_result.detail,
                    }
                )
                site_runs.append(
                    {
                        "seed": int(seed),
                        "status": site_result.status,
                        "adapted_checkpoint_path": str(site_result.adapted_checkpoint_path or ""),
                        "elapsed_seconds": site_result.elapsed_seconds,
                        "detail": site_result.detail,
                    }
                )
            primary_base = next((run for run in base_runs if run["status"] == "success"), base_runs[0])
            primary_site = next((run for run in site_runs if run["status"] == "success"), site_runs[0])
            status = (
                "success"
                if all(run["status"] == "success" for run in base_runs + site_runs)
                else "failed"
            )
            summary = {
                "claim_protocol": "fixed_same_facility_uplift",
                "policy_base": primary_base,
                "policy_site": primary_site,
                "dataset_lineage": dataset_lineage,
                "replicates": {
                    "generic_control": base_runs,
                    "site_trained": site_runs,
                },
            }
            metrics = {
                "policy_base_status": primary_base["status"],
                "policy_site_status": primary_site["status"],
                "policy_base_seconds": round(float(primary_base["elapsed_seconds"]), 2),
                "policy_site_seconds": round(float(primary_site["elapsed_seconds"]), 2),
                "num_training_seeds": len(base_runs),
                "num_successful_generic_control": sum(
                    1 for run in base_runs if run["status"] == "success"
                ),
                "num_successful_site_trained": sum(
                    1 for run in site_runs if run["status"] == "success"
                ),
            }
            detail = (
                f"generic_control={metrics['num_successful_generic_control']}/{len(base_runs)} "
                f"site_trained={metrics['num_successful_site_trained']}/{len(site_runs)}"
            )
        else:
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
                "dataset_lineage": dataset_lineage,
            }
            metrics = {
                "policy_base_status": base_result.status,
                "policy_site_status": site_result.status,
                "policy_base_seconds": round(base_result.elapsed_seconds, 2),
                "policy_site_seconds": round(site_result.elapsed_seconds, 2),
            }
            detail = f"base={base_result.detail} site={site_result.detail}"
        write_json(summary, train_root / "policy_pair_train_summary.json")

        return StageResult(
            stage_name=self.name,
            status=status,
            elapsed_seconds=0,
            outputs={
                "train_root": str(train_root),
                "policy_base_checkpoint": str(summary["policy_base"].get("adapted_checkpoint_path", "")),
                "policy_site_checkpoint": str(summary["policy_site"].get("adapted_checkpoint_path", "")),
                "summary_path": str(train_root / "policy_pair_train_summary.json"),
            },
            metrics=metrics,
            detail=detail,
        )


def _claim_dataset_lineage_error(*, summary_payload: dict, work_dir: Path) -> str | None:
    dataset_lineage = dict(summary_payload.get("dataset_lineage", {}) or {})
    if not dataset_lineage:
        return "Claim protocol dataset export is missing dataset_lineage metadata."

    claim_manifest_path = work_dir / "policy_eval" / "claim_manifest.json"
    claim_split_path = work_dir / "policy_eval" / "claim_split_manifest.json"
    if not claim_manifest_path.exists() or not claim_split_path.exists():
        return "Current claim manifest or split manifest is missing for lineage validation."

    claim_manifest = read_json(claim_manifest_path)
    expected_world_hash = str(claim_manifest.get("world_snapshot_hash", "") or "").strip()
    if expected_world_hash != str(dataset_lineage.get("world_snapshot_hash", "") or "").strip():
        return "Dataset lineage world snapshot hash does not match the current claim manifest."
    if _json_manifest_hash(claim_manifest_path) != str(
        dataset_lineage.get("claim_manifest_hash", "") or ""
    ).strip():
        return "Dataset lineage claim manifest hash does not match the current claim manifest."
    if _json_manifest_hash(claim_split_path) != str(
        dataset_lineage.get("claim_split_manifest_hash", "") or ""
    ).strip():
        return "Dataset lineage split manifest hash does not match the current claim split."
    return None


def _json_manifest_hash(path: Path) -> str:
    try:
        payload = json.dumps(read_json(path), sort_keys=True, separators=(",", ":"))
    except Exception:
        payload = str(path)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()
