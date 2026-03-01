"""Stage 4a: Export successful rollouts to RLDS TFRecord format for policy training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..training.rlds_export import (
    convert_jsonl_to_tfrecord,
    export_rollouts_to_rlds_jsonl,
)
from ..training.rollout_curriculum import sample_policy_curriculum
from .base import PipelineStage

logger = get_logger("stages.s4a_rlds_export")


class RLDSExportStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s4a_rlds_export"

    @property
    def description(self) -> str:
        return "Export mixed adapted rollouts (success/near-miss/hard-negative) to RLDS"

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

        if not config.rollout_dataset.enabled:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="rollout_dataset.enabled=false",
            )

        if not config.policy_finetune.enabled:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="policy_finetune.enabled=false; no need to export RLDS",
            )

        # Read Stage 4 scores
        prev_s4 = previous_results.get("s4_policy_eval")
        if not prev_s4 or prev_s4.status != "success":
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Stage 4 (policy_eval) did not succeed. Cannot export rollouts.",
            )

        scores_path = prev_s4.outputs.get("scores_path")
        if not scores_path or not Path(scores_path).exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=f"Scores file not found: {scores_path}",
            )

        scores_data = read_json(Path(scores_path))
        all_rollouts: List[Dict] = scores_data.get("scores", [])
        adapted_rollouts_all = [r for r in all_rollouts if r.get("condition") == "adapted"]
        if not adapted_rollouts_all:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="No adapted-condition rollouts found in Stage 4 output.",
            )

        curriculum_result = sample_policy_curriculum(
            adapted_rollouts_all,
            config.rollout_dataset,
            seed=int(config.rollout_dataset.seed),
        )
        filtered_rollouts = list(curriculum_result.get("adapted_rollouts", []))
        filter_counts = dict(curriculum_result.get("filter_counts", {}))
        # Backward-compat alias for existing reports/tests.
        filter_counts["num_adapted_after_filters"] = int(
            filter_counts.get("num_after_action_filters", len(filtered_rollouts))
        )
        train_rollouts = list(curriculum_result.get("train_rollouts", []))
        eval_rollouts = list(curriculum_result.get("eval_rollouts", []))
        curriculum_manifest = dict(curriculum_result.get("curriculum", {}))
        if not filtered_rollouts:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=(
                    "No adapted-condition rollouts passed action-quality filters. "
                    f"total_adapted={len(adapted_rollouts_all)} "
                    f"short={filter_counts.get('num_filtered_short', 0)} "
                    f"dim_mismatch={filter_counts.get('num_filtered_dim_mismatch', 0)} "
                    f"nonfinite={filter_counts.get('num_filtered_nonfinite', 0)} "
                    f"smoothness={filter_counts.get('num_filtered_smoothness', 0)}"
                ),
                metrics={
                    "total_adapted_rollouts": len(adapted_rollouts_all),
                    **filter_counts,
                },
            )

        stage_dir = work_dir / "rlds_export"
        stage_dir.mkdir(parents=True, exist_ok=True)

        dataset_name = config.rollout_dataset.adapted_dataset_name
        threshold = config.rollout_dataset.task_score_threshold
        min_steps = config.rollout_dataset.min_steps_per_rollout
        include_failed = bool(config.rollout_dataset.include_failed_rollouts)
        include_failed = include_failed or str(config.rollout_dataset.selection_mode) != "success_only"

        split_manifest_path = stage_dir / "split_manifest.json"
        split_manifest = {
            "train_pair_ids": list(curriculum_result.get("train_pair_ids", [])),
            "eval_pair_ids": list(curriculum_result.get("eval_pair_ids", [])),
        }
        write_json(split_manifest, split_manifest_path)
        curriculum_manifest_path = stage_dir / "curriculum_manifest.json"
        write_json(curriculum_manifest, curriculum_manifest_path)

        strict_disjoint_eval = bool(getattr(config.action_boost, "enabled", False)) and bool(
            getattr(config.action_boost, "strict_disjoint_eval", True)
        )
        if strict_disjoint_eval and not split_manifest["eval_pair_ids"]:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=(
                    "Strict disjoint eval is enabled but Stage 4a produced empty eval_pair_ids. "
                    "Increase rollout volume or adjust curriculum split/selection settings."
                ),
                outputs={
                    "split_manifest_path": str(split_manifest_path),
                    "curriculum_manifest_path": str(curriculum_manifest_path),
                },
                metrics={
                    "total_adapted_rollouts": len(adapted_rollouts_all),
                    "requested_rollouts_per_condition": int(config.eval_policy.num_rollouts),
                    **filter_counts,
                },
            )

        # Export train split to JSONL
        train_dir = stage_dir / "train"
        train_meta = export_rollouts_to_rlds_jsonl(
            rollouts=train_rollouts,
            output_dir=train_dir,
            condition="adapted",
            split="train",
            task_threshold=threshold,
            min_steps_per_rollout=min_steps,
            include_failed_rollouts=include_failed,
        )

        # Export eval split to JSONL
        eval_dir = stage_dir / "eval"
        eval_meta = export_rollouts_to_rlds_jsonl(
            rollouts=eval_rollouts,
            output_dir=eval_dir,
            condition="adapted",
            split="eval",
            task_threshold=threshold,
            min_steps_per_rollout=min_steps,
            include_failed_rollouts=include_failed,
        )

        num_train = train_meta.get("num_episodes", 0)
        num_eval = eval_meta.get("num_episodes", 0)

        if num_train == 0:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=(
                    f"No episodes passed filters (threshold={threshold}, "
                    f"min_steps={min_steps}). "
                    "Total adapted rollouts after action-quality gating: "
                    f"{len(filtered_rollouts)}"
                ),
                metrics={
                    "total_adapted_rollouts": len(adapted_rollouts_all),
                    **filter_counts,
                },
            )

        # Convert JSONL to TFRecords for OpenVLA-OFT
        dataset_dir = convert_jsonl_to_tfrecord(
            train_jsonl_path=train_dir / "episodes.jsonl",
            eval_jsonl_path=eval_dir / "episodes.jsonl" if num_eval > 0 else None,
            output_dir=config.rollout_dataset.export_dir,
            dataset_name=dataset_name,
        )

        logger.info(
            "Exported %d train + %d eval episodes to %s",
            num_train,
            num_eval,
            dataset_dir,
        )

        return StageResult(
            stage_name=self.name,
            status="success",
            elapsed_seconds=0,
            outputs={
                "rlds_dataset_dir": str(dataset_dir),
                "dataset_name": dataset_name,
                "train_jsonl": str(train_dir / "episodes.jsonl"),
                "eval_jsonl": str(eval_dir / "episodes.jsonl"),
                "split_manifest_path": str(split_manifest_path),
                "curriculum_manifest_path": str(curriculum_manifest_path),
            },
            metrics={
                "num_train_episodes": num_train,
                "num_eval_episodes": num_eval,
                "num_train_successes": train_meta.get("num_successes", 0),
                "task_score_threshold": threshold,
                "total_adapted_rollouts": len(adapted_rollouts_all),
                "num_success_candidates": int(
                    curriculum_manifest.get("candidate_counts", {}).get("success", 0)
                ),
                "num_near_miss_candidates": int(
                    curriculum_manifest.get("candidate_counts", {}).get("near_miss", 0)
                ),
                "num_hard_negative_candidates": int(
                    curriculum_manifest.get("candidate_counts", {}).get("hard_negative", 0)
                ),
                "num_train_near_miss": int(
                    curriculum_manifest.get("train_bucket_counts", {}).get("near_miss", 0)
                ),
                "num_train_hard_negative": int(
                    curriculum_manifest.get("train_bucket_counts", {}).get("hard_negative", 0)
                ),
                "max_action_delta_norm": config.rollout_dataset.max_action_delta_norm,
                "require_consistent_action_dim": (
                    config.rollout_dataset.require_consistent_action_dim
                ),
                **filter_counts,
            },
        )
