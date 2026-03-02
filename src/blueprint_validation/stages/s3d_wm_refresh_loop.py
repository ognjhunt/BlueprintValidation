"""Stage 3d: WM-only world-model refresh loop from scripted rollout outcomes."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import shutil
from typing import Dict, List, Optional

from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..evaluation.reasoning_conflicts import has_reasoning_conflict
from ..training.policy_rl_loop import refresh_world_model_from_bucketed_rollouts
from ..training.rollout_curriculum import (
    RolloutBucketThresholds,
    bucket_rollout,
    bucket_rollouts_by_quantile,
)
from ..video_io import decode_video_frame_count
from .base import PipelineStage

logger = get_logger("stages.s3d_wm_refresh_loop")


class WorldModelRefreshLoopStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s3d_wm_refresh_loop"

    @property
    def description(self) -> str:
        return "WM-only near-miss/success curriculum refresh for DreamDojo checkpoints"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        cfg = config.wm_refresh_loop
        if not cfg.enabled:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="wm_refresh_loop.enabled=false",
            )

        scope = (getattr(config.eval_policy, "headline_scope", "wm_only") or "wm_only").strip().lower()
        if scope != "wm_only":
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="wm_refresh_loop is only supported when eval_policy.headline_scope=wm_only",
            )

        prev_s4 = previous_results.get("s4_policy_eval")
        if not prev_s4 or prev_s4.status != "success":
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Stage 4 (wm-only eval) must succeed before wm_refresh_loop.",
            )

        scores_path = prev_s4.outputs.get("scores_path")
        if not scores_path or not Path(scores_path).exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=f"Stage 4 scores_path missing: {scores_path}",
            )

        scores_payload = read_json(Path(scores_path))
        all_scores = list(scores_payload.get("scores", []))
        source_condition = str(cfg.source_condition).strip().lower()
        source_scores = [
            row for row in all_scores if str(row.get("condition", "")).strip().lower() == source_condition
        ]
        source_scores = [
            row
            for row in source_scores
            if str(row.get("video_path", "")).strip() and Path(str(row.get("video_path"))).exists()
        ]
        num_vlm_floor_filtered_rollouts = 0
        num_vlm_floor_reasoning_conflicts = 0
        if bool(cfg.enforce_vlm_quality_floor):
            filtered_scores: List[Dict] = []
            for row in source_scores:
                task_score = float(row.get("task_score", 0.0) or 0.0)
                visual_score = float(row.get("visual_score", 0.0) or 0.0)
                spatial_score = float(row.get("spatial_score", 0.0) or 0.0)
                reasoning_conflict = has_reasoning_conflict(str(row.get("reasoning", "")))
                if reasoning_conflict:
                    num_vlm_floor_reasoning_conflicts += 1
                passes_floor = (
                    task_score >= float(cfg.min_refresh_task_score)
                    and visual_score >= float(cfg.min_refresh_visual_score)
                    and spatial_score >= float(cfg.min_refresh_spatial_score)
                    and (not bool(cfg.fail_on_reasoning_conflict) or not reasoning_conflict)
                )
                if passes_floor:
                    kept = dict(row)
                    kept["reasoning_conflict"] = bool(reasoning_conflict)
                    filtered_scores.append(kept)
                else:
                    num_vlm_floor_filtered_rollouts += 1
            source_scores = filtered_scores
        expected_decode_frames = max(2, int(config.eval_policy.reliability.min_rollout_frames))
        invalid_decode_rows: List[Dict] = []
        if bool(cfg.require_valid_video_decode):
            valid_rows: List[Dict] = []
            for row in source_scores:
                video_path = Path(str(row.get("video_path", "")).strip())
                decoded = decode_video_frame_count(video_path)
                if decoded < expected_decode_frames:
                    invalid_entry = dict(row)
                    invalid_entry["decoded_frames"] = int(decoded)
                    invalid_entry["required_frames"] = int(expected_decode_frames)
                    invalid_decode_rows.append(invalid_entry)
                    logger.warning(
                        "Decoded %d frames, expected %d for rollout video: %s",
                        decoded,
                        expected_decode_frames,
                        video_path,
                    )
                    continue
                valid_entry = dict(row)
                valid_entry["decoded_frames"] = int(decoded)
                valid_rows.append(valid_entry)
            source_scores = valid_rows
        if not source_scores and not bool(cfg.backfill_from_stage2_vlm_passed):
            decode_detail = ""
            if invalid_decode_rows:
                sample = invalid_decode_rows[0]
                decode_detail = (
                    " "
                    f"Decoded {int(sample.get('decoded_frames', 0))} frames, expected "
                    f"{int(sample.get('required_frames', expected_decode_frames))}."
                )
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=(
                    f"No rollout rows available for source_condition='{source_condition}' "
                    "with existing video paths."
                    f"{decode_detail}"
                ),
                metrics={
                    "source_condition": source_condition,
                    "num_source_rollouts": 0,
                    "num_vlm_floor_filtered_rollouts": num_vlm_floor_filtered_rollouts,
                    "num_vlm_floor_reasoning_conflicts": num_vlm_floor_reasoning_conflicts,
                    "stage2_backfill_considered": False,
                    "num_stage2_backfill_candidates": 0,
                    "effective_non_hard_count": 0,
                    "num_invalid_decode_rollouts": len(invalid_decode_rows),
                    "expected_decode_frames": int(expected_decode_frames),
                    "require_valid_video_decode": bool(cfg.require_valid_video_decode),
                    "degenerate_mix_detected": False,
                },
            )
        if not source_scores and bool(cfg.backfill_from_stage2_vlm_passed):
            logger.warning(
                "No rollout rows remained after pre-refresh filters; continuing with Stage-2 "
                "VLM-passed backfill viability checks."
            )

        thresholds = RolloutBucketThresholds(
            task_score_threshold=float(config.rollout_dataset.task_score_threshold),
            near_miss_min_task_score=float(config.rollout_dataset.near_miss_min_task_score),
            near_miss_max_task_score=float(config.rollout_dataset.near_miss_max_task_score),
        )
        selected_success: List[Dict] = []
        near_miss: List[Dict] = []
        hard_negative: List[Dict] = []
        for row in source_scores:
            bucket = bucket_rollout(row, thresholds)
            entry = dict(row)
            entry["rollout_bucket"] = bucket
            if bucket == "success":
                selected_success.append(entry)
            elif bucket == "near_miss":
                near_miss.append(entry)
            else:
                hard_negative.append(entry)
        threshold_bucket_counts = {
            "success": len(selected_success),
            "near_miss": len(near_miss),
            "hard_negative": len(hard_negative),
        }
        bucketing_strategy = "threshold"
        fallback_quantiles = {
            "success_threshold": None,
            "near_miss_threshold": None,
        }

        if (
            bool(cfg.quantile_fallback_enabled)
            and (len(selected_success) + len(near_miss) == 0)
            and len(source_scores) > 0
        ):
            fallback = bucket_rollouts_by_quantile(
                source_scores,
                success_quantile=float(cfg.quantile_success_threshold),
                near_miss_quantile=float(cfg.quantile_near_miss_threshold),
            )
            selected_success = list(fallback.get("success", []))
            near_miss = list(fallback.get("near_miss", []))
            hard_negative = list(fallback.get("hard_negative", []))
            bucketing_strategy = "quantile_fallback"
            fallback_quantiles = {
                "success_threshold": fallback.get("success_threshold"),
                "near_miss_threshold": fallback.get("near_miss_threshold"),
            }
        post_bucket_counts = {
            "success": len(selected_success),
            "near_miss": len(near_miss),
            "hard_negative": len(hard_negative),
        }
        degenerate_mix_detected = False
        max_hard_negative_fraction = float(cfg.max_hard_negative_fraction)
        total_bucketed = len(selected_success) + len(near_miss) + len(hard_negative)
        observed_hard_negative_fraction = (
            float(len(hard_negative)) / float(total_bucketed) if total_bucketed > 0 else 0.0
        )
        if (
            total_bucketed > 0
            and observed_hard_negative_fraction > max_hard_negative_fraction
        ):
            degenerate_mix_detected = True
            if max_hard_negative_fraction < 1.0:
                positive_count = len(selected_success) + len(near_miss)
                denom = max(1e-9, 1.0 - max_hard_negative_fraction)
                allowed_hard = int(
                    math.floor((max_hard_negative_fraction * float(positive_count)) / denom)
                )
                allowed_hard = max(0, allowed_hard)
                if len(hard_negative) > allowed_hard:
                    hard_negative = list(hard_negative[:allowed_hard])
            total_bucketed = len(selected_success) + len(near_miss) + len(hard_negative)
            observed_hard_negative_fraction = (
                float(len(hard_negative)) / float(total_bucketed) if total_bucketed > 0 else 0.0
            )
            post_bucket_counts = {
                "success": len(selected_success),
                "near_miss": len(near_miss),
                "hard_negative": len(hard_negative),
            }

        num_positive_rollouts = len(selected_success) + len(near_miss)
        stage2_backfill_considered = False
        num_stage2_backfill_candidates = 0
        effective_non_hard_count = num_positive_rollouts
        if (
            bool(cfg.backfill_from_stage2_vlm_passed)
            and effective_non_hard_count < int(cfg.min_non_hard_rollouts)
        ):
            stage2_backfill_considered = True
            stage2_manifest_path = work_dir / "enriched" / "enriched_manifest.json"
            if stage2_manifest_path.exists():
                try:
                    stage2_manifest = read_json(stage2_manifest_path)
                    for clip in stage2_manifest.get("clips", []):
                        output_video = Path(str(clip.get("output_video_path", "")).strip())
                        if not output_video.exists():
                            continue
                        if clip.get("vlm_quality_passed") is True:
                            num_stage2_backfill_candidates += 1
                except Exception:
                    logger.warning(
                        "Failed loading Stage-2 enriched manifest for WM refresh backfill: %s",
                        stage2_manifest_path,
                        exc_info=True,
                    )
            effective_non_hard_count = num_positive_rollouts + num_stage2_backfill_candidates

        if not (selected_success or near_miss or hard_negative) and effective_non_hard_count <= 0:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="No rollouts available after bucketing into success/near_miss/hard_negative.",
                metrics={
                    "source_condition": source_condition,
                    "num_source_rollouts": len(source_scores),
                    "num_success_rollouts": len(selected_success),
                    "num_near_miss_rollouts": len(near_miss),
                    "num_hard_negative_rollouts": len(hard_negative),
                    "bucketing_strategy": bucketing_strategy,
                    "threshold_bucket_counts": threshold_bucket_counts,
                    "post_bucket_counts": post_bucket_counts,
                    "fallback_quantiles": fallback_quantiles,
                    "min_non_hard_rollouts": int(cfg.min_non_hard_rollouts),
                    "max_hard_negative_fraction": max_hard_negative_fraction,
                    "observed_hard_negative_fraction": observed_hard_negative_fraction,
                    "degenerate_mix_detected": degenerate_mix_detected,
                    "num_vlm_floor_filtered_rollouts": num_vlm_floor_filtered_rollouts,
                    "num_vlm_floor_reasoning_conflicts": num_vlm_floor_reasoning_conflicts,
                    "stage2_backfill_considered": stage2_backfill_considered,
                    "num_stage2_backfill_candidates": num_stage2_backfill_candidates,
                    "effective_non_hard_count": effective_non_hard_count,
                    "num_invalid_decode_rollouts": len(invalid_decode_rows),
                    "expected_decode_frames": int(expected_decode_frames),
                    "require_valid_video_decode": bool(cfg.require_valid_video_decode),
                },
            )
        if bool(cfg.fail_on_degenerate_mix) and effective_non_hard_count < int(cfg.min_non_hard_rollouts):
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=(
                    "WM refresh blocked: degenerate rollout mix "
                    f"(effective_non_hard_count={effective_non_hard_count} < min_non_hard_rollouts="
                    f"{int(cfg.min_non_hard_rollouts)}). "
                    f"source_condition={source_condition}, bucketing_strategy={bucketing_strategy}, "
                    f"hard_negative={len(hard_negative)}, "
                    f"stage2_backfill_candidates={num_stage2_backfill_candidates}."
                ),
                metrics={
                    "source_condition": source_condition,
                    "num_source_rollouts": len(source_scores),
                    "num_success_rollouts": len(selected_success),
                    "num_near_miss_rollouts": len(near_miss),
                    "num_hard_negative_rollouts": len(hard_negative),
                    "bucketing_strategy": bucketing_strategy,
                    "threshold_bucket_counts": threshold_bucket_counts,
                    "post_bucket_counts": post_bucket_counts,
                    "fallback_quantiles": fallback_quantiles,
                    "min_non_hard_rollouts": int(cfg.min_non_hard_rollouts),
                    "max_hard_negative_fraction": max_hard_negative_fraction,
                    "observed_hard_negative_fraction": observed_hard_negative_fraction,
                    "degenerate_mix_detected": degenerate_mix_detected,
                    "num_vlm_floor_filtered_rollouts": num_vlm_floor_filtered_rollouts,
                    "num_vlm_floor_reasoning_conflicts": num_vlm_floor_reasoning_conflicts,
                    "stage2_backfill_considered": stage2_backfill_considered,
                    "num_stage2_backfill_candidates": num_stage2_backfill_candidates,
                    "effective_non_hard_count": effective_non_hard_count,
                    "num_invalid_decode_rollouts": len(invalid_decode_rows),
                    "expected_decode_frames": int(expected_decode_frames),
                    "require_valid_video_decode": bool(cfg.require_valid_video_decode),
                },
            )
        if (
            bool(cfg.fail_on_degenerate_mix)
            and observed_hard_negative_fraction > max_hard_negative_fraction
        ):
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=(
                    "WM refresh blocked: hard-negative fraction remains above cap "
                    f"after backfill attempts ({observed_hard_negative_fraction:.3f} > "
                    f"max_hard_negative_fraction={max_hard_negative_fraction:.3f})."
                ),
                metrics={
                    "source_condition": source_condition,
                    "num_source_rollouts": len(source_scores),
                    "num_success_rollouts": len(selected_success),
                    "num_near_miss_rollouts": len(near_miss),
                    "num_hard_negative_rollouts": len(hard_negative),
                    "bucketing_strategy": bucketing_strategy,
                    "threshold_bucket_counts": threshold_bucket_counts,
                    "post_bucket_counts": post_bucket_counts,
                    "fallback_quantiles": fallback_quantiles,
                    "min_non_hard_rollouts": int(cfg.min_non_hard_rollouts),
                    "max_hard_negative_fraction": max_hard_negative_fraction,
                    "observed_hard_negative_fraction": observed_hard_negative_fraction,
                    "degenerate_mix_detected": True,
                    "num_vlm_floor_filtered_rollouts": num_vlm_floor_filtered_rollouts,
                    "num_vlm_floor_reasoning_conflicts": num_vlm_floor_reasoning_conflicts,
                    "stage2_backfill_considered": stage2_backfill_considered,
                    "num_stage2_backfill_candidates": num_stage2_backfill_candidates,
                    "effective_non_hard_count": effective_non_hard_count,
                    "num_invalid_decode_rollouts": len(invalid_decode_rows),
                    "expected_decode_frames": int(expected_decode_frames),
                    "require_valid_video_decode": bool(cfg.require_valid_video_decode),
                },
            )

        stage_dir = work_dir / "wm_refresh_loop"
        stage_dir.mkdir(parents=True, exist_ok=True)
        iteration_summaries: List[Dict] = []
        refresh_manifest_paths: List[str] = []

        current_checkpoint = _resolve_current_world_checkpoint(previous_results, work_dir, config)
        if current_checkpoint is None or not current_checkpoint.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=(
                    "Could not resolve current world checkpoint from Stage 3 outputs "
                    f"or finetune.dreamdojo_checkpoint={config.finetune.dreamdojo_checkpoint}."
                ),
            )

        iterations = max(1, int(cfg.iterations))
        completed = 0
        latest_checkpoint = current_checkpoint
        final_selected_hard_negative_fraction: float | None = None
        for iteration in range(iterations):
            iter_dir = stage_dir / f"iter_{iteration:02d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            # Chain world-model refreshes by loading the previous adapted checkpoint.
            refresh_finetune_cfg = replace(
                config.finetune,
                dreamdojo_checkpoint=Path(latest_checkpoint),
                num_epochs=int(config.policy_rl_loop.world_model_refresh_epochs),
                learning_rate=float(config.policy_rl_loop.world_model_refresh_learning_rate),
            )
            loop_config = replace(config, finetune=refresh_finetune_cfg)

            refresh_result = refresh_world_model_from_bucketed_rollouts(
                config=loop_config,
                facility=facility,
                work_dir=work_dir,
                selected_success_rows=selected_success,
                near_miss_rows=near_miss,
                hard_negative_rows=hard_negative,
                output_dir=iter_dir,
                iteration=iteration,
            )
            refresh_result["input_checkpoint_path"] = str(latest_checkpoint)
            refresh_result["source_condition"] = source_condition
            iteration_summaries.append(refresh_result)
            mix_metrics = refresh_result.get("mix_metrics", {}) or {}
            selected_total = int(mix_metrics.get("selected_total", 0) or 0)
            selected_hard_negative = int(mix_metrics.get("selected_hard_negative", 0) or 0)
            selected_hard_negative_fraction = (
                float(selected_hard_negative) / float(selected_total)
                if selected_total > 0
                else 0.0
            )
            refresh_result["selected_hard_negative_fraction"] = selected_hard_negative_fraction
            final_selected_hard_negative_fraction = selected_hard_negative_fraction
            if selected_total > 0 and selected_hard_negative_fraction > max_hard_negative_fraction:
                degenerate_mix_detected = True
                detail = (
                    "WM refresh iteration "
                    f"{iteration} produced degenerate mix: selected_hard_negative="
                    f"{selected_hard_negative}/{selected_total} "
                    f"({selected_hard_negative_fraction:.3f}) exceeds "
                    f"max_hard_negative_fraction={max_hard_negative_fraction:.3f}."
                )
                if cfg.fail_if_refresh_fails:
                    return StageResult(
                        stage_name=self.name,
                        status="failed",
                        elapsed_seconds=0,
                        outputs={
                            "wm_refresh_loop_dir": str(stage_dir),
                            "iteration_summaries_path": str(
                                stage_dir / "wm_refresh_iteration_summaries.json"
                            ),
                            "refresh_manifest_paths": refresh_manifest_paths,
                            "final_adapted_checkpoint_path": str(latest_checkpoint),
                        },
                        metrics={
                            "iterations_requested": iterations,
                            "iterations_completed": completed,
                            "source_condition": source_condition,
                            "num_source_rollouts": len(source_scores),
                            "num_success_rollouts": len(selected_success),
                            "num_near_miss_rollouts": len(near_miss),
                            "num_hard_negative_rollouts": len(hard_negative),
                            "bucketing_strategy": bucketing_strategy,
                            "threshold_bucket_counts": threshold_bucket_counts,
                            "post_bucket_counts": post_bucket_counts,
                            "fallback_quantiles": fallback_quantiles,
                            "min_non_hard_rollouts": int(cfg.min_non_hard_rollouts),
                            "max_hard_negative_fraction": max_hard_negative_fraction,
                            "observed_hard_negative_fraction": observed_hard_negative_fraction,
                            "selected_hard_negative_fraction": selected_hard_negative_fraction,
                            "degenerate_mix_detected": degenerate_mix_detected,
                            "num_vlm_floor_filtered_rollouts": num_vlm_floor_filtered_rollouts,
                            "num_vlm_floor_reasoning_conflicts": num_vlm_floor_reasoning_conflicts,
                            "stage2_backfill_considered": stage2_backfill_considered,
                            "num_stage2_backfill_candidates": num_stage2_backfill_candidates,
                            "effective_non_hard_count": effective_non_hard_count,
                            "num_invalid_decode_rollouts": len(invalid_decode_rows),
                            "expected_decode_frames": int(expected_decode_frames),
                            "require_valid_video_decode": bool(cfg.require_valid_video_decode),
                        },
                        detail=detail,
                    )
                logger.warning(detail)
                break
            refresh_manifest = str(refresh_result.get("refresh_manifest_path", ""))
            if refresh_manifest:
                refresh_manifest_paths.append(refresh_manifest)

            candidate = str(refresh_result.get("adapted_checkpoint_path", "")).strip()
            if refresh_result.get("status") == "success" and candidate and Path(candidate).exists():
                latest_checkpoint = Path(candidate)
                completed += 1
                continue

            detail = (
                f"WM refresh iteration {iteration} failed: "
                f"{refresh_result.get('detail') or refresh_result.get('stderr') or 'unknown error'}"
            )
            if cfg.fail_if_refresh_fails:
                return StageResult(
                    stage_name=self.name,
                    status="failed",
                    elapsed_seconds=0,
                    outputs={
                        "wm_refresh_loop_dir": str(stage_dir),
                        "iteration_summaries_path": str(stage_dir / "wm_refresh_iteration_summaries.json"),
                        "refresh_manifest_paths": refresh_manifest_paths,
                        "final_adapted_checkpoint_path": str(latest_checkpoint),
                    },
                    metrics={
                        "iterations_requested": iterations,
                        "iterations_completed": completed,
                        "source_condition": source_condition,
                        "num_source_rollouts": len(source_scores),
                        "num_success_rollouts": len(selected_success),
                        "num_near_miss_rollouts": len(near_miss),
                        "num_hard_negative_rollouts": len(hard_negative),
                        "bucketing_strategy": bucketing_strategy,
                        "threshold_bucket_counts": threshold_bucket_counts,
                        "post_bucket_counts": post_bucket_counts,
                        "fallback_quantiles": fallback_quantiles,
                        "min_non_hard_rollouts": int(cfg.min_non_hard_rollouts),
                        "max_hard_negative_fraction": max_hard_negative_fraction,
                        "observed_hard_negative_fraction": observed_hard_negative_fraction,
                        "selected_hard_negative_fraction": final_selected_hard_negative_fraction,
                        "degenerate_mix_detected": degenerate_mix_detected,
                        "num_vlm_floor_filtered_rollouts": num_vlm_floor_filtered_rollouts,
                        "num_vlm_floor_reasoning_conflicts": num_vlm_floor_reasoning_conflicts,
                        "stage2_backfill_considered": stage2_backfill_considered,
                        "num_stage2_backfill_candidates": num_stage2_backfill_candidates,
                        "effective_non_hard_count": effective_non_hard_count,
                        "num_invalid_decode_rollouts": len(invalid_decode_rows),
                        "expected_decode_frames": int(expected_decode_frames),
                        "require_valid_video_decode": bool(cfg.require_valid_video_decode),
                    },
                    detail=detail,
                )
            logger.warning(detail)
            break

        summaries_path = stage_dir / "wm_refresh_iteration_summaries.json"
        write_json({"iterations": iteration_summaries}, summaries_path)
        prior_abs = _read_prior_absolute_point_differential(prev_s4)
        promoted_checkpoint = _promote_adapted_checkpoint_alias(
            source_checkpoint=latest_checkpoint,
            alias_path=work_dir / "finetune" / "adapted_checkpoint",
        )

        return StageResult(
            stage_name=self.name,
            status="success" if completed > 0 else "failed",
            elapsed_seconds=0,
            outputs={
                "wm_refresh_loop_dir": str(stage_dir),
                "iteration_summaries_path": str(summaries_path),
                "refresh_manifest_paths": refresh_manifest_paths,
                "final_adapted_checkpoint_path": str(latest_checkpoint),
                "promoted_adapted_checkpoint_path": str(promoted_checkpoint),
                # Compatibility aliases for downstream world-model consumers.
                "adapted_checkpoint_path": str(promoted_checkpoint),
                "lora_weights_path": str(promoted_checkpoint),
            },
            metrics={
                "iterations_requested": iterations,
                "iterations_completed": completed,
                "source_condition": source_condition,
                "num_source_rollouts": len(source_scores),
                "num_success_rollouts": len(selected_success),
                "num_near_miss_rollouts": len(near_miss),
                "num_hard_negative_rollouts": len(hard_negative),
                "num_positive_rollouts": num_positive_rollouts,
                "num_vlm_floor_filtered_rollouts": num_vlm_floor_filtered_rollouts,
                "num_vlm_floor_reasoning_conflicts": num_vlm_floor_reasoning_conflicts,
                "stage2_backfill_considered": stage2_backfill_considered,
                "num_stage2_backfill_candidates": num_stage2_backfill_candidates,
                "effective_non_hard_count": effective_non_hard_count,
                "bucketing_strategy": bucketing_strategy,
                "threshold_bucket_counts": threshold_bucket_counts,
                "post_bucket_counts": post_bucket_counts,
                "fallback_quantiles": fallback_quantiles,
                "min_non_hard_rollouts": int(cfg.min_non_hard_rollouts),
                "max_hard_negative_fraction": max_hard_negative_fraction,
                "observed_hard_negative_fraction": observed_hard_negative_fraction,
                "selected_hard_negative_fraction": final_selected_hard_negative_fraction,
                "degenerate_mix_detected": degenerate_mix_detected,
                "num_invalid_decode_rollouts": len(invalid_decode_rows),
                "expected_decode_frames": int(expected_decode_frames),
                "require_valid_video_decode": bool(cfg.require_valid_video_decode),
                "absolute_point_differential_pre_refresh": prior_abs,
            },
            detail="",
        )


def _resolve_current_world_checkpoint(
    previous_results: Dict[str, StageResult],
    work_dir: Path,
    config: ValidationConfig,
) -> Optional[Path]:
    s3 = previous_results.get("s3_finetune")
    if s3:
        candidate = s3.outputs.get("adapted_checkpoint_path") or s3.outputs.get("lora_weights_path")
        if candidate:
            path = Path(str(candidate))
            if path.exists():
                return path

    for candidate in [
        work_dir / "finetune" / "adapted_checkpoint",
        work_dir / "finetune" / "lora_weights",
        Path(config.finetune.dreamdojo_checkpoint),
    ]:
        if candidate.exists():
            return candidate
    return None


def _read_prior_absolute_point_differential(prev_s4: StageResult) -> Optional[float]:
    report_path = prev_s4.outputs.get("report_path")
    if not report_path:
        return None
    report_file = Path(str(report_path))
    if not report_file.exists():
        return None
    try:
        report = read_json(report_file)
    except Exception:
        return None
    value = report.get("absolute_point_differential", report.get("absolute_difference"))
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _promote_adapted_checkpoint_alias(source_checkpoint: Path, alias_path: Path) -> Path:
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    if alias_path.exists() or alias_path.is_symlink():
        try:
            if alias_path.is_symlink() or alias_path.is_file():
                alias_path.unlink()
            elif alias_path.is_dir():
                shutil.rmtree(alias_path)
        except Exception:
            # If cleanup fails, keep source path as the active checkpoint.
            return source_checkpoint
    try:
        alias_path.symlink_to(source_checkpoint, target_is_directory=True)
        return alias_path
    except Exception:
        # Symlink may be unavailable in some environments; fallback to source path.
        return source_checkpoint
