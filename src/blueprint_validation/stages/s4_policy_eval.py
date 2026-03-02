"""Stage 4: Policy evaluation — baseline vs site-adapted DreamDojo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Dict, List, Optional

import numpy as np

from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..evaluation.openvla_runner import (
    load_dreamdojo_world_model,
)
from ..evaluation.judge_audit import write_judge_audit_csv
from ..evaluation.action_overlay import overlay_scripted_trace_on_video
from ..evaluation.task_hints import (
    balance_eval_tasks,
    recommended_rollouts_per_condition,
    tasks_from_task_hints,
)
from ..evaluation.task_start_selector import (
    build_task_start_assignments,
    load_initial_frames_for_assignments,
    load_shared_task_start_manifest,
    save_shared_task_start_manifest,
    shared_manifest_is_compatible,
)
from ..evaluation.vlm_judge import (
    ManipulationJudgeScore,
    score_rollout,
    score_rollout_manipulation,
)
from ..evaluation.scripted_rollout_driver import (
    build_scripted_trace_manifest,
    run_scripted_rollout,
)
from ..evaluation.rollout_utils import run_rollout_with_adapter
from ..policy_adapters import get_policy_adapter
from .base import PipelineStage

logger = get_logger("stages.s4_policy_eval")


class PolicyEvalStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s4_policy_eval"

    @property
    def description(self) -> str:
        return "Evaluate selected policy adapter in baseline vs site-adapted DreamDojo world model"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        eval_dir = work_dir / "policy_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Check for fine-tuned checkpoint produced by Stage 3.
        adapted_dir = None
        prev_stage = previous_results.get("s3_finetune")
        if prev_stage:
            adapted_candidate = prev_stage.outputs.get(
                "adapted_checkpoint_path"
            ) or prev_stage.outputs.get("lora_weights_path")
            if adapted_candidate:
                adapted_dir = Path(adapted_candidate)
        if adapted_dir is None:
            # Backward-compatible fallbacks.
            for candidate in [
                work_dir / "finetune" / "adapted_checkpoint",
                work_dir / "finetune" / "lora_weights",
            ]:
                if candidate.exists():
                    adapted_dir = candidate
                    break
        if adapted_dir is None:
            adapted_dir = work_dir / "finetune" / "adapted_checkpoint"
        if not adapted_dir.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=f"Adapted checkpoint not found at {adapted_dir}. Run Stage 3 first.",
            )

        # Load render manifest for initial frames
        render_manifest_path = work_dir / "renders" / "render_manifest.json"
        if not render_manifest_path.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Render manifest not found. Run Stage 1 first.",
            )

        render_manifest = read_json(render_manifest_path)

        tasks, hint_count = _build_task_list(config, facility)
        unresolved_manip_tasks_dropped = 0
        if bool(config.eval_policy.require_object_grounded_manip_tasks):
            filtered_tasks: List[str] = []
            for task in tasks:
                if _is_manipulation_task(task) and not _is_object_grounded_manip_task(task):
                    unresolved_manip_tasks_dropped += 1
                    continue
                filtered_tasks.append(task)
            tasks = filtered_tasks
            if not tasks:
                tasks = [
                    "Navigate forward through the corridor",
                    "Turn left at the intersection",
                    "Approach the nearest obstacle",
                ]

        requested_rollouts = int(config.eval_policy.num_rollouts)
        planned_rollouts = recommended_rollouts_per_condition(
            num_unique_tasks=len(tasks),
            requested=requested_rollouts,
            profile="dreamdojo",
        )

        shared_manifest_path = eval_dir / "shared_task_start_manifest.json"
        selector_config = {
            "min_assignment_quality_score": float(config.eval_policy.min_assignment_quality_score),
            "require_object_grounded_manip_tasks": bool(
                config.eval_policy.require_object_grounded_manip_tasks
            ),
        }
        shared_manifest = load_shared_task_start_manifest(shared_manifest_path)
        reused_shared_manifest = False
        rollout_assignments: List[dict] = []
        if shared_manifest and shared_manifest_is_compatible(
            shared_manifest,
            facility_name=facility.name,
            render_manifest_path=render_manifest_path,
            render_manifest=render_manifest,
            tasks=tasks,
            video_orientation_fix=facility.video_orientation_fix,
            selector_config=selector_config,
        ):
            rollout_assignments = list(shared_manifest.get("assignments", []))
            reused_shared_manifest = bool(rollout_assignments)

        if not rollout_assignments:
            rollout_assignments = build_task_start_assignments(
                tasks=tasks,
                num_rollouts=planned_rollouts,
                render_manifest=render_manifest,
                task_hints_path=facility.task_hints_path,
                min_assignment_quality_score=float(config.eval_policy.min_assignment_quality_score),
                require_object_grounded_manip_tasks=bool(
                    config.eval_policy.require_object_grounded_manip_tasks
                ),
                video_orientation_fix=facility.video_orientation_fix,
            )
            save_shared_task_start_manifest(
                path=shared_manifest_path,
                facility_name=facility.name,
                render_manifest_path=render_manifest_path,
                task_profile="dreamdojo",
                requested_rollouts=requested_rollouts,
                planned_rollouts=planned_rollouts,
                tasks=tasks,
                assignments=rollout_assignments,
                render_manifest=render_manifest,
                video_orientation_fix=facility.video_orientation_fix,
                selector_config=selector_config,
            )
        else:
            # Ensure older manifests still carry explicit orientation metadata at runtime.
            normalized_fix = str(getattr(facility, "video_orientation_fix", "none"))
            for assignment in rollout_assignments:
                assignment.setdefault("video_orientation_fix", normalized_fix)

        invalid_assignments = [
            assignment
            for assignment in rollout_assignments
            if str(assignment.get("assignment_reject_reason") or "").strip()
        ]
        num_rejected_task_start_assignments = len(invalid_assignments)
        if invalid_assignments:
            logger.warning(
                "Dropping %d fallback task-start assignments that failed strict selector constraints.",
                num_rejected_task_start_assignments,
            )
            rollout_assignments = [
                assignment
                for assignment in rollout_assignments
                if not str(assignment.get("assignment_reject_reason") or "").strip()
            ]
        if not rollout_assignments:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=(
                    "No valid rollout assignments met task-start quality/grounding constraints. "
                    "Relax eval_policy.min_assignment_quality_score or "
                    "eval_policy.require_object_grounded_manip_tasks."
                ),
            )

        frame_cache = load_initial_frames_for_assignments(rollout_assignments)
        if not frame_cache:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Could not extract initial frames for rollout assignments.",
            )

        num_rollouts = len(rollout_assignments)
        headline_scope = _headline_scope(config)
        claim_mode = (config.eval_policy.mode or "claim").strip().lower() == "claim"
        required_dim = int(config.eval_policy.required_action_dim)
        if claim_mode and headline_scope == "dual" and config.policy_adapter.name.strip().lower() != "openvla_oft":
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Claim mode only supports policy_adapter.name=openvla_oft.",
            )
        if claim_mode and headline_scope == "dual" and not config.eval_policy.require_native_action_compat:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Claim mode requires eval_policy.require_native_action_compat=true.",
            )
        max_steps = int(config.eval_policy.max_steps_per_rollout)
        if claim_mode:
            max_steps = min(max_steps, int(config.eval_policy.reliability.max_horizon_steps))

        device = "cuda" if _has_cuda() else "cpu"
        if headline_scope == "wm_only":
            return _run_world_model_only_eval(
                config=config,
                facility=facility,
                work_dir=work_dir,
                eval_dir=eval_dir,
                adapted_dir=adapted_dir,
                tasks=tasks,
                hint_count=hint_count,
                requested_rollouts=requested_rollouts,
                planned_rollouts=planned_rollouts,
                rollout_assignments=rollout_assignments,
                frame_cache=frame_cache,
                num_rollouts=num_rollouts,
                shared_manifest_path=shared_manifest_path,
                reused_shared_manifest=reused_shared_manifest,
                max_steps=max_steps,
                device=device,
                claim_mode=claim_mode,
                unresolved_manip_tasks_dropped=unresolved_manip_tasks_dropped,
                num_rejected_task_start_assignments=num_rejected_task_start_assignments,
            )

        policy_adapter = get_policy_adapter(config.policy_adapter)
        base_model_name, base_checkpoint = policy_adapter.base_model_ref(config.eval_policy)
        adapted_policy_checkpoint = _resolve_adapted_policy_checkpoint(
            previous_results=previous_results,
            work_dir=work_dir,
            policy_adapter=policy_adapter,
        )

        all_scores: List[Dict] = []
        scoring_failures: List[str] = []
        observed_action_dims: set[int] = set()
        observed_policy_dims: set[int] = set()
        observed_world_dims: set[int] = set()

        conditions = list(config.eval_policy.conditions)
        for condition in conditions:
            logger.info("Running %s condition rollouts", condition)
            condition_dir = eval_dir / f"{condition}_rollouts"
            condition_dir.mkdir(exist_ok=True)

            # Load policy checkpoint for this condition.
            policy_checkpoint = base_checkpoint
            if condition == "adapted" and adapted_policy_checkpoint is not None:
                policy_checkpoint = adapted_policy_checkpoint
            policy_handle = policy_adapter.load_policy(
                model_name=base_model_name,
                checkpoint_path=policy_checkpoint,
                device=device,
            )

            # Load world model
            adapted = adapted_dir if condition == "adapted" else None
            world_model = load_dreamdojo_world_model(
                checkpoint_path=config.finetune.dreamdojo_checkpoint,
                adapted_checkpoint=adapted,
                configured_experiment=(
                    config.finetune.eval_world_experiment or config.finetune.experiment_config
                ),
                dreamdojo_repo=config.finetune.dreamdojo_repo,
                device=device,
            )
            world_dim = _extract_world_action_dim(world_model)
            if world_dim is not None:
                observed_world_dims.add(int(world_dim))
            policy_dim = _resolve_policy_action_dim(config)
            if policy_dim is not None:
                observed_policy_dims.add(int(policy_dim))
            if claim_mode and config.eval_policy.require_native_action_compat:
                if world_dim is None or policy_dim is None:
                    return StageResult(
                        stage_name=self.name,
                        status="failed",
                        elapsed_seconds=0,
                        detail=(
                            "Claim mode requires resolvable policy/world action dims. "
                            f"policy_dim={policy_dim}, world_dim={world_dim}"
                        ),
                    )
                if int(world_dim) != required_dim or int(policy_dim) != required_dim:
                    return StageResult(
                        stage_name=self.name,
                        status="failed",
                        elapsed_seconds=0,
                        detail=(
                            "Claim mode action contract failed: "
                            f"policy_dim={policy_dim}, world_dim={world_dim}, required={required_dim}."
                        ),
                    )

            for assignment in rollout_assignments:
                rollout_idx = int(assignment.get("rollout_index", 0))
                task = str(assignment.get("task", ""))
                clip_index = int(assignment.get("clip_index", -1))
                init_frame = frame_cache.get(clip_index)
                if init_frame is None:
                    msg = f"Initial frame missing for clip_index={clip_index} task='{task}'"
                    scoring_failures.append(msg)
                    logger.warning(msg)
                    continue
                clip_stub = str(assignment.get("clip_name", f"clip_{clip_index:03d}"))
                clip_name = f"{condition}_{clip_stub}_{rollout_idx:03d}".replace("/", "_").replace(
                    " ", "_"
                )

                rollout = run_rollout_with_adapter(
                    world_model=world_model,
                    policy_adapter=policy_adapter,
                    policy_handle=policy_handle,
                    initial_frame=init_frame,
                    task_prompt=task,
                    max_steps=max_steps,
                    unnorm_key=config.eval_policy.unnorm_key,
                    output_dir=condition_dir,
                    clip_name=clip_name,
                    device=device,
                    expected_action_dim=required_dim if claim_mode else None,
                    reanchor_every=(
                        int(config.eval_policy.reliability.keyframe_reanchor_every)
                        if claim_mode
                        else None
                    ),
                )
                action_contract = getattr(rollout, "action_contract", {}) or {}
                action_dim = action_contract.get("dataset_dim")
                if action_dim is not None:
                    observed_action_dims.add(int(action_dim))
                if claim_mode and config.eval_policy.require_native_action_compat:
                    if not bool(action_contract.get("compliant", False)):
                        return StageResult(
                            stage_name=self.name,
                            status="failed",
                            elapsed_seconds=0,
                            detail=(
                                "Claim mode action contract violation in rollout: "
                                f"{action_contract}"
                            ),
                        )

                if not rollout.video_path or not rollout.video_path.exists():
                    msg = f"Rollout video missing for {clip_name}"
                    scoring_failures.append(msg)
                    logger.warning(msg)
                    continue

                try:
                    if _is_manipulation_task(task):
                        score = score_rollout_manipulation(
                            video_path=rollout.video_path,
                            task_prompt=task,
                            config=config.eval_policy.vlm_judge,
                            facility_description=facility.description,
                        )
                    else:
                        score = score_rollout(
                            video_path=rollout.video_path,
                            task_prompt=task,
                            config=config.eval_policy.vlm_judge,
                            facility_description=facility.description,
                        )
                except Exception as e:
                    msg = f"VLM scoring failed for {clip_name}: {e}"
                    logger.warning(msg)
                    scoring_failures.append(msg)
                    continue

                all_scores.append(
                    {
                        "condition": condition,
                        "task": task,
                        "rollout_index": rollout_idx,
                        "task_score": score.task_score,
                        "visual_score": score.visual_score,
                        "spatial_score": score.spatial_score,
                        "reasoning": score.reasoning,
                        "video_path": str(rollout.video_path),
                        "num_steps": rollout.num_steps,
                        "action_sequence": getattr(rollout, "action_sequence", []),
                        "start_clip_index": clip_index,
                        "start_clip_name": clip_stub,
                        "start_path_type": str(assignment.get("path_type", "unknown")),
                        "target_instance_id": assignment.get("target_instance_id"),
                        "target_label": assignment.get("target_label"),
                        "target_grounded": bool(assignment.get("target_grounded", False)),
                        "assignment_quality_score": assignment.get("assignment_quality_score"),
                        "assignment_reject_reason": assignment.get("assignment_reject_reason"),
                        "start_frame_orientation_fix_applied": assignment.get(
                            "start_frame_orientation_fix_applied", "none"
                        ),
                        "is_manipulation_task": _is_manipulation_task(task),
                        "grasp_acquired": (
                            score.grasp_acquired
                            if isinstance(score, ManipulationJudgeScore)
                            else None
                        ),
                        "lifted_clear": (
                            score.lifted_clear
                            if isinstance(score, ManipulationJudgeScore)
                            else None
                        ),
                        "placed_in_target": (
                            score.placed_in_target
                            if isinstance(score, ManipulationJudgeScore)
                            else None
                        ),
                        "stable_after_place": (
                            score.stable_after_place
                            if isinstance(score, ManipulationJudgeScore)
                            else None
                        ),
                        "action_contract": action_contract,
                        "overlay_video_path": None,
                        "overlay_mode": "raw",
                    }
                )

        # Compute aggregate metrics per condition
        write_json({"scores": all_scores}, eval_dir / "vlm_scores.json")
        audit_csv_path = eval_dir / "judge_audit.csv"
        write_judge_audit_csv(all_scores, audit_csv_path)

        per_condition: Dict[str, Dict] = {}
        for cond in conditions:
            cond_scores = [s for s in all_scores if s["condition"] == cond]
            cond_manip = [s for s in cond_scores if s["is_manipulation_task"]]
            cond_mean = float(np.mean([s["task_score"] for s in cond_scores])) if cond_scores else 0
            per_condition[cond] = {
                "mean_task_score": round(cond_mean, 3),
                "num_rollouts": len(cond_scores),
                "manipulation_success_rate": _manipulation_success_rate(cond_manip),
            }

        # Pairwise comparisons between all condition pairs
        pairwise = _build_pairwise_metrics(all_scores, conditions)

        # Backward-compatible top-level metrics (baseline vs adapted)
        baseline_scores = [s for s in all_scores if s["condition"] == "baseline"]
        adapted_scores = [s for s in all_scores if s["condition"] == "adapted"]
        baseline_mean = per_condition.get("baseline", {}).get("mean_task_score", 0)
        adapted_mean = per_condition.get("adapted", {}).get("mean_task_score", 0)
        improvement = ((adapted_mean - baseline_mean) / max(baseline_mean, 1e-8)) * 100
        absolute_difference = adapted_mean - baseline_mean

        min_len = min(len(baseline_scores), len(adapted_scores))
        wins = sum(
            1
            for b, a in zip(baseline_scores[:min_len], adapted_scores[:min_len])
            if a["task_score"] > b["task_score"]
        )
        win_rate = wins / max(min_len, 1)

        p_value = None
        if min_len >= 2:
            try:
                from scipy import stats

                b_vals = [s["task_score"] for s in baseline_scores[:min_len]]
                a_vals = [s["task_score"] for s in adapted_scores[:min_len]]
                _, p_value = stats.ttest_rel(b_vals, a_vals)
                p_value = float(p_value)
            except ImportError:
                logger.warning("scipy not available; skipping p-value computation")

        policy_dim = _single_or_none(observed_policy_dims)
        world_dim = _single_or_none(observed_world_dims)
        dataset_dim = _single_or_none(observed_action_dims)
        action_contract = {
            "policy_dim": policy_dim,
            "world_dim": world_dim,
            "dataset_dim": dataset_dim,
            "compliant": (
                policy_dim is not None
                and world_dim is not None
                and dataset_dim is not None
                and policy_dim == world_dim == dataset_dim
            ),
            "reason": "",
        }
        if not action_contract["compliant"]:
            action_contract["reason"] = (
                "policy/world/dataset action dimensions are missing or inconsistent."
            )

        total_scoring_attempts = len(all_scores) + len(scoring_failures)
        scoring_failure_rate = len(scoring_failures) / max(total_scoring_attempts, 1)
        reliability_gate = _build_reliability_gate(
            config,
            all_scores,
            scoring_failure_rate=scoring_failure_rate,
        )
        manip_delta_pp = (
            (per_condition.get("adapted", {}).get("manipulation_success_rate", 0.0) or 0.0)
            - (per_condition.get("baseline", {}).get("manipulation_success_rate", 0.0) or 0.0)
        ) * 100.0
        claim_failure_reasons: List[str] = []
        if claim_mode:
            if not action_contract["compliant"]:
                claim_failure_reasons.append(
                    f"Action contract failed: {action_contract.get('reason')}"
                )
            if not reliability_gate["passed"]:
                claim_failure_reasons.append(
                    "Reliability gate failed: "
                    f"replay_pass_rate={reliability_gate['replay_pass_rate']:.3f}, "
                    f"controllability_pass_rate={reliability_gate['controllability_pass_rate']:.3f}"
                )
            if float(absolute_difference) < float(config.eval_policy.min_absolute_difference):
                claim_failure_reasons.append(
                    "Absolute task-score difference below threshold: "
                    f"{absolute_difference:.3f} < {config.eval_policy.min_absolute_difference:.3f}"
                )
            if float(manip_delta_pp) < float(config.eval_policy.min_manip_success_delta_pp):
                claim_failure_reasons.append(
                    "Manipulation success delta below threshold: "
                    f"{manip_delta_pp:.2f}pp < {config.eval_policy.min_manip_success_delta_pp:.2f}pp"
                )
        claim_passed = len(claim_failure_reasons) == 0

        metrics = {
            "headline_scope": headline_scope,
            "baseline_mean_task_score": round(float(baseline_mean), 3),
            "adapted_mean_task_score": round(float(adapted_mean), 3),
            "improvement_pct": round(float(improvement), 2),
            "absolute_difference": round(float(absolute_difference), 3),
            "absolute_point_differential": round(float(absolute_difference), 3),
            "win_rate": round(float(win_rate), 3),
            "p_value": round(p_value, 6) if p_value is not None else None,
            "num_rollouts_baseline": len(baseline_scores),
            "num_rollouts_adapted": len(adapted_scores),
            "num_scoring_failures": len(scoring_failures),
            "scoring_failure_rate": round(float(scoring_failure_rate), 6),
            "num_valid_scored_rows": len(all_scores),
            "used_adapted_policy_checkpoint": adapted_policy_checkpoint is not None,
            "adapted_policy_checkpoint": (
                str(adapted_policy_checkpoint) if adapted_policy_checkpoint else None
            ),
            "requested_rollouts_per_condition": requested_rollouts,
            "planned_rollouts_per_condition": planned_rollouts,
            "executed_rollouts_per_condition": num_rollouts,
            "num_unique_task_templates": len(tasks),
            "shared_task_start_manifest": str(shared_manifest_path),
            "shared_task_start_manifest_reused": reused_shared_manifest,
            "per_condition": per_condition,
            "pairwise": pairwise,
            "baseline_manipulation_success_rate": per_condition.get("baseline", {}).get(
                "manipulation_success_rate", 0.0
            ),
            "adapted_manipulation_success_rate": per_condition.get("adapted", {}).get(
                "manipulation_success_rate", 0.0
            ),
            "task_hints_injected": hint_count,
            "judge_audit_csv": str(audit_csv_path),
            "action_contract": action_contract,
            "reliability_gate": reliability_gate,
            "manipulation_success_delta_pp": round(float(manip_delta_pp), 3),
            "claim_mode": claim_mode,
            "claim_passed": claim_passed,
            "claim_failure_reasons": claim_failure_reasons,
            "deferred_claims": [],
            "confidence_intervals": _build_confidence_intervals(baseline_scores, adapted_scores),
            "heldout_manifest_hash": _manifest_hash(shared_manifest_path),
            "num_unresolved_manip_tasks_dropped": int(unresolved_manip_tasks_dropped),
            "num_rejected_task_start_assignments": int(num_rejected_task_start_assignments),
            "low_score_breakdown": _build_low_score_breakdown(all_scores),
        }

        write_json(metrics, eval_dir / "policy_eval_report.json")
        detail_lines = list(scoring_failures[:5])
        if (
            bool(config.eval_policy.reliability.enforce_stage_success)
            and not bool(reliability_gate.get("passed", False))
        ):
            detail_lines.append(f"Reliability gate failed: {reliability_gate.get('reason', '')}".strip())

        return StageResult(
            stage_name=self.name,
            status=(
                "success"
                if all_scores
                and (not claim_mode or claim_passed)
                and (
                    not bool(config.eval_policy.reliability.enforce_stage_success)
                    or bool(reliability_gate.get("passed", False))
                )
                else "failed"
            ),
            elapsed_seconds=0,
            outputs={
                "eval_dir": str(eval_dir),
                "scores_path": str(eval_dir / "vlm_scores.json"),
                "report_path": str(eval_dir / "policy_eval_report.json"),
                "shared_task_start_manifest": str(shared_manifest_path),
                "judge_audit_csv": str(audit_csv_path),
            },
            metrics=metrics,
            detail="\n".join(line for line in detail_lines if line),
        )


def _run_world_model_only_eval(
    *,
    config: ValidationConfig,
    facility: FacilityConfig,
    work_dir: Path,
    eval_dir: Path,
    adapted_dir: Path,
    tasks: List[str],
    hint_count: int,
    requested_rollouts: int,
    planned_rollouts: int,
    rollout_assignments: List[dict],
    frame_cache: Dict[int, np.ndarray],
    num_rollouts: int,
    shared_manifest_path: Path,
    reused_shared_manifest: bool,
    max_steps: int,
    device: str,
    claim_mode: bool,
    unresolved_manip_tasks_dropped: int,
    num_rejected_task_start_assignments: int,
) -> StageResult:
    rollout_driver = (config.eval_policy.rollout_driver or "scripted").strip().lower()
    if rollout_driver not in {"scripted", "both"}:
        return StageResult(
            stage_name="s4_policy_eval",
            status="failed",
            elapsed_seconds=0,
            detail=(
                "WM-only claim path currently supports rollout_driver in {'scripted','both'}; "
                f"got '{rollout_driver}'."
            ),
        )

    conditions = ["baseline", "adapted"]

    def _load_world_model_for_condition(condition: str):
        return load_dreamdojo_world_model(
            checkpoint_path=config.finetune.dreamdojo_checkpoint,
            adapted_checkpoint=(adapted_dir if condition == "adapted" else None),
            configured_experiment=(
                config.finetune.eval_world_experiment or config.finetune.experiment_config
            ),
            dreamdojo_repo=config.finetune.dreamdojo_repo,
            device=device,
        )

    def _release_world_model(model) -> None:
        try:
            pipe = getattr(model, "_pipe", None)
            inner = getattr(pipe, "model", None) if pipe is not None else None
            if inner is not None:
                for attr in ("net", "conditioner"):
                    comp = getattr(inner, attr, None)
                    if comp is not None and hasattr(comp, "to"):
                        try:
                            comp.to("cpu")
                        except Exception:
                            pass
                tokenizer = getattr(inner, "tokenizer", None)
                if tokenizer is not None:
                    for attr in ("encoder", "decoder"):
                        comp = getattr(tokenizer, attr, None)
                        if comp is not None and hasattr(comp, "to"):
                            try:
                                comp.to("cpu")
                            except Exception:
                                pass
                text_encoder = getattr(inner, "text_encoder", None)
                text_model = getattr(text_encoder, "model", None) if text_encoder is not None else None
                if text_model is not None and hasattr(text_model, "to"):
                    try:
                        text_model.to("cpu")
                    except Exception:
                        pass
            if pipe is not None and hasattr(pipe, "model"):
                try:
                    del pipe.model
                except Exception:
                    pass
            if hasattr(model, "_pipe"):
                try:
                    delattr(model, "_pipe")
                except Exception:
                    pass
        except Exception:
            pass
        del model
        try:
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    all_scores: List[Dict] = []
    scoring_failures: List[str] = []
    observed_action_dims: set[int] = set()
    observed_world_dims: set[int] = set()
    baseline_world_dim: Optional[int] = None
    trace_manifest: Optional[Dict[str, Dict]] = None

    for condition in conditions:
        logger.info("Running %s WM-only scripted rollouts", condition)
        condition_dir = eval_dir / f"{condition}_rollouts"
        condition_dir.mkdir(exist_ok=True)
        world_model = _load_world_model_for_condition(condition)
        world_dim = _extract_world_action_dim(world_model)
        if world_dim is not None:
            observed_world_dims.add(int(world_dim))
            if baseline_world_dim is None:
                baseline_world_dim = int(world_dim)
            elif condition == "adapted" and baseline_world_dim != int(world_dim):
                _release_world_model(world_model)
                return StageResult(
                    stage_name="s4_policy_eval",
                    status="failed",
                    elapsed_seconds=0,
                    detail=(
                        "World-model action dims differ across conditions: "
                        f"baseline={baseline_world_dim}, adapted={int(world_dim)}"
                    ),
                )

        if trace_manifest is None:
            action_dim = world_dim or _resolve_world_action_dim_from_config(config)
            if action_dim is None:
                _release_world_model(world_model)
                return StageResult(
                    stage_name="s4_policy_eval",
                    status="failed",
                    elapsed_seconds=0,
                    detail=(
                        "Could not resolve world-model action_dim for WM-only scripted rollouts. "
                        "Set finetune.eval_world_experiment or ensure checkpoint metadata exposes action dim."
                    ),
                )
            trace_manifest = build_scripted_trace_manifest(
                rollout_assignments,
                action_dim=int(action_dim),
                max_steps=max_steps,
            )

        for assignment in rollout_assignments:
            rollout_idx = int(assignment.get("rollout_index", 0))
            task = str(assignment.get("task", ""))
            clip_index = int(assignment.get("clip_index", -1))
            init_frame = frame_cache.get(clip_index)
            if init_frame is None:
                msg = f"Initial frame missing for clip_index={clip_index} task='{task}'"
                scoring_failures.append(msg)
                logger.warning(msg)
                continue
            trace = trace_manifest.get(_assignment_trace_key(assignment))
            if trace is None:
                msg = f"Missing scripted trace for rollout_index={rollout_idx}, task='{task}'"
                scoring_failures.append(msg)
                logger.warning(msg)
                continue
            clip_stub = str(assignment.get("clip_name", f"clip_{clip_index:03d}"))
            clip_name = f"{condition}_{clip_stub}_{rollout_idx:03d}".replace("/", "_").replace(
                " ", "_"
            )

            rollout = run_scripted_rollout(
                world_model=world_model,
                initial_frame=init_frame,
                action_sequence=trace["action_sequence"],
                output_dir=condition_dir,
                clip_name=clip_name,
                trace_id=str(trace["trace_id"]),
                reanchor_every=int(config.eval_policy.reliability.keyframe_reanchor_every),
            )
            action_contract = getattr(rollout, "action_contract", {}) or {}
            action_dim_row = action_contract.get("dataset_dim")
            if action_dim_row is not None:
                observed_action_dims.add(int(action_dim_row))

            if not rollout.video_path or not rollout.video_path.exists():
                msg = f"Rollout video missing for {clip_name}"
                scoring_failures.append(msg)
                logger.warning(msg)
                continue

            try:
                overlay_mode = "raw"
                scored_video_path = rollout.video_path
                overlay_video_path = None
                if _is_manipulation_task(task):
                    if str(config.eval_policy.manip_eval_mode).strip().lower() == "overlay_marker":
                        overlay_mode = "overlay_marker"
                        overlay_video_path = (
                            condition_dir / "overlay" / f"{clip_name}_overlay.mp4"
                        )
                        scored_video_path = overlay_scripted_trace_on_video(
                            input_video_path=rollout.video_path,
                            output_video_path=overlay_video_path,
                            action_sequence=trace.get("action_sequence", []),
                            target_label=str(assignment.get("target_label") or ""),
                        )
                    score = score_rollout_manipulation(
                        video_path=scored_video_path,
                        task_prompt=task,
                        config=config.eval_policy.vlm_judge,
                        facility_description=facility.description,
                    )
                else:
                    score = score_rollout(
                        video_path=scored_video_path,
                        task_prompt=task,
                        config=config.eval_policy.vlm_judge,
                        facility_description=facility.description,
                    )
            except Exception as e:
                msg = f"VLM scoring failed for {clip_name}: {e}"
                logger.warning(msg)
                scoring_failures.append(msg)
                continue

            all_scores.append(
                {
                    "condition": condition,
                    "task": task,
                    "rollout_index": rollout_idx,
                    "trace_id": trace["trace_id"],
                    "driver_type": getattr(rollout, "driver_type", "scripted"),
                    "task_score": score.task_score,
                    "visual_score": score.visual_score,
                    "spatial_score": score.spatial_score,
                    "reasoning": score.reasoning,
                    "video_path": str(rollout.video_path),
                    "num_steps": rollout.num_steps,
                    "action_sequence": getattr(rollout, "action_sequence", []),
                    "start_clip_index": clip_index,
                    "start_clip_name": clip_stub,
                    "start_path_type": str(assignment.get("path_type", "unknown")),
                    "target_instance_id": assignment.get("target_instance_id"),
                    "target_label": assignment.get("target_label"),
                    "target_grounded": bool(assignment.get("target_grounded", False)),
                    "assignment_quality_score": assignment.get("assignment_quality_score"),
                    "assignment_reject_reason": assignment.get("assignment_reject_reason"),
                    "start_frame_orientation_fix_applied": assignment.get(
                        "start_frame_orientation_fix_applied", "none"
                    ),
                    "is_manipulation_task": _is_manipulation_task(task),
                    "grasp_acquired": (
                        score.grasp_acquired if isinstance(score, ManipulationJudgeScore) else None
                    ),
                    "lifted_clear": (
                        score.lifted_clear if isinstance(score, ManipulationJudgeScore) else None
                    ),
                    "placed_in_target": (
                        score.placed_in_target if isinstance(score, ManipulationJudgeScore) else None
                    ),
                    "stable_after_place": (
                        score.stable_after_place
                        if isinstance(score, ManipulationJudgeScore)
                        else None
                    ),
                    "action_contract": action_contract,
                    "overlay_video_path": str(overlay_video_path) if overlay_video_path else None,
                    "overlay_mode": overlay_mode,
                }
            )
        _release_world_model(world_model)

    write_json({"scores": all_scores}, eval_dir / "vlm_scores.json")
    audit_csv_path = eval_dir / "judge_audit.csv"
    write_judge_audit_csv(all_scores, audit_csv_path)

    per_condition: Dict[str, Dict] = {}
    for cond in conditions:
        cond_scores = [s for s in all_scores if s["condition"] == cond]
        cond_manip = [s for s in cond_scores if s["is_manipulation_task"]]
        cond_mean = float(np.mean([s["task_score"] for s in cond_scores])) if cond_scores else 0
        per_condition[cond] = {
            "mean_task_score": round(cond_mean, 3),
            "num_rollouts": len(cond_scores),
            "manipulation_success_rate": _manipulation_success_rate(cond_manip),
        }

    pairwise = _build_pairwise_metrics(all_scores, conditions)
    baseline_scores = [s for s in all_scores if s["condition"] == "baseline"]
    adapted_scores = [s for s in all_scores if s["condition"] == "adapted"]
    baseline_mean = per_condition.get("baseline", {}).get("mean_task_score", 0)
    adapted_mean = per_condition.get("adapted", {}).get("mean_task_score", 0)
    improvement = ((adapted_mean - baseline_mean) / max(baseline_mean, 1e-8)) * 100
    absolute_difference = adapted_mean - baseline_mean

    min_len = min(len(baseline_scores), len(adapted_scores))
    wins = sum(
        1
        for b, a in zip(baseline_scores[:min_len], adapted_scores[:min_len])
        if a["task_score"] > b["task_score"]
    )
    win_rate = wins / max(min_len, 1)

    p_value = None
    if min_len >= 2:
        try:
            from scipy import stats

            b_vals = [s["task_score"] for s in baseline_scores[:min_len]]
            a_vals = [s["task_score"] for s in adapted_scores[:min_len]]
            _, p_value = stats.ttest_rel(b_vals, a_vals)
            p_value = float(p_value)
        except ImportError:
            logger.warning("scipy not available; skipping p-value computation")

    dataset_dim = _single_or_none(observed_action_dims)
    world_dim = _single_or_none(observed_world_dims)
    action_contract = {
        "policy_dim": None,
        "world_dim": world_dim,
        "dataset_dim": dataset_dim,
        "compliant": (
            world_dim is not None
            and dataset_dim is not None
            and int(world_dim) == int(dataset_dim)
        ),
        "reason": "",
    }
    if not action_contract["compliant"]:
        action_contract["reason"] = "world/dataset action dimensions are missing or inconsistent."

    total_scoring_attempts = len(all_scores) + len(scoring_failures)
    scoring_failure_rate = len(scoring_failures) / max(total_scoring_attempts, 1)
    reliability_gate = _build_reliability_gate(
        config,
        all_scores,
        scoring_failure_rate=scoring_failure_rate,
    )
    manip_delta_pp = (
        (per_condition.get("adapted", {}).get("manipulation_success_rate", 0.0) or 0.0)
        - (per_condition.get("baseline", {}).get("manipulation_success_rate", 0.0) or 0.0)
    ) * 100.0

    claim_failure_reasons: List[str] = []
    if claim_mode:
        if not action_contract["compliant"]:
            claim_failure_reasons.append(
                f"Action contract failed: {action_contract.get('reason')}"
            )
        if not reliability_gate["passed"]:
            claim_failure_reasons.append(
                "Reliability gate failed: "
                f"replay_pass_rate={reliability_gate['replay_pass_rate']:.3f}, "
                f"controllability_pass_rate={reliability_gate['controllability_pass_rate']:.3f}"
            )
        if float(absolute_difference) < float(config.eval_policy.min_absolute_difference):
            claim_failure_reasons.append(
                "Absolute task-score difference below threshold: "
                f"{absolute_difference:.3f} < {config.eval_policy.min_absolute_difference:.3f}"
            )
        if float(manip_delta_pp) < float(config.eval_policy.min_manip_success_delta_pp):
            claim_failure_reasons.append(
                "Manipulation success delta below threshold: "
                f"{manip_delta_pp:.2f}pp < {config.eval_policy.min_manip_success_delta_pp:.2f}pp"
            )
    claim_passed = len(claim_failure_reasons) == 0

    metrics = {
        "headline_scope": "wm_only",
        "rollout_driver": rollout_driver,
        "baseline_mean_task_score": round(float(baseline_mean), 3),
        "adapted_mean_task_score": round(float(adapted_mean), 3),
        "improvement_pct": round(float(improvement), 2),
        "absolute_difference": round(float(absolute_difference), 3),
        "absolute_point_differential": round(float(absolute_difference), 3),
        "win_rate": round(float(win_rate), 3),
        "p_value": round(p_value, 6) if p_value is not None else None,
        "num_rollouts_baseline": len(baseline_scores),
        "num_rollouts_adapted": len(adapted_scores),
        "num_scoring_failures": len(scoring_failures),
        "scoring_failure_rate": round(float(scoring_failure_rate), 6),
        "num_valid_scored_rows": len(all_scores),
        "used_adapted_policy_checkpoint": False,
        "adapted_policy_checkpoint": None,
        "requested_rollouts_per_condition": requested_rollouts,
        "planned_rollouts_per_condition": planned_rollouts,
        "executed_rollouts_per_condition": num_rollouts,
        "num_unique_task_templates": len(tasks),
        "shared_task_start_manifest": str(shared_manifest_path),
        "shared_task_start_manifest_reused": reused_shared_manifest,
        "per_condition": per_condition,
        "pairwise": pairwise,
        "baseline_manipulation_success_rate": per_condition.get("baseline", {}).get(
            "manipulation_success_rate", 0.0
        ),
        "adapted_manipulation_success_rate": per_condition.get("adapted", {}).get(
            "manipulation_success_rate", 0.0
        ),
        "task_hints_injected": hint_count,
        "judge_audit_csv": str(audit_csv_path),
        "action_contract": action_contract,
        "reliability_gate": reliability_gate,
        "manipulation_success_delta_pp": round(float(manip_delta_pp), 3),
        "claim_mode": claim_mode,
        "claim_passed": claim_passed,
        "claim_failure_reasons": claim_failure_reasons,
        "deferred_claims": [
            {
                "name": "openvla_in_loop",
                "status": "deferred",
                "reason": "eval_policy.headline_scope=wm_only; OpenVLA claim path intentionally deferred.",
            }
        ],
        "confidence_intervals": _build_confidence_intervals(baseline_scores, adapted_scores),
        "heldout_manifest_hash": _manifest_hash(shared_manifest_path),
        "num_unresolved_manip_tasks_dropped": int(unresolved_manip_tasks_dropped),
        "num_rejected_task_start_assignments": int(num_rejected_task_start_assignments),
        "low_score_breakdown": _build_low_score_breakdown(all_scores),
    }
    write_json(metrics, eval_dir / "policy_eval_report.json")
    detail_lines = list(scoring_failures[:5])
    if (
        bool(config.eval_policy.reliability.enforce_stage_success)
        and not bool(reliability_gate.get("passed", False))
    ):
        detail_lines.append(f"Reliability gate failed: {reliability_gate.get('reason', '')}".strip())

    return StageResult(
        stage_name="s4_policy_eval",
        status=(
            "success"
            if all_scores
            and (not claim_mode or claim_passed)
            and (
                not bool(config.eval_policy.reliability.enforce_stage_success)
                or bool(reliability_gate.get("passed", False))
            )
            else "failed"
        ),
        elapsed_seconds=0,
        outputs={
            "eval_dir": str(eval_dir),
            "scores_path": str(eval_dir / "vlm_scores.json"),
            "report_path": str(eval_dir / "policy_eval_report.json"),
            "shared_task_start_manifest": str(shared_manifest_path),
            "judge_audit_csv": str(audit_csv_path),
        },
        metrics=metrics,
        detail="\n".join(line for line in detail_lines if line),
    )


def _extract_initial_frames(render_manifest: dict) -> List[np.ndarray]:
    """Extract first frames from rendered video clips."""
    import cv2

    frames = []
    for clip in render_manifest.get("clips", []):
        video_path = Path(clip["video_path"])
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames


def _build_task_list(config: ValidationConfig, facility: FacilityConfig) -> tuple[List[str], int]:
    tasks = list(config.eval_policy.tasks or [])
    for task in config.eval_policy.manipulation_tasks:
        if task not in tasks:
            tasks.append(task)

    hint_tasks: List[str] = []
    if facility.task_hints_path is not None:
        try:
            hint_tasks = tasks_from_task_hints(
                facility.task_hints_path,
                profile="dreamdojo",
            )
        except Exception as exc:
            logger.warning("Failed loading task hints from %s: %s", facility.task_hints_path, exc)
    for task in hint_tasks:
        if task not in tasks:
            tasks.append(task)

    if not tasks:
        tasks = [
            "Navigate forward through the corridor",
            "Turn left at the intersection",
            "Approach the nearest obstacle",
        ]
    return balance_eval_tasks(tasks, profile="dreamdojo"), len(hint_tasks)


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _build_rollout_plan(tasks: List[str], num_rollouts: int) -> List[str]:
    """Build exactly num_rollouts task prompts in round-robin order."""
    if num_rollouts <= 0:
        return []
    if not tasks:
        return []
    return [tasks[i % len(tasks)] for i in range(num_rollouts)]


def _is_manipulation_task(task: str) -> bool:
    lowered = task.lower()
    keywords = ["pick", "grasp", "lift", "place", "stack", "regrasp", "tote", "bin"]
    return any(k in lowered for k in keywords)


def _is_object_grounded_manip_task(task: str) -> bool:
    if not _is_manipulation_task(task):
        return True
    lowered = str(task or "").lower()
    # Grounded object token patterns: bowl_101, trash_can_157, etc.
    return bool(re.search(r"\b[a-z0-9_]+_[0-9]{2,}\b", lowered))


def _manipulation_success_rate(scores: List[Dict]) -> float:
    if not scores:
        return 0.0
    successes = 0
    for s in scores:
        has_flags = s.get("grasp_acquired") is not None
        if has_flags:
            if (
                bool(s.get("grasp_acquired"))
                and bool(s.get("lifted_clear"))
                and bool(s.get("placed_in_target"))
            ):
                successes += 1
        elif float(s.get("task_score", 0.0)) >= 7.0:
            successes += 1
    return round(successes / len(scores), 3)


def _resolve_adapted_policy_checkpoint(
    previous_results: Dict[str, StageResult],
    work_dir: Path,
    policy_adapter=None,
) -> Path | None:
    if policy_adapter is None:
        from ..training.openvla_finetune import resolve_latest_openvla_checkpoint

        class _LegacyResolver:
            def resolve_latest_checkpoint(self, run_root: Path):
                return resolve_latest_openvla_checkpoint(run_root)

        policy_adapter = _LegacyResolver()

    prev_rl = previous_results.get("s3c_policy_rl_loop")
    if prev_rl:
        candidate = prev_rl.outputs.get("adapted_policy_checkpoint_rl") or prev_rl.outputs.get(
            "adapted_openvla_checkpoint_rl"
        )
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path

    prev = previous_results.get("s3b_policy_finetune")
    if prev:
        candidate = prev.outputs.get("adapted_policy_checkpoint") or prev.outputs.get(
            "adapted_openvla_checkpoint"
        )
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path
    rl_root = work_dir / "policy_rl_loop"
    if rl_root.exists():
        for iter_dir in sorted(rl_root.glob("iter_*"), reverse=True):
            candidate = policy_adapter.resolve_latest_checkpoint(
                iter_dir / "policy_refine" / "runs"
            )
            if candidate is not None:
                return candidate
    fallback = policy_adapter.resolve_latest_checkpoint(work_dir / "policy_finetune" / "runs")
    if fallback is not None:
        return fallback
    return None


def _build_pairwise_metrics(all_scores: List[Dict], conditions: List[str]) -> Dict:
    """Compute improvement, win rate, and p-value for each pair of conditions."""
    pairwise = {}
    for i, c1 in enumerate(conditions):
        for c2 in conditions[i + 1 :]:
            s1 = [s for s in all_scores if s["condition"] == c1]
            s2 = [s for s in all_scores if s["condition"] == c2]
            if not s1 or not s2:
                continue
            mean1 = float(np.mean([s["task_score"] for s in s1]))
            mean2 = float(np.mean([s["task_score"] for s in s2]))
            improvement = ((mean2 - mean1) / max(mean1, 1e-8)) * 100
            abs_diff = mean2 - mean1
            min_len = min(len(s1), len(s2))
            wins = sum(
                1 for a, b in zip(s1[:min_len], s2[:min_len]) if b["task_score"] > a["task_score"]
            )
            win_rate = wins / max(min_len, 1)
            p_value = None
            if min_len >= 2:
                try:
                    from scipy import stats

                    v1 = [s["task_score"] for s in s1[:min_len]]
                    v2 = [s["task_score"] for s in s2[:min_len]]
                    _, p_value = stats.ttest_rel(v1, v2)
                    p_value = float(p_value)
                except ImportError:
                    pass
            pairwise[f"{c1}_vs_{c2}"] = {
                f"{c1}_mean": round(mean1, 3),
                f"{c2}_mean": round(mean2, 3),
                "improvement_pct": round(improvement, 2),
                "absolute_difference": round(abs_diff, 3),
                "win_rate": round(win_rate, 3),
                "p_value": round(p_value, 6) if p_value is not None else None,
            }
    return pairwise


def _headline_scope(config: ValidationConfig) -> str:
    scope = (getattr(config.eval_policy, "headline_scope", "wm_only") or "wm_only").strip().lower()
    return scope if scope in {"wm_only", "dual"} else "wm_only"


def _resolve_world_action_dim_from_config(config: ValidationConfig) -> int | None:
    token = (
        config.finetune.eval_world_experiment
        or config.finetune.experiment_config
        or ""
    ).strip()
    if not token:
        return None
    if token.lower().startswith("cosmos_predict2"):
        mapping = {
            "cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame": 384,
            "cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320": 7,
            "cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_480_640_": 7,
        }
        return mapping.get(token)
    maybe = Path(token)
    if maybe.is_absolute() or "/" in token or token.endswith(".yaml"):
        candidate = maybe if maybe.is_absolute() else (config.finetune.dreamdojo_repo / "configs" / maybe)
        if candidate.suffix != ".yaml":
            yaml_candidate = candidate.with_suffix(".yaml")
            if yaml_candidate.exists():
                candidate = yaml_candidate
    else:
        stem = token.lower()
        if stem.startswith("dreamdojo_"):
            stem = stem[len("dreamdojo_") :]
        candidate = config.finetune.dreamdojo_repo / "configs" / f"{Path(stem).stem}.yaml"
    if not candidate.exists():
        return None
    try:
        text = candidate.read_text(encoding="utf-8")
    except OSError:
        return None
    import re

    match = re.search(r"^\s*action_dim\s*:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def _assignment_trace_key(assignment: dict) -> str:
    return (
        f"{assignment.get('rollout_index', 0)}::"
        f"{assignment.get('clip_index', -1)}::"
        f"{assignment.get('task', '')}"
    )


def _resolve_policy_action_dim(config: ValidationConfig) -> int | None:
    adapter = (config.policy_adapter.name or "").strip().lower()
    if adapter == "openvla_oft":
        return int(config.policy_adapter.openvla.policy_action_dim)
    if adapter == "pi05":
        return int(config.policy_adapter.pi05.policy_action_dim)
    return None


def _extract_world_action_dim(world_model) -> int | None:
    value = getattr(world_model, "expected_action_dim", None)
    if value is None:
        value = getattr(world_model, "_expected_action_dim", None)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _single_or_none(values: set[int]) -> int | None:
    if not values:
        return None
    if len(values) == 1:
        return next(iter(values))
    return None


def _build_reliability_gate(
    config: ValidationConfig,
    scores: List[Dict],
    *,
    scoring_failure_rate: float = 0.0,
) -> Dict:
    if not scores:
        return {
            "replay_pass_rate": 0.0,
            "controllability_pass_rate": 0.0,
            "passed": False,
            "reason": "No valid scored rollouts available.",
            "scoring_failure_rate": round(float(scoring_failure_rate), 6),
            "max_scoring_failure_rate": round(
                float(config.eval_policy.reliability.max_scoring_failure_rate), 6
            ),
            "num_valid_scores": 0,
        }
    replay_passes = 0
    controllability_passes = 0
    for row in scores:
        visual = float(row.get("visual_score", 0.0))
        spatial = float(row.get("spatial_score", 0.0))
        if visual >= 5.0 and spatial >= 5.0:
            replay_passes += 1
        actions = row.get("action_sequence") or []
        if len(actions) >= 2:
            try:
                arr = np.asarray(actions, dtype=np.float32)
                deltas = np.diff(arr, axis=0)
                if float(np.max(np.linalg.norm(deltas, axis=1))) > 1e-4:
                    controllability_passes += 1
            except Exception:
                pass
    replay_rate = replay_passes / max(len(scores), 1)
    ctrl_rate = controllability_passes / max(len(scores), 1)
    failure_rate_ok = float(scoring_failure_rate) <= float(
        config.eval_policy.reliability.max_scoring_failure_rate
    )
    passed = (
        replay_rate >= float(config.eval_policy.reliability.min_replay_pass_rate)
        and ctrl_rate >= float(config.eval_policy.reliability.min_controllability_pass_rate)
        and failure_rate_ok
    )
    reasons: List[str] = []
    if replay_rate < float(config.eval_policy.reliability.min_replay_pass_rate):
        reasons.append("replay_pass_rate below threshold")
    if ctrl_rate < float(config.eval_policy.reliability.min_controllability_pass_rate):
        reasons.append("controllability_pass_rate below threshold")
    if not failure_rate_ok:
        reasons.append("scoring_failure_rate above threshold")
    reason = "" if passed else "; ".join(reasons)
    return {
        "replay_pass_rate": round(float(replay_rate), 6),
        "controllability_pass_rate": round(float(ctrl_rate), 6),
        "passed": bool(passed),
        "reason": reason,
        "scoring_failure_rate": round(float(scoring_failure_rate), 6),
        "max_scoring_failure_rate": round(
            float(config.eval_policy.reliability.max_scoring_failure_rate), 6
        ),
        "num_valid_scores": len(scores),
    }


def _build_confidence_intervals(baseline_scores: List[Dict], adapted_scores: List[Dict]) -> Dict:
    min_len = min(len(baseline_scores), len(adapted_scores))
    if min_len < 2:
        return {"paired_mean_delta": None, "paired_95ci_low": None, "paired_95ci_high": None}
    b_vals = np.asarray([s["task_score"] for s in baseline_scores[:min_len]], dtype=np.float32)
    a_vals = np.asarray([s["task_score"] for s in adapted_scores[:min_len]], dtype=np.float32)
    diffs = a_vals - b_vals
    mean = float(np.mean(diffs))
    std = float(np.std(diffs, ddof=1))
    sem = std / np.sqrt(float(min_len))
    margin = 1.96 * sem
    return {
        "paired_mean_delta": round(mean, 6),
        "paired_95ci_low": round(mean - margin, 6),
        "paired_95ci_high": round(mean + margin, 6),
    }


def _build_low_score_breakdown(scores: List[Dict]) -> Dict:
    low_rows = [
        row
        for row in scores
        if float(row.get("task_score", 0.0)) <= 1.0
        or float(row.get("visual_score", 0.0)) <= 1.0
        or float(row.get("spatial_score", 0.0)) <= 1.0
    ]
    categories = {
        "ceiling_or_upside_down": 0,
        "off_target_or_not_visible": 0,
        "blur_or_indiscernible": 0,
        "missing_robot_or_target_object": 0,
    }
    examples: List[Dict] = []
    for row in low_rows:
        reasoning = str(row.get("reasoning", "")).lower()
        if any(tok in reasoning for tok in ("ceiling", "upside down", "upside-down")):
            categories["ceiling_or_upside_down"] += 1
        if any(
            tok in reasoning
            for tok in (
                "does not show",
                "do not show",
                "not visible",
                "impossible to evaluate",
                "cannot evaluate",
            )
        ):
            categories["off_target_or_not_visible"] += 1
        if any(tok in reasoning for tok in ("blur", "blurry", "indiscernible", "abstract", "unclear")):
            categories["blur_or_indiscernible"] += 1
        if any(tok in reasoning for tok in ("no robot", "target zone", "target object", "no visible")):
            categories["missing_robot_or_target_object"] += 1
        if len(examples) < 8:
            examples.append(
                {
                    "condition": row.get("condition"),
                    "task": row.get("task"),
                    "start_clip_name": row.get("start_clip_name"),
                    "reasoning_excerpt": str(row.get("reasoning", ""))[:240],
                }
            )
    return {
        "num_low_score_rows": len(low_rows),
        "category_counts": categories,
        "examples": examples,
    }


def _manifest_hash(path: Path) -> str:
    try:
        payload = json.dumps(read_json(path), sort_keys=True, separators=(",", ":"))
    except Exception:
        payload = str(path)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()
