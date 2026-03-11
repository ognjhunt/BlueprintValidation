"""Stage 4: Policy evaluation — baseline vs site-adapted DreamDojo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..intake_metadata import (
    resolve_intake_lineage,
    resolve_scene_memory_runtime_metadata,
    summarize_scene_memory_runtime,
)
from ..evaluation.openvla_runner import (
    load_dreamdojo_world_model,
)
from ..evaluation.judge_audit import write_judge_audit_csv
from ..evaluation.action_overlay import overlay_scripted_trace_on_video
from ..evaluation.claim_benchmark import (
    claim_benchmark_alignment_failures,
    claim_benchmark_strictness_failures,
    load_pinned_claim_benchmark,
)
from ..evaluation.claim_protocol import (
    build_claim_split_payload,
    checkpoint_content_hash,
    claim_manifest_payload,
    claim_protocol_enabled,
    deterministic_claim_task_failures,
    validate_claim_split_payload,
)
from ..evaluation.rollout_reliability import (
    build_reliability_gate as shared_build_reliability_gate,
    single_or_none as shared_single_or_none,
)
from ..evaluation.policy_eval_metrics import (
    attach_claim_row_metadata as _attach_claim_row_metadata,
    build_confidence_intervals as _build_confidence_intervals,
    build_low_score_breakdown as _build_low_score_breakdown,
    build_pairwise_metrics as _build_pairwise_metrics,
    is_manipulation_task as _is_manipulation_task,
    is_object_grounded_manip_task as _is_object_grounded_manip_task,
    manipulation_success_rate as _manipulation_success_rate,
)
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
from ..evaluation.video_orientation import (
    normalize_video_orientation_fix,
    transform_video_orientation,
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
from ..evaluation.stats_utils import paired_ttest_p_value
from ..evaluation.rollout_utils import run_rollout_with_adapter
from ..policy_adapters import get_policy_adapter
from .base import PipelineStage
from .render_backend import resolve_stage1_render_manifest_source

logger = get_logger("stages.s4_policy_eval")


def _resolve_stage_lineage(
    *,
    config: ValidationConfig,
    facility: FacilityConfig,
    work_dir: Path,
    render_manifest: dict,
    previous_results: Dict[str, StageResult],
) -> tuple[Dict[str, object], Dict[str, object]]:
    intake_lineage = resolve_intake_lineage(facility)
    if isinstance(render_manifest.get("intake_lineage"), dict):
        intake_lineage = dict(render_manifest.get("intake_lineage", {}) or {})
    runtime_summary = summarize_scene_memory_runtime(
        resolve_scene_memory_runtime_metadata(
            config,
            facility,
            work_dir=work_dir,
            previous_results=previous_results,
        )
    )
    if isinstance(render_manifest.get("scene_memory_runtime"), dict):
        runtime_summary = dict(render_manifest.get("scene_memory_runtime", {}) or {})
    return intake_lineage, runtime_summary


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
        fixed_claim_protocol = claim_protocol_enabled(config)
        if fixed_claim_protocol:
            if not bool(config.eval_policy.freeze_world_snapshot):
                return StageResult(
                    stage_name=self.name,
                    status="failed",
                    elapsed_seconds=0,
                    detail=(
                        "Fixed-world claim protocol requires "
                        "eval_policy.freeze_world_snapshot=true."
                    ),
                )
            if str(config.eval_policy.primary_endpoint).strip().lower() != "task_success":
                return StageResult(
                    stage_name=self.name,
                    status="failed",
                    elapsed_seconds=0,
                    detail=(
                        "Fixed-world claim protocol requires "
                        "eval_policy.primary_endpoint=task_success."
                    ),
                )
            for moving_stage in ("s3c_policy_rl_loop", "s3d_wm_refresh_loop"):
                stage_result = previous_results.get(moving_stage)
                if stage_result and stage_result.status == "success":
                    return StageResult(
                        stage_name=self.name,
                        status="failed",
                        elapsed_seconds=0,
                        detail=(
                            "Fixed-world claim protocol rejects refreshed-world artifacts from "
                            f"{moving_stage}. Freeze a single Stage 3 adapted checkpoint."
                        ),
                    )

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

        # Load render manifest for initial frames
        render_source = resolve_stage1_render_manifest_source(work_dir, previous_results)
        if render_source is None:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Render manifest not found. Run Stage 1 first.",
            )
        render_manifest_path = render_source.source_manifest_path
        render_manifest = read_json(render_source.source_manifest_path)
        intake_lineage, runtime_summary = _resolve_stage_lineage(
            config=config,
            facility=facility,
            work_dir=work_dir,
            render_manifest=render_manifest,
            previous_results=previous_results,
        )

        requested_rollouts = int(config.eval_policy.num_rollouts)
        shared_manifest_path = eval_dir / "shared_task_start_manifest.json"
        selector_config = {
            "min_assignment_quality_score": float(config.eval_policy.min_assignment_quality_score),
            "require_object_grounded_manip_tasks": bool(
                config.eval_policy.require_object_grounded_manip_tasks
            ),
        }
        claim_benchmark_path = (
            Path(facility.claim_benchmark_path)
            if facility.claim_benchmark_path is not None
            else None
        )
        claim_benchmark_manifest_hash = ""
        task_specs: List[Dict[str, object]] = []
        task_specs_by_prompt: Dict[str, Dict[str, object]] = {}
        tasks: List[str] = []
        hint_count = 0
        unresolved_manip_tasks_dropped = 0
        shared_manifest = load_shared_task_start_manifest(shared_manifest_path)
        reused_shared_manifest = False
        rollout_assignments: List[dict] = []
        if fixed_claim_protocol:
            if claim_benchmark_path is None:
                return StageResult(
                    stage_name=self.name,
                    status="failed",
                    elapsed_seconds=0,
                    detail=(
                        "Fixed-world claim protocol requires facility.claim_benchmark_path to "
                        "point to a pinned benchmark manifest."
                    ),
                )
            benchmark_alignment_failures = claim_benchmark_alignment_failures(
                benchmark_path=claim_benchmark_path,
                render_manifest=render_manifest,
            )
            if benchmark_alignment_failures:
                return StageResult(
                    stage_name=self.name,
                    status="failed",
                    elapsed_seconds=0,
                    metrics={
                        "claim_protocol": "fixed_same_facility_uplift",
                        "claim_outcome": "INCONCLUSIVE",
                        "headline_eligible": False,
                        "claim_failure_reasons": benchmark_alignment_failures,
                    },
                    detail=benchmark_alignment_failures[0],
                )
            try:
                claim_benchmark = load_pinned_claim_benchmark(
                    benchmark_path=claim_benchmark_path,
                    render_manifest=render_manifest,
                    video_orientation_fix=facility.video_orientation_fix,
                )
            except ValueError as exc:
                return StageResult(
                    stage_name=self.name,
                    status="failed",
                    elapsed_seconds=0,
                    detail=f"Invalid claim benchmark manifest: {exc}",
                )
            strictness_failures = claim_benchmark_strictness_failures(
                benchmark=claim_benchmark,
                min_eval_task_specs=int(config.eval_policy.claim_strictness.min_eval_task_specs),
                min_eval_start_clips=int(config.eval_policy.claim_strictness.min_eval_start_clips),
                min_common_eval_cells=int(
                    config.eval_policy.claim_strictness.min_common_eval_cells
                ),
            )
            if strictness_failures:
                return StageResult(
                    stage_name=self.name,
                    status="failed",
                    elapsed_seconds=0,
                    metrics={
                        "claim_protocol": "fixed_same_facility_uplift",
                        "claim_outcome": "INCONCLUSIVE",
                        "headline_eligible": False,
                        "claim_failure_reasons": strictness_failures,
                    },
                    detail=strictness_failures[0],
                )
            claim_benchmark_manifest_hash = claim_benchmark.manifest_hash
            task_specs = [dict(spec) for spec in claim_benchmark.task_specs]
            task_specs_by_prompt = {
                str(spec.get("task_prompt", "")).strip(): dict(spec) for spec in task_specs
            }
            tasks = list(claim_benchmark.tasks)
            planned_rollouts = len(claim_benchmark.assignments)
            if shared_manifest and shared_manifest_is_compatible(
                shared_manifest,
                facility_name=facility.name,
                render_manifest_path=render_manifest_path,
                render_manifest=render_manifest,
                tasks=tasks,
                video_orientation_fix=facility.video_orientation_fix,
                selector_config=selector_config,
                benchmark_manifest_hash=claim_benchmark_manifest_hash,
            ):
                rollout_assignments = list(shared_manifest.get("assignments", []))
                reused_shared_manifest = bool(rollout_assignments)
            if not rollout_assignments:
                rollout_assignments = [
                    dict(assignment) for assignment in claim_benchmark.assignments
                ]
                save_shared_task_start_manifest(
                    path=shared_manifest_path,
                    facility_name=facility.name,
                    render_manifest_path=render_manifest_path,
                    task_profile="claim",
                    requested_rollouts=requested_rollouts,
                    planned_rollouts=planned_rollouts,
                    tasks=tasks,
                    assignments=rollout_assignments,
                    render_manifest=render_manifest,
                    video_orientation_fix=facility.video_orientation_fix,
                    selector_config=selector_config,
                    benchmark_manifest_hash=claim_benchmark_manifest_hash,
                )
            else:
                normalized_fix = str(getattr(facility, "video_orientation_fix", "none"))
                for assignment in rollout_assignments:
                    assignment.setdefault("video_orientation_fix", normalized_fix)
        else:
            tasks, hint_count = _build_task_list(config, facility)
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

            planned_rollouts = recommended_rollouts_per_condition(
                num_unique_tasks=len(tasks),
                requested=requested_rollouts,
                profile="dreamdojo",
            )
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
                    min_assignment_quality_score=float(
                        config.eval_policy.min_assignment_quality_score
                    ),
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
                normalized_fix = str(getattr(facility, "video_orientation_fix", "none"))
                for assignment in rollout_assignments:
                    assignment.setdefault("video_orientation_fix", normalized_fix)

        invalid_assignments = [
            assignment
            for assignment in rollout_assignments
            if str(assignment.get("assignment_reject_reason") or "").strip()
        ]
        num_rejected_task_start_assignments = len(invalid_assignments)
        if invalid_assignments and not fixed_claim_protocol:
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

        if not adapted_dir.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=f"Adapted checkpoint not found at {adapted_dir}. Run Stage 3 first.",
            )

        world_snapshot_hash = ""
        claim_manifest_path = eval_dir / "claim_manifest.json"
        task_specs_path = eval_dir / "task_specs.json"
        claim_split_manifest_path = eval_dir / "claim_split_manifest.json"
        if fixed_claim_protocol:
            world_snapshot_hash = checkpoint_content_hash(adapted_dir)
            deterministic_failures = deterministic_claim_task_failures(task_specs)
            if deterministic_failures:
                return StageResult(
                    stage_name=self.name,
                    status="failed",
                    elapsed_seconds=0,
                    metrics={
                        "claim_protocol": "fixed_same_facility_uplift",
                        "claim_outcome": "INCONCLUSIVE",
                        "headline_eligible": False,
                        "claim_failure_reasons": deterministic_failures,
                    },
                    detail=deterministic_failures[0],
                )
            task_specs_by_prompt = {str(spec["task_prompt"]): spec for spec in task_specs}
            try:
                claim_split_payload = build_claim_split_payload(
                    task_specs=task_specs,
                    assignments=rollout_assignments,
                    world_snapshot_hash=world_snapshot_hash,
                    train_split=float(config.rollout_dataset.train_split),
                    split_strategy=str(config.eval_policy.split_strategy),
                    min_eval_task_specs=int(
                        config.eval_policy.claim_strictness.min_eval_task_specs
                    ),
                    min_eval_start_clips=int(
                        config.eval_policy.claim_strictness.min_eval_start_clips
                    ),
                    min_common_eval_cells=int(
                        config.eval_policy.claim_strictness.min_common_eval_cells
                    ),
                )
            except ValueError as exc:
                return StageResult(
                    stage_name=self.name,
                    status="failed",
                    elapsed_seconds=0,
                    metrics={
                        "claim_protocol": "fixed_same_facility_uplift",
                        "claim_outcome": "INCONCLUSIVE",
                        "headline_eligible": False,
                        "claim_failure_reasons": [str(exc)],
                    },
                    detail=f"Failed building fixed-world claim split: {exc}",
                )
            split_failures = validate_claim_split_payload(
                payload=claim_split_payload,
                config=config,
            )
            if split_failures:
                return StageResult(
                    stage_name=self.name,
                    status="failed",
                    elapsed_seconds=0,
                    metrics={
                        "claim_protocol": "fixed_same_facility_uplift",
                        "claim_outcome": "INCONCLUSIVE",
                        "headline_eligible": False,
                        "claim_failure_reasons": split_failures,
                    },
                    detail=split_failures[0],
                )
            write_json(task_specs, task_specs_path)
            write_json(claim_split_payload, claim_split_manifest_path)
            write_json(
                claim_manifest_payload(
                    config=config,
                    facility=facility,
                    adapted_checkpoint=adapted_dir,
                    world_snapshot_hash=world_snapshot_hash,
                    task_specs_path=task_specs_path,
                    split_manifest_path=claim_split_manifest_path,
                    benchmark_manifest_path=claim_benchmark_path,
                    benchmark_manifest_hash=claim_benchmark_manifest_hash,
                ),
                claim_manifest_path,
            )
            claim_cell_lookup = {
                (int(cell.get("rollout_index", -1)), str(cell.get("task_prompt", "")).strip()): cell
                for cell in list(claim_split_payload.get("cells", []))
            }
            for assignment in rollout_assignments:
                claim_cell = claim_cell_lookup.get(
                    (
                        int(assignment.get("rollout_index", -1)),
                        str(assignment.get("task", "")).strip(),
                    )
                )
                if not claim_cell:
                    continue
                assignment["eval_cell_id"] = claim_cell.get("eval_cell_id")
                assignment["task_spec_id"] = claim_cell.get("task_spec_id")
                assignment["start_clip_id"] = claim_cell.get("start_clip_id")
                assignment["start_region_id"] = claim_cell.get("start_region_id")
                assignment["world_snapshot_hash"] = world_snapshot_hash

        num_rollouts = len(rollout_assignments)
        headline_scope = _headline_scope(config)
        claim_mode = (config.eval_policy.mode or "claim").strip().lower() == "claim"
        fixed_world_claim_mode = claim_mode and fixed_claim_protocol
        required_dim = int(config.eval_policy.required_action_dim)
        if (
            claim_mode
            and headline_scope in {"wm_uplift", "dual"}
            and config.policy_adapter.name.strip().lower() != "openvla_oft"
        ):
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Claim mode only supports policy_adapter.name=openvla_oft.",
            )
        if (
            claim_mode
            and headline_scope in {"wm_uplift", "dual"}
            and not config.eval_policy.require_native_action_compat
        ):
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Claim mode requires eval_policy.require_native_action_compat=true.",
            )
        max_steps = int(config.eval_policy.max_steps_per_rollout)
        if claim_mode:
            max_steps = min(max_steps, int(config.eval_policy.reliability.max_horizon_steps))
        min_rollout_frames = int(config.eval_policy.reliability.min_rollout_frames)
        min_rollout_steps = int(config.eval_policy.reliability.min_rollout_steps)
        fail_on_short_rollout = bool(config.eval_policy.reliability.fail_on_short_rollout)
        if fail_on_short_rollout and (max_steps + 1) < min_rollout_frames:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=(
                    "Configured rollout horizon is too short for reliability checks: "
                    f"max_steps={max_steps} -> max_frames={max_steps + 1}, "
                    f"required_min_rollout_frames={min_rollout_frames}."
                ),
            )
        if fail_on_short_rollout and max_steps < min_rollout_steps:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=(
                    "Configured rollout horizon is too short for reliability checks: "
                    f"max_steps={max_steps}, required_min_rollout_steps={min_rollout_steps}."
                ),
            )

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
                fixed_claim_protocol=fixed_claim_protocol,
                task_specs_by_prompt=task_specs_by_prompt,
                world_snapshot_hash=world_snapshot_hash,
                claim_manifest_path=claim_manifest_path,
                task_specs_path=task_specs_path,
                claim_split_manifest_path=claim_split_manifest_path,
                claim_benchmark_path=claim_benchmark_path,
                claim_benchmark_manifest_hash=claim_benchmark_manifest_hash,
                unresolved_manip_tasks_dropped=unresolved_manip_tasks_dropped,
                num_rejected_task_start_assignments=num_rejected_task_start_assignments,
                min_rollout_frames=min_rollout_frames,
                min_rollout_steps=min_rollout_steps,
                fail_on_short_rollout=fail_on_short_rollout,
                intake_lineage=intake_lineage,
                runtime_summary=runtime_summary,
            )

        policy_adapter = get_policy_adapter(config.policy_adapter)
        base_model_name, base_checkpoint = policy_adapter.base_model_ref(config.eval_policy)
        all_scores: List[Dict] = []
        scoring_failures: List[str] = []
        short_rollout_failures: List[str] = []
        short_step_rollout_failures: List[str] = []
        min_observed_rollout_frames: Optional[int] = None
        min_observed_rollout_steps: Optional[int] = None
        observed_action_dims: set[int] = set()
        observed_policy_dims: set[int] = set()
        observed_world_dims: set[int] = set()
        claim_state_failures: List[str] = []

        conditions = list(config.eval_policy.conditions)
        for condition in conditions:
            logger.info("Running %s condition rollouts", condition)
            condition_dir = eval_dir / f"{condition}_rollouts"
            condition_dir.mkdir(exist_ok=True)

            # Load policy checkpoint for this condition.
            policy_handle = policy_adapter.load_policy(
                model_name=base_model_name,
                checkpoint_path=base_checkpoint,
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
                    rollout_context=dict(assignment),
                    task_spec=task_specs_by_prompt.get(task),
                    require_native_task_state=fixed_claim_protocol,
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
                rollout_step_count = int(getattr(rollout, "num_steps", 0) or 0)
                if (
                    min_observed_rollout_steps is None
                    or rollout_step_count < min_observed_rollout_steps
                ):
                    min_observed_rollout_steps = rollout_step_count
                if fail_on_short_rollout and rollout_step_count < min_rollout_steps:
                    msg = (
                        "Rollout trajectory too short for reliable scoring: "
                        f"{clip_name} has {rollout_step_count} steps "
                        f"(required >= {min_rollout_steps})"
                    )
                    logger.warning(msg)
                    scoring_failures.append(msg)
                    short_step_rollout_failures.append(msg)
                    continue
                rollout_frame_count: Optional[int] = None
                if fail_on_short_rollout:
                    rollout_frame_count = _video_frame_count(rollout.video_path)
                    if (
                        min_observed_rollout_frames is None
                        or rollout_frame_count < min_observed_rollout_frames
                    ):
                        min_observed_rollout_frames = rollout_frame_count
                    if rollout_frame_count < min_rollout_frames:
                        msg = (
                            "Rollout video too short for reliable scoring: "
                            f"{clip_name} has {rollout_frame_count} frames "
                            f"(required >= {min_rollout_frames})"
                        )
                        logger.warning(msg)
                        scoring_failures.append(msg)
                        short_rollout_failures.append(msg)
                        continue

                try:
                    _orient_mode = normalize_video_orientation_fix(
                        str(getattr(facility, "video_orientation_fix", "none"))
                    )
                    _raw_rollout_path = rollout.video_path
                    _oriented_rollout_path: Optional[Path] = None
                    if _orient_mode != "none":
                        _oriented_rollout_path = _raw_rollout_path.with_name(
                            _raw_rollout_path.stem + f"_oriented_{_orient_mode}.mp4"
                        )
                        try:
                            transform_video_orientation(
                                input_path=_raw_rollout_path,
                                output_path=_oriented_rollout_path,
                                orientation_fix=_orient_mode,
                                force_grayscale=False,
                            )
                            _rollout_path_for_scoring = _oriented_rollout_path
                        except Exception as _oe:
                            logger.warning(
                                "orientation transform failed for %s: %s",
                                _raw_rollout_path,
                                _oe,
                            )
                            _rollout_path_for_scoring = _raw_rollout_path
                            _oriented_rollout_path = None
                    else:
                        _rollout_path_for_scoring = _raw_rollout_path
                    if _is_manipulation_task(task):
                        score = score_rollout_manipulation(
                            video_path=_rollout_path_for_scoring,
                            task_prompt=task,
                            config=config.eval_policy.vlm_judge,
                            facility_description=facility.description,
                        )
                    else:
                        score = score_rollout(
                            video_path=_rollout_path_for_scoring,
                            task_prompt=task,
                            config=config.eval_policy.vlm_judge,
                            facility_description=facility.description,
                        )
                except Exception as e:
                    msg = f"VLM scoring failed for {clip_name}: {e}"
                    logger.warning(msg)
                    scoring_failures.append(msg)
                    continue
                finally:
                    if (
                        "_oriented_rollout_path" in dir()
                        and _oriented_rollout_path is not None
                        and _oriented_rollout_path.exists()
                    ):
                        try:
                            _oriented_rollout_path.unlink()
                        except Exception:
                            pass

                all_scores.append(
                    _attach_claim_row_metadata(
                        base_row={
                            "condition": condition,
                            "task": task,
                            "rollout_index": rollout_idx,
                            "task_score": score.task_score,
                            "visual_score": score.visual_score,
                            "spatial_score": score.spatial_score,
                            "reasoning": score.reasoning,
                            "video_path": str(rollout.video_path),
                            "num_steps": rollout_step_count,
                            "rollout_frame_count": rollout_frame_count,
                            "action_sequence": getattr(rollout, "action_sequence", []),
                            "start_clip_index": clip_index,
                            "start_clip_name": clip_stub,
                            "start_path_type": str(assignment.get("path_type", "unknown")),
                            "target_instance_id": assignment.get("target_instance_id"),
                            "target_label": assignment.get("target_label"),
                            "initial_camera": dict(assignment.get("initial_camera", {}) or {}),
                            "path_context": dict(assignment.get("path_context", {}) or {}),
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
                        },
                        assignment=assignment,
                        rollout=rollout,
                        task_specs_by_prompt=task_specs_by_prompt,
                        fixed_claim_protocol=fixed_claim_protocol,
                        claim_state_failures=claim_state_failures,
                    )
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
            b_vals = [s["task_score"] for s in baseline_scores[:min_len]]
            a_vals = [s["task_score"] for s in adapted_scores[:min_len]]
            p_value = paired_ttest_p_value(b_vals, a_vals)

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
            if not fixed_world_claim_mode and float(absolute_difference) < float(
                config.eval_policy.min_absolute_difference
            ):
                claim_failure_reasons.append(
                    "Absolute task-score difference below threshold: "
                    f"{absolute_difference:.3f} < {config.eval_policy.min_absolute_difference:.3f}"
                )
            if not fixed_world_claim_mode and float(manip_delta_pp) < float(
                config.eval_policy.min_manip_success_delta_pp
            ):
                claim_failure_reasons.append(
                    "Manipulation success delta below threshold: "
                    f"{manip_delta_pp:.2f}pp < {config.eval_policy.min_manip_success_delta_pp:.2f}pp"
                )
        claim_passed = len(claim_failure_reasons) == 0

        metrics = {
            "headline_scope": headline_scope,
            "claim_protocol": ("fixed_same_facility_uplift" if fixed_claim_protocol else "none"),
            "primary_endpoint": str(config.eval_policy.primary_endpoint),
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
            "num_short_rollouts": len(short_rollout_failures),
            "num_short_step_rollouts": len(short_step_rollout_failures),
            "min_rollout_frames_required": int(min_rollout_frames),
            "min_rollout_steps_required": int(min_rollout_steps),
            "fail_on_short_rollout": bool(fail_on_short_rollout),
            "min_observed_rollout_frames": (
                int(min_observed_rollout_frames)
                if min_observed_rollout_frames is not None
                else None
            ),
            "min_observed_rollout_steps": (
                int(min_observed_rollout_steps) if min_observed_rollout_steps is not None else None
            ),
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
            "headline_eligible": False,
            "deferred_claims": [],
            "confidence_intervals": _build_confidence_intervals(baseline_scores, adapted_scores),
            "heldout_manifest_hash": _manifest_hash(shared_manifest_path),
            "num_unresolved_manip_tasks_dropped": int(unresolved_manip_tasks_dropped),
            "num_rejected_task_start_assignments": int(num_rejected_task_start_assignments),
            "low_score_breakdown": _build_low_score_breakdown(all_scores),
            "world_snapshot_hash": world_snapshot_hash or None,
            "claim_manifest_path": str(claim_manifest_path) if fixed_claim_protocol else "",
            "task_specs_path": str(task_specs_path) if fixed_claim_protocol else "",
            "claim_split_manifest_path": (
                str(claim_split_manifest_path) if fixed_claim_protocol else ""
            ),
            "claim_benchmark_path": str(claim_benchmark_path) if claim_benchmark_path else "",
            "claim_benchmark_manifest_hash": claim_benchmark_manifest_hash or "",
            "num_claim_state_failures": len(claim_state_failures),
            "intake_kind": intake_lineage.get("preferred_intake_kind"),
            "intake_lineage": intake_lineage,
            "scene_memory_runtime": runtime_summary,
            "scene_memory_runtime_backend": runtime_summary.get("selected_backend"),
        }

        write_json(metrics, eval_dir / "policy_eval_report.json")
        detail_lines = list(scoring_failures[:5])
        if fixed_claim_protocol and claim_state_failures:
            detail_lines.append(
                f"Claim protocol missing deterministic task-state evidence for "
                f"{len(claim_state_failures)} rollouts."
            )
        if bool(config.eval_policy.reliability.enforce_stage_success) and not bool(
            reliability_gate.get("passed", False)
        ):
            detail_lines.append(
                f"Reliability gate failed: {reliability_gate.get('reason', '')}".strip()
            )
        if fail_on_short_rollout and short_rollout_failures:
            detail_lines.append(
                f"Short rollout guard failed for {len(short_rollout_failures)} rollouts "
                f"(required_min_rollout_frames={min_rollout_frames})."
            )
        if fail_on_short_rollout and short_step_rollout_failures:
            detail_lines.append(
                f"Short step-rollout guard failed for {len(short_step_rollout_failures)} rollouts "
                f"(required_min_rollout_steps={min_rollout_steps})."
            )

        return StageResult(
            stage_name=self.name,
            status=(
                "success"
                if all_scores
                and (not fixed_claim_protocol or not claim_state_failures)
                and (not claim_mode or claim_passed)
                and (
                    not bool(config.eval_policy.reliability.enforce_stage_success)
                    or bool(reliability_gate.get("passed", False))
                )
                and (not fail_on_short_rollout or not short_rollout_failures)
                and (not fail_on_short_rollout or not short_step_rollout_failures)
                else "failed"
            ),
            elapsed_seconds=0,
            outputs={
                "eval_dir": str(eval_dir),
                "scores_path": str(eval_dir / "vlm_scores.json"),
                "report_path": str(eval_dir / "policy_eval_report.json"),
                "shared_task_start_manifest": str(shared_manifest_path),
                "judge_audit_csv": str(audit_csv_path),
                "intake_kind": intake_lineage.get("preferred_intake_kind"),
                "intake_lineage": intake_lineage,
                "scene_memory_runtime": runtime_summary,
                "scene_memory_runtime_backend": runtime_summary.get("selected_backend"),
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
    fixed_claim_protocol: bool,
    task_specs_by_prompt: Dict[str, Dict[str, object]],
    world_snapshot_hash: str,
    claim_manifest_path: Path,
    task_specs_path: Path,
    claim_split_manifest_path: Path,
    claim_benchmark_path: Path | None,
    claim_benchmark_manifest_hash: str,
    unresolved_manip_tasks_dropped: int,
    num_rejected_task_start_assignments: int,
    min_rollout_frames: int,
    min_rollout_steps: int,
    fail_on_short_rollout: bool,
    intake_lineage: Dict[str, object],
    runtime_summary: Dict[str, object],
) -> StageResult:
    rollout_driver = (config.eval_policy.rollout_driver or "scripted").strip().lower()
    claim_state_failures: List[str] = []
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
                text_model = (
                    getattr(text_encoder, "model", None) if text_encoder is not None else None
                )
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
    short_rollout_failures: List[str] = []
    short_step_rollout_failures: List[str] = []
    min_observed_rollout_frames: Optional[int] = None
    min_observed_rollout_steps: Optional[int] = None
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
                rollout_context=dict(assignment),
                task_prompt=task,
                task_spec=task_specs_by_prompt.get(task),
                require_native_task_state=fixed_claim_protocol,
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
            rollout_step_count = int(getattr(rollout, "num_steps", 0) or 0)
            if (
                min_observed_rollout_steps is None
                or rollout_step_count < min_observed_rollout_steps
            ):
                min_observed_rollout_steps = rollout_step_count
            if fail_on_short_rollout and rollout_step_count < min_rollout_steps:
                msg = (
                    "Rollout trajectory too short for reliable scoring: "
                    f"{clip_name} has {rollout_step_count} steps "
                    f"(required >= {min_rollout_steps})"
                )
                logger.warning(msg)
                scoring_failures.append(msg)
                short_step_rollout_failures.append(msg)
                continue
            rollout_frame_count: Optional[int] = None
            if fail_on_short_rollout:
                rollout_frame_count = _video_frame_count(rollout.video_path)
                if (
                    min_observed_rollout_frames is None
                    or rollout_frame_count < min_observed_rollout_frames
                ):
                    min_observed_rollout_frames = rollout_frame_count
                if rollout_frame_count < min_rollout_frames:
                    msg = (
                        "Rollout video too short for reliable scoring: "
                        f"{clip_name} has {rollout_frame_count} frames "
                        f"(required >= {min_rollout_frames})"
                    )
                    logger.warning(msg)
                    scoring_failures.append(msg)
                    short_rollout_failures.append(msg)
                    continue

            try:
                overlay_mode = "raw"
                scored_video_path = rollout.video_path
                overlay_video_path = None
                if _is_manipulation_task(task):
                    if str(config.eval_policy.manip_eval_mode).strip().lower() == "overlay_marker":
                        overlay_mode = "overlay_marker"
                        overlay_video_path = condition_dir / "overlay" / f"{clip_name}_overlay.mp4"
                        scored_video_path = overlay_scripted_trace_on_video(
                            input_video_path=rollout.video_path,
                            output_video_path=overlay_video_path,
                            action_sequence=trace.get("action_sequence", []),
                            target_label=str(assignment.get("target_label") or ""),
                        )
                _orient_mode_s = normalize_video_orientation_fix(
                    str(getattr(facility, "video_orientation_fix", "none"))
                )
                _oriented_scored_path: Optional[Path] = None
                if _orient_mode_s != "none":
                    _oriented_scored_path = scored_video_path.with_name(
                        scored_video_path.stem + f"_oriented_{_orient_mode_s}.mp4"
                    )
                    try:
                        transform_video_orientation(
                            input_path=scored_video_path,
                            output_path=_oriented_scored_path,
                            orientation_fix=_orient_mode_s,
                            force_grayscale=False,
                        )
                        _scored_path_for_scoring = _oriented_scored_path
                    except Exception as _oe:
                        logger.warning(
                            "orientation transform failed for %s: %s",
                            scored_video_path,
                            _oe,
                        )
                        _scored_path_for_scoring = scored_video_path
                        _oriented_scored_path = None
                else:
                    _scored_path_for_scoring = scored_video_path
                if _is_manipulation_task(task):
                    score = score_rollout_manipulation(
                        video_path=_scored_path_for_scoring,
                        task_prompt=task,
                        config=config.eval_policy.vlm_judge,
                        facility_description=facility.description,
                    )
                else:
                    score = score_rollout(
                        video_path=_scored_path_for_scoring,
                        task_prompt=task,
                        config=config.eval_policy.vlm_judge,
                        facility_description=facility.description,
                    )
            except Exception as e:
                msg = f"VLM scoring failed for {clip_name}: {e}"
                logger.warning(msg)
                scoring_failures.append(msg)
                continue
            finally:
                if (
                    "_oriented_scored_path" in dir()
                    and _oriented_scored_path is not None
                    and _oriented_scored_path.exists()
                ):
                    try:
                        _oriented_scored_path.unlink()
                    except Exception:
                        pass

            all_scores.append(
                _attach_claim_row_metadata(
                    base_row={
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
                        "num_steps": rollout_step_count,
                        "rollout_frame_count": rollout_frame_count,
                        "action_sequence": getattr(rollout, "action_sequence", []),
                        "start_clip_index": clip_index,
                        "start_clip_name": clip_stub,
                        "start_path_type": str(assignment.get("path_type", "unknown")),
                        "target_instance_id": assignment.get("target_instance_id"),
                        "target_label": assignment.get("target_label"),
                        "initial_camera": dict(assignment.get("initial_camera", {}) or {}),
                        "path_context": dict(assignment.get("path_context", {}) or {}),
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
                        "overlay_video_path": str(overlay_video_path)
                        if overlay_video_path
                        else None,
                        "overlay_mode": overlay_mode,
                    },
                    assignment=assignment,
                    rollout=rollout,
                    task_specs_by_prompt=task_specs_by_prompt,
                    fixed_claim_protocol=fixed_claim_protocol,
                    claim_state_failures=claim_state_failures,
                )
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
        b_vals = [s["task_score"] for s in baseline_scores[:min_len]]
        a_vals = [s["task_score"] for s in adapted_scores[:min_len]]
        p_value = paired_ttest_p_value(b_vals, a_vals)

    dataset_dim = _single_or_none(observed_action_dims)
    world_dim = _single_or_none(observed_world_dims)
    action_contract = {
        "policy_dim": None,
        "world_dim": world_dim,
        "dataset_dim": dataset_dim,
        "compliant": (
            world_dim is not None and dataset_dim is not None and int(world_dim) == int(dataset_dim)
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
            claim_failure_reasons.append(f"Action contract failed: {action_contract.get('reason')}")
        if not reliability_gate["passed"]:
            claim_failure_reasons.append(
                "Reliability gate failed: "
                f"replay_pass_rate={reliability_gate['replay_pass_rate']:.3f}, "
                f"controllability_pass_rate={reliability_gate['controllability_pass_rate']:.3f}"
            )
        if not fixed_claim_protocol and float(absolute_difference) < float(
            config.eval_policy.min_absolute_difference
        ):
            claim_failure_reasons.append(
                "Absolute task-score difference below threshold: "
                f"{absolute_difference:.3f} < {config.eval_policy.min_absolute_difference:.3f}"
            )
        if not fixed_claim_protocol and float(manip_delta_pp) < float(
            config.eval_policy.min_manip_success_delta_pp
        ):
            claim_failure_reasons.append(
                "Manipulation success delta below threshold: "
                f"{manip_delta_pp:.2f}pp < {config.eval_policy.min_manip_success_delta_pp:.2f}pp"
            )
    claim_passed = len(claim_failure_reasons) == 0

    metrics = {
        "headline_scope": "wm_only",
        "rollout_driver": rollout_driver,
        "claim_protocol": "fixed_same_facility_uplift" if fixed_claim_protocol else "none",
        "primary_endpoint": str(config.eval_policy.primary_endpoint),
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
        "num_short_rollouts": len(short_rollout_failures),
        "num_short_step_rollouts": len(short_step_rollout_failures),
        "min_rollout_frames_required": int(min_rollout_frames),
        "min_rollout_steps_required": int(min_rollout_steps),
        "fail_on_short_rollout": bool(fail_on_short_rollout),
        "min_observed_rollout_frames": (
            int(min_observed_rollout_frames) if min_observed_rollout_frames is not None else None
        ),
        "min_observed_rollout_steps": (
            int(min_observed_rollout_steps) if min_observed_rollout_steps is not None else None
        ),
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
        "headline_eligible": False,
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
        "world_snapshot_hash": world_snapshot_hash or None,
        "claim_manifest_path": str(claim_manifest_path) if fixed_claim_protocol else "",
        "task_specs_path": str(task_specs_path) if fixed_claim_protocol else "",
        "claim_split_manifest_path": (
            str(claim_split_manifest_path) if fixed_claim_protocol else ""
        ),
        "claim_benchmark_path": str(claim_benchmark_path) if claim_benchmark_path else "",
        "claim_benchmark_manifest_hash": claim_benchmark_manifest_hash or "",
        "num_claim_state_failures": len(claim_state_failures),
        "intake_kind": intake_lineage.get("preferred_intake_kind"),
        "intake_lineage": intake_lineage,
        "scene_memory_runtime": runtime_summary,
        "scene_memory_runtime_backend": runtime_summary.get("selected_backend"),
    }
    write_json(metrics, eval_dir / "policy_eval_report.json")
    detail_lines = list(scoring_failures[:5])
    if fixed_claim_protocol and claim_state_failures:
        detail_lines.append(
            f"Claim protocol missing deterministic task-state evidence for "
            f"{len(claim_state_failures)} rollouts."
        )
    if bool(config.eval_policy.reliability.enforce_stage_success) and not bool(
        reliability_gate.get("passed", False)
    ):
        detail_lines.append(
            f"Reliability gate failed: {reliability_gate.get('reason', '')}".strip()
        )
    if fail_on_short_rollout and short_rollout_failures:
        detail_lines.append(
            f"Short rollout guard failed for {len(short_rollout_failures)} rollouts "
            f"(required_min_rollout_frames={min_rollout_frames})."
        )
    if fail_on_short_rollout and short_step_rollout_failures:
        detail_lines.append(
            f"Short step-rollout guard failed for {len(short_step_rollout_failures)} rollouts "
            f"(required_min_rollout_steps={min_rollout_steps})."
        )

    return StageResult(
        stage_name="s4_policy_eval",
        status=(
            "success"
            if all_scores
            and (not fixed_claim_protocol or not claim_state_failures)
            and (not claim_mode or claim_passed)
            and (
                not bool(config.eval_policy.reliability.enforce_stage_success)
                or bool(reliability_gate.get("passed", False))
            )
            and (not fail_on_short_rollout or not short_rollout_failures)
            and (not fail_on_short_rollout or not short_step_rollout_failures)
            else "failed"
        ),
        elapsed_seconds=0,
            outputs={
                "eval_dir": str(eval_dir),
                "scores_path": str(eval_dir / "vlm_scores.json"),
                "report_path": str(eval_dir / "policy_eval_report.json"),
                "shared_task_start_manifest": str(shared_manifest_path),
                "judge_audit_csv": str(audit_csv_path),
                "intake_kind": intake_lineage.get("preferred_intake_kind"),
                "intake_lineage": intake_lineage,
                "scene_memory_runtime": runtime_summary,
                "scene_memory_runtime_backend": runtime_summary.get("selected_backend"),
            },
        metrics=metrics,
        detail="\n".join(line for line in detail_lines if line),
    )


def _video_frame_count(video_path: Path) -> int:
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 0:
            cap.release()
            return frame_count
        # Fallback when container metadata is missing.
        decoded = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            decoded += 1
        cap.release()
        return int(decoded)
    except Exception:
        return 0


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
    task_profile = "claim" if claim_protocol_enabled(config) else "dreamdojo"
    tasks = list(config.eval_policy.tasks or [])
    for task in config.eval_policy.manipulation_tasks:
        if task not in tasks:
            tasks.append(task)

    hint_tasks: List[str] = []
    if facility.task_hints_path is not None:
        try:
            hint_tasks = tasks_from_task_hints(
                facility.task_hints_path,
                profile=task_profile,
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
    return balance_eval_tasks(tasks, profile=task_profile), len(hint_tasks)


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


def _headline_scope(config: ValidationConfig) -> str:
    scope = (getattr(config.eval_policy, "headline_scope", "wm_only") or "wm_only").strip().lower()
    return scope if scope in {"wm_only", "wm_uplift", "dual"} else "wm_only"


def _resolve_world_action_dim_from_config(config: ValidationConfig) -> int | None:
    token = (
        config.finetune.eval_world_experiment or config.finetune.experiment_config or ""
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
        candidate = (
            maybe if maybe.is_absolute() else (config.finetune.dreamdojo_repo / "configs" / maybe)
        )
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
    return shared_single_or_none(values)


def _build_reliability_gate(
    config: ValidationConfig,
    scores: List[Dict],
    *,
    scoring_failure_rate: float = 0.0,
) -> Dict:
    return dict(
        shared_build_reliability_gate(
            config,
            scores,
            scoring_failure_rate=scoring_failure_rate,
        )
    )


def _manifest_hash(path: Path) -> str:
    try:
        payload = json.dumps(read_json(path), sort_keys=True, separators=(",", ":"))
    except Exception:
        payload = str(path)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()
