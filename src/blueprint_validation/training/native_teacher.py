"""Native benchmark-hydrated teacher and correction rollout generation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..common import get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..evaluation.claim_benchmark import load_pinned_claim_benchmark
from ..evaluation.openvla_runner import load_dreamdojo_world_model
from ..evaluation.rollout_state_proxy import (
    build_correction_action_candidates,
    build_teacher_action_candidates,
    teacher_demo_quota,
)
from ..evaluation.scripted_rollout_driver import run_scripted_rollout
from ..evaluation.task_start_selector import load_initial_frames_for_assignments
from ..evaluation.task_success import evaluate_task_success
from ..stages.render_backend import resolve_stage1_render_manifest_source

logger = get_logger("training.native_teacher")


def generate_teacher_rollouts(
    *,
    config: ValidationConfig,
    facility: FacilityConfig,
    work_dir: Path,
    output_dir: Path,
    condition: str,
    mode: str,
    max_steps: int,
) -> Tuple[List[Dict], Dict[str, object]]:
    task_specs_by_id, train_assignments = _load_train_claim_assignments(
        config=config,
        facility=facility,
        work_dir=work_dir,
    )
    if not train_assignments:
        return [], {"num_candidates": 0, "num_successful": 0, "num_assignments": 0}

    frame_cache = load_initial_frames_for_assignments(train_assignments)
    if not frame_cache:
        return [], {"num_candidates": 0, "num_successful": 0, "num_assignments": len(train_assignments)}

    output_dir.mkdir(parents=True, exist_ok=True)
    world_model = _load_world_model_for_condition(
        config=config,
        work_dir=work_dir,
        condition=condition,
    )
    rows: List[Dict] = []
    total_candidates = 0
    for assignment in train_assignments:
        task_spec_id = str(assignment.get("task_spec_id", "")).strip()
        task_spec = task_specs_by_id.get(task_spec_id)
        if task_spec is None:
            continue
        init_frame = frame_cache.get(int(assignment.get("clip_index", -1)))
        if init_frame is None:
            continue
        candidates = build_teacher_action_candidates(
            task_spec=task_spec,
            rollout_context=assignment,
            max_steps=max_steps,
            mode=mode,
            action_dim=int(config.eval_policy.required_action_dim),
        )
        candidate_limit, success_limit = teacher_demo_quota(str(task_spec.get("task_family", "")))
        total_candidates += min(len(candidates), candidate_limit)
        successes = 0
        seen_motion_families: set[str] = set()
        for candidate_idx, candidate in enumerate(candidates[:candidate_limit]):
            motion_family = str(candidate.get("motion_family", "")).strip() or f"candidate_{candidate_idx:02d}"
            if motion_family in seen_motion_families:
                continue
            rollout = run_scripted_rollout(
                world_model=world_model,
                initial_frame=init_frame,
                action_sequence=list(candidate.get("action_sequence", [])),
                output_dir=output_dir,
                clip_name=_teacher_clip_name(
                    assignment=assignment,
                    prefix=f"{condition}_{mode}",
                    candidate_idx=candidate_idx,
                ),
                trace_id=_trace_id(
                    assignment=assignment,
                    motion_family=motion_family,
                    prefix=f"{condition}_{mode}",
                ),
                rollout_context=dict(assignment),
                task_prompt=str(task_spec.get("task_prompt", assignment.get("task", ""))),
                task_spec=dict(task_spec),
            )
            row = _teacher_row_from_rollout(
                assignment=assignment,
                task_spec=task_spec,
                rollout=rollout,
                rollout_index=100000
                + (1000 if condition == "baseline" else 0)
                + int(assignment.get("rollout_index", 0)) * 10
                + candidate_idx,
                condition=condition,
                source_type="planner_demo",
                parent_rollout_id=None,
                motion_family=motion_family,
            )
            if not bool(row.get("task_success", False)):
                continue
            seen_motion_families.add(motion_family)
            rows.append(row)
            successes += 1
            if successes >= success_limit:
                break

    summary = {
        "num_assignments": len(train_assignments),
        "num_candidates": total_candidates,
        "num_successful": len(rows),
        "condition": condition,
        "mode": mode,
        "rows_manifest_path": str(output_dir / f"{condition}_{mode}_teacher_rows.json"),
    }
    write_json({"rows": rows}, output_dir / f"{condition}_{mode}_teacher_rows.json")
    write_json(summary, output_dir / f"{condition}_{mode}_teacher_summary.json")
    return rows, summary


def generate_correction_rollouts(
    *,
    config: ValidationConfig,
    facility: FacilityConfig,
    work_dir: Path,
    output_dir: Path,
    failed_rows: List[Dict],
    condition: str,
    mode: str,
    max_steps: int,
) -> Tuple[List[Dict], Dict[str, object]]:
    task_specs_by_id, train_assignments = _load_train_claim_assignments(
        config=config,
        facility=facility,
        work_dir=work_dir,
    )
    assignment_by_eval = {
        str(item.get("eval_cell_id", "")).strip(): dict(item)
        for item in train_assignments
        if str(item.get("eval_cell_id", "")).strip()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    world_model = _load_world_model_for_condition(
        config=config,
        work_dir=work_dir,
        condition=condition,
    )
    rows: List[Dict] = []
    attempted = 0
    for parent_idx, failed_row in enumerate(failed_rows):
        eval_cell_id = str(failed_row.get("eval_cell_id", "")).strip()
        assignment = assignment_by_eval.get(eval_cell_id)
        task_spec_id = str(failed_row.get("task_spec_id", "")).strip()
        task_spec = task_specs_by_id.get(task_spec_id)
        if assignment is None or task_spec is None:
            continue
        recovery_step, candidates = build_correction_action_candidates(
            task_spec=task_spec,
            rollout_context=assignment,
            state_trace=list(failed_row.get("state_trace", []) or []),
            max_steps=max_steps,
            mode=mode,
            action_dim=int(config.eval_policy.required_action_dim),
        )
        init_frame = _read_video_frame(
            video_path=Path(str(failed_row.get("video_path", "")).strip()),
            frame_index=recovery_step,
        )
        if init_frame is None:
            continue
        attempted += 1
        parent_rollout_id = _parent_rollout_id(failed_row)
        for candidate_idx, candidate in enumerate(candidates):
            motion_family = str(candidate.get("motion_family", "")).strip() or f"recovery_{candidate_idx:02d}"
            rollout = run_scripted_rollout(
                world_model=world_model,
                initial_frame=init_frame,
                action_sequence=list(candidate.get("action_sequence", [])),
                output_dir=output_dir,
                clip_name=f"{condition}_{mode}_recovery_{parent_idx:04d}_{candidate_idx:02d}",
                trace_id=f"recovery_{hashlib.md5((parent_rollout_id + motion_family).encode('utf-8')).hexdigest()[:12]}",
                rollout_context=dict(assignment),
                task_prompt=str(task_spec.get("task_prompt", assignment.get("task", ""))),
                task_spec=dict(task_spec),
            )
            row = _teacher_row_from_rollout(
                assignment=assignment,
                task_spec=task_spec,
                rollout=rollout,
                rollout_index=300000 + parent_idx * 10 + candidate_idx,
                condition=condition,
                source_type="planner_correction",
                parent_rollout_id=parent_rollout_id,
                motion_family=motion_family,
            )
            row["recovery_start_step"] = int(recovery_step)
            if bool(row.get("task_success", False)):
                rows.append(row)
                break
    summary = {
        "num_failed_rows_considered": len(failed_rows),
        "num_attempted_recoveries": attempted,
        "num_successful_corrections": len(rows),
        "condition": condition,
        "mode": mode,
        "rows_manifest_path": str(output_dir / f"{condition}_{mode}_correction_rows.json"),
    }
    write_json({"rows": rows}, output_dir / f"{condition}_{mode}_correction_rows.json")
    write_json(summary, output_dir / f"{condition}_{mode}_correction_summary.json")
    return rows, summary


def _load_train_claim_assignments(
    *,
    config: ValidationConfig,
    facility: FacilityConfig,
    work_dir: Path,
) -> Tuple[Dict[str, dict], List[dict]]:
    claim_manifest_path = work_dir / "policy_eval" / "claim_manifest.json"
    claim_split_path = work_dir / "policy_eval" / "claim_split_manifest.json"
    render_source = resolve_stage1_render_manifest_source(work_dir, previous_results={})
    if (
        facility.claim_benchmark_path is None
        or not claim_manifest_path.exists()
        or not claim_split_path.exists()
        or render_source is None
    ):
        return {}, []
    render_manifest = read_json(render_source.source_manifest_path)
    benchmark = load_pinned_claim_benchmark(
        benchmark_path=Path(facility.claim_benchmark_path),
        render_manifest=render_manifest,
        video_orientation_fix=getattr(facility, "video_orientation_fix", "none"),
    )
    split_payload = read_json(claim_split_path)
    cells = {
        str(cell.get("eval_cell_id", "")).strip(): dict(cell)
        for cell in list(split_payload.get("cells", []) or [])
        if str(cell.get("eval_cell_id", "")).strip()
    }
    train_ids = {
        str(value).strip()
        for value in list(split_payload.get("train_eval_cell_ids", []) or [])
        if str(value).strip()
    }
    annotated: List[dict] = []
    cell_by_key = {
        (
            int(cell.get("rollout_index", -1)),
            str(cell.get("task_spec_id", "")).strip(),
            str(cell.get("start_clip_id", "")).strip(),
        ): cell
        for cell in cells.values()
    }
    for assignment in benchmark.assignments:
        key = (
            int(assignment.get("rollout_index", -1)),
            str(assignment.get("task_spec_id", "")).strip(),
            str(assignment.get("start_clip_id", "")).strip(),
        )
        cell = cell_by_key.get(key)
        if cell is None:
            continue
        eval_cell_id = str(cell.get("eval_cell_id", "")).strip()
        if eval_cell_id not in train_ids:
            continue
        item = dict(assignment)
        item["eval_cell_id"] = eval_cell_id
        item["world_snapshot_hash"] = str(split_payload.get("world_snapshot_hash", "") or "")
        annotated.append(item)
    task_specs_by_id = {
        str(spec.get("task_spec_id", "")).strip(): dict(spec)
        for spec in benchmark.task_specs
        if str(spec.get("task_spec_id", "")).strip()
    }
    return task_specs_by_id, annotated


def _load_world_model_for_condition(
    *,
    config: ValidationConfig,
    work_dir: Path,
    condition: str,
):
    adapted_checkpoint = None
    if str(condition).strip().lower() != "baseline":
        adapted_checkpoint = work_dir / "finetune" / "adapted_checkpoint"
        if not adapted_checkpoint.exists():
            adapted_checkpoint = work_dir / "finetune" / "lora_weights"
    return load_dreamdojo_world_model(
        checkpoint_path=config.finetune.dreamdojo_checkpoint,
        adapted_checkpoint=adapted_checkpoint if adapted_checkpoint and adapted_checkpoint.exists() else None,
        configured_experiment=(
            config.finetune.eval_world_experiment or config.finetune.experiment_config
        ),
        dreamdojo_repo=config.finetune.dreamdojo_repo,
        device="cuda" if _has_cuda() else "cpu",
    )


def _teacher_row_from_rollout(
    *,
    assignment: dict,
    task_spec: dict,
    rollout,
    rollout_index: int,
    condition: str,
    source_type: str,
    parent_rollout_id: str | None,
    motion_family: str,
) -> Dict:
    row = {
        "condition": condition,
        "task": str(task_spec.get("task_prompt", assignment.get("task", ""))),
        "rollout_index": int(rollout_index),
        "video_path": str(rollout.video_path),
        "num_steps": int(getattr(rollout, "num_steps", 0) or 0),
        "action_sequence": list(getattr(rollout, "action_sequence", []) or []),
        "state_trace": list(getattr(rollout, "state_trace", []) or []),
        "is_manipulation_task": str(task_spec.get("task_family", "")).strip().lower() == "manipulation",
        "eval_cell_id": str(assignment.get("eval_cell_id", "") or ""),
        "task_spec_id": str(task_spec.get("task_spec_id", "") or ""),
        "start_clip_id": str(assignment.get("start_clip_id", "") or assignment.get("clip_name", "") or ""),
        "start_region_id": str(assignment.get("start_region_id", "") or ""),
        "start_frame_hash": str(assignment.get("start_frame_hash", "") or ""),
        "world_snapshot_hash": str(assignment.get("world_snapshot_hash", "") or ""),
        "target_instance_id": assignment.get("target_instance_id") or task_spec.get("target_instance_id"),
        "target_label": assignment.get("target_label") or task_spec.get("target_label"),
        "initial_camera": dict(assignment.get("initial_camera", {}) or {}),
        "path_context": dict(assignment.get("path_context", {}) or {}),
        "source_type": source_type,
        "sim_backend": "native_benchmark_sim",
        "success_source": "sim_ground_truth",
        "parent_rollout_id": parent_rollout_id,
        "motion_family": motion_family,
    }
    success_payload = evaluate_task_success(
        task_spec=task_spec,
        rollout_row=row,
        state_trace=row["state_trace"],
    )
    row.update(success_payload)
    row["task_score"] = 10.0 if bool(row.get("task_success", False)) else 0.0
    row["visual_score"] = 10.0
    row["spatial_score"] = 10.0
    row["reasoning"] = "native_benchmark_sim"
    row["grasp_acquired"] = _latest_bool(row["state_trace"], "grasp_acquired")
    row["lifted_clear"] = _latest_bool(row["state_trace"], "lifted_clear")
    row["placed_in_target"] = _latest_bool(row["state_trace"], "placed_in_target")
    row["stable_after_place"] = _latest_bool(row["state_trace"], "stable_after_place")
    return row


def _teacher_clip_name(*, assignment: dict, prefix: str, candidate_idx: int) -> str:
    return (
        f"{prefix}_teacher_{int(assignment.get('rollout_index', 0)):04d}_"
        f"{candidate_idx:02d}_{str(assignment.get('task_spec_id', 'task')).replace('/', '_')}"
    )


def _trace_id(*, assignment: dict, motion_family: str, prefix: str) -> str:
    digest = hashlib.md5(
        f"{prefix}|{assignment.get('rollout_index', 0)}|{assignment.get('task_spec_id', '')}|{motion_family}".encode("utf-8")
    ).hexdigest()
    return f"trace_{digest[:12]}"


def _latest_bool(state_trace: List[Dict], key: str) -> bool | None:
    for state in reversed(state_trace):
        if key in state:
            return bool(state.get(key))
    return None


def _parent_rollout_id(row: Dict) -> str:
    eval_cell_id = str(row.get("eval_cell_id", "")).strip()
    if eval_cell_id:
        return eval_cell_id
    return f"{int(row.get('rollout_index', 0))}::{str(row.get('task', '')).strip()}"


def _read_video_frame(video_path: Path, frame_index: int) -> np.ndarray | None:
    if not video_path.exists():
        return None
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        return None
    return None


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False
