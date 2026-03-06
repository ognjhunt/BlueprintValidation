"""Pinned fixed-world claim benchmark loading and hydration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, List

from ..common import read_json
from .task_start_selector import normalize_video_orientation_fix


@dataclass
class ClaimBenchmark:
    task_specs: List[Dict[str, object]]
    tasks: List[str]
    assignments: List[dict]
    manifest_hash: str


def claim_benchmark_manifest_hash(path: Path) -> str:
    try:
        payload = json.dumps(read_json(path), sort_keys=True, separators=(",", ":"))
    except Exception:
        payload = str(path)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_pinned_claim_benchmark(
    *,
    benchmark_path: Path,
    render_manifest: dict,
    video_orientation_fix: str = "none",
) -> ClaimBenchmark:
    if not benchmark_path.exists():
        raise ValueError(f"Claim benchmark manifest not found: {benchmark_path}")
    payload = read_json(benchmark_path)
    if not isinstance(payload, dict):
        raise ValueError("Claim benchmark manifest must be a JSON object.")
    version = int(payload.get("version", 1) or 1)
    if version != 1:
        raise ValueError(f"Unsupported claim benchmark manifest version: {version}")

    task_specs = _validate_task_specs(payload.get("task_specs"))
    spec_by_id = {str(spec["task_spec_id"]).strip(): dict(spec) for spec in task_specs}
    tasks = [str(spec["task_prompt"]).strip() for spec in task_specs]

    clips = list(render_manifest.get("clips", []) or [])
    clip_by_name = {
        str(clip.get("clip_name", "")).strip(): dict(clip)
        for clip in clips
        if str(clip.get("clip_name", "")).strip()
    }
    clip_by_index = {
        int(clip.get("clip_index", -1)): dict(clip)
        for clip in clips
        if int(clip.get("clip_index", -1)) >= 0
    }

    assignments_raw = payload.get("assignments")
    if not isinstance(assignments_raw, list) or not assignments_raw:
        raise ValueError("Claim benchmark manifest must include non-empty assignments.")
    assignments: List[dict] = []
    seen_rollout_indices: set[int] = set()
    normalized_fix = normalize_video_orientation_fix(video_orientation_fix)

    for default_idx, raw_assignment in enumerate(assignments_raw):
        if not isinstance(raw_assignment, dict):
            raise ValueError("Claim benchmark assignments must be objects.")
        rollout_index = int(raw_assignment.get("rollout_index", default_idx))
        if rollout_index in seen_rollout_indices:
            raise ValueError(
                f"Claim benchmark rollout_index must be unique; duplicate={rollout_index}."
            )
        seen_rollout_indices.add(rollout_index)

        task_spec_id = str(raw_assignment.get("task_spec_id", "")).strip()
        if not task_spec_id:
            raise ValueError("Claim benchmark assignment is missing task_spec_id.")
        task_spec = spec_by_id.get(task_spec_id)
        if task_spec is None:
            raise ValueError(
                f"Claim benchmark assignment references unknown task_spec_id='{task_spec_id}'."
            )

        clip = _resolve_clip(
            raw_assignment=raw_assignment,
            clip_by_name=clip_by_name,
            clip_by_index=clip_by_index,
        )
        start_clip_id = str(raw_assignment.get("start_clip_id", "")).strip()
        start_region_id = str(raw_assignment.get("start_region_id", "")).strip()
        if not start_clip_id:
            raise ValueError("Claim benchmark assignment is missing start_clip_id.")
        if not start_region_id:
            raise ValueError("Claim benchmark assignment is missing start_region_id.")

        assignment = {
            "rollout_index": rollout_index,
            "task_spec_id": task_spec_id,
            "task": str(task_spec["task_prompt"]).strip(),
            "task_kind": str(task_spec.get("task_family", "")).strip() or _infer_task_kind(task_spec),
            "clip_index": int(clip.get("clip_index", -1)),
            "clip_name": str(clip.get("clip_name", "")).strip() or start_clip_id,
            "path_type": str(raw_assignment.get("path_type", clip.get("path_type", "unknown"))),
            "video_path": str(clip.get("video_path", "")),
            "initial_camera": dict(clip.get("initial_camera", {}) or {}),
            "path_context": dict(clip.get("path_context", {}) or {}),
            "target_instance_id": raw_assignment.get(
                "target_instance_id",
                task_spec.get("target_instance_id"),
            ),
            "target_label": raw_assignment.get("target_label", task_spec.get("target_label")),
            "target_grounded": bool(raw_assignment.get("target_grounded", True)),
            "assignment_quality_score": round(
                float(raw_assignment.get("assignment_quality_score", 1.0)),
                6,
            ),
            "assignment_reject_reason": raw_assignment.get("assignment_reject_reason"),
            "video_orientation_fix": normalized_fix,
            "start_clip_id": start_clip_id,
            "start_region_id": start_region_id,
        }
        assignments.append(assignment)

    assignments.sort(key=lambda item: int(item.get("rollout_index", 0)))
    return ClaimBenchmark(
        task_specs=task_specs,
        tasks=tasks,
        assignments=assignments,
        manifest_hash=claim_benchmark_manifest_hash(benchmark_path),
    )


def _validate_task_specs(raw_specs: object) -> List[Dict[str, object]]:
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("Claim benchmark manifest must include non-empty task_specs.")
    task_specs: List[Dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, dict):
            raise ValueError("Claim benchmark task_specs entries must be objects.")
        task_spec_id = str(raw_spec.get("task_spec_id", "")).strip()
        task_prompt = str(raw_spec.get("task_prompt", "")).strip()
        if not task_spec_id:
            raise ValueError("Claim benchmark task spec is missing task_spec_id.")
        if not task_prompt:
            raise ValueError("Claim benchmark task spec is missing task_prompt.")
        if task_spec_id in seen_ids:
            raise ValueError(
                f"Claim benchmark task_spec_id must be unique; duplicate='{task_spec_id}'."
            )
        if task_prompt in seen_prompts:
            raise ValueError(
                f"Claim benchmark task_prompt must be unique; duplicate='{task_prompt}'."
            )
        seen_ids.add(task_spec_id)
        seen_prompts.add(task_prompt)
        task_specs.append(dict(raw_spec))
    return task_specs


def _resolve_clip(
    *,
    raw_assignment: dict,
    clip_by_name: Dict[str, dict],
    clip_by_index: Dict[int, dict],
) -> dict:
    clip_name = str(raw_assignment.get("clip_name", "")).strip()
    if clip_name:
        clip = clip_by_name.get(clip_name)
        if clip is None:
            raise ValueError(f"Claim benchmark clip_name '{clip_name}' was not found in render_manifest.")
        return clip
    if "clip_index" not in raw_assignment:
        raise ValueError("Claim benchmark assignment must include clip_name or clip_index.")
    clip_index = int(raw_assignment.get("clip_index", -1))
    clip = clip_by_index.get(clip_index)
    if clip is None:
        raise ValueError(f"Claim benchmark clip_index '{clip_index}' was not found in render_manifest.")
    return clip


def _infer_task_kind(task_spec: dict) -> str:
    family = str(task_spec.get("task_family", "")).strip().lower()
    if family in {"navigation", "articulation", "manipulation"}:
        return family
    return "other"
