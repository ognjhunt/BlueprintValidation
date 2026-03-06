"""Helpers for the fixed-world same-facility claim protocol."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
import re
from typing import Dict, Iterable, List, Tuple

from ..common import get_logger, read_json
from ..config import FacilityConfig, ValidationConfig

logger = get_logger("evaluation.claim_protocol")


def claim_protocol_enabled(config: ValidationConfig) -> bool:
    return (
        str(getattr(config.eval_policy, "claim_protocol", "none") or "none").strip().lower()
        == "fixed_same_facility_uplift"
    )


def checkpoint_content_hash(path: Path) -> str:
    """Return a stable recursive content hash for a file or directory checkpoint."""
    if not path.exists():
        return ""
    digest = hashlib.sha1()
    if path.is_file():
        digest.update(path.name.encode("utf-8"))
        _update_digest_with_file(digest, path)
        return digest.hexdigest()

    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = str(file_path.relative_to(path)).replace("\\", "/")
        digest.update(rel.encode("utf-8"))
        _update_digest_with_file(digest, file_path)
    return digest.hexdigest()


def _update_digest_with_file(digest: "hashlib._Hash", file_path: Path) -> None:
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)


def build_task_specs(
    *,
    config: ValidationConfig,
    facility: FacilityConfig,
    tasks: List[str],
) -> List[Dict[str, object]]:
    target_index = _load_target_index(facility.task_hints_path)
    specs: List[Dict[str, object]] = []
    for task in tasks:
        prompt = str(task).strip()
        family = _task_family(prompt)
        target = _resolve_task_target(prompt, target_index)
        goal_target = _resolve_goal_target(
            prompt=prompt,
            family=family,
            target=target,
            target_index=target_index,
            facility=facility,
        )
        target_label = target.get("label") if isinstance(target, dict) else None
        target_instance_id = target.get("instance_id") if isinstance(target, dict) else None
        target_center_xyz = _center_xyz(target)
        goal_target_label = goal_target.get("label") if isinstance(goal_target, dict) else None
        goal_target_instance_id = (
            goal_target.get("instance_id") if isinstance(goal_target, dict) else None
        )
        goal_point_xyz = _goal_point_xyz(
            family=family,
            prompt=prompt,
            target=target,
            goal_target=goal_target,
            facility=facility,
        )
        goal_region_id = _goal_region_id_for_task(prompt, family, target)
        predicate = _success_predicate_for_family(
            prompt=prompt,
            family=family,
            target=target,
            target_center_xyz=target_center_xyz,
            goal_point_xyz=goal_point_xyz,
            goal_region_id=goal_region_id,
        )
        task_spec_id = _stable_protocol_id(
            "task_spec",
            {
                "task_prompt": prompt,
                "family": family,
                "target_label": target_label,
                "target_instance_id": target_instance_id,
                "target_center_xyz": target_center_xyz,
                "goal_target_label": goal_target_label,
                "goal_target_instance_id": goal_target_instance_id,
                "goal_point_xyz": goal_point_xyz,
                "goal_region_id": goal_region_id,
                "predicate": predicate,
            },
        )
        specs.append(
            {
                "task_spec_id": task_spec_id,
                "task_prompt": prompt,
                "task_family": family,
                "target_label": target_label,
                "target_instance_id": target_instance_id,
                "target_center_xyz": target_center_xyz,
                "goal_target_label": goal_target_label,
                "goal_target_instance_id": goal_target_instance_id,
                "goal_point_xyz": goal_point_xyz,
                "goal_region_id": goal_region_id,
                "forbidden_events": _forbidden_events_for_family(family),
                "success_predicate": predicate,
                "primary_endpoint": str(config.eval_policy.primary_endpoint),
            }
        )
    return specs


def build_claim_split_payload(
    *,
    task_specs: List[Dict[str, object]],
    assignments: List[dict],
    world_snapshot_hash: str,
    train_split: float,
    split_strategy: str,
    min_eval_task_specs: int | None = None,
    min_eval_start_clips: int | None = None,
    min_common_eval_cells: int | None = None,
) -> Dict[str, object]:
    specs_by_prompt = {str(spec["task_prompt"]): spec for spec in task_specs}
    cells: List[Dict[str, object]] = []
    for assignment in assignments:
        prompt = str(assignment.get("task", "")).strip()
        spec = specs_by_prompt.get(prompt)
        if not spec:
            continue
        start_clip_id = _start_clip_id(assignment)
        start_region_id = _start_region_id(assignment)
        target_instance_id = str(spec.get("target_instance_id") or assignment.get("target_instance_id") or "").strip() or None
        eval_cell_id = _stable_protocol_id(
            "eval_cell",
            {
                "task_spec_id": spec["task_spec_id"],
                "start_clip_id": start_clip_id,
                "start_region_id": start_region_id,
                "target_instance_id": target_instance_id,
                "world_snapshot_hash": world_snapshot_hash,
                "rollout_index": int(assignment.get("rollout_index", 0)),
            },
        )
        cells.append(
            {
                "eval_cell_id": eval_cell_id,
                "task_spec_id": spec["task_spec_id"],
                "task_prompt": prompt,
                "task_family": spec["task_family"],
                "start_clip_id": start_clip_id,
                "start_region_id": start_region_id,
                "start_clip_index": int(assignment.get("clip_index", -1)),
                "start_frame_hash": str(assignment.get("start_frame_hash", "")).strip(),
                "target_instance_id": target_instance_id,
                "target_label": spec.get("target_label"),
                "world_snapshot_hash": world_snapshot_hash,
                "rollout_index": int(assignment.get("rollout_index", 0)),
            }
        )

    if str(split_strategy).strip().lower() != "disjoint_tasks_and_starts":
        train_ids = [str(cell["eval_cell_id"]) for cell in cells]
        return {
            "split_strategy": str(split_strategy),
            "world_snapshot_hash": world_snapshot_hash,
            "cells": cells,
            "train_eval_cell_ids": train_ids,
            "eval_eval_cell_ids": [],
            "unused_eval_cell_ids": [],
            "heldout_task_spec_ids": [],
            "heldout_task_families": [],
            "heldout_start_clip_ids": [],
            "heldout_start_region_ids": [],
        }

    if not cells:
        return {
            "split_strategy": "disjoint_tasks_and_starts",
            "world_snapshot_hash": world_snapshot_hash,
            "cells": [],
            "train_eval_cell_ids": [],
            "eval_eval_cell_ids": [],
            "unused_eval_cell_ids": [],
            "heldout_task_spec_ids": [],
            "heldout_task_families": [],
            "heldout_start_clip_ids": [],
            "heldout_start_region_ids": [],
            "task_split_axis": "task_spec_id",
            "start_split_axis": "start_clip_id",
        }

    task_axis = "task_spec_id"
    task_ids = sorted(
        {
            str(cell.get("task_spec_id", "")).strip()
            for cell in cells
            if str(cell.get("task_spec_id", "")).strip()
        }
    )
    start_axis = "start_clip_id"
    start_ids = sorted(
        {
            str(cell.get("start_clip_id", "")).strip()
            for cell in cells
            if str(cell.get("start_clip_id", "")).strip()
        }
    )
    if len(task_ids) < 2:
        raise ValueError("Claim split requires at least two disjoint task specs.")
    if len(start_ids) < 2:
        raise ValueError("Claim split requires at least two disjoint start clips.")
    holdout_task_count = max(
        _holdout_count(len(task_ids), train_split),
        max(1, int(min_eval_task_specs or 1)),
    )
    holdout_start_count = max(
        _holdout_count(len(start_ids), train_split),
        max(1, int(min_eval_start_clips or 1)),
    )
    target_eval_cell_count = max(1, int(min_common_eval_cells or 1))

    heldout_task_ids, heldout_start_ids = _plan_disjoint_holdouts(
        cells=cells,
        task_axis=task_axis,
        start_axis=start_axis,
        required_task_count=holdout_task_count,
        required_start_count=holdout_start_count,
        target_eval_cell_count=target_eval_cell_count,
    )

    train_cells: List[Dict[str, object]] = []
    eval_cells: List[Dict[str, object]] = []
    unused_cells: List[Dict[str, object]] = []
    for cell in cells:
        task_holdout = str(cell.get(task_axis, "")) in heldout_task_ids
        start_holdout = str(cell.get(start_axis, "")) in heldout_start_ids
        if task_holdout and start_holdout:
            eval_cells.append(cell)
        elif not task_holdout and not start_holdout:
            train_cells.append(cell)
        else:
            unused_cells.append(cell)

    if not eval_cells or not train_cells:
        raise ValueError(
            "Claim split could not produce non-empty disjoint train/eval cells under "
            "split_strategy=disjoint_tasks_and_starts."
        )

    if not {
        str(cell.get("task_spec_id", ""))
        for cell in eval_cells
    }.isdisjoint({str(cell.get("task_spec_id", "")) for cell in train_cells}):
        raise ValueError("Claim split leaked task specs between train and eval cells.")
    if not {
        str(cell.get(start_axis, ""))
        for cell in eval_cells
    }.isdisjoint({str(cell.get(start_axis, "")) for cell in train_cells}):
        raise ValueError("Claim split leaked start groups between train and eval cells.")

    heldout_task_spec_ids = {
        str(cell["task_spec_id"])
        for cell in eval_cells
        if str(cell.get("task_spec_id", "")).strip()
    }
    heldout_start_clip_ids = {
        str(cell["start_clip_id"])
        for cell in eval_cells
        if str(cell.get("start_clip_id", "")).strip()
    }
    heldout_start_region_ids = {
        str(cell["start_region_id"])
        for cell in eval_cells
        if str(cell.get("start_region_id", "")).strip()
    }

    heldout_task_families = sorted(
        {
            str(cell["task_family"])
            for cell in eval_cells
            if str(cell.get("task_family", "")).strip()
        }
    )
    return {
        "split_strategy": "disjoint_tasks_and_starts",
        "world_snapshot_hash": world_snapshot_hash,
        "cells": cells,
        "train_eval_cell_ids": [str(cell["eval_cell_id"]) for cell in train_cells],
        "eval_eval_cell_ids": [str(cell["eval_cell_id"]) for cell in eval_cells],
        "unused_eval_cell_ids": [str(cell["eval_cell_id"]) for cell in unused_cells],
        "heldout_task_spec_ids": sorted(heldout_task_spec_ids),
        "heldout_task_families": heldout_task_families,
        "heldout_start_clip_ids": sorted(heldout_start_clip_ids),
        "heldout_start_region_ids": sorted(heldout_start_region_ids),
        "task_split_axis": task_axis,
        "start_split_axis": start_axis,
    }


def _plan_disjoint_holdouts(
    *,
    cells: List[Dict[str, object]],
    task_axis: str,
    start_axis: str,
    required_task_count: int,
    required_start_count: int,
    target_eval_cell_count: int,
) -> Tuple[set[str], set[str]]:
    task_ids = sorted(
        {
            str(cell.get(task_axis, "")).strip()
            for cell in cells
            if str(cell.get(task_axis, "")).strip()
        }
    )
    start_ids = sorted(
        {
            str(cell.get(start_axis, "")).strip()
            for cell in cells
            if str(cell.get(start_axis, "")).strip()
        }
    )
    required_task_count = min(max(1, int(required_task_count)), max(1, len(task_ids) - 1))
    required_start_count = min(max(1, int(required_start_count)), max(1, len(start_ids) - 1))
    target_eval_cell_count = max(1, int(target_eval_cell_count))

    cells_by_task: Dict[str, List[Dict[str, object]]] = {}
    cells_by_start: Dict[str, List[Dict[str, object]]] = {}
    for cell in cells:
        task_id = str(cell.get(task_axis, "")).strip()
        start_id = str(cell.get(start_axis, "")).strip()
        if task_id:
            cells_by_task.setdefault(task_id, []).append(cell)
        if start_id:
            cells_by_start.setdefault(start_id, []).append(cell)

    ranked_tasks = sorted(task_ids, key=lambda task_id: (-len(cells_by_task.get(task_id, [])), task_id))
    task_pool_limit = min(len(ranked_tasks), max(required_task_count + 8, 14))
    task_pool = ranked_tasks[:task_pool_limit]
    max_task_count = min(len(task_pool), max(required_task_count, required_task_count + 8))
    max_start_count = min(len(start_ids) - 1, max(required_start_count, required_start_count + 12))

    best: Tuple[Tuple[int, int, int, int, int, int], set[str], set[str]] | None = None

    for task_count in range(required_task_count, max_task_count + 1):
        for task_subset_tuple in combinations(task_pool, task_count):
            task_subset = set(task_subset_tuple)
            start_counts: Dict[str, int] = {}
            for cell in cells:
                task_id = str(cell.get(task_axis, "")).strip()
                start_id = str(cell.get(start_axis, "")).strip()
                if task_id in task_subset and start_id:
                    start_counts[start_id] = start_counts.get(start_id, 0) + 1
            ordered_starts = sorted(
                start_ids,
                key=lambda start_id: (-start_counts.get(start_id, 0), -len(cells_by_start.get(start_id, [])), start_id),
            )
            for start_count in range(required_start_count, max_start_count + 1):
                start_subset = set(ordered_starts[:start_count])
                eval_count = 0
                train_count = 0
                for cell in cells:
                    task_id = str(cell.get(task_axis, "")).strip()
                    start_id = str(cell.get(start_axis, "")).strip()
                    task_holdout = task_id in task_subset
                    start_holdout = start_id in start_subset
                    if task_holdout and start_holdout:
                        eval_count += 1
                    elif not task_holdout and not start_holdout:
                        train_count += 1
                if eval_count <= 0 or train_count <= 0:
                    continue
                score = (
                    1 if eval_count >= target_eval_cell_count else 0,
                    min(eval_count, target_eval_cell_count),
                    eval_count,
                    train_count,
                    -len(task_subset),
                    -len(start_subset),
                )
                if best is None or score > best[0]:
                    best = (score, set(task_subset), set(start_subset))
                if eval_count >= target_eval_cell_count and train_count >= target_eval_cell_count:
                    break

    if best is not None:
        return best[1], best[2]

    return (set(task_ids[-required_task_count:]), set(start_ids[-required_start_count:]))


def claim_manifest_payload(
    *,
    config: ValidationConfig,
    facility: FacilityConfig,
    adapted_checkpoint: Path,
    world_snapshot_hash: str,
    task_specs_path: Path,
    split_manifest_path: Path,
) -> Dict[str, object]:
    return {
        "claim_protocol": "fixed_same_facility_uplift",
        "facility_name": facility.name,
        "facility_description": facility.description,
        "primary_endpoint": str(config.eval_policy.primary_endpoint),
        "freeze_world_snapshot": bool(config.eval_policy.freeze_world_snapshot),
        "world_snapshot_path": str(adapted_checkpoint),
        "world_snapshot_hash": world_snapshot_hash,
        "headline_scope": str(config.eval_policy.headline_scope),
        "task_specs_path": str(task_specs_path),
        "split_manifest_path": str(split_manifest_path),
        "training_seeds": [int(v) for v in list(config.eval_policy.claim_replication.training_seeds)],
        "control_arms": [str(v) for v in list(config.policy_compare.control_arms)],
        "claim_strictness": {
            "min_eval_task_specs": int(config.eval_policy.claim_strictness.min_eval_task_specs),
            "min_eval_start_clips": int(config.eval_policy.claim_strictness.min_eval_start_clips),
            "min_common_eval_cells": int(config.eval_policy.claim_strictness.min_common_eval_cells),
            "min_positive_training_seeds": int(
                config.eval_policy.claim_strictness.min_positive_training_seeds
            ),
            "p_value_threshold": float(config.eval_policy.claim_strictness.p_value_threshold),
            "require_site_specific_advantage": bool(
                config.eval_policy.claim_strictness.require_site_specific_advantage
            ),
            "site_vs_generic_min_lift_pp": float(
                config.eval_policy.claim_strictness.site_vs_generic_min_lift_pp
            ),
        },
        "config_hash": _stable_protocol_id("config", config.eval_policy.__dict__),
    }


def paired_eval_key(row: Dict[str, object]) -> str:
    eval_cell_id = str(row.get("eval_cell_id", "")).strip()
    if eval_cell_id:
        return eval_cell_id
    return f"{int(row.get('rollout_index', 0))}::{str(row.get('task', '')).strip()}"


def group_rows_by_eval_cell(rows: Iterable[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(paired_eval_key(row), []).append(row)
    return grouped


def deterministic_claim_task_failures(task_specs: List[Dict[str, object]]) -> List[str]:
    failures: List[str] = []
    for spec in task_specs:
        predicate = dict(spec.get("success_predicate", {}) or {})
        predicate_type = str(predicate.get("type", "")).strip().lower()
        if predicate_type in {
            "navigation_reach_and_hold",
            "articulation_open_close_sequence",
            "manipulation_pick_place_stable",
        }:
            continue
        failures.append(
            f"Task '{spec.get('task_prompt', '')}' lacks a deterministic claim endpoint "
            f"(predicate={predicate_type or 'missing'})."
        )
    return failures


def validate_claim_split_payload(
    *,
    payload: Dict[str, object],
    config: ValidationConfig,
) -> List[str]:
    strictness = config.eval_policy.claim_strictness
    eval_cells = [
        dict(cell)
        for cell in list(payload.get("cells", []) or [])
        if str(cell.get("eval_cell_id", "")).strip()
        and str(cell.get("eval_cell_id", "")).strip()
        in {
            str(v).strip()
            for v in list(payload.get("eval_eval_cell_ids", []) or [])
            if str(v).strip()
        }
    ]
    failures: List[str] = []
    heldout_task_specs = {
        str(cell.get("task_spec_id", "")).strip()
        for cell in eval_cells
        if str(cell.get("task_spec_id", "")).strip()
    }
    heldout_start_clips = {
        str(cell.get("start_clip_id", "")).strip()
        for cell in eval_cells
        if str(cell.get("start_clip_id", "")).strip()
    }
    if len(heldout_task_specs) < int(strictness.min_eval_task_specs):
        failures.append(
            "Claim split does not contain enough heldout task specs: "
            f"{len(heldout_task_specs)} < {int(strictness.min_eval_task_specs)}."
        )
    if len(heldout_start_clips) < int(strictness.min_eval_start_clips):
        failures.append(
            "Claim split does not contain enough heldout start clips: "
            f"{len(heldout_start_clips)} < {int(strictness.min_eval_start_clips)}."
        )
    if len(eval_cells) < int(strictness.min_common_eval_cells):
        failures.append(
            "Claim split does not contain enough heldout eval cells: "
            f"{len(eval_cells)} < {int(strictness.min_common_eval_cells)}."
        )
    return failures


def _task_family(prompt: str) -> str:
    lowered = prompt.lower()
    if any(token in lowered for token in ("pick up", "grasp", "lift", "place", "stack", "regrasp")):
        return "manipulation"
    if any(token in lowered for token in ("open and close", "open ", "close ", "turn on", "turn off")):
        return "articulation"
    if any(token in lowered for token in ("navigate", "approach", "go to", "move toward")):
        return "navigation"
    return "other"


def _load_target_index(task_hints_path: Path | None) -> Dict[str, Dict[str, object]]:
    index = {"by_instance": {}, "by_label": {}}
    if task_hints_path is None or not task_hints_path.exists():
        return index
    try:
        payload = read_json(task_hints_path)
    except Exception as exc:
        logger.warning("Failed loading task hints for claim protocol: %s", exc)
        return index
    for key in ("manipulation_candidates", "articulation_hints", "navigation_hints"):
        for entry in payload.get(key, []):
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("label", "")).strip()
            instance_id = str(entry.get("instance_id", "")).strip()
            center_xyz = _entry_center_xyz(entry)
            item = {
                "label": label,
                "instance_id": instance_id,
                "center_xyz": center_xyz,
            }
            if instance_id:
                index["by_instance"][instance_id] = item
            label_key = _normalize_label(label)
            if label_key:
                index["by_label"].setdefault(label_key, []).append(item)
    return index


def _resolve_task_target(prompt: str, target_index: Dict[str, Dict[str, object]]) -> Dict[str, object] | None:
    family = _task_family(prompt)
    token = _extract_primary_task_token(prompt, family)
    if not token:
        return None
    return _resolve_token(token, target_index)


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_")


def _entry_center_xyz(entry: Dict[str, object]) -> List[float] | None:
    bbox = entry.get("boundingBox") or entry.get("obb")
    if not isinstance(bbox, dict):
        return None
    center = bbox.get("center")
    if not isinstance(center, (list, tuple)) or len(center) < 3:
        return None
    try:
        values = [float(center[0]), float(center[1]), float(center[2])]
    except Exception:
        return None
    return values if all((value == value) and abs(value) < 1.0e9 for value in values) else None


def _center_xyz(target: Dict[str, object] | None) -> List[float] | None:
    if not isinstance(target, dict):
        return None
    center = target.get("center_xyz")
    if not isinstance(center, (list, tuple)) or len(center) < 3:
        return None
    try:
        return [float(center[0]), float(center[1]), float(center[2])]
    except Exception:
        return None


def _extract_primary_task_token(prompt: str, family: str) -> str:
    lowered = prompt.lower().strip()
    if family == "manipulation":
        patterns = (
            r"(?:pick up|grasp|lift)\s+(?:the\s+)?([a-z0-9_ ]+?)(?:\s+from\b|\s+and\b|\s+into\b|\s+in\b|\s+on\b|\s+to\b|$)",
        )
    elif family == "articulation":
        patterns = (
            r"(?:open and close|close and open|open|close|turn on|turn off)\s+(?:the\s+)?([a-z0-9_ ]+?)(?:\s+and\b|$)",
        )
    elif family == "navigation":
        patterns = (
            r"(?:navigate to|approach|go to|move toward)\s+(?:the\s+)?([a-z0-9_ ]+?)(?:$)",
        )
    else:
        patterns = tuple()
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        token = str(match.group(1) or "").strip()
        if token:
            return token
    return ""


def _resolve_token(token: str, target_index: Dict[str, Dict[str, object]]) -> Dict[str, object] | None:
    cleaned = str(token or "").strip().lower()
    if not cleaned:
        return None
    if "_" in cleaned:
        maybe_instance = cleaned.split("_")[-1]
        item = target_index["by_instance"].get(maybe_instance)
        if item is not None:
            return item
    options = target_index["by_label"].get(_normalize_label(cleaned), [])
    if options:
        return options[0]
    # Fall back to progressively shorter label prefixes.
    words = [word for word in re.split(r"[^a-z0-9]+", cleaned) if word]
    for cutoff in range(len(words), 0, -1):
        options = target_index["by_label"].get(_normalize_label(" ".join(words[:cutoff])), [])
        if options:
            return options[0]
    return None


def _resolve_goal_target(
    *,
    prompt: str,
    family: str,
    target: Dict[str, object] | None,
    target_index: Dict[str, Dict[str, object]],
    facility: FacilityConfig,
) -> Dict[str, object] | None:
    if family in {"navigation", "articulation"}:
        return target
    if family != "manipulation":
        return None
    lowered = prompt.lower().strip()
    match = re.search(
        r"(?:place|put|drop)\s+(?:it|the [a-z0-9_ ]+|[a-z0-9_ ]+?)\s+(?:in|into|on|onto|to)\s+(?:the\s+)?([a-z0-9_ ]+?)(?:$)",
        lowered,
    )
    if match:
        goal_token = str(match.group(1) or "").strip()
        if "target zone" not in goal_token:
            resolved = _resolve_token(goal_token, target_index)
            if resolved is not None:
                return resolved
    if facility.manipulation_zones:
        zone = facility.manipulation_zones[0]
        return {
            "label": zone.name,
            "instance_id": f"zone::{_normalize_label(zone.name)}",
            "center_xyz": [float(v) for v in list(zone.target_point[:3])],
        }
    return None


def _goal_point_xyz(
    *,
    family: str,
    prompt: str,
    target: Dict[str, object] | None,
    goal_target: Dict[str, object] | None,
    facility: FacilityConfig,
) -> List[float] | None:
    if family == "navigation":
        return _center_xyz(target)
    if family == "articulation":
        return _center_xyz(target)
    if family != "manipulation":
        return None
    goal_center = _center_xyz(goal_target)
    if goal_center is not None:
        return goal_center
    if facility.manipulation_zones:
        zone = facility.manipulation_zones[0]
        return [float(v) for v in list(zone.target_point[:3])]
    target_center = _center_xyz(target)
    if target_center is None:
        return None
    # Stable deterministic fallback when no explicit placement zone exists.
    return [target_center[0] + 0.35, target_center[1], target_center[2] + 0.10]


def _goal_region_id_for_task(
    prompt: str,
    family: str,
    target: Dict[str, object] | None,
) -> str:
    if family == "manipulation":
        return "target_zone"
    if family == "articulation":
        return f"joint::{str(target.get('instance_id') if target else '').strip() or _normalize_label(prompt)}"
    if family == "navigation":
        return f"region::{_normalize_label(str(target.get('label') if target else prompt))}"
    return f"goal::{_normalize_label(prompt)}"


def _success_predicate_for_family(
    *,
    prompt: str,
    family: str,
    target: Dict[str, object] | None,
    target_center_xyz: List[float] | None,
    goal_point_xyz: List[float] | None,
    goal_region_id: str,
) -> Dict[str, object]:
    target_instance_id = str(target.get("instance_id") if target else "").strip() or None
    if family == "navigation":
        return {
            "type": "navigation_reach_and_hold",
            "goal_region_id": goal_region_id,
            "goal_point_xyz": goal_point_xyz,
            "goal_radius_m": 0.75,
            "hold_steps": 2,
            "collision_key": "invalid_collision",
        }
    if family == "articulation":
        return {
            "type": "articulation_open_close_sequence",
            "target_instance_id": target_instance_id,
            "target_center_xyz": target_center_xyz,
            "joint_key": "joint_position",
            "open_threshold": 0.8,
            "close_threshold": 0.2,
            "sequence": _articulation_sequence(prompt),
        }
    if family == "manipulation":
        return {
            "type": "manipulation_pick_place_stable",
            "target_instance_id": target_instance_id,
            "target_center_xyz": target_center_xyz,
            "goal_point_xyz": goal_point_xyz,
            "goal_region_id": goal_region_id,
            "grasp_radius_m": 0.22,
            "goal_radius_m": 0.30,
            "lift_height_m": 0.14,
            "require_stable_after_place": True,
        }
    return {
        "type": "task_score_threshold",
        "threshold": 7.0,
    }


def _forbidden_events_for_family(family: str) -> List[str]:
    if family == "navigation":
        return ["invalid_collision"]
    if family == "manipulation":
        return ["invalid_collision", "target_dropped"]
    if family == "articulation":
        return ["invalid_collision"]
    return []


def _start_clip_id(assignment: dict) -> str:
    clip_index = int(assignment.get("clip_index", -1))
    clip_name = str(assignment.get("clip_name", f"clip_{clip_index:03d}")).strip()
    return clip_name or f"clip_{clip_index:03d}"


def _start_region_id(assignment: dict) -> str:
    path_type = str(assignment.get("path_type", "unknown")).strip() or "unknown"
    target_instance_id = str(assignment.get("target_instance_id", "")).strip()
    clip_index = int(assignment.get("clip_index", -1))
    token = target_instance_id or f"clip_{clip_index:03d}"
    return f"{path_type}:{token}"


def _holdout_count(total: int, train_split: float) -> int:
    if total <= 1:
        return 0
    eval_ratio = max(0.0, min(1.0, 1.0 - float(train_split)))
    count = int(round(total * eval_ratio))
    return max(1, min(total - 1, count))


def _stable_protocol_id(prefix: str, payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _articulation_sequence(prompt: str) -> List[str]:
    lowered = prompt.lower()
    if "close and open" in lowered or "turn off and on" in lowered:
        return ["close", "open"]
    if "open and close" in lowered or "turn on and off" in lowered:
        return ["open", "close"]
    if "turn off" in lowered or re.search(r"\bclose\b", lowered):
        return ["close"]
    if "turn on" in lowered or re.search(r"\bopen\b", lowered):
        return ["open"]
    return ["open", "close"]
