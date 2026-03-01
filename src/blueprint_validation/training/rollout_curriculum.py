"""Curriculum helpers for mixed rollout sampling and world-refresh composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class RolloutBucketThresholds:
    task_score_threshold: float = 7.0
    near_miss_min_task_score: float = 5.0
    near_miss_max_task_score: float = 6.99


def pair_id_for_rollout(entry: Dict[str, Any]) -> str:
    return f"{entry.get('rollout_index')}::{entry.get('task', '')}"


def bucket_rollout(
    entry: Dict[str, Any],
    thresholds: RolloutBucketThresholds,
) -> str:
    """Classify a rollout into success / near_miss / hard_negative."""
    is_manip = bool(entry.get("is_manipulation_task", False))
    grasp = entry.get("grasp_acquired")
    lifted = entry.get("lifted_clear")
    placed = entry.get("placed_in_target")

    has_flags = grasp is not None and lifted is not None and placed is not None
    if is_manip and has_flags:
        all_true = bool(grasp) and bool(lifted) and bool(placed)
        any_true = bool(grasp) or bool(lifted) or bool(placed)
        if all_true:
            return "success"
        if any_true:
            return "near_miss"

    task_score = float(entry.get("task_score", 0.0) or 0.0)
    if task_score >= float(thresholds.task_score_threshold):
        return "success"
    if float(thresholds.near_miss_min_task_score) <= task_score <= float(
        thresholds.near_miss_max_task_score
    ):
        return "near_miss"
    return "hard_negative"


def validate_action_sequence(
    *,
    actions: List[object],
    require_consistent_action_dim: bool,
    max_action_delta_norm: float,
) -> Tuple[bool, str | None]:
    dims: set[int] = set()
    for action in actions:
        vec_size = action_vector_size(action)
        if vec_size is None:
            return False, "nonfinite"
        dims.add(vec_size)
    if require_consistent_action_dim and len(dims) != 1:
        return False, "dim_mismatch"

    try:
        arr = np.asarray(actions, dtype=np.float64)
    except Exception:
        return False, "smoothness"
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        return False, "smoothness"
    if not np.isfinite(arr).all():
        return False, "nonfinite"
    if len(arr) < 2:
        return True, None
    deltas = np.diff(arr, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    if not np.isfinite(norms).all():
        return False, "nonfinite"
    if float(np.max(norms)) > max_action_delta_norm:
        return False, "smoothness"
    return True, None


def action_vector_size(action: object) -> int | None:
    try:
        arr = np.asarray(action, dtype=np.float64)
    except Exception:
        return None
    if arr.ndim == 0:
        return 1
    return int(arr.size)


def filter_rollouts_for_curriculum(
    rollouts: List[Dict[str, Any]],
    *,
    min_steps_per_rollout: int,
    require_consistent_action_dim: bool,
    max_action_delta_norm: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    kept: List[Dict[str, Any]] = []
    counts = {
        "num_filtered_short": 0,
        "num_filtered_dim_mismatch": 0,
        "num_filtered_nonfinite": 0,
        "num_filtered_smoothness": 0,
        "num_after_action_filters": 0,
    }

    for entry in rollouts:
        actions = entry.get("action_sequence") or []
        if len(actions) < int(min_steps_per_rollout):
            counts["num_filtered_short"] += 1
            continue
        valid, reason = validate_action_sequence(
            actions=actions,
            require_consistent_action_dim=bool(require_consistent_action_dim),
            max_action_delta_norm=float(max_action_delta_norm),
        )
        if not valid:
            if reason == "dim_mismatch":
                counts["num_filtered_dim_mismatch"] += 1
            elif reason == "nonfinite":
                counts["num_filtered_nonfinite"] += 1
            else:
                counts["num_filtered_smoothness"] += 1
            continue
        kept.append(dict(entry))

    counts["num_after_action_filters"] = len(kept)
    return kept, counts


def sample_policy_curriculum(
    rollouts: List[Dict[str, Any]],
    cfg: Any,
    *,
    seed: int,
) -> Dict[str, Any]:
    """Sample deterministic train/eval policy curriculum with bucket quotas."""
    filtered, filter_counts = filter_rollouts_for_curriculum(
        rollouts,
        min_steps_per_rollout=int(cfg.min_steps_per_rollout),
        require_consistent_action_dim=bool(cfg.require_consistent_action_dim),
        max_action_delta_norm=float(cfg.max_action_delta_norm),
    )

    adapted = [r for r in filtered if str(r.get("condition", "")) == "adapted"]
    thresholds = RolloutBucketThresholds(
        task_score_threshold=float(cfg.task_score_threshold),
        near_miss_min_task_score=float(cfg.near_miss_min_task_score),
        near_miss_max_task_score=float(cfg.near_miss_max_task_score),
    )
    for row in adapted:
        row["rollout_bucket"] = bucket_rollout(row, thresholds)

    by_bucket = {
        "success": [r for r in adapted if r.get("rollout_bucket") == "success"],
        "near_miss": [r for r in adapted if r.get("rollout_bucket") == "near_miss"],
        "hard_negative": [r for r in adapted if r.get("rollout_bucket") == "hard_negative"],
    }

    for rows in by_bucket.values():
        rows.sort(key=pair_id_for_rollout)

    total = len(adapted)
    target_train = int(total * float(cfg.train_split))
    if total > 0 and target_train <= 0:
        target_train = 1
    target_train = min(total, target_train)

    if str(cfg.selection_mode) == "success_only":
        frac_success = 1.0
        frac_near = 0.0
        frac_hard = 0.0
    elif str(cfg.selection_mode) == "success_near_miss":
        frac_near = float(cfg.near_miss_target_fraction)
        frac_hard = 0.0
        frac_success = max(0.0, 1.0 - frac_near)
    else:
        frac_near = float(cfg.near_miss_target_fraction)
        frac_hard = float(cfg.hard_negative_target_fraction)
        frac_success = max(0.0, 1.0 - frac_near - frac_hard)

    success_target = int(round(target_train * frac_success))
    near_target = int(round(target_train * frac_near))
    hard_target = int(round(target_train * frac_hard))

    rng = Random(int(seed))
    train_selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    per_task_counts: Dict[str, int] = {}
    per_task_max = int(cfg.per_task_max_episodes)

    train_selected.extend(
        _sample_bucket(
            by_bucket["success"],
            success_target,
            rng,
            selected_ids,
            per_task_counts,
            per_task_max,
        )
    )
    train_selected.extend(
        _sample_bucket(
            by_bucket["near_miss"],
            near_target,
            rng,
            selected_ids,
            per_task_counts,
            per_task_max,
        )
    )
    train_selected.extend(
        _sample_bucket(
            by_bucket["hard_negative"],
            hard_target,
            rng,
            selected_ids,
            per_task_counts,
            per_task_max,
        )
    )

    # Backfill from any remaining candidates if initial quotas underfill.
    remaining = [
        r
        for bucket in ("success", "near_miss", "hard_negative")
        for r in by_bucket[bucket]
        if pair_id_for_rollout(r) not in selected_ids
    ]
    rng.shuffle(remaining)
    for row in remaining:
        if len(train_selected) >= target_train:
            break
        if not _allow_task_quota(row, per_task_counts, per_task_max):
            continue
        pid = pair_id_for_rollout(row)
        selected_ids.add(pid)
        _increment_task_quota(row, per_task_counts)
        train_selected.append(row)

    eval_selected = [r for r in adapted if pair_id_for_rollout(r) not in selected_ids]

    # Guarantee non-empty disjoint eval when data exists.
    if adapted and not eval_selected and len(train_selected) > 1:
        moved = train_selected.pop()
        selected_ids.discard(pair_id_for_rollout(moved))
        eval_selected.append(moved)

    train_pair_ids = sorted({pair_id_for_rollout(r) for r in train_selected})
    eval_pair_ids = sorted({pair_id_for_rollout(r) for r in eval_selected})

    curriculum = {
        "selection_mode": str(cfg.selection_mode),
        "target_train_count": target_train,
        "bucket_targets": {
            "success": success_target,
            "near_miss": near_target,
            "hard_negative": hard_target,
        },
        "candidate_counts": {
            "success": len(by_bucket["success"]),
            "near_miss": len(by_bucket["near_miss"]),
            "hard_negative": len(by_bucket["hard_negative"]),
        },
        "train_bucket_counts": {
            "success": sum(1 for r in train_selected if r.get("rollout_bucket") == "success"),
            "near_miss": sum(1 for r in train_selected if r.get("rollout_bucket") == "near_miss"),
            "hard_negative": sum(
                1 for r in train_selected if r.get("rollout_bucket") == "hard_negative"
            ),
        },
        "eval_bucket_counts": {
            "success": sum(1 for r in eval_selected if r.get("rollout_bucket") == "success"),
            "near_miss": sum(1 for r in eval_selected if r.get("rollout_bucket") == "near_miss"),
            "hard_negative": sum(
                1 for r in eval_selected if r.get("rollout_bucket") == "hard_negative"
            ),
        },
        "train_pair_ids": train_pair_ids,
        "eval_pair_ids": eval_pair_ids,
    }

    return {
        "filtered_rollouts": filtered,
        "adapted_rollouts": adapted,
        "train_rollouts": train_selected,
        "eval_rollouts": eval_selected,
        "train_pair_ids": train_pair_ids,
        "eval_pair_ids": eval_pair_ids,
        "curriculum": curriculum,
        "filter_counts": filter_counts,
    }


def build_world_refresh_mix(
    stage2_manifest: Dict[str, Any] | None,
    selected_success: List[Dict[str, Any]],
    near_miss: List[Dict[str, Any]],
    hard_negative: List[Dict[str, Any]],
    cfg: Any,
    *,
    seed: int,
) -> Dict[str, Any]:
    """Build a mixed refresh manifest from Stage-2 data + rollout buckets."""
    rng = Random(int(seed))

    stage2_rows = _normalize_stage2_clips(stage2_manifest or {})
    success_rows = _normalize_rollout_rows(selected_success, source_bucket="selected")
    near_rows = _normalize_rollout_rows(near_miss, source_bucket="near_miss")
    hard_rows = _normalize_rollout_rows(hard_negative, source_bucket="hard_negative")

    all_unique = {}
    for row in stage2_rows + success_rows + near_rows + hard_rows:
        key = str(row.get("output_video_path", "")).strip()
        if not key:
            continue
        if key in all_unique:
            continue
        all_unique[key] = row

    available_total = len(all_unique)
    if available_total == 0:
        return {
            "clips": [],
            "mix_metrics": {
                "available_total": 0,
                "selected_total": 0,
                "selected_stage2": 0,
                "selected_success": 0,
                "selected_near_miss": 0,
                "selected_hard_negative": 0,
            },
        }

    floor_target = max(1, int(cfg.world_model_refresh_min_total_clips))
    cap_target = max(1, int(cfg.world_model_refresh_max_total_clips))
    desired_base = len(success_rows) + len(near_rows) + len(hard_rows)
    desired = max(floor_target, desired_base)
    target_total = min(cap_target, desired, available_total)

    stage2_target = int(round(target_total * float(cfg.world_model_refresh_stage2_fraction)))
    success_target = int(round(target_total * float(cfg.world_model_refresh_success_fraction)))
    near_target = int(round(target_total * float(cfg.world_model_refresh_near_miss_fraction)))
    hard_target = max(0, target_total - stage2_target - success_target - near_target)

    selected: List[Dict[str, Any]] = []
    used_paths: set[str] = set()
    selected.extend(_sample_paths(stage2_rows, stage2_target, rng, used_paths))
    selected.extend(_sample_paths(success_rows, success_target, rng, used_paths))
    selected.extend(_sample_paths(near_rows, near_target, rng, used_paths))
    selected.extend(_sample_paths(hard_rows, hard_target, rng, used_paths))

    leftovers = []
    for src in (stage2_rows, success_rows, near_rows, hard_rows):
        for row in src:
            path = str(row.get("output_video_path", "")).strip()
            if not path or path in used_paths:
                continue
            leftovers.append(row)
    rng.shuffle(leftovers)
    for row in leftovers:
        if len(selected) >= target_total:
            break
        path = str(row.get("output_video_path", "")).strip()
        used_paths.add(path)
        selected.append(row)

    metrics = {
        "available_total": available_total,
        "selected_total": len(selected),
        "selected_stage2": sum(1 for r in selected if r.get("source_bucket") == "stage2"),
        "selected_success": sum(1 for r in selected if r.get("source_bucket") == "selected"),
        "selected_near_miss": sum(
            1 for r in selected if r.get("source_bucket") == "near_miss"
        ),
        "selected_hard_negative": sum(
            1 for r in selected if r.get("source_bucket") == "hard_negative"
        ),
        "target_total": target_total,
        "target_stage2": stage2_target,
        "target_success": success_target,
        "target_near_miss": near_target,
        "target_hard_negative": hard_target,
    }
    return {"clips": selected, "mix_metrics": metrics}


def _sample_bucket(
    rows: List[Dict[str, Any]],
    target: int,
    rng: Random,
    selected_ids: set[str],
    per_task_counts: Dict[str, int],
    per_task_max: int,
) -> List[Dict[str, Any]]:
    rows = list(rows)
    rng.shuffle(rows)
    selected = []
    for row in rows:
        if len(selected) >= max(0, int(target)):
            break
        pid = pair_id_for_rollout(row)
        if pid in selected_ids:
            continue
        if not _allow_task_quota(row, per_task_counts, per_task_max):
            continue
        selected_ids.add(pid)
        _increment_task_quota(row, per_task_counts)
        selected.append(row)
    return selected


def _allow_task_quota(
    row: Dict[str, Any],
    per_task_counts: Dict[str, int],
    per_task_max: int,
) -> bool:
    if per_task_max <= 0:
        return True
    task = str(row.get("task", ""))
    return per_task_counts.get(task, 0) < per_task_max


def _increment_task_quota(row: Dict[str, Any], per_task_counts: Dict[str, int]) -> None:
    task = str(row.get("task", ""))
    per_task_counts[task] = per_task_counts.get(task, 0) + 1


def _sample_paths(
    rows: List[Dict[str, Any]],
    target: int,
    rng: Random,
    used_paths: set[str],
) -> List[Dict[str, Any]]:
    rows = list(rows)
    rng.shuffle(rows)
    selected = []
    for row in rows:
        if len(selected) >= max(0, int(target)):
            break
        path = str(row.get("output_video_path", "")).strip()
        if not path or path in used_paths:
            continue
        used_paths.add(path)
        selected.append(row)
    return selected


def _normalize_stage2_clips(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for clip in manifest.get("clips", []):
        output_video = str(clip.get("output_video_path") or "").strip()
        if not output_video:
            continue
        if not Path(output_video).exists():
            continue
        clip_name = str(clip.get("clip_name") or Path(output_video).stem)
        rows.append(
            {
                "clip_name": clip_name,
                "variant_name": str(clip.get("variant_name") or "stage2"),
                "prompt": str(clip.get("prompt") or ""),
                "output_video_path": output_video,
                "input_video_path": str(clip.get("input_video_path") or output_video),
                "source_bucket": "stage2",
            }
        )
    return rows


def _normalize_rollout_rows(rows: List[Dict[str, Any]], *, source_bucket: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        output_video = str(row.get("video_path") or "").strip()
        if not output_video:
            continue
        if not Path(output_video).exists():
            continue
        rollout_index = int(row.get("rollout_index", 0) or 0)
        task = str(row.get("task") or "")
        out.append(
            {
                "clip_name": f"{source_bucket}_{rollout_index:04d}",
                "variant_name": source_bucket,
                "prompt": f"{source_bucket} rollout for task: {task}",
                "output_video_path": output_video,
                "input_video_path": output_video,
                "source_bucket": source_bucket,
            }
        )
    return out
