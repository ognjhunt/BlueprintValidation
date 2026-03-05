"""Deterministic primary-endpoint evaluation for fixed-world claim runs."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def evaluate_task_success(
    *,
    task_spec: Dict[str, object],
    rollout_row: Dict[str, object],
    state_trace: List[Dict[str, object]] | None,
) -> Dict[str, object]:
    trace = list(state_trace or [])
    predicate = dict(task_spec.get("success_predicate", {}) or {})
    predicate_type = str(predicate.get("type", "")).strip().lower()
    if predicate_type == "navigation_reach_and_hold":
        return _evaluate_navigation(predicate, trace)
    if predicate_type == "articulation_open_close_sequence":
        return _evaluate_articulation(predicate, trace)
    if predicate_type == "manipulation_pick_place_stable":
        return _evaluate_manipulation(predicate, trace)
    return _fallback_score_threshold(predicate, rollout_row)


def rollout_state_trace_available(state_trace: Iterable[Dict[str, object]] | None) -> bool:
    return bool(list(state_trace or []))


def summarize_task_success_rows(rows: List[Dict[str, object]]) -> float:
    if not rows:
        return 0.0
    vals = [1.0 if bool(row.get("task_success", False)) else 0.0 for row in rows]
    return round(float(np.mean(vals)), 6)


def _evaluate_navigation(
    predicate: Dict[str, object],
    trace: List[Dict[str, object]],
) -> Dict[str, object]:
    if not trace:
        return _unavailable("Missing state_trace for navigation predicate.")
    goal_region_id = str(predicate.get("goal_region_id", "")).strip()
    hold_steps = max(1, int(predicate.get("hold_steps", 1)))
    collision_key = str(predicate.get("collision_key", "invalid_collision")).strip() or "invalid_collision"
    max_consecutive = 0
    consecutive = 0
    for state in trace:
        if bool(state.get(collision_key, False)):
            return {
                "task_success": False,
                "task_success_reason": "invalid_collision",
                "task_success_available": True,
            }
        current_region = str(state.get("active_region_id", "")).strip()
        if current_region == goal_region_id:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    return {
        "task_success": bool(max_consecutive >= hold_steps),
        "task_success_reason": (
            "reached_goal_region" if max_consecutive >= hold_steps else "goal_region_not_reached"
        ),
        "task_success_available": True,
    }


def _evaluate_articulation(
    predicate: Dict[str, object],
    trace: List[Dict[str, object]],
) -> Dict[str, object]:
    if not trace:
        return _unavailable("Missing state_trace for articulation predicate.")
    target_instance_id = str(predicate.get("target_instance_id", "")).strip()
    joint_key = str(predicate.get("joint_key", "joint_position")).strip() or "joint_position"
    open_threshold = float(predicate.get("open_threshold", 0.8))
    close_threshold = float(predicate.get("close_threshold", 0.2))
    opened = False
    closed_after_open = False
    for state in trace:
        if bool(state.get("invalid_collision", False)):
            return {
                "task_success": False,
                "task_success_reason": "invalid_collision",
                "task_success_available": True,
            }
        value = _joint_position_for_state(state, target_instance_id, joint_key)
        if value is None:
            continue
        if value >= open_threshold:
            opened = True
        if opened and value <= close_threshold:
            closed_after_open = True
            break
    return {
        "task_success": bool(opened and closed_after_open),
        "task_success_reason": (
            "opened_and_closed" if opened and closed_after_open else "sequence_incomplete"
        ),
        "task_success_available": True,
    }


def _evaluate_manipulation(
    predicate: Dict[str, object],
    trace: List[Dict[str, object]],
) -> Dict[str, object]:
    if not trace:
        return _unavailable("Missing state_trace for manipulation predicate.")
    require_stable = bool(predicate.get("require_stable_after_place", True))
    grasp = any(bool(state.get("grasp_acquired", False)) for state in trace)
    lifted = any(bool(state.get("lifted_clear", False)) for state in trace)
    placed = any(bool(state.get("placed_in_target", False)) for state in trace)
    stable = any(bool(state.get("stable_after_place", False)) for state in trace)
    if any(bool(state.get("invalid_collision", False)) for state in trace):
        return {
            "task_success": False,
            "task_success_reason": "invalid_collision",
            "task_success_available": True,
        }
    success = grasp and lifted and placed and (stable if require_stable else True)
    return {
        "task_success": bool(success),
        "task_success_reason": (
            "pick_place_stable"
            if success
            else "missing_grasp_or_lift_or_place_or_stability"
        ),
        "task_success_available": True,
    }


def _fallback_score_threshold(
    predicate: Dict[str, object],
    rollout_row: Dict[str, object],
) -> Dict[str, object]:
    threshold = float(predicate.get("threshold", 7.0))
    score = float(rollout_row.get("task_score", 0.0) or 0.0)
    return {
        "task_success": bool(score >= threshold),
        "task_success_reason": "score_threshold",
        "task_success_available": False,
    }


def _joint_position_for_state(
    state: Dict[str, object],
    target_instance_id: str,
    joint_key: str,
) -> float | None:
    if target_instance_id:
        joints = state.get("joint_positions", {})
        if isinstance(joints, dict) and target_instance_id in joints:
            try:
                return float(joints[target_instance_id])
            except Exception:
                return None
    value = state.get(joint_key)
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def _unavailable(reason: str) -> Dict[str, object]:
    return {
        "task_success": False,
        "task_success_reason": str(reason),
        "task_success_available": False,
    }
