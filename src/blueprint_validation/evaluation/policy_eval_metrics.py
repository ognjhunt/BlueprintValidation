"""Shared metric and artifact helpers for policy evaluation stages."""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

from .claim_protocol import paired_eval_key
from .stats_utils import paired_ttest_p_value
from .task_success import evaluate_task_success


def is_manipulation_task(task: str) -> bool:
    lowered = task.lower()
    keywords = ["pick", "grasp", "lift", "place", "stack", "regrasp", "tote", "bin"]
    return any(k in lowered for k in keywords)


def is_object_grounded_manip_task(task: str) -> bool:
    if not is_manipulation_task(task):
        return True
    lowered = str(task or "").lower()
    return bool(re.search(r"\b[a-z0-9_]+_[0-9]{2,}\b", lowered))


def manipulation_success_rate(scores: List[Dict]) -> float:
    if not scores:
        return 0.0
    successes = 0
    for score in scores:
        has_flags = score.get("grasp_acquired") is not None
        if has_flags:
            if (
                bool(score.get("grasp_acquired"))
                and bool(score.get("lifted_clear"))
                and bool(score.get("placed_in_target"))
            ):
                successes += 1
        elif float(score.get("task_score", 0.0)) >= 7.0:
            successes += 1
    return round(successes / len(scores), 3)


def attach_claim_row_metadata(
    *,
    base_row: Dict[str, object],
    assignment: dict,
    rollout,
    task_specs_by_prompt: Dict[str, Dict[str, object]],
    fixed_claim_protocol: bool,
    claim_state_failures: List[str],
) -> Dict[str, object]:
    row = dict(base_row)
    row["eval_cell_id"] = assignment.get("eval_cell_id")
    row["task_spec_id"] = assignment.get("task_spec_id")
    row["start_clip_id"] = assignment.get("start_clip_id") or assignment.get("clip_name")
    row["start_region_id"] = assignment.get("start_region_id")
    row["start_frame_hash"] = assignment.get("start_frame_hash")
    row["world_snapshot_hash"] = assignment.get("world_snapshot_hash")
    row["state_trace"] = list(getattr(rollout, "state_trace", []) or [])
    if not fixed_claim_protocol:
        row["task_success"] = None
        row["task_success_reason"] = ""
        row["task_success_available"] = False
        return row

    task_spec = task_specs_by_prompt.get(str(row.get("task", "")).strip())
    if not task_spec:
        claim_state_failures.append(f"Missing task spec for task='{row.get('task', '')}'")
        row["task_success"] = False
        row["task_success_reason"] = "missing_task_spec"
        row["task_success_available"] = False
        return row

    success_payload = evaluate_task_success(
        task_spec=task_spec,
        rollout_row=row,
        state_trace=row.get("state_trace") or [],
    )
    row.update(success_payload)
    if not bool(success_payload.get("task_success_available", False)):
        claim_state_failures.append(
            f"Unavailable task-success endpoint for {str(row.get('condition', ''))}:"
            f"{str(row.get('task', ''))}"
        )
    return row


def build_pairwise_metrics(all_scores: List[Dict], conditions: List[str]) -> Dict:
    """Compute improvement, win rate, and p-value for each pair of conditions."""
    pairwise = {}
    has_explicit_pairing = any(str(row.get("eval_cell_id", "")).strip() for row in all_scores)
    for idx, left_condition in enumerate(conditions):
        for right_condition in conditions[idx + 1 :]:
            if has_explicit_pairing:
                grouped: Dict[str, Dict[str, Dict]] = {}
                for row in all_scores:
                    condition = str(row.get("condition", "")).strip()
                    if condition not in {left_condition, right_condition}:
                        continue
                    grouped.setdefault(paired_eval_key(row), {})[condition] = row
                pairs = [
                    (rows[left_condition], rows[right_condition])
                    for rows in grouped.values()
                    if left_condition in rows and right_condition in rows
                ]
                if not pairs:
                    continue
                left_rows = [left for left, _ in pairs]
                right_rows = [right for _, right in pairs]
            else:
                left_rows = [score for score in all_scores if score["condition"] == left_condition]
                right_rows = [score for score in all_scores if score["condition"] == right_condition]
                if not left_rows or not right_rows:
                    continue
                min_len = min(len(left_rows), len(right_rows))
                pairs = list(zip(left_rows[:min_len], right_rows[:min_len]))
            mean1 = float(np.mean([score["task_score"] for score in left_rows]))
            mean2 = float(np.mean([score["task_score"] for score in right_rows]))
            improvement = ((mean2 - mean1) / max(mean1, 1e-8)) * 100
            abs_diff = mean2 - mean1
            wins = sum(
                1
                for left_row, right_row in pairs
                if float(right_row["task_score"]) > float(left_row["task_score"])
            )
            win_rate = wins / max(len(pairs), 1)
            p_value = None
            if len(pairs) >= 2:
                left_values = [left["task_score"] for left, _ in pairs]
                right_values = [right["task_score"] for _, right in pairs]
                p_value = paired_ttest_p_value(left_values, right_values)
            pairwise[f"{left_condition}_vs_{right_condition}"] = {
                f"{left_condition}_mean": round(mean1, 3),
                f"{right_condition}_mean": round(mean2, 3),
                "improvement_pct": round(improvement, 2),
                "absolute_difference": round(abs_diff, 3),
                "win_rate": round(win_rate, 3),
                "p_value": round(p_value, 6) if p_value is not None else None,
            }
    return pairwise


def build_confidence_intervals(baseline_scores: List[Dict], adapted_scores: List[Dict]) -> Dict:
    min_len = min(len(baseline_scores), len(adapted_scores))
    if min_len < 2:
        return {"paired_mean_delta": None, "paired_95ci_low": None, "paired_95ci_high": None}
    baseline_values = np.asarray(
        [score["task_score"] for score in baseline_scores[:min_len]],
        dtype=np.float32,
    )
    adapted_values = np.asarray(
        [score["task_score"] for score in adapted_scores[:min_len]],
        dtype=np.float32,
    )
    diffs = adapted_values - baseline_values
    mean = float(np.mean(diffs))
    std = float(np.std(diffs, ddof=1))
    sem = std / np.sqrt(float(min_len))
    margin = 1.96 * sem
    return {
        "paired_mean_delta": round(mean, 6),
        "paired_95ci_low": round(mean - margin, 6),
        "paired_95ci_high": round(mean + margin, 6),
    }


def build_low_score_breakdown(scores: List[Dict]) -> Dict:
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
        if any(token in reasoning for token in ("ceiling", "upside down", "upside-down")):
            categories["ceiling_or_upside_down"] += 1
        if any(
            token in reasoning
            for token in ("off target", "not visible", "out of frame", "off-center")
        ):
            categories["off_target_or_not_visible"] += 1
        if any(
            token in reasoning
            for token in ("blur", "blurry", "indiscernible", "unclear", "artifact")
        ):
            categories["blur_or_indiscernible"] += 1
        if any(
            token in reasoning
            for token in ("missing robot", "missing target", "missing object", "no robot")
        ):
            categories["missing_robot_or_target_object"] += 1
        if len(examples) < 5:
            examples.append(
                {
                    "condition": row.get("condition"),
                    "task": row.get("task"),
                    "reasoning": row.get("reasoning"),
                }
            )
    return {
        "num_low_score_rows": len(low_rows),
        "categories": categories,
        "examples": examples,
    }
