"""Task-hint ingestion from BlueprintCapturePipeline task_targets artifacts."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Dict, List

from ..common import get_logger, read_json

logger = get_logger("evaluation.task_hints")


_TASK_ID_PROMPT_MAP = {
    "pick_place_manipulation": "Pick up a target object and place it in the target zone",
    "open_close_access_points": "Open and close a nearby door or cabinet",
    "open_close_articulation": "Open and close a nearby door or cabinet",
    "navigate_to_target": "Navigate to the target region while avoiding obstacles",
}

_TASK_PROFILE_SPECS: Dict[str, Dict[str, int]] = {
    # World-model comparison (baseline vs adapted):
    # 12-18 unique templates with 3-5 repeats each.
    "dreamdojo": {
        "unique_min": 12,
        "unique_max": 18,
        "manip_quota": 8,
        "artic_quota": 6,
        "nav_quota": 4,
        "max_per_family": 2,
    },
    # Policy comparison (base vs finetuned policy):
    # 20-30 unique templates with 4-5 repeats each.
    "policy": {
        "unique_min": 20,
        "unique_max": 30,
        "manip_quota": 14,
        "artic_quota": 10,
        "nav_quota": 6,
        "max_per_family": 2,
    },
}


def tasks_from_task_hints(
    path: Path,
    max_per_category: int = 4,
    profile: str = "policy",
) -> List[str]:
    """Extract concrete policy task prompts from task_targets.json style payload."""
    if not path.exists():
        raise RuntimeError(f"Task hints file not found: {path}")

    profile_key = _normalize_profile(profile)
    spec = _TASK_PROFILE_SPECS[profile_key]

    payload = read_json(path)
    tasks: List[str] = []

    # Prefer explicit task IDs emitted by BlueprintCapturePipeline.
    for task in payload.get("tasks", []):
        task_raw = str(task.get("task_id") or "").strip()
        task_id = task_raw.lower()
        mapped = _TASK_ID_PROMPT_MAP.get(task_id)
        if mapped:
            tasks.append(mapped)
        elif task_raw and " " in task_raw:
            # Synthetic bootstrap may store direct prompt text in task_id.
            tasks.append(task_raw)

    lowered_existing = [t.lower() for t in tasks]
    has_explicit_manip = any(t.startswith("pick up ") and "_" in t for t in lowered_existing)
    has_explicit_artic = any(t.startswith("open and close ") and "_" in t for t in lowered_existing)

    # Add object-specific manipulation prompts using candidate labels.
    if not has_explicit_manip:
        manip_refs = _top_object_refs(
            payload.get("manipulation_candidates", []),
            limit=max(max_per_category, spec["manip_quota"]),
            exclude_categories={"navigation"},
            max_per_family=spec["max_per_family"],
        )
        for label, instance_id in manip_refs:
            if instance_id:
                token = _prompt_token(label, instance_id)
                tasks.append(f"Pick up {token} and place it in the target zone")
            else:
                tasks.append(f"Pick up the {label} and place it in the target zone")

    if not has_explicit_artic:
        artic_refs = _top_object_refs(
            payload.get("articulation_hints", []),
            limit=max(max_per_category, spec["artic_quota"]),
            max_per_family=spec["max_per_family"],
        )
        for label, instance_id in artic_refs:
            if instance_id:
                token = _prompt_token(label, instance_id)
                tasks.append(f"Open and close {token}")
            else:
                tasks.append(f"Open and close the {label}")

    nav_labels = _top_labels(
        payload.get("navigation_hints", []),
        limit=max(max_per_category, spec["nav_quota"]),
    )
    for label in nav_labels:
        tasks.append(f"Navigate to the {label}")

    deduped = _dedupe(tasks)
    balanced = balance_eval_tasks(deduped, profile=profile_key)
    logger.info(
        "Loaded %d task hints from %s (profile=%s, balanced=%d)",
        len(deduped),
        path,
        profile_key,
        len(balanced),
    )
    return balanced


def balance_eval_tasks(tasks: List[str], profile: str = "policy") -> List[str]:
    """Balance tasks by category and label family for robust eval comparisons."""
    profile_key = _normalize_profile(profile)
    spec = _TASK_PROFILE_SPECS[profile_key]
    deduped = _dedupe(tasks)
    if not deduped:
        return []

    buckets: Dict[str, List[str]] = {
        "manipulation": [],
        "articulation": [],
        "navigation": [],
        "other": [],
    }
    for task in deduped:
        buckets[_task_kind(task)].append(task)

    selected_manip = _select_tasks_with_family_cap(
        buckets["manipulation"],
        limit=spec["manip_quota"],
        max_per_family=spec["max_per_family"],
    )
    selected_artic = _select_tasks_with_family_cap(
        buckets["articulation"],
        limit=spec["artic_quota"],
        max_per_family=spec["max_per_family"],
    )
    selected_nav = _select_tasks_with_family_cap(
        buckets["navigation"],
        limit=spec["nav_quota"],
        max_per_family=spec["max_per_family"],
    )
    selected_other = buckets["other"]

    balanced = _interleave_task_groups(
        [selected_manip, selected_artic, selected_nav, selected_other]
    )

    if len(balanced) < spec["unique_min"]:
        leftovers = [t for t in deduped if t not in balanced]
        for task in leftovers:
            balanced.append(task)
            if len(balanced) >= spec["unique_min"]:
                break

    # Keep evaluation set within the recommended unique-task range.
    return balanced[: spec["unique_max"]]


def recommended_rollouts_per_condition(
    num_unique_tasks: int,
    requested: int,
    profile: str,
) -> int:
    """Choose rollouts/condition to hit repeat-count and total-rollout guidance."""
    debug_override = os.environ.get("BLUEPRINT_EVAL_ALLOW_SMALL_ROLLOUTS", "").strip().lower()
    if debug_override in {"1", "true", "yes", "on"}:
        return max(1, int(requested))

    profile_key = _normalize_profile(profile)
    if profile_key == "dreamdojo":
        repeats_target = 4
        min_per_condition = 80  # 2 conditions -> 160 total (strong range starts at 150)
        max_per_condition = 125  # 2 conditions -> 250 total (strong range upper bound)
    else:
        repeats_target = 4
        min_per_condition = 80  # adapted + trained -> 160 total minimum
        max_per_condition = 200  # adapted + trained -> 400 total upper bound

    task_based_target = max(1, num_unique_tasks) * repeats_target
    planned = max(int(requested), min_per_condition, task_based_target)
    return min(planned, max_per_condition)


def _top_labels(
    entries: list,
    limit: int,
    exclude_categories: set[str] | None = None,
) -> List[str]:
    excluded = {c.strip().lower() for c in (exclude_categories or set())}
    seen = set()
    out: List[str] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "").strip().lower()
        if category and category in excluded:
            continue
        label = str(item.get("label") or "").strip().lower()
        if not label or label in {"object", "unknown"}:
            continue
        if label in seen:
            continue
        seen.add(label)
        out.append(label.replace("_", " "))
        if len(out) >= max(0, limit):
            break
    return out


def _top_object_refs(
    entries: list,
    limit: int,
    exclude_categories: set[str] | None = None,
    max_per_family: int = 2,
) -> List[tuple[str, str]]:
    excluded = {c.strip().lower() for c in (exclude_categories or set())}
    seen = set()
    family_counts: Dict[str, int] = {}
    out: List[tuple[str, str]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "").strip().lower()
        if category and category in excluded:
            continue
        label = str(item.get("label") or "").strip().lower()
        if not label or label in {"object", "unknown"}:
            continue
        label_norm = label.replace("_", " ")
        instance_id = str(item.get("instance_id") or "").strip()
        family = _label_family(label_norm)
        if family_counts.get(family, 0) >= max(1, max_per_family):
            continue
        key = (label_norm, instance_id)
        if key in seen:
            continue
        seen.add(key)
        family_counts[family] = family_counts.get(family, 0) + 1
        out.append((label_norm, instance_id))
        if len(out) >= max(0, limit):
            break
    return out


def _prompt_token(label: str, instance_id: str) -> str:
    label_token = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "object"
    iid_token = re.sub(r"[^a-zA-Z0-9]+", "", instance_id)
    if iid_token:
        return f"{label_token}_{iid_token}"
    return label_token


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def _normalize_profile(profile: str) -> str:
    key = str(profile or "").strip().lower()
    if key not in _TASK_PROFILE_SPECS:
        return "policy"
    return key


def _task_kind(task: str) -> str:
    t = task.strip().lower()
    if any(k in t for k in ("pick up", "grasp", "lift", "place", "stack", "regrasp")):
        return "manipulation"
    if any(k in t for k in ("open and close", "open ", "close ", "turn on", "turn off", "toggle")):
        return "articulation"
    if any(k in t for k in ("navigate", "approach", "go to", "move toward")):
        return "navigation"
    return "other"


def _task_family(task: str) -> str:
    t = task.strip().lower()
    patterns = [
        r"pick up ([a-z0-9_]+)",
        r"open and close ([a-z0-9_]+)",
        r"turn on ([a-z0-9_]+)",
        r"navigate to (?:the )?([a-z0-9_ ]+)",
    ]
    for pat in patterns:
        match = re.search(pat, t)
        if not match:
            continue
        token = match.group(1).strip().replace(" ", "_")
        token = re.sub(r"_[0-9]+$", "", token)
        if token:
            return _label_family(token)
    words = [w for w in re.split(r"[^a-z0-9]+", t) if w]
    if not words:
        return "misc"
    return "_".join(words[:4])


def _label_family(label: str) -> str:
    tokens = [t for t in re.split(r"[^a-z0-9]+", label.lower()) if t]
    if not tokens:
        return "object"
    return tokens[-1]


def _select_tasks_with_family_cap(
    tasks: List[str],
    limit: int,
    max_per_family: int,
) -> List[str]:
    out: List[str] = []
    family_counts: Dict[str, int] = {}
    for task in tasks:
        family = _task_family(task)
        if family_counts.get(family, 0) >= max(1, max_per_family):
            continue
        out.append(task)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(out) >= max(0, limit):
            break
    return out


def _interleave_task_groups(groups: List[List[str]]) -> List[str]:
    out: List[str] = []
    idx = 0
    while True:
        added = False
        for group in groups:
            if idx < len(group):
                out.append(group[idx])
                added = True
        if not added:
            break
        idx += 1
    return out
