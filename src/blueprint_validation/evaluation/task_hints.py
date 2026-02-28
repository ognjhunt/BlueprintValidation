"""Task-hint ingestion from BlueprintCapturePipeline task_targets artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import List

from ..common import get_logger, read_json

logger = get_logger("evaluation.task_hints")


_TASK_ID_PROMPT_MAP = {
    "pick_place_manipulation": "Pick up a target object and place it in the target zone",
    "open_close_access_points": "Open and close a nearby door or cabinet",
    "open_close_articulation": "Open and close a nearby door or cabinet",
    "navigate_to_target": "Navigate to the target region while avoiding obstacles",
}


def tasks_from_task_hints(path: Path, max_per_category: int = 4) -> List[str]:
    """Extract concrete policy task prompts from task_targets.json style payload."""
    if not path.exists():
        raise RuntimeError(f"Task hints file not found: {path}")

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

    # Add object-specific manipulation prompts using candidate labels.
    manip_labels = _top_labels(
        payload.get("manipulation_candidates", []),
        limit=max_per_category,
        exclude_categories={"navigation"},
    )
    for label in manip_labels:
        tasks.append(f"Pick up the {label} and place it in the target zone")

    artic_labels = _top_labels(payload.get("articulation_hints", []), limit=max_per_category)
    for label in artic_labels:
        tasks.append(f"Open and close the {label}")

    nav_labels = _top_labels(payload.get("navigation_hints", []), limit=max_per_category)
    for label in nav_labels:
        tasks.append(f"Navigate to the {label}")

    deduped = _dedupe(tasks)
    logger.info("Loaded %d task hints from %s", len(deduped), path)
    return deduped


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
