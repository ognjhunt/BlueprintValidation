"""Task-conditioned camera-start selection and shared manifest helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Dict, List, Optional

import numpy as np

from ..common import get_logger, read_json, write_json

logger = get_logger("evaluation.task_start_selector")


@dataclass
class TargetRegion:
    instance_id: str
    label: str
    category: str
    center: np.ndarray


def build_task_start_assignments(
    *,
    tasks: List[str],
    num_rollouts: int,
    render_manifest: dict,
    task_hints_path: Optional[Path],
) -> List[dict]:
    """Assign each rollout to a task and the best matching initial camera clip."""
    clips = list(render_manifest.get("clips", []))
    if not clips or not tasks or num_rollouts <= 0:
        return []

    target_index = _load_target_index(task_hints_path)
    clip_usage: Dict[int, int] = {}
    assignments: List[dict] = []

    for rollout_index in range(num_rollouts):
        task = tasks[rollout_index % len(tasks)]
        task_kind = _task_kind(task)
        target = _resolve_task_target(task, target_index)
        clip = _select_best_clip(
            clips=clips,
            task_kind=task_kind,
            target=target,
            clip_usage=clip_usage,
        )
        clip_index = int(clip.get("clip_index", rollout_index))
        clip_usage[clip_index] = clip_usage.get(clip_index, 0) + 1
        assignments.append(
            {
                "rollout_index": rollout_index,
                "task": task,
                "task_kind": task_kind,
                "clip_index": clip_index,
                "clip_name": str(clip.get("clip_name", f"clip_{clip_index:03d}")),
                "path_type": str(clip.get("path_type", "unknown")),
                "video_path": str(clip.get("video_path", "")),
                "target_instance_id": target.instance_id if target else None,
                "target_label": target.label if target else None,
            }
        )
    return assignments


def load_initial_frames_for_assignments(assignments: List[dict]) -> Dict[int, np.ndarray]:
    """Decode initial RGB frame for each unique clip referenced by assignments."""
    frames: Dict[int, np.ndarray] = {}
    for item in assignments:
        clip_index = int(item.get("clip_index", -1))
        if clip_index < 0 or clip_index in frames:
            continue
        video_path_raw = str(item.get("video_path", ""))
        if not video_path_raw:
            continue
        video_path = Path(video_path_raw)
        if not video_path.exists():
            continue
        frame = _load_first_frame(video_path)
        if frame is None:
            continue
        frames[clip_index] = frame
    return frames


def _load_first_frame(video_path: Path) -> Optional[np.ndarray]:
    # Prefer OpenCV when available (fast), but fall back to imageio to keep
    # evaluation functional in minimal test/runtime environments.
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        pass

    try:
        import imageio.v2 as imageio

        reader = imageio.get_reader(str(video_path))
        try:
            frame = reader.get_data(0)
        finally:
            reader.close()
        if frame is not None:
            return np.asarray(frame)
    except Exception:
        return None
    return None


def load_shared_task_start_manifest(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception:
        logger.warning("Failed to load shared task-start manifest from %s", path, exc_info=True)
        return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("assignments"), list):
        return None
    return payload


def save_shared_task_start_manifest(
    *,
    path: Path,
    facility_name: str,
    render_manifest_path: Path,
    task_profile: str,
    requested_rollouts: int,
    planned_rollouts: int,
    tasks: List[str],
    assignments: List[dict],
) -> None:
    payload = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "facility": facility_name,
        "render_manifest_path": str(render_manifest_path),
        "task_profile": task_profile,
        "requested_rollouts": int(requested_rollouts),
        "planned_rollouts": int(planned_rollouts),
        "num_unique_tasks": len(tasks),
        "tasks": tasks,
        "assignments": assignments,
    }
    write_json(payload, path)


def shared_manifest_is_compatible(
    manifest: dict,
    *,
    facility_name: str,
    render_manifest_path: Path,
) -> bool:
    if str(manifest.get("facility", "")) != facility_name:
        return False
    if str(manifest.get("render_manifest_path", "")) != str(render_manifest_path):
        return False
    return bool(manifest.get("assignments"))


def _load_target_index(task_hints_path: Optional[Path]) -> dict:
    out = {"by_instance": {}, "by_label": {}}
    if task_hints_path is None or not task_hints_path.exists():
        return out

    try:
        payload = read_json(task_hints_path)
    except Exception:
        logger.warning("Failed to load task hints from %s", task_hints_path, exc_info=True)
        return out

    for category, key in (
        ("manipulation", "manipulation_candidates"),
        ("articulation", "articulation_hints"),
        ("navigation", "navigation_hints"),
    ):
        for entry in payload.get(key, []):
            if not isinstance(entry, dict):
                continue
            bbox = entry.get("boundingBox") or entry.get("obb")
            if not isinstance(bbox, dict):
                continue
            center = bbox.get("center")
            if not isinstance(center, list) or len(center) != 3:
                continue
            try:
                center_arr = np.asarray([float(v) for v in center], dtype=np.float64)
            except (TypeError, ValueError):
                continue
            instance_id = str(entry.get("instance_id", "")).strip()
            label = str(entry.get("label", "unknown")).strip()
            region = TargetRegion(
                instance_id=instance_id,
                label=label,
                category=category,
                center=center_arr,
            )
            if instance_id:
                out["by_instance"][instance_id] = region
            label_key = _normalize_label(label)
            out["by_label"].setdefault(label_key, []).append(region)
    return out


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _task_kind(task: str) -> str:
    lowered = task.lower()
    if any(k in lowered for k in ("pick up", "grasp", "lift", "place", "regrasp")):
        return "manipulation"
    if any(k in lowered for k in ("open and close", "open ", "close ", "turn on", "turn off", "toggle")):
        return "articulation"
    if any(k in lowered for k in ("navigate", "approach", "go to", "move toward")):
        return "navigation"
    return "other"


def _resolve_task_target(task: str, target_index: dict) -> Optional[TargetRegion]:
    lowered = task.lower().strip()

    token_match = re.search(r"(?:pick up|open and close|turn on)\s+([a-z0-9_]+)", lowered)
    if token_match:
        token = token_match.group(1)
        target = _resolve_token(token, target_index)
        if target is not None:
            return target

    nav_match = re.search(r"navigate to (?:the )?([a-z0-9_ ]+)", lowered)
    if nav_match:
        nav_label = _normalize_label(nav_match.group(1))
        options = target_index["by_label"].get(nav_label)
        if options:
            return options[0]
    return None


def _resolve_token(token: str, target_index: dict) -> Optional[TargetRegion]:
    token = token.strip("_")
    if not token:
        return None

    # Exact instance-id match.
    by_instance = target_index["by_instance"]
    if token in by_instance:
        return by_instance[token]

    # Common pattern: label_123 -> instance_id=123.
    m = re.match(r"(.+?)_([0-9]+)$", token)
    if m:
        instance_id = m.group(2)
        if instance_id in by_instance:
            return by_instance[instance_id]
        label_key = _normalize_label(m.group(1))
        options = target_index["by_label"].get(label_key)
        if options:
            return options[0]

    options = target_index["by_label"].get(_normalize_label(token))
    if options:
        return options[0]
    return None


def _select_best_clip(
    *,
    clips: List[dict],
    task_kind: str,
    target: Optional[TargetRegion],
    clip_usage: Dict[int, int],
) -> dict:
    ranked: List[tuple[float, int, dict]] = []
    for clip in clips:
        clip_index = int(clip.get("clip_index", 0))
        usage_penalty = 0.20 * clip_usage.get(clip_index, 0)
        score = _score_clip_for_task(clip, task_kind=task_kind, target=target) - usage_penalty
        ranked.append((score, clip_index, clip))

    ranked.sort(key=lambda x: (-x[0], x[1]))
    return ranked[0][2]


def _score_clip_for_task(clip: dict, *, task_kind: str, target: Optional[TargetRegion]) -> float:
    path_type = str(clip.get("path_type", "")).strip().lower()
    if task_kind == "manipulation":
        score = {"manipulation": 2.0, "orbit": 0.8, "sweep": 0.5}.get(path_type, 0.2)
    elif task_kind == "articulation":
        score = {"manipulation": 1.8, "orbit": 1.0, "sweep": 0.6}.get(path_type, 0.2)
    elif task_kind == "navigation":
        score = {"sweep": 2.0, "orbit": 1.8, "manipulation": 0.2}.get(path_type, 0.5)
    else:
        score = 0.3

    if target is None:
        return score

    cam = clip.get("initial_camera")
    if not isinstance(cam, dict):
        return score
    position = _as_vec3(cam.get("position"))
    forward = _as_vec3(cam.get("forward"))
    if position is None:
        return score

    target_vec = target.center - position
    dist = float(np.linalg.norm(target_vec))
    score -= 0.18 * dist

    if forward is not None and dist > 1e-6:
        facing = float(np.dot(target_vec / dist, forward / (np.linalg.norm(forward) + 1e-8)))
        score += 2.5 * facing
        if facing < -0.2:
            score -= 2.0

    path_context = clip.get("path_context")
    if isinstance(path_context, dict):
        approach_point = _as_vec3(path_context.get("approach_point"))
        if approach_point is not None:
            score -= 0.08 * float(np.linalg.norm(target.center - approach_point))
    return score


def _as_vec3(value) -> Optional[np.ndarray]:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        return np.asarray([float(v) for v in value], dtype=np.float64)
    except (TypeError, ValueError):
        return None
