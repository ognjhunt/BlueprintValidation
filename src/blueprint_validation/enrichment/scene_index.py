"""Lightweight pose-based scene index scaffolding for Stage-2 context retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from ..common import get_logger, read_json, write_json

logger = get_logger("enrichment.scene_index")


def build_scene_index(
    *,
    render_manifest: dict,
    output_path: Path,
    sample_every_n_frames: int = 8,
) -> dict:
    """Build a compact scene index JSON from render manifest clips and camera poses."""
    step = max(1, int(sample_every_n_frames))
    entries: List[Dict[str, object]] = []

    for clip in list(render_manifest.get("clips", [])):
        clip_name = str(clip.get("clip_name", "")).strip()
        if not clip_name:
            continue
        video_path = str(clip.get("video_path", "")).strip()
        camera_path = Path(str(clip.get("camera_path", "")))
        frames = _load_camera_frames(camera_path)
        if not frames:
            entries.append(
                {
                    "clip_name": clip_name,
                    "video_path": video_path,
                    "frame_index": 0,
                    "has_pose": False,
                }
            )
            continue

        for frame_index in range(0, len(frames), step):
            c2w = _parse_c2w(frames[frame_index].get("camera_to_world"))
            if c2w is None:
                continue
            position = c2w[:3, 3]
            forward = -c2w[:3, 2]
            entries.append(
                {
                    "clip_name": clip_name,
                    "video_path": video_path,
                    "frame_index": int(frame_index),
                    "has_pose": True,
                    "position": [float(v) for v in position.tolist()],
                    "forward": [float(v) for v in forward.tolist()],
                }
            )

    payload = {
        "version": 1,
        "sample_every_n_frames": step,
        "num_entries": len(entries),
        "num_clips": len(list(render_manifest.get("clips", []))),
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(payload, output_path)
    return payload


def query_nearest_context_candidates(
    *,
    scene_index: dict,
    anchor_clip_name: str,
    anchor_frame_index: int | None,
    k: int = 3,
) -> List[dict]:
    """Query nearest scene-context candidates using pose distance, then temporal fallback."""
    max_k = max(0, int(k))
    if max_k <= 0:
        return []

    entries = [e for e in list(scene_index.get("entries", [])) if isinstance(e, dict)]
    if not entries:
        return []

    anchor_idx = int(anchor_frame_index) if anchor_frame_index is not None else 0
    anchor_pose = _select_anchor_pose(entries, anchor_clip_name=anchor_clip_name, anchor_idx=anchor_idx)

    ranked: List[tuple[float, str, int, dict]] = []
    for item in entries:
        clip_name = str(item.get("clip_name", "")).strip()
        frame_index = int(item.get("frame_index", 0))
        if clip_name == anchor_clip_name and frame_index == anchor_idx:
            continue

        if anchor_pose is not None:
            score = _pose_distance(anchor_pose, item)
            if score is None:
                score = 1_000_000.0 + abs(frame_index - anchor_idx)
            else:
                score += 0.01 * abs(frame_index - anchor_idx)
        else:
            clip_bias = 0.0 if clip_name == anchor_clip_name else 1000.0
            score = clip_bias + abs(frame_index - anchor_idx)

        ranked.append((float(score), clip_name, frame_index, item))

    ranked.sort(key=lambda x: (x[0], x[1], x[2]))
    results: List[dict] = []
    seen = set()
    for score, clip_name, frame_index, item in ranked:
        key = (clip_name, frame_index)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "clip_name": clip_name,
                "video_path": str(item.get("video_path", "")),
                "frame_index": int(frame_index),
                "score": float(score),
                "has_pose": bool(item.get("has_pose", False)),
            }
        )
        if len(results) >= max_k:
            break
    return results


def _load_camera_frames(camera_path: Path) -> List[dict]:
    if not camera_path.exists():
        return []
    try:
        payload = read_json(camera_path)
    except Exception:
        logger.debug("Failed loading camera path for scene index: %s", camera_path, exc_info=True)
        return []
    frames = payload.get("camera_path", [])
    if not isinstance(frames, list):
        return []
    return [frame for frame in frames if isinstance(frame, dict)]


def _parse_c2w(value) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != 16:
        return None
    try:
        return np.asarray([float(v) for v in value], dtype=np.float64).reshape(4, 4)
    except (TypeError, ValueError):
        return None


def _select_anchor_pose(
    entries: List[dict],
    *,
    anchor_clip_name: str,
    anchor_idx: int,
) -> dict | None:
    candidates = []
    for item in entries:
        if str(item.get("clip_name", "")).strip() != anchor_clip_name:
            continue
        if not bool(item.get("has_pose", False)):
            continue
        frame_index = int(item.get("frame_index", 0))
        candidates.append((abs(frame_index - anchor_idx), item))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _pose_distance(anchor_item: dict, candidate_item: dict) -> float | None:
    if not bool(candidate_item.get("has_pose", False)):
        return None
    anchor_pos = _vec3(anchor_item.get("position"))
    cand_pos = _vec3(candidate_item.get("position"))
    anchor_fwd = _vec3(anchor_item.get("forward"))
    cand_fwd = _vec3(candidate_item.get("forward"))
    if anchor_pos is None or cand_pos is None:
        return None

    position_term = float(np.linalg.norm(anchor_pos - cand_pos))
    direction_term = 0.0
    if anchor_fwd is not None and cand_fwd is not None:
        a = anchor_fwd / (np.linalg.norm(anchor_fwd) + 1e-8)
        b = cand_fwd / (np.linalg.norm(cand_fwd) + 1e-8)
        direction_term = float(1.0 - np.clip(np.dot(a, b), -1.0, 1.0))
    return position_term + 0.25 * direction_term


def _vec3(value) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        return np.asarray([float(v) for v in value], dtype=np.float64)
    except (TypeError, ValueError):
        return None
