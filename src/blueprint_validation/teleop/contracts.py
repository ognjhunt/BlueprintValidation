"""Contracts for scene packages and action-labeled teleop sessions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from ..common import read_json, write_json


class TeleopManifestError(ValueError):
    """Raised when a teleop or scene package manifest is invalid."""


_SCENE_HANDOFF_REQUIRED = (
    Path("assets/scene_manifest.json"),
    Path("usd/scene.usda"),
)

_TELEOP_REQUIRED_SESSION_KEYS = {
    "session_id",
    "scene_id",
    "task_id",
    "task_text",
    "demo_index",
    "success",
    "sim_backend",
    "teleop_device",
    "robot_type",
    "robot_asset_ref",
    "action_space",
    "action_dim",
    "state_keys",
    "camera_ids",
    "video_paths",
    "calibration_refs",
    "lerobot_root",
    "episode_ref",
    "start_state_hash",
    "action_sequence_path",
    "state_sequence_path",
}


def _require_text(payload: Mapping[str, Any], key: str, *, where: str) -> str:
    value = str(payload.get(key, "") or "").strip()
    if not value:
        raise TeleopManifestError(f"Missing non-empty '{key}'{where}")
    return value


def _require_bool(payload: Mapping[str, Any], key: str, *, where: str) -> bool:
    if key not in payload:
        raise TeleopManifestError(f"Missing boolean '{key}'{where}")
    return bool(payload[key])


def _require_int(payload: Mapping[str, Any], key: str, *, where: str) -> int:
    if key not in payload:
        raise TeleopManifestError(f"Missing integer '{key}'{where}")
    try:
        return int(payload[key])
    except Exception as exc:  # pragma: no cover - defensive
        raise TeleopManifestError(f"Invalid integer '{key}'{where}") from exc


def _require_text_list(payload: Mapping[str, Any], key: str, *, where: str) -> List[str]:
    values = payload.get(key)
    if not isinstance(values, list) or not values:
        raise TeleopManifestError(f"Missing non-empty list '{key}'{where}")
    normalized = [str(v).strip() for v in values if str(v).strip()]
    if not normalized:
        raise TeleopManifestError(f"Missing non-empty list '{key}'{where}")
    return normalized


def _require_mapping(payload: Mapping[str, Any], key: str, *, where: str) -> Dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict) or not value:
        raise TeleopManifestError(f"Missing non-empty object '{key}'{where}")
    return dict(value)


def _load_json_sequence(path: Path, *, where: str) -> List[Any]:
    try:
        payload = read_json(path)
    except Exception as exc:  # pragma: no cover - defensive
        raise TeleopManifestError(f"Could not load JSON file '{path}'{where}: {exc}") from exc
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("actions", "states", "sequence", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise TeleopManifestError(f"Expected JSON sequence in '{path}'{where}")


def load_and_validate_scene_package(scene_root: Path) -> Dict[str, Any]:
    """Validate the minimal scene handoff contract for sim teleop."""
    root = scene_root.resolve()
    missing = [str(rel) for rel in _SCENE_HANDOFF_REQUIRED if not (root / rel).exists()]
    if missing:
        joined = ", ".join(missing)
        raise TeleopManifestError(f"Scene package missing required files: {joined}")

    scene_manifest_path = root / "assets" / "scene_manifest.json"
    scene_manifest = read_json(scene_manifest_path)
    return {
        "scene_root": str(root),
        "scene_manifest_path": str(scene_manifest_path),
        "scene_manifest": scene_manifest,
        "usd_scene_path": str(root / "usd" / "scene.usda"),
        "has_isaac_lab": (root / "isaac_lab").exists(),
        "has_geniesim_task_config": (root / "geniesim" / "task_config.json").exists(),
    }


def summarize_teleop_session_quality(session: Mapping[str, Any], *, require_existing_paths: bool) -> Dict[str, Any]:
    """Validate one teleop session and return a structured quality report."""
    where = f" (session_id={session.get('session_id', '<missing>')})"
    missing_keys = sorted(_TELEOP_REQUIRED_SESSION_KEYS - set(session.keys()))
    if missing_keys:
        raise TeleopManifestError(f"Teleop session missing keys{where}: {', '.join(missing_keys)}")

    _require_text(session, "session_id", where=where)
    _require_text(session, "scene_id", where=where)
    _require_text(session, "task_id", where=where)
    _require_text(session, "task_text", where=where)
    _require_bool(session, "success", where=where)
    _require_text(session, "sim_backend", where=where)
    _require_text(session, "teleop_device", where=where)
    _require_text(session, "robot_type", where=where)
    _require_text(session, "robot_asset_ref", where=where)
    _require_text(session, "action_space", where=where)
    action_dim = _require_int(session, "action_dim", where=where)
    if action_dim <= 0:
        raise TeleopManifestError(f"action_dim must be > 0{where}")

    camera_ids = _require_text_list(session, "camera_ids", where=where)
    _require_text_list(session, "state_keys", where=where)
    video_paths = _require_mapping(session, "video_paths", where=where)
    calibration_refs = _require_mapping(session, "calibration_refs", where=where)
    lerobot_root = Path(_require_text(session, "lerobot_root", where=where))
    action_path = Path(_require_text(session, "action_sequence_path", where=where))
    state_path = Path(_require_text(session, "state_sequence_path", where=where))
    _require_text(session, "episode_ref", where=where)
    _require_text(session, "start_state_hash", where=where)
    demo_index = _require_int(session, "demo_index", where=where)
    if demo_index < 0:
        raise TeleopManifestError(f"demo_index must be >= 0{where}")

    if require_existing_paths:
        required_paths = [lerobot_root, action_path, state_path]
        for path in required_paths:
            if not path.exists():
                raise TeleopManifestError(f"Referenced path does not exist: {path}{where}")
        for camera_id in camera_ids:
            video_path = Path(str(video_paths.get(camera_id, "") or ""))
            calib_path = Path(str(calibration_refs.get(camera_id, "") or ""))
            if not str(video_path):
                raise TeleopManifestError(f"Missing video path for camera '{camera_id}'{where}")
            if not str(calib_path):
                raise TeleopManifestError(f"Missing calibration ref for camera '{camera_id}'{where}")
            if not video_path.exists():
                raise TeleopManifestError(f"Video path does not exist: {video_path}{where}")
            if not calib_path.exists():
                raise TeleopManifestError(f"Calibration ref does not exist: {calib_path}{where}")

    actions = _load_json_sequence(action_path, where=where)
    states = _load_json_sequence(state_path, where=where)
    if not actions:
        raise TeleopManifestError(f"Empty action sequence{where}")
    if not states:
        raise TeleopManifestError(f"Empty state sequence{where}")
    if len(actions) != len(states):
        raise TeleopManifestError(
            f"Action/state length mismatch{where}: actions={len(actions)} states={len(states)}"
        )

    inferred_dims = {len(step) for step in actions if isinstance(step, list)}
    if inferred_dims != {action_dim}:
        raise TeleopManifestError(
            f"Action dimension mismatch{where}: expected={action_dim} inferred={sorted(inferred_dims)}"
        )

    digest = hashlib.sha256(
        json.dumps(
            {
                "session_id": session["session_id"],
                "scene_id": session["scene_id"],
                "task_id": session["task_id"],
                "demo_index": demo_index,
                "action_dim": action_dim,
                "num_steps": len(actions),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    return {
        "passed": bool(session.get("success", False)),
        "session_id": str(session["session_id"]),
        "scene_id": str(session["scene_id"]),
        "task_id": str(session["task_id"]),
        "robot_type": str(session["robot_type"]),
        "action_dim": action_dim,
        "num_steps": len(actions),
        "camera_ids": camera_ids,
        "quality_hash": digest,
    }


def load_and_validate_teleop_manifest(path: Path, *, require_existing_paths: bool = True) -> Dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise TeleopManifestError(f"Teleop manifest must be an object ({path})")
    schema_version = str(payload.get("schema_version", "") or "").strip()
    if schema_version != "v1":
        raise TeleopManifestError(f"Unsupported teleop schema_version '{schema_version}' ({path})")
    sessions = payload.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise TeleopManifestError(f"Teleop manifest must contain non-empty 'sessions' ({path})")
    reports = []
    normalized = []
    for raw in sessions:
        if not isinstance(raw, dict):
            raise TeleopManifestError(f"Each teleop session must be an object ({path})")
        report = summarize_teleop_session_quality(
            raw,
            require_existing_paths=require_existing_paths,
        )
        reports.append(report)
        normalized.append(dict(raw))
    return {
        "schema_version": "v1",
        "source_name": str(payload.get("source_name", "teleop") or "teleop"),
        "sessions": normalized,
        "quality_reports": reports,
    }


def build_stage1_source_manifest(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Derive a Stage 1f-compatible video manifest from validated teleop sessions."""
    sessions = list(payload.get("sessions", []))
    clips: List[Dict[str, Any]] = []
    for session in sessions:
        task_id = str(session.get("task_id", "") or "").strip() or "task"
        demo_index = int(session.get("demo_index", 0) or 0)
        for camera_id, video_path in sorted(dict(session.get("video_paths", {}) or {}).items()):
            video_text = str(video_path or "").strip()
            if not video_text:
                continue
            clips.append(
                {
                    "clip_name": f"teleop_{task_id}_{demo_index:03d}_{camera_id}",
                    "video_path": video_text,
                    "source_name": str(payload.get("source_name", "teleop") or "teleop"),
                    "source_stage": "teleop_stage1_source_manifest",
                    "augmentation_type": "external_teleop",
                    "task_id": task_id,
                    "camera_id": str(camera_id),
                    "robot_type": str(session.get("robot_type", "")),
                    "teleop_device": str(session.get("teleop_device", "")),
                    "sim_backend": str(session.get("sim_backend", "")),
                }
            )
    return {
        "source_name": str(payload.get("source_name", "teleop") or "teleop"),
        "num_clips": len(clips),
        "clips": clips,
    }


def write_teleop_manifests(
    *,
    output_dir: Path,
    source_name: str,
    sessions: Iterable[Mapping[str, Any]],
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    teleop_manifest = {
        "schema_version": "v1",
        "source_name": str(source_name or "teleop"),
        "sessions": [dict(session) for session in sessions],
    }
    validated = load_and_validate_teleop_manifest(
        _write_temp_and_reload(output_dir / "teleop_session_manifest.json", teleop_manifest),
        require_existing_paths=True,
    )
    teleop_path = output_dir / "teleop_session_manifest.json"
    write_json(validated, teleop_path)
    stage1_path = output_dir / "teleop_stage1_source_manifest.json"
    write_json(build_stage1_source_manifest(validated), stage1_path)
    quality_path = output_dir / "teleop_quality_report.json"
    write_json({"reports": validated["quality_reports"]}, quality_path)
    return {
        "teleop_manifest_path": teleop_path,
        "stage1_source_manifest_path": stage1_path,
        "quality_report_path": quality_path,
    }


def _write_temp_and_reload(path: Path, payload: Mapping[str, Any]) -> Path:
    write_json(dict(payload), path)
    return path
