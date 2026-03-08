"""Helpers for loading external teleop rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

from ..common import read_json, write_json
from ..config import ExternalRolloutsConfig, ValidationConfig
from ..teleop.contracts import load_and_validate_teleop_manifest


def external_rollouts_enabled_for_policy(cfg: ExternalRolloutsConfig) -> bool:
    return bool(cfg.enabled) and str(cfg.mode or "wm_and_policy").strip().lower() in {
        "policy_only",
        "wm_and_policy",
    }


def external_rollouts_enabled_for_wm(cfg: ExternalRolloutsConfig) -> bool:
    return bool(cfg.enabled) and str(cfg.mode or "wm_and_policy").strip().lower() in {
        "wm_only",
        "wm_and_policy",
    }


def convert_teleop_sessions_to_rollout_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rollout_index, session in enumerate(list(payload.get("sessions", []))):
        action_path = Path(str(session["action_sequence_path"]))
        state_path = Path(str(session["state_sequence_path"]))
        actions = _load_json_sequence(action_path)
        states = _load_json_sequence(state_path)
        video_paths = dict(session.get("video_paths", {}) or {})
        primary_camera_id = str(session.get("camera_ids", [""])[0] or "").strip()
        primary_video = str(video_paths.get(primary_camera_id) or next(iter(video_paths.values()), "")).strip()
        rows.append(
            {
                "condition": "adapted",
                "task": str(session.get("task_text", "") or session.get("task_id", "")).strip(),
                "task_spec_id": str(session.get("task_id", "")).strip(),
                "rollout_index": int(session.get("demo_index", rollout_index)),
                "video_path": primary_video,
                "action_sequence": actions,
                "state_sequence": states,
                "task_score": 10.0 if bool(session.get("success", False)) else 0.0,
                "task_success": bool(session.get("success", False)),
                "task_success_available": True,
                "task_success_reason": "external_teleop",
                "source_type": "external_teleop",
                "sim_backend": str(session.get("sim_backend", "") or "isaac_sim"),
                "success_source": "external_teleop",
                "is_manipulation_task": True,
                "robot_type": str(session.get("robot_type", "") or ""),
                "robot_asset_ref": str(session.get("robot_asset_ref", "") or ""),
                "action_space": str(session.get("action_space", "") or ""),
                "action_dim": int(session.get("action_dim", 0) or 0),
                "camera_ids": list(session.get("camera_ids", []) or []),
                "video_paths": video_paths,
                "calibration_refs": dict(session.get("calibration_refs", {}) or {}),
                "teleop_device": str(session.get("teleop_device", "") or ""),
                "scene_id": str(session.get("scene_id", "") or ""),
                "episode_ref": str(session.get("episode_ref", "") or ""),
                "start_state_hash": str(session.get("start_state_hash", "") or ""),
                "world_snapshot_hash": str(session.get("scene_id", "") or ""),
                "initial_camera": {"camera_id": primary_camera_id} if primary_camera_id else {},
                "path_context": {
                    "source_name": str(payload.get("source_name", "teleop") or "teleop"),
                    "session_id": str(session.get("session_id", "") or ""),
                },
                "eval_cell_id": (
                    f"external::{session.get('task_id', '')}::{int(session.get('demo_index', rollout_index))}"
                ),
            }
        )
    return rows


def write_external_rollout_rows(output_path: Path, rows: List[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json({"rows": rows}, output_path)


def load_external_rollout_rows(path: Path) -> List[Dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    return [dict(row) for row in rows if isinstance(row, dict)]


def load_external_rollouts_from_config(config: ValidationConfig) -> List[Dict[str, Any]]:
    ext_cfg = config.external_rollouts
    if not external_rollouts_enabled_for_policy(ext_cfg):
        return []
    manifest_path = ext_cfg.manifest_path
    if manifest_path is None or not manifest_path.exists():
        return []
    payload = load_and_validate_teleop_manifest(manifest_path, require_existing_paths=True)
    return convert_teleop_sessions_to_rollout_rows(payload)


def load_external_rollouts_for_policy(
    config: ValidationConfig,
    previous_results: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    ext_cfg = config.external_rollouts
    if not external_rollouts_enabled_for_policy(ext_cfg):
        return []

    previous = previous_results or {}
    stage_result = previous.get("s1g_external_rollout_ingest")
    rows_path = None
    if stage_result is not None:
        rows_path = str(getattr(stage_result, "outputs", {}).get("rollout_rows_path", "") or "")
    if rows_path:
        path = Path(rows_path)
        if path.exists():
            return load_external_rollout_rows(path)
    return load_external_rollouts_from_config(config)


def _load_json_sequence(path: Path) -> List[Any]:
    payload = read_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("actions", "states", "sequence", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Expected JSON sequence in {path}")
