"""Service-driven session helpers for NeoVerse site-world runtime orchestration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np

from .config import PolicyAdapterConfig, ValidationConfig
from .neoverse_runtime_client import NeoVerseRuntimeClient, NeoVerseRuntimeClientConfig
from .policy_adapters import get_policy_adapter
from .public_contract import public_runtime_label
from .training.rlds_export import export_rollouts_to_rlds_jsonl


class HostedSessionError(RuntimeError):
    pass


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _session_state_path(work_dir: Path) -> Path:
    return work_dir / "session_state.json"


def _episode_state_path(work_dir: Path) -> Path:
    return work_dir / "episode_state.json"


def _rollouts_dir(work_dir: Path) -> Path:
    return work_dir / "rollouts"


def _canonical_adapter_name(name: str) -> str:
    key = (name or "").strip().lower()
    if key in {"openvla_oft", "openvla-oft", "oft", "openvla", "open-vla"}:
        return "openvla_oft"
    if key in {"pi05", "pi0.5", "openpi"}:
        return "pi05"
    if key in {"dreamzero", "dream-zero", "dz"}:
        return "dreamzero"
    if key in {"mock", "session_mock", "test"}:
        return "mock"
    return key


def _env_truthy(name: str) -> bool:
    return (str(__import__("os").environ.get(name, "") or "").strip().lower() in {"1", "true", "yes", "on"})


def _camera_catalog(robot_profile: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cameras = robot_profile.get("observation_cameras")
    if not isinstance(cameras, list) or not cameras:
        raise HostedSessionError(f"Robot profile {robot_profile.get('id')} is missing observation cameras.")
    return [dict(item) for item in cameras if isinstance(item, Mapping)]


def _save_frame(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def _load_frame(path: str) -> np.ndarray:
    frame = cv2.imread(str(path))
    if frame is None:
        raise HostedSessionError(f"Could not load observation frame from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _decode_png(payload: bytes) -> np.ndarray:
    array = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HostedSessionError("NeoVerse runtime returned invalid render bytes.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _write_video(frame_paths: Sequence[str], output_path: Path) -> None:
    if not frame_paths:
        raise HostedSessionError(f"No frames available for video export: {output_path}")
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise HostedSessionError(f"Could not read rollout frame: {frame_paths[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        writer.write(frame)
    writer.release()


def _adjacent_site_world_paths(registration_path: Path) -> tuple[Path, Path]:
    root = registration_path.parent
    return root / "site_world_health.json", root / "site_world_spec.json"


def _optional_path(value: Any) -> Optional[Path]:
    text = str(value or "").strip()
    if not text or text.startswith("gs://") or text.startswith("http://") or text.startswith("https://"):
        return None
    return Path(text).resolve()


def _grounding_summary(spec: Mapping[str, Any]) -> Dict[str, Any]:
    conditioning = dict(spec.get("conditioning") or {}) if isinstance(spec.get("conditioning"), Mapping) else {}
    local_paths = dict(conditioning.get("local_paths") or {}) if isinstance(conditioning.get("local_paths"), Mapping) else {}
    geometry = dict(spec.get("geometry") or {}) if isinstance(spec.get("geometry"), Mapping) else {}
    qualification_references = (
        dict(spec.get("qualification_references") or {})
        if isinstance(spec.get("qualification_references"), Mapping)
        else {}
    )
    visuals = [
        _optional_path(local_paths.get("keyframe_path")),
        _optional_path(local_paths.get("raw_video_path")),
        _optional_path(conditioning.get("keyframe_uri")),
        _optional_path(conditioning.get("raw_video_uri")),
    ]
    arkit_poses = _optional_path(local_paths.get("arkit_poses_path")) or _optional_path(conditioning.get("arkit_poses_uri"))
    arkit_intrinsics = _optional_path(local_paths.get("arkit_intrinsics_path")) or _optional_path(conditioning.get("arkit_intrinsics_uri"))
    depth_path = _optional_path(local_paths.get("depth_path")) or _optional_path(conditioning.get("depth_uri"))
    occupancy_path = _optional_path(local_paths.get("occupancy_path")) or _optional_path(geometry.get("occupancy_path"))
    collision_path = _optional_path(local_paths.get("collision_path")) or _optional_path(geometry.get("collision_path"))
    object_index_path = _optional_path(local_paths.get("object_index_path")) or _optional_path(geometry.get("object_index_path"))
    object_geometry_path = _optional_path(local_paths.get("object_geometry_manifest_path")) or _optional_path(geometry.get("object_geometry_manifest_path"))
    checks = {
        "visual_source": any(path is not None and path.exists() for path in visuals),
        "arkit_poses": bool(arkit_poses and arkit_poses.exists()),
        "arkit_intrinsics": bool(arkit_intrinsics and arkit_intrinsics.exists()),
        "depth": bool(depth_path and depth_path.exists()),
        "occupancy": bool(occupancy_path and occupancy_path.exists()),
        "collision": bool(collision_path and collision_path.exists()),
        "object_index": bool(object_index_path and object_index_path.exists()),
        "object_geometry": bool(object_geometry_path and object_geometry_path.exists()),
        "qualification_refs": bool(qualification_references),
    }
    missing_required = [key for key in ("visual_source", "arkit_poses", "arkit_intrinsics") if not checks[key]]
    missing_optional = [key for key in ("depth", "occupancy", "collision", "object_index", "object_geometry") if not checks[key]]
    return {
        "checks": checks,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "qualification_state": str(spec.get("qualification_state") or ""),
        "downstream_evaluation_eligibility": spec.get("downstream_evaluation_eligibility"),
        "task_catalog_count": len(list(spec.get("task_catalog", []) or [])),
        "scenario_catalog_count": len(list(spec.get("scenario_catalog", []) or [])),
        "start_state_catalog_count": len(list(spec.get("start_state_catalog", []) or [])),
        "robot_profile_count": len(list(spec.get("robot_profiles", []) or [])),
    }


def _load_site_world_bundle(registration_path: Path) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    registration = _read_json(registration_path)
    health_path, spec_path = _adjacent_site_world_paths(registration_path)
    health = _read_json(health_path) if health_path.exists() else {}
    spec = _read_json(spec_path) if spec_path.exists() else {}
    return registration, health, spec, _grounding_summary(spec) if spec else {
        "checks": {},
        "missing_required": [],
        "missing_optional": [],
        "qualification_state": "",
        "downstream_evaluation_eligibility": None,
        "task_catalog_count": 0,
        "scenario_catalog_count": 0,
        "start_state_catalog_count": 0,
        "robot_profile_count": 0,
    }


def _resolve_runtime_client(
    config: ValidationConfig,
    registration: Mapping[str, Any],
) -> NeoVerseRuntimeClient:
    service_cfg = config.scene_memory_runtime.neoverse_service
    service_url = str(service_cfg.service_url or "").strip() or str(registration.get("runtime_base_url") or "").strip()
    if not service_url:
        raise HostedSessionError("NeoVerse runtime service URL is not configured in config or site-world registration.")
    api_key = ""
    if service_cfg.api_key_env:
        api_key = str(__import__("os").environ.get(service_cfg.api_key_env, "") or "").strip()
    return NeoVerseRuntimeClient(
        NeoVerseRuntimeClientConfig(
            service_url=service_url.rstrip("/"),
            api_key=api_key,
            timeout_seconds=max(1, int(service_cfg.timeout_seconds)),
        )
    )


def _resolve_policy_adapter(
    config: ValidationConfig,
    policy_payload: Mapping[str, Any],
    *,
    robot_profile: Mapping[str, Any],
):
    adapter_name = _canonical_adapter_name(
        str(policy_payload.get("adapter_name") or robot_profile.get("default_policy_adapter") or config.policy_adapter.name)
    )
    if adapter_name == "mock" and not _env_truthy("BLUEPRINT_ALLOW_MOCK_POLICY_ADAPTER"):
        raise HostedSessionError("Mock policy adapter is disabled for production hosted sessions.")
    allowed = {
        _canonical_adapter_name(str(item))
        for item in robot_profile.get("allowed_policy_adapters", []) or []
        if str(item).strip()
    }
    if allowed and adapter_name not in allowed:
        raise HostedSessionError(
            f"Robot profile {robot_profile.get('id')} does not allow policy adapter {adapter_name}."
        )
    adapter_config = PolicyAdapterConfig(
        name=adapter_name,
        openvla=config.policy_adapter.openvla,
        pi05=config.policy_adapter.pi05,
        dreamzero=config.policy_adapter.dreamzero,
    )
    return get_policy_adapter(adapter_config), adapter_config


def _policy_load_args(
    config: ValidationConfig,
    policy_payload: Mapping[str, Any],
) -> tuple[str, Optional[Path], str]:
    model_name = str(policy_payload.get("model_name") or config.eval_policy.model_name or "").strip()
    checkpoint_value = str(policy_payload.get("checkpoint_path") or "").strip()
    checkpoint_path = Path(checkpoint_value).resolve() if checkpoint_value else config.eval_policy.checkpoint_path
    device = str(policy_payload.get("device") or "cpu")
    return model_name, checkpoint_path, device


def _materialize_observation(
    *,
    client: NeoVerseRuntimeClient,
    remote_session_id: str,
    robot_profile: Mapping[str, Any],
    remote_episode: Mapping[str, Any],
    episode_dir: Path,
    step_index: int,
) -> Dict[str, Any]:
    remote_observation = dict(remote_episode.get("observation", {}) or {})
    camera_catalog = {str(item.get("id") or ""): dict(item) for item in _camera_catalog(robot_profile)}
    remote_cameras = list(remote_observation.get("cameraFrames", []) or [])
    primary_camera_id = str(remote_observation.get("primaryCameraId") or "")
    if not primary_camera_id and remote_cameras:
        primary_camera_id = str((remote_cameras[0] or {}).get("cameraId") or "")
    local_summaries: List[Dict[str, Any]] = []
    local_paths: Dict[str, str] = {}

    for camera in remote_cameras:
        if not isinstance(camera, Mapping):
            continue
        camera_id = str(camera.get("cameraId") or "").strip()
        if not camera_id:
            continue
        remote_path = str(camera.get("framePath") or "")
        payload = client.render_bytes(remote_session_id, camera_id=camera_id)
        frame = _decode_png(payload)
        output_path = episode_dir / "cameras" / camera_id / f"frame_{step_index:03d}.png"
        _save_frame(output_path, frame)
        local_paths[camera_id] = str(output_path)
        catalog_entry = camera_catalog.get(camera_id, {})
        local_summaries.append(
            {
                "cameraId": camera_id,
                "role": str(camera.get("role") or catalog_entry.get("role") or ""),
                "required": bool(camera.get("required", catalog_entry.get("required", False))),
                "available": True,
                "framePath": str(output_path),
                "remoteFramePath": remote_path,
            }
        )

    if not primary_camera_id:
        raise HostedSessionError("NeoVerse runtime observation did not include any cameras.")
    if primary_camera_id not in local_paths:
        payload = client.render_bytes(remote_session_id, camera_id=primary_camera_id)
        frame = _decode_png(payload)
        output_path = episode_dir / "cameras" / primary_camera_id / f"frame_{step_index:03d}.png"
        _save_frame(output_path, frame)
        local_paths[primary_camera_id] = str(output_path)

    return {
        "stepIndex": int(remote_episode.get("stepIndex", step_index) or step_index),
        "primaryCameraId": primary_camera_id,
        "frame_path": local_paths[primary_camera_id],
        "cameraFrames": local_summaries,
        "camera_frame_paths": local_paths,
        "runtimeMetadata": dict(remote_observation.get("runtimeMetadata", {}) or {}),
        "worldSnapshot": dict(remote_observation.get("worldSnapshot", {}) or {}),
        "remoteObservation": remote_observation,
    }


def _episode_payload(episode_state: Mapping[str, Any]) -> Dict[str, Any]:
    runtime_metadata = dict(((episode_state.get("observation") or {}).get("runtimeMetadata") or {}))
    return {
        "episodeId": episode_state["episode_id"],
        "taskId": episode_state["task_id"],
        "task": episode_state["task"],
        "scenarioId": episode_state["scenario_id"],
        "scenario": episode_state["scenario"],
        "startStateId": episode_state["start_state_id"],
        "startState": episode_state["start_state"],
        "status": episode_state["status"],
        "stepIndex": int(episode_state["step_index"]),
        "done": bool(episode_state["done"]),
        "reward": episode_state.get("reward"),
        "success": episode_state.get("success"),
        "failureReason": episode_state.get("failure_reason"),
        "observation": episode_state.get("observation"),
        "actionTrace": episode_state.get("action_trace", []),
        "observationCameras": episode_state.get("observation_cameras", []),
        "artifactUris": episode_state.get("artifact_uris", {}),
        "canonicalPackageVersion": runtime_metadata.get("canonical_package_version"),
        "presentationConfig": runtime_metadata.get("presentation_config"),
        "qualityFlags": runtime_metadata.get("quality_flags", {}),
        "protectedRegionViolations": runtime_metadata.get("protected_region_violations", {}),
        "debugArtifacts": runtime_metadata.get("debug_artifacts", {}),
    }


def _finalize_episode_artifacts(episode_state: Dict[str, Any], *, session_work_dir: Path) -> None:
    episode_dir = _rollouts_dir(session_work_dir) / str(episode_state["episode_id"])
    primary_camera_id = str(episode_state["observation"].get("primaryCameraId") or "")
    camera_frame_paths = episode_state.get("camera_frame_paths", {}) or {}
    primary_frame_paths = camera_frame_paths.get(primary_camera_id) if isinstance(camera_frame_paths, Mapping) else None
    if isinstance(primary_frame_paths, list) and primary_frame_paths and episode_state["artifact_uris"].get("rollout_video") is None:
        video_path = episode_dir / f"{episode_dir.name}.mp4"
        _write_video([str(item) for item in primary_frame_paths], video_path)
        episode_state["artifact_uris"]["rollout_video"] = str(video_path)
    score_path = episode_dir / "score.json"
    actions_path = episode_dir / "actions.json"
    _write_json(
        score_path,
        {
            "success": bool(episode_state.get("success", False)),
            "reward": float(episode_state.get("reward", 0.0) or 0.0),
            "failure_reason": episode_state.get("failure_reason"),
            "num_steps": int(episode_state.get("step_index", 0)),
        },
    )
    _write_json(actions_path, {"actions": episode_state.get("action_trace", [])})
    episode_state["artifact_uris"]["score"] = str(score_path)
    episode_state["artifact_uris"]["actions"] = str(actions_path)
    _write_json(episode_dir / "episode_state.json", episode_state)


def create_session(
    *,
    config: ValidationConfig,
    session_id: str,
    session_work_dir: Path,
    registration_path: Path,
    robot_profile_id: str,
    task_id: str,
    scenario_id: str,
    start_state_id: str,
    policy_payload: Optional[Mapping[str, Any]] = None,
    export_modes: Sequence[str] = (),
    robot_profile_override: Optional[Mapping[str, Any]] = None,
    notes: str = "",
    unsafe_allow_blocked_site_world: bool = False,
) -> Dict[str, Any]:
    registration, health, spec, grounding = _load_site_world_bundle(registration_path)
    if not registration.get("site_world_id"):
        raise HostedSessionError(f"Invalid site-world registration: {registration_path}")
    allow_blocked_site_world = bool(unsafe_allow_blocked_site_world) or _env_truthy(
        "BLUEPRINT_UNSAFE_ALLOW_BLOCKED_SITE_WORLD"
    )
    if health and not bool(health.get("launchable", False)) and not allow_blocked_site_world:
        blockers = ", ".join(str(item) for item in health.get("blockers", []) if str(item).strip())
        raise HostedSessionError(f"Site world is not launchable: {blockers or 'unknown blockers'}")

    robot_profiles = [dict(item) for item in registration.get("robot_profiles", []) if isinstance(item, Mapping)]
    task_entries = [dict(item) for item in registration.get("task_catalog", []) if isinstance(item, Mapping)]
    scenario_entries = [dict(item) for item in registration.get("scenario_catalog", []) if isinstance(item, Mapping)]
    start_state_entries = [dict(item) for item in registration.get("start_state_catalog", []) if isinstance(item, Mapping)]

    def _find(entries: Sequence[Mapping[str, Any]], selected_id: str, label: str) -> Dict[str, Any]:
        for entry in entries:
            if str(entry.get("id") or entry.get("task_id") or "").strip() == selected_id:
                return dict(entry)
        raise HostedSessionError(f"Unsupported {label}: {selected_id}")

    robot_profile = _find(robot_profiles, robot_profile_id, "robot profile")
    if isinstance(robot_profile_override, Mapping):
        robot_profile.update(dict(robot_profile_override))
    task_entry = _find(task_entries, task_id, "task")
    scenario_entry = _find(scenario_entries, scenario_id, "scenario")
    start_state_entry = _find(start_state_entries, start_state_id, "start state")

    session_work_dir.mkdir(parents=True, exist_ok=True)
    _rollouts_dir(session_work_dir).mkdir(parents=True, exist_ok=True)

    client = _resolve_runtime_client(config, registration)
    create_payload = client.create_session(
        str(registration["site_world_id"]),
        session_id=session_id,
        robot_profile_id=robot_profile_id,
        task_id=task_id,
        scenario_id=scenario_id,
        start_state_id=start_state_id,
        notes=notes,
        canonical_package_uri=str((policy_payload or {}).get("canonical_package_uri") or "") or None,
        canonical_package_version=str((policy_payload or {}).get("canonical_package_version") or "") or None,
        prompt=str((policy_payload or {}).get("prompt") or "") or None,
        trajectory=(policy_payload or {}).get("trajectory"),
        presentation_model=str((policy_payload or {}).get("presentation_model") or "") or None,
        debug_mode=bool((policy_payload or {}).get("debug_mode", False)),
        unsafe_allow_blocked_site_world=allow_blocked_site_world,
    )
    runtime_probe = client.probe_runtime()

    session_state = {
        "schema_version": "v1",
        "session_id": session_id,
        "remote_session_id": str(create_payload.get("session_id") or session_id),
        "site_world_registration_path": str(registration_path.resolve()),
        "site_world_id": registration.get("site_world_id"),
        "build_id": registration.get("build_id"),
        "runtime_backend_selected": "neoverse_service",
        "runtime_backend_public_name": public_runtime_label("neoverse_service"),
        "runtime_service_url": str(client.config.service_url),
        "status": "ready",
        "scene_id": registration.get("scene_id"),
        "capture_id": registration.get("capture_id"),
        "site_submission_id": registration.get("site_submission_id"),
        "robot_profile": robot_profile,
        "task": dict(task_entry),
        "scenario": dict(scenario_entry),
        "start_state": dict(start_state_entry),
        "notes": notes,
        "policy": dict(policy_payload or {}),
        "unsafe_allow_blocked_site_world": allow_blocked_site_world,
        "export_modes": [str(item) for item in export_modes if str(item).strip()],
        "current_episode_id": None,
        "latest_episode_path": None,
        "runtime_smoke_path": None,
        "batch_summary_path": None,
        "artifact_uris": {
            "session_state": str(_session_state_path(session_work_dir)),
        },
        "dataset_artifacts": {},
        "grounding_summary": grounding,
        "runtime_probe": runtime_probe,
        "canonical_package_version": create_payload.get("canonical_package_version"),
        "presentation_config": create_payload.get("presentation_config"),
        "unsafe_allow_blocked_site_world": bool(
            create_payload.get("unsafe_allow_blocked_site_world", allow_blocked_site_world)
        ),
        "quality_flags": create_payload.get("quality_flags", {}),
        "protected_region_violations": create_payload.get("protected_region_violations", {}),
        "debug_artifacts": create_payload.get("debug_artifacts", {}),
    }
    _write_json(_session_state_path(session_work_dir), session_state)
    return {
        "session_id": session_id,
        "runtime_backend_selected": "neoverse_service",
        "runtime_backend_public_name": public_runtime_label("neoverse_service"),
        "status": "ready",
        "siteWorldId": registration.get("site_world_id"),
        "robotProfile": robot_profile,
        "observationCameras": list(create_payload.get("observation_cameras", []) or _camera_catalog(robot_profile)),
        "artifact_uris": session_state["artifact_uris"],
        "dataset_artifacts": session_state["dataset_artifacts"],
        "grounding_summary": grounding,
        "canonical_package_version": create_payload.get("canonical_package_version"),
        "presentation_config": create_payload.get("presentation_config"),
        "unsafe_allow_blocked_site_world": bool(
            create_payload.get("unsafe_allow_blocked_site_world", allow_blocked_site_world)
        ),
        "quality_flags": create_payload.get("quality_flags", {}),
        "protected_region_violations": create_payload.get("protected_region_violations", {}),
        "debug_artifacts": create_payload.get("debug_artifacts", {}),
    }


def reset_session(
    *,
    config: ValidationConfig,
    session_id: str,
    session_work_dir: Path,
    task_id: Optional[str],
    scenario_id: Optional[str],
    start_state_id: Optional[str],
    seed: Optional[int],
) -> Dict[str, Any]:
    del seed
    session_state = _read_json(_session_state_path(session_work_dir))
    registration_path = Path(str(session_state["site_world_registration_path"]))
    registration, _health, _spec, grounding = _load_site_world_bundle(registration_path)
    client = _resolve_runtime_client(config, registration)
    remote_session_id = str(session_state.get("remote_session_id") or session_id)

    task_entry = dict(session_state.get("task") or {})
    scenario_entry = dict(session_state.get("scenario") or {})
    start_state_entry = dict(session_state.get("start_state") or {})
    if task_id:
        task_entry = next(
            (dict(item) for item in registration.get("task_catalog", []) if isinstance(item, Mapping) and str(item.get("id") or item.get("task_id") or "").strip() == task_id),
            task_entry,
        )
    if scenario_id:
        scenario_entry = next(
            (dict(item) for item in registration.get("scenario_catalog", []) if isinstance(item, Mapping) and str(item.get("id") or "").strip() == scenario_id),
            scenario_entry,
        )
    if start_state_id:
        start_state_entry = next(
            (dict(item) for item in registration.get("start_state_catalog", []) if isinstance(item, Mapping) and str(item.get("id") or "").strip() == start_state_id),
            start_state_entry,
        )

    reset_payload = client.reset_session(
        remote_session_id,
        task_id=str(task_entry.get("id") or task_entry.get("task_id") or "") or None,
        scenario_id=str(scenario_entry.get("id") or "") or None,
        start_state_id=str(start_state_entry.get("id") or "") or None,
    )
    remote_episode = dict(reset_payload.get("episode", {}) or {})
    episode_id = f"episode-{hashlib.sha256('::'.join([session_id, str(task_entry.get('id') or ''), str(start_state_entry.get('id') or '')]).encode('utf-8')).hexdigest()[:8]}"
    episode_dir = _rollouts_dir(session_work_dir) / episode_id
    observation = _materialize_observation(
        client=client,
        remote_session_id=remote_session_id,
        robot_profile=dict(session_state["robot_profile"]),
        remote_episode=remote_episode,
        episode_dir=episode_dir,
        step_index=0,
    )
    observation_cameras = observation.get("cameraFrames", [])
    camera_frame_paths = {
        camera["cameraId"]: [camera["framePath"]]
        for camera in observation_cameras
        if isinstance(camera, Mapping) and camera.get("framePath")
    }
    episode_state = {
        "schema_version": "v1",
        "session_id": session_id,
        "remote_session_id": remote_session_id,
        "episode_id": episode_id,
        "site_world_id": session_state.get("site_world_id"),
        "build_id": session_state.get("build_id"),
        "task_id": str(task_entry.get("id") or task_entry.get("task_id") or ""),
        "task": str(task_entry.get("task_text") or task_entry.get("name") or ""),
        "scenario_id": str(scenario_entry.get("id") or ""),
        "scenario": str(scenario_entry.get("name") or ""),
        "start_state_id": str(start_state_entry.get("id") or ""),
        "start_state": str(start_state_entry.get("name") or ""),
        "status": str(remote_episode.get("status") or "ready"),
        "step_index": int(remote_episode.get("stepIndex", 0) or 0),
        "done": bool(remote_episode.get("done", False)),
        "reward": float(remote_episode.get("reward", 0.0) or 0.0),
        "success": remote_episode.get("success"),
        "failure_reason": remote_episode.get("failureReason"),
        "action_trace": [],
        "observation": observation,
        "observation_cameras": observation_cameras,
        "camera_frame_paths": camera_frame_paths,
        "artifact_uris": {
            "episode_state": str(episode_dir / "episode_state.json"),
        },
        "remote_episode": remote_episode,
    }
    runtime_smoke_path = session_work_dir / "runtime_smoke.json"
    _write_json(
        runtime_smoke_path,
        {
            "schema_version": "v1",
            "session_id": session_id,
            "remote_session_id": remote_session_id,
            "site_world_id": session_state.get("site_world_id"),
            "service_url": session_state.get("runtime_service_url"),
            "grounding_summary": grounding,
            "initial_episode": _episode_payload(episode_state),
        },
    )
    _write_json(_episode_state_path(session_work_dir), episode_state)
    session_state["current_episode_id"] = episode_id
    session_state["latest_episode_path"] = str(_episode_state_path(session_work_dir))
    session_state["runtime_smoke_path"] = str(runtime_smoke_path)
    session_state["status"] = "running"
    session_state["task"] = dict(task_entry)
    session_state["scenario"] = dict(scenario_entry)
    session_state["start_state"] = dict(start_state_entry)
    _write_json(_session_state_path(session_work_dir), session_state)
    return {"session_id": session_id, "episode": _episode_payload(episode_state)}


def step_session(
    *,
    config: ValidationConfig,
    session_work_dir: Path,
    episode_id: str,
    action: Optional[List[float]],
    auto_policy: bool,
) -> Dict[str, Any]:
    session_state = _read_json(_session_state_path(session_work_dir))
    episode_state = _read_json(_episode_state_path(session_work_dir))
    if episode_state.get("episode_id") != episode_id:
        raise HostedSessionError("Episode ID does not match current session episode state")
    registration = _read_json(Path(str(session_state["site_world_registration_path"])))
    client = _resolve_runtime_client(config, registration)

    robot_profile = dict(session_state["robot_profile"])
    policy_payload = dict(session_state.get("policy") or {})
    if auto_policy:
        policy_adapter, _adapter_config = _resolve_policy_adapter(config, policy_payload, robot_profile=robot_profile)
        model_name, checkpoint_path, device = _policy_load_args(config, policy_payload)
        handle = policy_adapter.load_policy(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        current_frame = _load_frame(str((episode_state.get("observation") or {}).get("frame_path") or ""))
        next_action = policy_adapter.predict_action(
            handle=handle,
            frame=current_frame,
            task_prompt=str(episode_state.get("task") or ""),
            unnorm_key=None,
            device=device,
        )
        action_list = np.asarray(next_action, dtype=np.float32).reshape(-1).tolist()
    else:
        action_list = list(action or [])

    expected_action_dim = int(((robot_profile.get("action_space") or {}).get("dim")) or len(action_list) or 7)
    if len(action_list) != expected_action_dim:
        raise HostedSessionError(
            f"Action-space mismatch: received {len(action_list)} values, expected {expected_action_dim}."
        )

    remote_session_id = str(session_state.get("remote_session_id") or session_state["session_id"])
    step_payload = client.step_session(remote_session_id, action=action_list)
    remote_episode = dict(step_payload.get("episode", {}) or {})
    next_step_index = int(remote_episode.get("stepIndex", int(episode_state.get("step_index", 0)) + 1) or 0)
    episode_dir = _rollouts_dir(session_work_dir) / str(episode_state["episode_id"])
    observation = _materialize_observation(
        client=client,
        remote_session_id=remote_session_id,
        robot_profile=robot_profile,
        remote_episode=remote_episode,
        episode_dir=episode_dir,
        step_index=next_step_index,
    )
    for camera in observation.get("cameraFrames", []):
        if not isinstance(camera, Mapping):
            continue
        camera_id = str(camera.get("cameraId") or "")
        frame_path = str(camera.get("framePath") or "")
        if not camera_id or not frame_path:
            continue
        episode_state.setdefault("camera_frame_paths", {}).setdefault(camera_id, []).append(frame_path)

    episode_state["step_index"] = next_step_index
    episode_state["reward"] = float(remote_episode.get("reward", episode_state.get("reward", 0.0)) or 0.0)
    episode_state["done"] = bool(remote_episode.get("done", False))
    episode_state["success"] = remote_episode.get("success")
    episode_state["failure_reason"] = remote_episode.get("failureReason")
    episode_state["status"] = str(remote_episode.get("status") or ("completed" if episode_state["done"] else "running"))
    episode_state["action_trace"].append(action_list)
    episode_state["observation"] = observation
    episode_state["observation_cameras"] = observation.get("cameraFrames", [])
    episode_state["remote_episode"] = remote_episode
    episode_state.setdefault("artifact_uris", {})
    _finalize_episode_artifacts(episode_state, session_work_dir=session_work_dir)
    _write_json(_episode_state_path(session_work_dir), episode_state)
    return {"session_id": session_state["session_id"], "episode": _episode_payload(episode_state)}


def run_batch(
    *,
    config: ValidationConfig,
    session_work_dir: Path,
    num_episodes: int,
    task_id: Optional[str],
    scenario_id: Optional[str],
    start_state_id: Optional[str],
    seed: Optional[int],
    max_steps: Optional[int],
) -> Dict[str, Any]:
    del seed
    session_state = _read_json(_session_state_path(session_work_dir))
    assignments: List[Dict[str, Any]] = []
    failures: List[str] = []
    max_steps_value = int(max_steps or 6)
    for rollout_index in range(num_episodes):
        reset_payload = reset_session(
            config=config,
            session_id=str(session_state["session_id"]),
            session_work_dir=session_work_dir,
            task_id=task_id,
            scenario_id=scenario_id,
            start_state_id=start_state_id,
            seed=None,
        )
        episode = reset_payload["episode"]
        episode_payload = episode
        for _ in range(max_steps_value):
            if bool(episode_payload.get("done")):
                break
            step_payload = step_session(
                config=config,
                session_work_dir=session_work_dir,
                episode_id=str(episode_payload["episodeId"]),
                action=None,
                auto_policy=True,
            )
            episode_payload = step_payload["episode"]
        assignments.append(
            {
                "episode_id": episode_payload["episodeId"],
                "rollout_index": rollout_index,
                "task_id": episode_payload["taskId"],
                "scenario_id": episode_payload["scenarioId"],
                "scenario": episode_payload["scenario"],
                "start_state_id": episode_payload["startStateId"],
                "start_state": episode_payload["startState"],
                "video_path": str((episode_payload.get("artifactUris") or {}).get("rollout_video") or ""),
                "score_path": str((episode_payload.get("artifactUris") or {}).get("score") or ""),
                "success": bool(episode_payload.get("success", False)),
            }
        )
        if episode_payload.get("failureReason"):
            failures.append(str(episode_payload["failureReason"]))
    num_success = sum(1 for item in assignments if item["success"])
    summary = {
        "batchRunId": f"batch-{hashlib.sha256(str(session_state['session_id']).encode('utf-8')).hexdigest()[:8]}",
        "status": "completed",
        "numEpisodes": num_episodes,
        "numSuccess": num_success,
        "numFailure": num_episodes - num_success,
        "successRate": round(num_success / float(max(num_episodes, 1)), 4),
        "commonFailureModes": sorted(set(failures)),
    }
    manifest_path = session_work_dir / "runtime_batch_manifest.json"
    _write_json(manifest_path, {"assignments": assignments, "summary": summary})
    session_state["batch_summary_path"] = str(manifest_path)
    _write_json(_session_state_path(session_work_dir), session_state)
    return {
        "session_id": session_state["session_id"],
        "batchRunId": summary["batchRunId"],
        "status": "completed",
        "assignments": assignments,
        "summary": summary,
        "artifact_uris": {"runtime_batch_manifest": str(manifest_path)},
    }


def stop_session(*, session_work_dir: Path) -> Dict[str, Any]:
    session_state = _read_json(_session_state_path(session_work_dir))
    session_state["status"] = "stopped"
    _write_json(_session_state_path(session_work_dir), session_state)
    return {"sessionId": session_state["session_id"], "status": "stopped"}


def export_session(*, session_work_dir: Path) -> Dict[str, Any]:
    session_state = _read_json(_session_state_path(session_work_dir))
    raw_bundle_dir = session_work_dir / "exports" / "raw_bundle"
    raw_bundle_dir.mkdir(parents=True, exist_ok=True)
    rollouts: List[Dict[str, Any]] = []
    rlds_rollouts: List[Dict[str, Any]] = []
    for rollout_dir in sorted(_rollouts_dir(session_work_dir).glob("*")):
        if not rollout_dir.is_dir():
            continue
        episode_state_path = rollout_dir / "episode_state.json"
        if not episode_state_path.exists():
            continue
        episode_state = _read_json(episode_state_path)
        if not episode_state.get("artifact_uris", {}).get("rollout_video"):
            _finalize_episode_artifacts(episode_state, session_work_dir=session_work_dir)
            _write_json(episode_state_path, episode_state)
        rollouts.append(
            {
                "episode_id": rollout_dir.name,
                "video_path": str(episode_state["artifact_uris"].get("rollout_video") or ""),
                "actions_path": str(episode_state["artifact_uris"].get("actions") or ""),
                "score_path": str(episode_state["artifact_uris"].get("score") or ""),
                "task_id": episode_state.get("task_id"),
                "task": episode_state.get("task"),
                "scenario_id": episode_state.get("scenario_id"),
                "scenario": episode_state.get("scenario"),
                "start_state_id": episode_state.get("start_state_id"),
                "start_state": episode_state.get("start_state"),
            }
        )
        rlds_rollouts.append(
            {
                "rollout_index": len(rlds_rollouts),
                "video_path": str(episode_state["artifact_uris"].get("rollout_video") or ""),
                "action_sequence": episode_state.get("action_trace", []),
                "task": str(episode_state.get("task") or ""),
                "eval_cell_id": str(episode_state.get("scenario_id") or ""),
                "task_spec_id": str(episode_state.get("task_id") or ""),
                "start_region_id": str(episode_state.get("start_state_id") or ""),
                "sim_backend": "neoverse_service",
                "success": bool(episode_state.get("success", False)),
                "task_success": bool(episode_state.get("success", False)),
                "task_success_available": True,
                "task_success_reason": str(episode_state.get("failure_reason") or ""),
                "task_score": float(episode_state.get("reward", 0.0) or 0.0),
            }
        )
    raw_manifest = {
        "schema_version": "v1",
        "session_id": session_state["session_id"],
        "rollouts": rollouts,
        "session_state_path": str(_session_state_path(session_work_dir)),
        "batch_summary_path": session_state.get("batch_summary_path"),
        "runtime_smoke_path": session_state.get("runtime_smoke_path"),
    }
    raw_manifest_path = raw_bundle_dir / "raw_session_bundle.json"
    _write_json(raw_manifest_path, raw_manifest)

    rlds_dir = session_work_dir / "exports" / "rlds"
    train_meta = export_rollouts_to_rlds_jsonl(
        rlds_rollouts,
        rlds_dir / "train",
        condition="hosted_session",
        split="train",
        task_threshold=0.5,
        min_steps_per_rollout=1,
        include_failed_rollouts=True,
    )
    rlds_manifest = {
        "schema_version": "v1",
        "session_id": session_state["session_id"],
        "train_jsonl": train_meta["episodes_jsonl"],
        "train_meta_path": str(rlds_dir / "train" / "episodes_meta.json"),
    }
    rlds_manifest_path = rlds_dir / "rlds_manifest.json"
    _write_json(rlds_manifest_path, rlds_manifest)

    export_path = session_work_dir / "export_manifest.json"
    manifest = {
        "schema_version": "v1",
        "session_id": session_state["session_id"],
        "raw_bundle": {
            "manifest_path": str(raw_manifest_path),
            "rollout_count": len(rollouts),
        },
        "rlds_dataset": {
            "manifest_path": str(rlds_manifest_path),
            "train_jsonl": train_meta["episodes_jsonl"],
            "format": "rlds_style_jsonl",
        },
        "rollouts": rollouts,
    }
    _write_json(export_path, manifest)
    return {
        "exportId": f"export-{session_state['session_id']}",
        "manifestUri": str(export_path),
        "artifact_uris": {
            "export_manifest": str(export_path),
            "raw_bundle": str(raw_manifest_path),
            "rlds_dataset": str(rlds_manifest_path),
        },
        "dataset_artifacts": {
            "rlds": {
                "manifestUri": str(rlds_manifest_path),
                "trainJsonl": train_meta["episodes_jsonl"],
            }
        },
    }
