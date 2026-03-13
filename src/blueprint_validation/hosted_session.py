"""NeoVerse hosted-session helpers for built site-world packages."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from blueprint_contracts.site_world_contract import SiteWorldIntakeError, load_site_world_bundle

from .config import ValidationConfig
from .neoverse_runtime_client import NeoVerseRuntimeClient, NeoVerseRuntimeClientConfig
from .optional_dependencies import require_optional_dependency
from .public_contract import public_runtime_label


class HostedSessionError(RuntimeError):
    pass


def _require_cv2():
    try:
        return require_optional_dependency("cv2", extra="vision", purpose="hosted session image/video IO")
    except RuntimeError as exc:  # pragma: no cover - optional dependency
        raise HostedSessionError(str(exc)) from exc


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row)) + "\n" for row in rows), encoding="utf-8")


def _session_state_path(work_dir: Path) -> Path:
    return work_dir / "session_state.json"


def _episode_state_path(work_dir: Path) -> Path:
    return work_dir / "episode_state.json"


def _rollouts_dir(work_dir: Path) -> Path:
    return work_dir / "rollouts"


def _resolve_runtime_client(config: ValidationConfig, registration: Mapping[str, Any]) -> NeoVerseRuntimeClient:
    service_cfg = config.scene_memory_runtime.neoverse_service
    service_url = str(service_cfg.service_url or "").strip() or str(registration.get("runtime_base_url") or "").strip()
    if not service_url:
        service_url = str(os.environ.get("NEOVERSE_RUNTIME_SERVICE_URL") or "").strip()
    if not service_url:
        raise HostedSessionError("NeoVerse runtime service URL is not configured in config, registration, or env.")
    api_key = ""
    if service_cfg.api_key_env:
        api_key = str(os.environ.get(service_cfg.api_key_env, "") or "").strip()
    return NeoVerseRuntimeClient(
        NeoVerseRuntimeClientConfig(
            service_url=service_url.rstrip("/"),
            api_key=api_key,
            timeout_seconds=max(1, int(service_cfg.timeout_seconds)),
        )
    )


def _catalog_entry(entries: Sequence[Mapping[str, Any]], selected_id: str, label: str) -> Dict[str, Any]:
    for entry in entries:
        if str(entry.get("id") or entry.get("task_id") or "").strip() == selected_id:
            return dict(entry)
    raise HostedSessionError(f"Unsupported {label}: {selected_id}")


def _camera_catalog(robot_profile: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    cameras = robot_profile.get("observation_cameras")
    if not isinstance(cameras, list) or not cameras:
        raise HostedSessionError(f"Robot profile {robot_profile.get('id')} is missing observation cameras.")
    return {
        str(item.get("id") or ""): dict(item)
        for item in cameras
        if isinstance(item, Mapping) and str(item.get("id") or "").strip()
    }


def _decode_png(payload: bytes) -> np.ndarray:
    cv2 = _require_cv2()
    array = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HostedSessionError("NeoVerse runtime returned invalid PNG bytes.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _save_frame(path: Path, frame: np.ndarray) -> None:
    cv2 = _require_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def _write_video(frame_paths: Sequence[str], output_path: Path) -> None:
    if not frame_paths:
        return
    cv2 = _require_cv2()
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            writer.write(frame)
    writer.release()


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
    camera_catalog = _camera_catalog(robot_profile)
    remote_cameras = list(remote_observation.get("cameraFrames", []) or [])
    primary_camera_id = str(remote_observation.get("primaryCameraId") or "")
    if not primary_camera_id and remote_cameras:
        primary_camera_id = str((remote_cameras[0] or {}).get("cameraId") or "")

    local_paths: Dict[str, str] = {}
    local_summaries: List[Dict[str, Any]] = []
    for camera in remote_cameras:
        if not isinstance(camera, Mapping):
            continue
        camera_id = str(camera.get("cameraId") or "").strip()
        if not camera_id:
            continue
        payload = client.render_bytes(remote_session_id, camera_id=camera_id)
        frame = _decode_png(payload)
        output_path = episode_dir / "cameras" / camera_id / f"frame_{step_index:03d}.png"
        _save_frame(output_path, frame)
        local_paths[camera_id] = str(output_path)
        local_summaries.append(
            {
                "cameraId": camera_id,
                "role": str(camera.get("role") or camera_catalog.get(camera_id, {}).get("role") or ""),
                "required": bool(camera.get("required", camera_catalog.get(camera_id, {}).get("required", False))),
                "available": True,
                "framePath": str(output_path),
                "remoteFramePath": str(camera.get("framePath") or ""),
            }
        )

    if primary_camera_id and primary_camera_id not in local_paths:
        payload = client.render_bytes(remote_session_id, camera_id=primary_camera_id)
        frame = _decode_png(payload)
        output_path = episode_dir / "cameras" / primary_camera_id / f"frame_{step_index:03d}.png"
        _save_frame(output_path, frame)
        local_paths[primary_camera_id] = str(output_path)

    if not primary_camera_id:
        raise HostedSessionError("NeoVerse runtime observation did not include any cameras.")

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
    primary_camera_id = str((episode_state.get("observation") or {}).get("primaryCameraId") or "")
    camera_frame_paths = episode_state.get("camera_frame_paths", {}) if isinstance(episode_state.get("camera_frame_paths"), Mapping) else {}
    primary_frame_paths = camera_frame_paths.get(primary_camera_id)
    if isinstance(primary_frame_paths, list) and primary_frame_paths and episode_state.get("artifact_uris", {}).get("rollout_video") is None:
        video_path = episode_dir / f"{episode_dir.name}.mp4"
        _write_video([str(item) for item in primary_frame_paths], video_path)
        episode_state.setdefault("artifact_uris", {})["rollout_video"] = str(video_path)
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
    episode_state.setdefault("artifact_uris", {})["score"] = str(score_path)
    episode_state.setdefault("artifact_uris", {})["actions"] = str(actions_path)
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
    notes: str = "",
    canonical_package_uri: Optional[str] = None,
    canonical_package_version: Optional[str] = None,
    prompt: Optional[str] = None,
    trajectory: Mapping[str, Any] | str | None = None,
    presentation_model: Optional[str] = None,
    debug_mode: bool = False,
    unsafe_allow_blocked_site_world: bool = False,
) -> Dict[str, Any]:
    try:
        bundle = load_site_world_bundle(registration_path, require_spec=True)
    except SiteWorldIntakeError as exc:
        raise HostedSessionError(str(exc)) from exc

    registration = bundle.registration
    health = bundle.health
    site_world = bundle.resolved
    if not registration.get("site_world_id"):
        raise HostedSessionError(f"Invalid site-world registration: {registration_path}")
    allow_blocked_site_world = bool(unsafe_allow_blocked_site_world) or str(os.environ.get("BLUEPRINT_UNSAFE_ALLOW_BLOCKED_SITE_WORLD", "")).strip().lower() in {"1", "true", "yes", "on"}
    if not allow_blocked_site_world and not bool(health.get("launchable", False)):
        blockers = ", ".join(str(item) for item in health.get("blockers", []) if str(item).strip())
        raise HostedSessionError(f"Site world is not launchable: {blockers or 'unknown blockers'}")

    robot_profiles = [dict(item) for item in site_world.get("robot_profiles", []) if isinstance(item, Mapping)]
    task_entries = [dict(item) for item in site_world.get("task_catalog", []) if isinstance(item, Mapping)]
    scenario_entries = [dict(item) for item in site_world.get("scenario_catalog", []) if isinstance(item, Mapping)]
    start_state_entries = [dict(item) for item in site_world.get("start_state_catalog", []) if isinstance(item, Mapping)]

    robot_profile = _catalog_entry(robot_profiles, robot_profile_id, "robot profile")
    task_entry = _catalog_entry(task_entries, task_id, "task")
    scenario_entry = _catalog_entry(scenario_entries, scenario_id, "scenario")
    start_state_entry = _catalog_entry(start_state_entries, start_state_id, "start state")

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
        canonical_package_uri=canonical_package_uri,
        canonical_package_version=canonical_package_version,
        prompt=prompt,
        trajectory=trajectory,
        presentation_model=presentation_model,
        debug_mode=bool(debug_mode),
        unsafe_allow_blocked_site_world=allow_blocked_site_world,
    )
    runtime_probe = client.probe_runtime()

    session_state = {
        "schema_version": "v1",
        "session_id": session_id,
        "remote_session_id": str(create_payload.get("session_id") or session_id),
        "site_world_registration_path": str(registration_path.resolve()),
        "site_world_spec_path": str(bundle.spec_path.resolve()),
        "site_world_id": registration.get("site_world_id"),
        "scene_id": registration.get("scene_id"),
        "capture_id": registration.get("capture_id"),
        "build_id": registration.get("build_id"),
        "status": "ready",
        "runtime_backend_selected": "neoverse_service",
        "runtime_backend_public_name": public_runtime_label("neoverse_service"),
        "runtime_service_url": str(client.config.service_url),
        "robot_profile": robot_profile,
        "task": task_entry,
        "scenario": scenario_entry,
        "start_state": start_state_entry,
        "notes": notes,
        "runtime_probe": runtime_probe,
        "current_episode_id": None,
        "latest_episode_path": None,
        "runtime_smoke_path": None,
        "batch_summary_path": None,
        "artifact_uris": {"session_state": str(_session_state_path(session_work_dir))},
    }
    _write_json(_session_state_path(session_work_dir), session_state)
    return {
        "session_id": session_id,
        "status": "ready",
        "siteWorldId": registration.get("site_world_id"),
        "runtime_backend_selected": "neoverse_service",
        "runtime_backend_public_name": public_runtime_label("neoverse_service"),
        "robotProfile": robot_profile,
        "observationCameras": list(create_payload.get("observation_cameras", []) or _camera_catalog(robot_profile).values()),
        "artifact_uris": session_state["artifact_uris"],
        "runtime_probe": runtime_probe,
    }


def reset_session(
    *,
    config: ValidationConfig,
    session_id: str,
    session_work_dir: Path,
    task_id: Optional[str] = None,
    scenario_id: Optional[str] = None,
    start_state_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    del seed
    session_state = _read_json(_session_state_path(session_work_dir))
    registration_path = Path(str(session_state["site_world_registration_path"]))
    try:
        bundle = load_site_world_bundle(registration_path, require_spec=True)
    except SiteWorldIntakeError as exc:
        raise HostedSessionError(str(exc)) from exc

    site_world = bundle.resolved
    registration = bundle.registration
    client = _resolve_runtime_client(config, registration)
    remote_session_id = str(session_state.get("remote_session_id") or session_id)

    task_entry = dict(session_state.get("task") or {})
    scenario_entry = dict(session_state.get("scenario") or {})
    start_state_entry = dict(session_state.get("start_state") or {})
    if task_id:
        task_entry = _catalog_entry([dict(item) for item in site_world.get("task_catalog", []) if isinstance(item, Mapping)], task_id, "task")
    if scenario_id:
        scenario_entry = _catalog_entry([dict(item) for item in site_world.get("scenario_catalog", []) if isinstance(item, Mapping)], scenario_id, "scenario")
    if start_state_id:
        start_state_entry = _catalog_entry([dict(item) for item in site_world.get("start_state_catalog", []) if isinstance(item, Mapping)], start_state_id, "start state")

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
    camera_frame_paths = {
        camera["cameraId"]: [camera["framePath"]]
        for camera in observation.get("cameraFrames", [])
        if isinstance(camera, Mapping) and camera.get("framePath")
    }
    episode_state = {
        "schema_version": "v1",
        "session_id": session_id,
        "remote_session_id": remote_session_id,
        "episode_id": episode_id,
        "site_world_id": session_state.get("site_world_id"),
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
        "observation_cameras": observation.get("cameraFrames", []),
        "camera_frame_paths": camera_frame_paths,
        "artifact_uris": {"episode_state": str(episode_dir / "episode_state.json")},
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
            "initial_episode": _episode_payload(episode_state),
        },
    )
    _write_json(_episode_state_path(session_work_dir), episode_state)
    session_state["current_episode_id"] = episode_id
    session_state["latest_episode_path"] = str(_episode_state_path(session_work_dir))
    session_state["runtime_smoke_path"] = str(runtime_smoke_path)
    session_state["status"] = "running"
    session_state["task"] = task_entry
    session_state["scenario"] = scenario_entry
    session_state["start_state"] = start_state_entry
    _write_json(_session_state_path(session_work_dir), session_state)
    return {"session_id": session_id, "episode": _episode_payload(episode_state)}


def step_session(
    *,
    config: ValidationConfig,
    session_work_dir: Path,
    episode_id: str,
    action: Sequence[float],
) -> Dict[str, Any]:
    session_state = _read_json(_session_state_path(session_work_dir))
    episode_state = _read_json(_episode_state_path(session_work_dir))
    if episode_state.get("episode_id") != episode_id:
        raise HostedSessionError("Episode ID does not match current session episode state.")
    registration = _read_json(Path(str(session_state["site_world_registration_path"])))
    client = _resolve_runtime_client(config, registration)

    robot_profile = dict(session_state["robot_profile"])
    expected_action_dim = int(((robot_profile.get("action_space") or {}).get("dim")) or len(action) or 7)
    action_list = [float(value) for value in action]
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
        if isinstance(camera, Mapping) and camera.get("cameraId") and camera.get("framePath"):
            episode_state.setdefault("camera_frame_paths", {}).setdefault(str(camera["cameraId"]), []).append(str(camera["framePath"]))

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
    _finalize_episode_artifacts(episode_state, session_work_dir=session_work_dir)
    _write_json(_episode_state_path(session_work_dir), episode_state)
    return {"session_id": session_state["session_id"], "episode": _episode_payload(episode_state)}


def run_batch(
    *,
    config: ValidationConfig,
    session_work_dir: Path,
    num_episodes: int,
    task_id: Optional[str] = None,
    scenario_id: Optional[str] = None,
    start_state_id: Optional[str] = None,
    seed: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    del seed
    session_state = _read_json(_session_state_path(session_work_dir))
    action_dim = int(((session_state.get("robot_profile") or {}).get("action_space") or {}).get("dim") or 7)
    default_action = [0.0] * action_dim
    max_steps_value = max(1, int(max_steps or 6))

    assignments: List[Dict[str, Any]] = []
    failures: List[str] = []
    for rollout_index in range(max(0, int(num_episodes))):
        reset_payload = reset_session(
            config=config,
            session_id=str(session_state["session_id"]),
            session_work_dir=session_work_dir,
            task_id=task_id,
            scenario_id=scenario_id,
            start_state_id=start_state_id,
        )
        episode_payload = reset_payload["episode"]
        for _ in range(max_steps_value):
            if bool(episode_payload.get("done")):
                break
            step_payload = step_session(
                config=config,
                session_work_dir=session_work_dir,
                episode_id=str(episode_payload["episodeId"]),
                action=default_action,
            )
            episode_payload = step_payload["episode"]
        assignments.append(
            {
                "episode_id": episode_payload["episodeId"],
                "rollout_index": rollout_index,
                "task_id": episode_payload["taskId"],
                "scenario_id": episode_payload["scenarioId"],
                "start_state_id": episode_payload["startStateId"],
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
        "numEpisodes": len(assignments),
        "numSuccess": num_success,
        "numFailure": len(assignments) - num_success,
        "successRate": round(num_success / float(max(len(assignments), 1)), 4),
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
    rlds_rows: List[Dict[str, Any]] = []
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
        rollout_row = {
            "episode_id": rollout_dir.name,
            "video_path": str(episode_state.get("artifact_uris", {}).get("rollout_video") or ""),
            "actions_path": str(episode_state.get("artifact_uris", {}).get("actions") or ""),
            "score_path": str(episode_state.get("artifact_uris", {}).get("score") or ""),
            "task_id": episode_state.get("task_id"),
            "task": episode_state.get("task"),
            "scenario_id": episode_state.get("scenario_id"),
            "scenario": episode_state.get("scenario"),
            "start_state_id": episode_state.get("start_state_id"),
            "start_state": episode_state.get("start_state"),
        }
        rollouts.append(rollout_row)
        rlds_rows.append(
            {
                "rollout_index": len(rlds_rows),
                "episode_id": rollout_dir.name,
                "video_path": rollout_row["video_path"],
                "task_spec_id": episode_state.get("task_id"),
                "task": str(episode_state.get("task") or ""),
                "eval_cell_id": str(episode_state.get("scenario_id") or ""),
                "start_region_id": str(episode_state.get("start_state_id") or ""),
                "sim_backend": "neoverse_service",
                "action_sequence": episode_state.get("action_trace", []),
                "success": bool(episode_state.get("success", False)),
                "task_success": bool(episode_state.get("success", False)),
                "task_score": float(episode_state.get("reward", 0.0) or 0.0),
                "failure_reason": str(episode_state.get("failure_reason") or ""),
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
    rlds_jsonl_path = rlds_dir / "episodes.jsonl"
    _write_jsonl(rlds_jsonl_path, rlds_rows)
    rlds_manifest_path = rlds_dir / "rlds_manifest.json"
    _write_json(
        rlds_manifest_path,
        {
            "schema_version": "v1",
            "session_id": session_state["session_id"],
            "episodes_jsonl": str(rlds_jsonl_path),
            "episode_count": len(rlds_rows),
        },
    )

    export_path = session_work_dir / "export_manifest.json"
    manifest = {
        "schema_version": "v1",
        "session_id": session_state["session_id"],
        "raw_bundle": {"manifest_path": str(raw_manifest_path), "rollout_count": len(rollouts)},
        "rlds_dataset": {"manifest_path": str(rlds_manifest_path), "episodes_jsonl": str(rlds_jsonl_path)},
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
    }
