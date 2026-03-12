"""Hosted session runtime helpers for WebApp-driven evaluation control."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np

from .config import FacilityConfig, PolicyAdapterConfig, ValidationConfig
from .neoverse_hosted_runtime import (
    NeoVerseHostedRuntime,
)
from .policy_adapters import get_policy_adapter
from .public_contract import public_runtime_label
from .scene_memory_runtime import resolve_scene_memory_runtime_plan
from .training.rlds_export import export_rollouts_to_rlds_jsonl


class HostedSessionError(RuntimeError):
    pass


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _env_truthy(name: str) -> bool:
    return (str(os.environ.get(name, "") or "").strip().lower() in {"1", "true", "yes", "on"})


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


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


def _task_entries(runtime_manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    task_catalog = runtime_manifest.get("task_catalog")
    if isinstance(task_catalog, list) and task_catalog:
        return [dict(item) for item in task_catalog if isinstance(item, Mapping)]
    task_anchor_path = Path(str(runtime_manifest.get("task_anchor_manifest_uri") or ""))
    if not task_anchor_path.exists():
        raise HostedSessionError("Hosted session runtime is missing task_anchor_manifest.json")
    anchor = _read_json(task_anchor_path)
    tasks = anchor.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise HostedSessionError("Hosted session runtime has no task entries")
    return [dict(item) for item in tasks if isinstance(item, Mapping)]


def _default_robot_profiles() -> List[Dict[str, Any]]:
    return [
        {
            "id": "mobile_manipulator_rgb_v1",
            "display_name": "Mobile manipulator",
            "embodiment_type": "mobile_manipulator",
            "action_space": {
                "name": "ee_delta_pose_gripper",
                "dim": 7,
                "labels": [
                    "base_x",
                    "base_y",
                    "base_yaw",
                    "ee_x",
                    "ee_y",
                    "ee_z",
                    "gripper",
                ],
            },
            "observation_cameras": [
                {"id": "head_rgb", "role": "head", "required": True, "default_enabled": True},
                {"id": "wrist_rgb", "role": "wrist", "required": False, "default_enabled": True},
                {"id": "site_context_rgb", "role": "context", "required": False, "default_enabled": True},
            ],
            "base_semantics": "holonomic_mobile_base",
            "gripper_semantics": "parallel_jaw_gripper",
            "urdf_uri": None,
            "usd_uri": None,
            "allowed_policy_adapters": ["openvla_oft", "pi05", "dreamzero"],
            "default_policy_adapter": "openvla_oft",
        },
        {
            "id": "humanoid_dual_camera_v1",
            "display_name": "Humanoid",
            "embodiment_type": "humanoid",
            "action_space": {
                "name": "whole_body_delta_pose_gripper",
                "dim": 7,
                "labels": [
                    "body_x",
                    "body_y",
                    "body_yaw",
                    "hand_x",
                    "hand_y",
                    "hand_z",
                    "gripper",
                ],
            },
            "observation_cameras": [
                {"id": "head_rgb", "role": "head", "required": True, "default_enabled": True},
                {"id": "left_wrist_rgb", "role": "wrist_left", "required": False, "default_enabled": True},
                {"id": "right_wrist_rgb", "role": "wrist_right", "required": False, "default_enabled": True},
                {"id": "site_context_rgb", "role": "context", "required": False, "default_enabled": True},
            ],
            "base_semantics": "bipedal_base",
            "gripper_semantics": "multi_finger_gripper",
            "urdf_uri": None,
            "usd_uri": None,
            "allowed_policy_adapters": ["openvla_oft", "dreamzero"],
            "default_policy_adapter": "openvla_oft",
        },
        {
            "id": "fixed_arm_cell_v1",
            "display_name": "Fixed arm cell",
            "embodiment_type": "fixed_arm",
            "action_space": {
                "name": "joint_delta_gripper",
                "dim": 7,
                "labels": [
                    "joint_1",
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                    "joint_6",
                    "gripper",
                ],
            },
            "observation_cameras": [
                {"id": "cell_rgb", "role": "head", "required": True, "default_enabled": True},
                {"id": "wrist_rgb", "role": "wrist", "required": False, "default_enabled": True},
            ],
            "base_semantics": "fixed_base",
            "gripper_semantics": "parallel_jaw_gripper",
            "urdf_uri": None,
            "usd_uri": None,
            "allowed_policy_adapters": ["openvla_oft", "pi05"],
            "default_policy_adapter": "pi05",
        },
    ]


def _robot_profiles(runtime_manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    profiles = runtime_manifest.get("robot_profiles")
    if isinstance(profiles, list) and profiles:
        return [dict(item) for item in profiles if isinstance(item, Mapping)]
    return _default_robot_profiles()


def _catalog_entries(runtime_manifest: Mapping[str, Any], key: str, fallback_values: Sequence[str]) -> List[Dict[str, Any]]:
    entries = runtime_manifest.get(key)
    if isinstance(entries, list) and entries:
        return [dict(item) for item in entries if isinstance(item, Mapping)]
    out: List[Dict[str, Any]] = []
    prefix = key.replace("_catalog", "")
    for index, value in enumerate(fallback_values):
        text = str(value or "").strip()
        if not text:
            continue
        out.append({"id": f"{prefix}_{index}", "name": text})
    return out


def _find_catalog_entry(
    entries: Sequence[Mapping[str, Any]],
    *,
    requested_id: Optional[str],
    fallback_name: Optional[str] = None,
    label: str,
) -> Dict[str, Any]:
    requested = str(requested_id or "").strip()
    if requested:
        for item in entries:
            if str(item.get("id") or "").strip() == requested:
                return dict(item)
        raise HostedSessionError(f"Unsupported {label}: {requested}")
    if fallback_name:
        fallback = str(fallback_name or "").strip().lower()
        for item in entries:
            if str(item.get("name") or item.get("task_text") or "").strip().lower() == fallback:
                return dict(item)
    if not entries:
        raise HostedSessionError(f"Hosted session runtime is missing {label} catalog entries")
    return dict(entries[0])


def _runtime_plan(config: ValidationConfig, runtime_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    bundle_manifest = Path(str(runtime_manifest.get("scene_memory_manifest_uri") or ""))
    bundle_dir = bundle_manifest.parent if bundle_manifest.name else bundle_manifest
    adapter_uris = runtime_manifest.get("adapter_manifest_uris")
    adapter_paths = (
        {
            str(key): Path(str(value))
            for key, value in adapter_uris.items()
            if isinstance(adapter_uris, Mapping)
            and str(key).strip()
            and str(value).strip()
        }
        if isinstance(adapter_uris, Mapping)
        else {}
    )
    facility = FacilityConfig(
        name="hosted_session",
        scene_memory_bundle_path=bundle_dir if bundle_dir.exists() else None,
        preview_simulation_path=(
            Path(str(runtime_manifest.get("preview_simulation_manifest_uri"))).parent
            if str(runtime_manifest.get("preview_simulation_manifest_uri") or "").strip()
            else None
        ),
        scene_memory_adapter_manifests=adapter_paths,
    )
    try:
        return resolve_scene_memory_runtime_plan(config, facility)
    except Exception:
        return {
            "selected_backend": runtime_manifest.get("default_backend"),
            "available_backends": runtime_manifest.get("available_backends", []),
            "selection_reason": "runtime_manifest_fallback",
        }


def _resolve_backend(config: ValidationConfig, runtime_manifest: Mapping[str, Any]) -> str:
    if not bool(runtime_manifest.get("launchable", False)):
        blockers = ", ".join(str(item) for item in runtime_manifest.get("launch_blockers", []))
        raise HostedSessionError(f"Hosted session runtime is not launchable: {blockers}")
    runtime_plan = _runtime_plan(config, runtime_manifest)
    backend = str(runtime_plan.get("selected_backend") or runtime_manifest.get("default_backend") or "").strip()
    if not backend:
        raise HostedSessionError("Hosted session runtime has no executable backend selected")
    if backend != "neoverse":
        raise HostedSessionError(
            f"Hosted sessions require neoverse as the execution backend; got {backend or 'unset'}."
        )
    return backend


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


def _save_frame(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def _load_frame(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.uint8)
    if isinstance(value, str) and value.strip():
        frame = cv2.imread(value)
        if frame is None:
            raise HostedSessionError(f"Could not load observation frame from {value}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    raise HostedSessionError("Hosted runtime returned an invalid frame payload.")


class HostedWorldModelAdapter:
    def create_session(self, *, session_context: Mapping[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def reset_episode(self, *, session_context: Mapping[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def step_episode(
        self,
        *,
        session_context: Mapping[str, Any],
        action: Sequence[float],
        current_observation: Mapping[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def run_batch(self, *, session_context: Mapping[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def export_rollouts(self, *, session_context: Mapping[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class NeoVerseHostedAdapter(HostedWorldModelAdapter):
    def __init__(self, *, config: ValidationConfig, runtime_manifest: Mapping[str, Any]) -> None:
        self.config = config
        self.runtime_manifest = runtime_manifest
        runtime_cfg = config.scene_memory_runtime.neoverse
        if not bool(runtime_cfg.allow_runtime_execution):
            raise HostedSessionError("NeoVerse hosted runtime execution is disabled in Validation config.")
        if runtime_cfg.repo_path is None or not runtime_cfg.repo_path.exists():
            raise HostedSessionError("NeoVerse hosted runtime repo_path is missing or does not exist.")
        kwargs = {
            "scene_memory_manifest_uri": runtime_manifest.get("scene_memory_manifest_uri"),
            "conditioning_bundle_uri": runtime_manifest.get("conditioning_bundle_uri"),
            "preview_simulation_manifest_uri": runtime_manifest.get("preview_simulation_manifest_uri"),
            "runtime_manifest": dict(runtime_manifest),
            "repo_path": str(runtime_cfg.repo_path),
            "python_executable": (
                str(runtime_cfg.python_executable) if runtime_cfg.python_executable is not None else None
            ),
            "inference_script": runtime_cfg.inference_script,
            "checkpoint_path": str(runtime_cfg.checkpoint_path) if runtime_cfg.checkpoint_path else None,
        }
        try:
            self.runtime = NeoVerseHostedRuntime(**kwargs)
        except Exception as exc:
            raise HostedSessionError(f"NeoVerse hosted runtime initialization failed: {exc}") from exc

    def _call(self, method_names: Sequence[str], *, session_context: Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
        for method_name in method_names:
            method = getattr(self.runtime, method_name, None)
            if not callable(method):
                continue
            try:
                payload = method(session_context=session_context, **kwargs)
            except TypeError:
                try:
                    payload = method(**kwargs)
                except TypeError:
                    try:
                        payload = method(session_context)
                    except Exception as exc:
                        raise HostedSessionError(
                            f"NeoVerse runtime method {method_name} failed: {exc}"
                        ) from exc
                except Exception as exc:
                    raise HostedSessionError(
                        f"NeoVerse runtime method {method_name} failed: {exc}"
                    ) from exc
            except Exception as exc:
                raise HostedSessionError(
                    f"NeoVerse runtime method {method_name} failed: {exc}"
                ) from exc
            if isinstance(payload, Mapping):
                return dict(payload)
            raise HostedSessionError(f"NeoVerse runtime method {method_name} returned non-mapping payload.")
        raise HostedSessionError(
            f"NeoVerse runtime missing required method. Tried: {', '.join(method_names)}"
        )

    def create_session(self, *, session_context: Mapping[str, Any]) -> Dict[str, Any]:
        return self._call(("create_session", "initialize_session"), session_context=session_context)

    def reset_episode(self, *, session_context: Mapping[str, Any]) -> Dict[str, Any]:
        return self._call(("reset_episode", "initial_observation"), session_context=session_context)

    def step_episode(
        self,
        *,
        session_context: Mapping[str, Any],
        action: Sequence[float],
        current_observation: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return self._call(
            ("step_episode", "predict_next_observation"),
            session_context=session_context,
            action=list(action),
            current_observation=dict(current_observation),
        )

    def run_batch(self, *, session_context: Mapping[str, Any]) -> Dict[str, Any]:
        return {"status": "supported", "session_context": dict(session_context)}

    def export_rollouts(self, *, session_context: Mapping[str, Any]) -> Dict[str, Any]:
        return {"status": "supported", "session_context": dict(session_context)}


def _resolve_world_model_adapter(
    *,
    config: ValidationConfig,
    runtime_manifest: Mapping[str, Any],
) -> HostedWorldModelAdapter:
    backend = _resolve_backend(config, runtime_manifest)
    if backend == "neoverse":
        return NeoVerseHostedAdapter(config=config, runtime_manifest=runtime_manifest)
    raise HostedSessionError(f"Unsupported hosted-session backend: {backend}")


def _camera_catalog(robot_profile: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cameras = robot_profile.get("observation_cameras")
    if isinstance(cameras, list) and cameras:
        return [dict(item) for item in cameras if isinstance(item, Mapping)]
    raise HostedSessionError(f"Robot profile {robot_profile.get('id')} is missing observation cameras.")


def _normalize_runtime_observation(
    payload: Mapping[str, Any],
    *,
    robot_profile: Mapping[str, Any],
    episode_dir: Path,
    step_index: int,
) -> Dict[str, Any]:
    camera_catalog = _camera_catalog(robot_profile)
    raw_camera_frames = payload.get("camera_frames")
    camera_frames: Dict[str, np.ndarray] = {}
    if isinstance(raw_camera_frames, Mapping):
        for camera_id, frame in raw_camera_frames.items():
            camera_frames[str(camera_id)] = _load_frame(frame)
    elif payload.get("frame") is not None:
        primary_id = str(camera_catalog[0].get("id") or "head_rgb")
        camera_frames[primary_id] = _load_frame(payload.get("frame"))
    else:
        raise HostedSessionError("NeoVerse runtime reset/step payload missing camera frame data.")

    saved_paths: Dict[str, str] = {}
    summaries: List[Dict[str, Any]] = []
    primary_camera_id = ""
    for camera in camera_catalog:
        camera_id = str(camera.get("id") or "").strip()
        if not camera_id:
            continue
        frame = camera_frames.get(camera_id)
        if frame is not None:
            frame_path = episode_dir / "cameras" / camera_id / f"frame_{step_index:03d}.png"
            _save_frame(frame_path, frame)
            saved_paths[camera_id] = str(frame_path)
            if not primary_camera_id:
                primary_camera_id = camera_id
        summaries.append(
            {
                "cameraId": camera_id,
                "role": str(camera.get("role") or ""),
                "required": bool(camera.get("required", False)),
                "available": frame is not None,
                "framePath": saved_paths.get(camera_id),
            }
        )
    if not primary_camera_id:
        raise HostedSessionError("NeoVerse runtime did not return any usable observation camera frames.")

    return {
        "stepIndex": step_index,
        "primaryCameraId": primary_camera_id,
        "frame_path": saved_paths[primary_camera_id],
        "cameraFrames": summaries,
        "camera_frame_paths": saved_paths,
        "task_instruction": str(payload.get("task_instruction") or ""),
        "runtimeMetadata": dict(payload.get("runtime_metadata", {}) or {}),
        "worldSnapshot": dict(payload.get("world_snapshot", {}) or {}),
    }


def _session_context_from_state(
    session_state: Mapping[str, Any],
    *,
    task_entry: Mapping[str, Any],
    scenario_entry: Mapping[str, Any],
    start_state_entry: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "session_id": session_state["session_id"],
        "scene_id": session_state["scene_id"],
        "capture_id": session_state["capture_id"],
        "site_submission_id": session_state["site_submission_id"],
        "runtime_backend_selected": session_state["runtime_backend_selected"],
        "runtime_manifest_path": session_state["runtime_manifest_path"],
        "robot_profile": session_state["robot_profile"],
        "task": dict(task_entry),
        "scenario": dict(scenario_entry),
        "start_state": dict(start_state_entry),
        "policy": dict(session_state["policy"]),
    }


def _episode_payload(episode_state: Mapping[str, Any]) -> Dict[str, Any]:
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
    }


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


def _finalize_episode_artifacts(episode_state: Dict[str, Any], *, session_work_dir: Path) -> None:
    episode_dir = _rollouts_dir(session_work_dir) / str(episode_state["episode_id"])
    primary_camera_id = str(episode_state["observation"].get("primaryCameraId") or "")
    camera_frame_paths = episode_state.get("camera_frame_paths", {}) or {}
    primary_frame_paths = (
        camera_frame_paths.get(primary_camera_id)
        if isinstance(camera_frame_paths, Mapping)
        else None
    )
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
    runtime_manifest_path: Path,
    robot_profile_id: str,
    task_id: str,
    scenario_id: str,
    start_state_id: str,
    policy_payload: Mapping[str, Any],
    export_modes: Sequence[str],
    robot_profile_override: Optional[Mapping[str, Any]] = None,
    notes: str = "",
) -> Dict[str, Any]:
    runtime_manifest = _read_json(runtime_manifest_path)
    runtime_manifest.setdefault("runtime_manifest_path", str(runtime_manifest_path))
    backend = _resolve_backend(config, runtime_manifest)
    adapter = _resolve_world_model_adapter(config=config, runtime_manifest=runtime_manifest)
    task_entries = _task_entries(runtime_manifest)
    scenario_entries = _catalog_entries(
        runtime_manifest,
        "scenario_catalog",
        runtime_manifest.get("scenario_variants", []) if isinstance(runtime_manifest.get("scenario_variants"), list) else [],
    )
    start_state_entries = _catalog_entries(
        runtime_manifest,
        "start_state_catalog",
        runtime_manifest.get("start_states", []) if isinstance(runtime_manifest.get("start_states"), list) else [],
    )
    robot_profiles = _robot_profiles(runtime_manifest)
    robot_profile = _find_catalog_entry(robot_profiles, requested_id=robot_profile_id, label="robot profile")
    if isinstance(robot_profile_override, Mapping):
        robot_profile.update(dict(robot_profile_override))
    task_entry = _find_catalog_entry(
        task_entries,
        requested_id=task_id,
        fallback_name=None,
        label="task",
    )
    scenario_entry = _find_catalog_entry(scenario_entries, requested_id=scenario_id, label="scenario")
    start_state_entry = _find_catalog_entry(start_state_entries, requested_id=start_state_id, label="start state")
    _resolve_policy_adapter(config, policy_payload, robot_profile=robot_profile)

    session_work_dir.mkdir(parents=True, exist_ok=True)
    _rollouts_dir(session_work_dir).mkdir(parents=True, exist_ok=True)

    session_state = {
        "schema_version": "v1",
        "session_id": session_id,
        "runtime_manifest_path": str(runtime_manifest_path),
        "runtime_backend_selected": backend,
        "runtime_backend_public_name": public_runtime_label(backend),
        "status": "ready",
        "scene_id": runtime_manifest.get("scene_id"),
        "capture_id": runtime_manifest.get("capture_id"),
        "site_submission_id": runtime_manifest.get("site_submission_id"),
        "robot_profile": robot_profile,
        "task": dict(task_entry),
        "scenario": dict(scenario_entry),
        "start_state": dict(start_state_entry),
        "notes": notes,
        "policy": dict(policy_payload),
        "export_modes": [str(item) for item in export_modes if str(item).strip()],
        "current_episode_id": None,
        "latest_episode_path": None,
        "batch_summary_path": None,
        "artifact_uris": {
            "session_state": str(_session_state_path(session_work_dir)),
        },
        "dataset_artifacts": {},
    }
    session_context = _session_context_from_state(
        session_state,
        task_entry=task_entry,
        scenario_entry=scenario_entry,
        start_state_entry=start_state_entry,
    )
    create_payload = adapter.create_session(session_context=session_context)
    session_state["runtime_session_metadata"] = dict(create_payload.get("runtime_session_metadata", {}) or {})
    _write_json(_session_state_path(session_work_dir), session_state)
    return {
        "session_id": session_id,
        "runtime_backend_selected": backend,
        "runtime_backend_public_name": public_runtime_label(backend),
        "status": "ready",
        "robotProfile": robot_profile,
        "observationCameras": _camera_catalog(robot_profile),
        "artifact_uris": session_state["artifact_uris"],
        "dataset_artifacts": session_state["dataset_artifacts"],
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
    runtime_manifest_path = Path(str(session_state["runtime_manifest_path"]))
    runtime_manifest = _read_json(runtime_manifest_path)
    runtime_manifest.setdefault("runtime_manifest_path", str(runtime_manifest_path))
    adapter = _resolve_world_model_adapter(config=config, runtime_manifest=runtime_manifest)
    task_entries = _task_entries(runtime_manifest)
    scenario_entries = _catalog_entries(
        runtime_manifest,
        "scenario_catalog",
        runtime_manifest.get("scenario_variants", []) if isinstance(runtime_manifest.get("scenario_variants"), list) else [],
    )
    start_state_entries = _catalog_entries(
        runtime_manifest,
        "start_state_catalog",
        runtime_manifest.get("start_states", []) if isinstance(runtime_manifest.get("start_states"), list) else [],
    )
    task_entry = _find_catalog_entry(
        task_entries,
        requested_id=task_id,
        fallback_name=str((session_state.get("task") or {}).get("task_text") or (session_state.get("task") or {}).get("task") or ""),
        label="task",
    )
    scenario_entry = _find_catalog_entry(
        scenario_entries,
        requested_id=scenario_id,
        fallback_name=str((session_state.get("scenario") or {}).get("name") or ""),
        label="scenario",
    )
    start_state_entry = _find_catalog_entry(
        start_state_entries,
        requested_id=start_state_id,
        fallback_name=str((session_state.get("start_state") or {}).get("name") or ""),
        label="start state",
    )
    session_context = _session_context_from_state(
        session_state,
        task_entry=task_entry,
        scenario_entry=scenario_entry,
        start_state_entry=start_state_entry,
    )
    reset_payload = adapter.reset_episode(session_context=session_context)
    episode_id = f"episode-{hashlib.sha256('::'.join([session_id, str(task_entry.get('id') or ''), str(start_state_entry.get('id') or '')]).encode('utf-8')).hexdigest()[:8]}"
    episode_dir = _rollouts_dir(session_work_dir) / episode_id
    observation = _normalize_runtime_observation(
        reset_payload,
        robot_profile=session_state["robot_profile"],
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
        "episode_id": episode_id,
        "task_id": str(task_entry.get("id") or task_entry.get("task_id") or ""),
        "task": str(task_entry.get("task_text") or task_entry.get("name") or ""),
        "scenario_id": str(scenario_entry.get("id") or ""),
        "scenario": str(scenario_entry.get("name") or ""),
        "start_state_id": str(start_state_entry.get("id") or ""),
        "start_state": str(start_state_entry.get("name") or ""),
        "status": "ready",
        "step_index": 0,
        "done": False,
        "reward": float(reset_payload.get("reward", 0.0) or 0.0),
        "success": None,
        "failure_reason": None,
        "action_trace": [],
        "observation": observation,
        "observation_cameras": observation_cameras,
        "camera_frame_paths": camera_frame_paths,
        "artifact_uris": {
            "episode_state": str(episode_dir / "episode_state.json"),
        },
        "runtime_metadata": dict(reset_payload.get("runtime_metadata", {}) or {}),
    }
    _write_json(_episode_state_path(session_work_dir), episode_state)
    session_state["current_episode_id"] = episode_id
    session_state["latest_episode_path"] = str(_episode_state_path(session_work_dir))
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
    runtime_manifest_path = Path(str(session_state["runtime_manifest_path"]))
    runtime_manifest = _read_json(runtime_manifest_path)
    runtime_manifest.setdefault("runtime_manifest_path", str(runtime_manifest_path))
    adapter = _resolve_world_model_adapter(config=config, runtime_manifest=runtime_manifest)
    episode_state = _read_json(_episode_state_path(session_work_dir))
    if episode_state.get("episode_id") != episode_id:
        raise HostedSessionError("Episode ID does not match current session episode state")

    robot_profile = session_state["robot_profile"]
    policy_payload = (
        session_state.get("policy")
        if isinstance(session_state.get("policy"), Mapping)
        else {}
    )
    policy_adapter, _ = _resolve_policy_adapter(config, policy_payload, robot_profile=robot_profile)
    policy_handle = policy_adapter.load_policy(
        model_name=str(policy_payload.get("model_name") or ""),
        checkpoint_path=(
            Path(str(policy_payload.get("checkpoint_path")))
            if str(policy_payload.get("checkpoint_path") or "").strip()
            else None
        ),
        device=str(policy_payload.get("device") or "cpu"),
    )

    primary_frame_path = str((episode_state.get("observation") or {}).get("frame_path") or "")
    if not primary_frame_path:
        raise HostedSessionError("Current episode observation is missing a primary frame_path.")
    current_frame = _load_frame(primary_frame_path)
    if auto_policy:
        next_action = policy_adapter.predict_action(
            handle=policy_handle,
            frame=current_frame,
            task_prompt=str(episode_state.get("task") or ""),
            unnorm_key=None,
            device=str(policy_payload.get("device") or "cpu"),
        )
        action_list = np.asarray(next_action, dtype=np.float32).reshape(-1).tolist()
    else:
        action_list = list(action or [])
    expected_action_dim = int(((robot_profile.get("action_space") or {}).get("dim")) or len(action_list) or 7)
    if len(action_list) != expected_action_dim:
        raise HostedSessionError(
            f"Action-space mismatch: received {len(action_list)} values, expected {expected_action_dim}."
        )

    session_context = _session_context_from_state(
        session_state,
        task_entry=session_state["task"],
        scenario_entry=session_state["scenario"],
        start_state_entry=session_state["start_state"],
    )
    step_payload = adapter.step_episode(
        session_context=session_context,
        action=action_list,
        current_observation=episode_state["observation"],
    )
    next_step_index = int(episode_state.get("step_index", 0)) + 1
    episode_dir = _rollouts_dir(session_work_dir) / str(episode_state["episode_id"])
    observation = _normalize_runtime_observation(
        step_payload,
        robot_profile=robot_profile,
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
    episode_state["reward"] = float(step_payload.get("reward", episode_state.get("reward", 0.0)) or 0.0)
    episode_state["done"] = bool(step_payload.get("done", False))
    episode_state["success"] = (
        bool(step_payload.get("success"))
        if step_payload.get("success") is not None
        else (True if episode_state["done"] and not step_payload.get("failure_reason") else False if episode_state["done"] else None)
    )
    episode_state["failure_reason"] = step_payload.get("failure_reason")
    episode_state["status"] = "completed" if episode_state["done"] else "running"
    episode_state["action_trace"].append(action_list)
    episode_state["observation"] = observation
    episode_state["observation_cameras"] = observation.get("cameraFrames", [])
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
    runtime_manifest_path = Path(str(session_state["runtime_manifest_path"]))
    runtime_manifest = _read_json(runtime_manifest_path)
    runtime_manifest.setdefault("runtime_manifest_path", str(runtime_manifest_path))
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
        "artifactManifestUri": str(session_work_dir / "batch_run_summary.json"),
    }
    _write_json(session_work_dir / "batch_run_summary.json", {"assignments": assignments, "summary": summary})
    session_state["batch_summary_path"] = str(session_work_dir / "batch_run_summary.json")
    _write_json(_session_state_path(session_work_dir), session_state)
    return {
        "session_id": session_state["session_id"],
        "batchRunId": summary["batchRunId"],
        "status": "completed",
        "assignments": assignments,
        "summary": summary,
        "artifact_uris": {"batch_summary": str(session_work_dir / "batch_run_summary.json")},
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
                "sim_backend": "neoverse_hosted_session",
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
