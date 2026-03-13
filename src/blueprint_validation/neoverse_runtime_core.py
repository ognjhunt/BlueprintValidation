"""Persistent downstream site-world runtime store used by the NeoVerse service.

This store consumes already-built site-world package artifacts and derives local runtime
cache/session state from them. Shared contract ownership stays in BlueprintContracts.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from blueprint_contracts.site_world_contract import (
    SiteWorldIntakeError,
    load_site_world_bundle,
    normalize_trajectory_payload,
)

from .optional_dependencies import require_optional_dependency
from .runtime_layer_grounding import (
    composite_runtime_layer,
    load_runtime_layer_bundle,
    snapshot_runtime_layer_bundle,
    update_presentation_session_manifest,
    validate_runtime_layer_spec,
    verify_canonical_package_version,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:12]}"


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _dedupe_strings(*collections: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for collection in collections:
        for item in collection:
            text = str(item or "").strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
    return out


def _blocked_status(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text and text not in {"ready", "healthy"}:
            return text
    return "blocked"


def _load_frame(path: Path) -> np.ndarray:
    cv2 = require_optional_dependency(
        "cv2",
        extra="vision",
        purpose="NeoVerse runtime frame IO",
    )
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"failed to load frame at {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _save_frame(path: Path, frame: np.ndarray) -> None:
    cv2 = require_optional_dependency(
        "cv2",
        extra="vision",
        purpose="NeoVerse runtime frame IO",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def _extract_first_frame(video_path: Path) -> np.ndarray:
    cv2 = require_optional_dependency(
        "cv2",
        extra="vision",
        purpose="NeoVerse runtime frame IO",
    )
    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read video frame from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _coerce_camera_frame(frame: np.ndarray, camera_id: str) -> np.ndarray:
    cv2 = require_optional_dependency(
        "cv2",
        extra="vision",
        purpose="NeoVerse runtime frame IO",
    )
    variant = frame.copy()
    if "wrist" in camera_id:
        variant = cv2.resize(
            variant,
            (max(variant.shape[1] - 24, 32), max(variant.shape[0] - 24, 32)),
        )
        variant = cv2.resize(variant, (frame.shape[1], frame.shape[0]))
    elif "context" in camera_id:
        variant = np.clip((variant.astype(np.float32) * 0.92) + 10.0, 0, 255).astype(np.uint8)
    return variant


def _apply_action(frame: np.ndarray, action: Sequence[float], step_index: int) -> np.ndarray:
    cv2 = require_optional_dependency(
        "cv2",
        extra="vision",
        purpose="NeoVerse runtime frame IO",
    )
    padded = [float(value) for value in action] + [0.0] * max(0, 7 - len(action))
    dx = int(np.clip(padded[0] * 18.0, -32.0, 32.0))
    dy = int(np.clip(padded[1] * 18.0, -32.0, 32.0))
    brightness = np.clip(padded[2] * 20.0, -32.0, 32.0)
    scale = float(np.clip(1.0 + (abs(padded[5]) * 0.04), 1.0, 1.2))

    height, width = frame.shape[:2]
    matrix = np.float32([[scale, 0, dx], [0, scale, dy]])
    transformed = cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    if brightness:
        transformed = np.clip(transformed.astype(np.float32) + brightness, 0, 255).astype(np.uint8)

    cv2.putText(
        transformed,
        f"step {step_index:03d}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return transformed


def _preferred_local_path(*values: Any) -> Optional[Path]:
    for raw in values:
        value = str(raw or "").strip()
        if not value or value.startswith(("gs://", "http://", "https://")):
            continue
        path = Path(value).resolve()
        if path.exists():
            return path
    return None


def _sensor_truth(value: Mapping[str, Any], key: str) -> bool:
    return bool(value.get(key) is True)


@dataclass(frozen=True)
class RuntimeValidationResult:
    launchable: bool
    blockers: list[str]
    warnings: list[str]


class NeoVerseRuntimeStore:
    """Disk-backed store for site worlds and sessions."""

    def __init__(self, root_dir: str | Path, *, base_url: str, ws_base_url: Optional[str] = None) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.base_url = base_url.rstrip("/")
        self.ws_base_url = (ws_base_url or self.base_url.replace("http://", "ws://").replace("https://", "wss://")).rstrip("/")
        self.site_worlds_dir = self.root_dir / "site_worlds"
        self.sessions_dir = self.root_dir / "sessions"
        self.site_worlds_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _site_world_dir(self, site_world_id: str) -> Path:
        return self.site_worlds_dir / site_world_id

    def _session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def _site_world_state_path(self, site_world_id: str) -> Path:
        return self._site_world_dir(site_world_id) / "site_world_registration.json"

    def _site_world_health_path(self, site_world_id: str) -> Path:
        return self._site_world_dir(site_world_id) / "site_world_health.json"

    def _site_world_spec_path(self, site_world_id: str) -> Path:
        return self._site_world_dir(site_world_id) / "site_world_spec.json"

    def _session_state_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "session_state.json"

    def validate_spec(
        self,
        spec: Mapping[str, Any],
        *,
        protected_regions_manifest: Mapping[str, Any] | None = None,
    ) -> RuntimeValidationResult:
        blockers: list[str] = []
        warnings: list[str] = []
        blockers.extend(validate_runtime_layer_spec(spec))
        runtime_layer_policy = (
            dict(spec.get("runtime_layer_policy") or {})
            if isinstance(spec.get("runtime_layer_policy"), Mapping)
            else {}
        )

        qualification_state = str(spec.get("qualification_state") or "").strip().lower()
        if qualification_state != "ready":
            blockers.append(f"qualification_state:{qualification_state or 'missing'}")
        if not bool(spec.get("downstream_evaluation_eligibility")):
            blockers.append("downstream_evaluation_eligibility:false")
        grounding_status = str(
            (protected_regions_manifest or {}).get("grounding_status")
            or runtime_layer_policy.get("grounding_status")
            or spec.get("grounding_status")
            or ""
        ).strip().lower()
        if grounding_status == "ungrounded":
            reason = str(
                (protected_regions_manifest or {}).get("ungrounded_reason")
                or runtime_layer_policy.get("ungrounded_reason")
                or spec.get("ungrounded_reason")
                or "ungrounded"
            ).strip()
            blockers.append(f"runtime_grounding:{reason or 'ungrounded'}")

        capture_source = str(spec.get("capture_source") or "").strip().lower()
        if capture_source in {"glasses", "video_only", "glasses_video_only"}:
            blockers.append("video_only_capture:not_launchable")

        conditioning_map = dict(spec.get("conditioning") or {}) if isinstance(spec.get("conditioning"), Mapping) else {}
        sensor_map = dict(conditioning_map.get("sensor_availability") or {}) if isinstance(conditioning_map.get("sensor_availability"), Mapping) else {}
        if not _sensor_truth(sensor_map, "arkit_poses"):
            blockers.append("missing_spatial_conditioning:arkit_poses")
        if not _sensor_truth(sensor_map, "arkit_intrinsics"):
            blockers.append("missing_spatial_conditioning:arkit_intrinsics")

        local_map = dict(conditioning_map.get("local_paths") or {}) if isinstance(conditioning_map.get("local_paths"), Mapping) else {}
        poses_path = _preferred_local_path(local_map.get("arkit_poses_path"), conditioning_map.get("arkit_poses_uri"))
        intrinsics_path = _preferred_local_path(local_map.get("arkit_intrinsics_path"), conditioning_map.get("arkit_intrinsics_uri"))
        if poses_path is None:
            blockers.append("missing_local_conditioning:arkit_poses")
        if intrinsics_path is None:
            blockers.append("missing_local_conditioning:arkit_intrinsics")

        conditioning_source = _preferred_local_path(
            local_map.get("keyframe_path"),
            local_map.get("raw_video_path"),
            conditioning_map.get("keyframe_uri"),
            conditioning_map.get("raw_video_uri"),
        )
        if conditioning_source is None:
            blockers.append("missing_local_conditioning:visual_source")

        geometry_map = dict(spec.get("geometry") or {}) if isinstance(spec.get("geometry"), Mapping) else {}
        occupancy_path = _preferred_local_path(local_map.get("occupancy_path"), geometry_map.get("occupancy_path"))
        object_index_path = _preferred_local_path(local_map.get("object_index_path"), geometry_map.get("object_index_path"))
        if occupancy_path is None:
            warnings.append("occupancy_path_missing")
        if object_index_path is None:
            warnings.append("object_index_path_missing")

        return RuntimeValidationResult(
            launchable=len(blockers) == 0,
            blockers=blockers,
            warnings=warnings,
        )

    def build_site_world(self, spec: Mapping[str, Any]) -> Dict[str, Any]:
        """Deprecated compatibility wrapper for spec-only site-world registration."""
        scene_id = str(spec.get("scene_id") or "").strip()
        capture_id = str(spec.get("capture_id") or "").strip()
        if not scene_id or not capture_id:
            raise RuntimeError("site world build requires scene_id and capture_id")

        site_world_id = _stable_id("siteworld", scene_id, capture_id)
        registration = {
            "schema_version": "v1",
            "site_world_id": site_world_id,
            "scene_id": scene_id,
            "capture_id": capture_id,
            "site_submission_id": spec.get("site_submission_id"),
            "status": "ready",
            "task_catalog": list(spec.get("task_catalog") or []),
            "scenario_catalog": list(spec.get("scenario_catalog") or []),
            "start_state_catalog": list(spec.get("start_state_catalog") or []),
            "robot_profiles": list(spec.get("robot_profiles") or []),
            "canonical_package_uri": spec.get("canonical_package_uri"),
            "canonical_package_version": spec.get("canonical_package_version"),
            "blockers": [],
            "warnings": [],
            "runtime_capabilities": {
                "supports_step_rollout": True,
                "supports_batch_rollout": True,
                "supports_camera_views": True,
                "supports_stream": True,
                "protected_region_locking": True,
                "runtime_layer_compositing": True,
                "debug_render_outputs": True,
            },
            "registration_mode": "legacy_spec_build",
            "intake_source": "legacy_spec_only_payload",
            "compatibility_notice": (
                "Deprecated compatibility path. Prefer registering built "
                "site_world_spec/site_world_registration/site_world_health artifacts."
            ),
        }
        health = {
            "schema_version": "v1",
            "site_world_id": site_world_id,
            "scene_id": scene_id,
            "capture_id": capture_id,
            "site_submission_id": spec.get("site_submission_id"),
            "healthy": True,
            "launchable": True,
            "status": "healthy",
            "blockers": [],
            "warnings": [],
            "runtime_capabilities": dict(registration["runtime_capabilities"]),
            "canonical_package_version": spec.get("canonical_package_version"),
            "registration_mode": "legacy_spec_build",
            "intake_source": "legacy_spec_only_payload",
            "compatibility_notice": registration["compatibility_notice"],
        }
        return self.register_site_world_package(
            spec=spec,
            registration=registration,
            health=health,
        )

    def register_site_world_package(
        self,
        *,
        spec: Mapping[str, Any],
        registration: Mapping[str, Any],
        health: Mapping[str, Any],
    ) -> Dict[str, Any]:
        scene_id = str(spec.get("scene_id") or registration.get("scene_id") or "").strip()
        capture_id = str(spec.get("capture_id") or registration.get("capture_id") or "").strip()
        if not scene_id or not capture_id:
            raise RuntimeError("site world registration requires scene_id and capture_id")

        site_world_id = _stable_id("siteworld", scene_id, capture_id)
        upstream_site_world_id = str(registration.get("site_world_id") or "").strip()
        if upstream_site_world_id:
            site_world_id = upstream_site_world_id
        build_id = str(registration.get("build_id") or "").strip() or _stable_id(
            "build",
            scene_id,
            capture_id,
            _utc_now_iso(),
        )
        site_world_dir = self._site_world_dir(site_world_id)
        cache_dir = site_world_dir / "cache" / build_id
        cache_dir.mkdir(parents=True, exist_ok=True)

        conditioning_map = dict(spec.get("conditioning") or {}) if isinstance(spec.get("conditioning"), Mapping) else {}
        local_map = dict(conditioning_map.get("local_paths") or {}) if isinstance(conditioning_map.get("local_paths"), Mapping) else {}
        conditioning_source = _preferred_local_path(
            local_map.get("keyframe_path"),
            local_map.get("raw_video_path"),
            conditioning_map.get("keyframe_uri"),
            conditioning_map.get("raw_video_uri"),
        )
        if conditioning_source is None:
            raise RuntimeError("site world build requires a local conditioning source")

        source_frame = (
            _extract_first_frame(conditioning_source)
            if conditioning_source.suffix.lower() in {".mp4", ".mov", ".m4v", ".avi"}
            else _load_frame(conditioning_source)
        )
        seed_frame_path = cache_dir / "seed_frame.png"
        _save_frame(seed_frame_path, source_frame)

        runtime_layer_bundle = load_runtime_layer_bundle(spec)
        validation = self.validate_spec(
            spec,
            protected_regions_manifest=runtime_layer_bundle["protected_regions_manifest"],
        )
        version_error = verify_canonical_package_version(
            spec=spec,
            protected_regions_manifest=runtime_layer_bundle["protected_regions_manifest"],
            canonical_render_policy=runtime_layer_bundle["canonical_render_policy"],
            presentation_variance_policy=runtime_layer_bundle["presentation_variance_policy"],
        )
        if version_error is not None:
            raise RuntimeError(version_error)
        runtime_layer_snapshots = snapshot_runtime_layer_bundle(runtime_layer_bundle, cache_dir)

        supported_cameras = list(registration.get("supported_cameras") or [])
        if not supported_cameras:
            seen: set[str] = set()
            for profile in spec.get("robot_profiles", []) or []:
                if not isinstance(profile, Mapping):
                    continue
                for camera in profile.get("observation_cameras", []) or []:
                    if not isinstance(camera, Mapping):
                        continue
                    camera_id = str(camera.get("id") or "").strip()
                    if camera_id and camera_id not in seen:
                        seen.add(camera_id)
                        supported_cameras.append(camera_id)
        if not supported_cameras:
            supported_cameras = ["head_rgb"]

        launchable = bool(health.get("launchable", True)) and validation.launchable
        runtime_blockers = _dedupe_strings(
            list(registration.get("blockers") or []),
            list(health.get("blockers") or []),
            validation.blockers,
        )
        runtime_warnings = _dedupe_strings(
            list(registration.get("warnings") or []),
            list(health.get("warnings") or []),
            validation.warnings,
        )

        registration_payload = {
            **dict(registration),
            "schema_version": "v1",
            "site_world_id": site_world_id,
            "build_id": build_id,
            "scene_id": scene_id,
            "capture_id": capture_id,
            "site_submission_id": spec.get("site_submission_id"),
            "status": (
                "ready"
                if launchable
                else _blocked_status(registration.get("status"), health.get("status"))
            ),
            "runtime_base_url": self.base_url,
            "websocket_base_url": self.ws_base_url,
            "vm_instance_id": os.getenv("VASTAI_INSTANCE_ID") or os.getenv("HOSTNAME") or "local-vm",
            "canonical_package_uri": spec.get("canonical_package_uri"),
            "canonical_package_version": spec.get("canonical_package_version"),
            "cache_path": str(cache_dir),
            "conditioning_source_path": str(conditioning_source),
            "seed_frame_path": str(seed_frame_path),
            "runtime_layer_policy_snapshots": runtime_layer_snapshots,
            "supported_cameras": supported_cameras,
            "scenario_catalog": list(spec.get("scenario_catalog") or []),
            "start_state_catalog": list(spec.get("start_state_catalog") or []),
            "task_catalog": list(spec.get("task_catalog") or []),
            "robot_profiles": list(spec.get("robot_profiles") or []),
            "runtime_capabilities": {
                "supports_step_rollout": True,
                "supports_batch_rollout": True,
                "supports_camera_views": True,
                "supports_stream": True,
                "protected_region_locking": True,
                "runtime_layer_compositing": True,
                "debug_render_outputs": True,
            },
            "registration_mode": str(registration.get("registration_mode") or "package_registration"),
            "intake_source": str(registration.get("intake_source") or "built_site_world_package"),
            "compatibility_notice": str(registration.get("compatibility_notice") or ""),
            "health_uri": f"{self.base_url}/v1/site-worlds/{site_world_id}/health",
            "generated_at": _utc_now_iso(),
            "blockers": runtime_blockers,
            "warnings": runtime_warnings,
            "grounding_status": spec.get("grounding_status")
            or (spec.get("runtime_layer_policy") or {}).get("grounding_status"),
            "ungrounded_reason": spec.get("ungrounded_reason")
            or (spec.get("runtime_layer_policy") or {}).get("ungrounded_reason"),
            "empty_index_cause": spec.get("empty_index_cause"),
        }
        health_payload = {
            **dict(health),
            "schema_version": "v1",
            "site_world_id": site_world_id,
            "build_id": build_id,
            "scene_id": scene_id,
            "capture_id": capture_id,
            "site_submission_id": spec.get("site_submission_id"),
            "healthy": launchable,
            "launchable": launchable,
            "status": (
                "healthy"
                if launchable
                else _blocked_status(health.get("status"), registration.get("status"))
            ),
            "blockers": runtime_blockers,
            "warnings": runtime_warnings,
            "canonical_package_version": spec.get("canonical_package_version"),
            "last_heartbeat_at": _utc_now_iso(),
            "runtime_base_url": self.base_url,
            "websocket_base_url": self.ws_base_url,
            "vm_instance_id": os.getenv("VASTAI_INSTANCE_ID") or os.getenv("HOSTNAME") or "local-vm",
            "supported_cameras": supported_cameras,
            "scenario_catalog": list(spec.get("scenario_catalog") or []),
            "start_state_catalog": list(spec.get("start_state_catalog") or []),
            "task_catalog": list(spec.get("task_catalog") or []),
            "robot_profiles": list(spec.get("robot_profiles") or []),
            "runtime_capabilities": dict(registration_payload.get("runtime_capabilities") or {}),
            "registration_mode": registration_payload.get("registration_mode"),
            "intake_source": registration_payload.get("intake_source"),
            "compatibility_notice": registration_payload.get("compatibility_notice"),
            "grounding_status": registration_payload.get("grounding_status"),
            "ungrounded_reason": registration_payload.get("ungrounded_reason"),
            "empty_index_cause": registration_payload.get("empty_index_cause"),
        }
        _write_json(self._site_world_spec_path(site_world_id), dict(spec))
        _write_json(self._site_world_state_path(site_world_id), registration_payload)
        _write_json(self._site_world_health_path(site_world_id), health_payload)
        return registration_payload

    def load_site_world(self, site_world_id: str) -> Dict[str, Any]:
        path = self._site_world_state_path(site_world_id)
        if not path.is_file():
            raise FileNotFoundError(site_world_id)
        return _read_json(path)

    def _load_site_world_bundle(self, site_world_id: str) -> Dict[str, Any]:
        try:
            bundle = load_site_world_bundle(self._site_world_state_path(site_world_id), require_spec=True)
        except SiteWorldIntakeError as exc:
            raise RuntimeError(str(exc)) from exc
        return {
            "registration": bundle.registration,
            "health": bundle.health,
            "spec": bundle.spec,
            "resolved": bundle.resolved,
        }

    def load_site_world_health(self, site_world_id: str) -> Dict[str, Any]:
        path = self._site_world_health_path(site_world_id)
        if not path.is_file():
            raise FileNotFoundError(site_world_id)
        payload = _read_json(path)
        payload["last_heartbeat_at"] = _utc_now_iso()
        _write_json(path, payload)
        return payload

    def _catalog_entry(self, entries: Sequence[Any], selected_id: str, *, label: str) -> Dict[str, Any]:
        for item in entries:
            if isinstance(item, Mapping) and (
                str(item.get("id") or "").strip() == selected_id
                or str(item.get("task_id") or "").strip() == selected_id
            ):
                return dict(item)
        raise RuntimeError(f"unknown {label}: {selected_id}")

    def create_session(
        self,
        site_world_id: str,
        *,
        session_id: Optional[str] = None,
        robot_profile_id: str,
        task_id: str,
        scenario_id: str,
        start_state_id: str,
        notes: str = "",
        canonical_package_uri: str | None = None,
        canonical_package_version: str | None = None,
        prompt: str | None = None,
        trajectory: Mapping[str, Any] | str | None = None,
        presentation_model: str | None = None,
        debug_mode: bool | None = None,
        unsafe_allow_blocked_site_world: bool = False,
    ) -> Dict[str, Any]:
        bundle = self._load_site_world_bundle(site_world_id)
        registration = bundle["registration"]
        site_world = bundle["resolved"]
        site_world_spec = bundle["spec"]
        health = self.load_site_world_health(site_world_id)
        allow_blocked_site_world = bool(unsafe_allow_blocked_site_world) or _env_truthy(
            "BLUEPRINT_UNSAFE_ALLOW_BLOCKED_SITE_WORLD"
        )
        if not bool(health.get("launchable")) and not allow_blocked_site_world:
            raise RuntimeError(f"site world {site_world_id} is not launchable")

        expected_package_uri = str(site_world.get("canonical_package_uri") or "").strip()
        expected_package_version = str(site_world.get("canonical_package_version") or "").strip()
        if canonical_package_uri and expected_package_uri and str(canonical_package_uri).strip() != expected_package_uri:
            raise RuntimeError("canonical_package_uri_mismatch")
        if canonical_package_version and expected_package_version and str(canonical_package_version).strip() != expected_package_version:
            raise RuntimeError("canonical_package_version_mismatch")

        robot_profile = self._catalog_entry(site_world.get("robot_profiles", []), robot_profile_id, label="robot profile")
        task_entry = self._catalog_entry(site_world.get("task_catalog", []), task_id, label="task")
        scenario_entry = self._catalog_entry(site_world.get("scenario_catalog", []), scenario_id, label="scenario")
        start_state_entry = self._catalog_entry(site_world.get("start_state_catalog", []), start_state_id, label="start state")
        trajectory_payload = normalize_trajectory_payload(trajectory)
        session_id = str(session_id or _stable_id("session", site_world_id, robot_profile_id, task_id, _utc_now_iso())).strip()
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        session_state = {
            "schema_version": "v1",
            "session_id": session_id,
            "site_world_id": site_world_id,
            "build_id": registration.get("build_id"),
            "scene_id": registration.get("scene_id"),
            "capture_id": registration.get("capture_id"),
            "runtime_base_url": registration.get("runtime_base_url"),
            "status": "ready",
            "canonical_package_uri": expected_package_uri,
            "canonical_package_version": expected_package_version,
            "robot_profile": robot_profile,
            "task": task_entry,
            "scenario": scenario_entry,
            "start_state": start_state_entry,
            "notes": notes,
            "presentation_config": {
                "prompt": prompt,
                "trajectory": trajectory_payload,
                "presentation_model": presentation_model or "runtime_default",
                "debug_mode": bool(debug_mode),
                "unsafe_allow_blocked_site_world": allow_blocked_site_world,
            },
            "unsafe_allow_blocked_site_world": allow_blocked_site_world,
            "quality_flags": {"presentation_quality": "normal", "editable_ratio": 0.0, "locked_ratio": 0.0},
            "protected_region_violations": [],
            "debug_artifacts": {},
            "step_index": 0,
            "done": False,
            "success": None,
            "failure_reason": None,
            "reward": 0.0,
            "action_trace": [],
            "latest_render_paths": {},
            "created_at": _utc_now_iso(),
        }
        _write_json(self._session_state_path(session_id), session_state)
        update_presentation_session_manifest(
            session_dir=session_dir,
            payload={
                "schema_version": "v1",
                "session_id": session_id,
                "site_world_id": site_world_id,
                "canonical_package_uri": expected_package_uri,
                "canonical_package_version": expected_package_version,
                "prompt": prompt,
                "trajectory": trajectory_payload,
                "presentation_model": presentation_model or "runtime_default",
                "debug_mode": bool(debug_mode),
                "quality_flags": session_state["quality_flags"],
                "protected_region_violations": session_state["protected_region_violations"],
                "policy_refs": dict(site_world_spec.get("runtime_layer_policy") or {}),
                "latest_debug_artifacts": {},
            },
        )
        return {
            "session_id": session_id,
            "site_world_id": site_world_id,
            "build_id": registration.get("build_id"),
            "status": "ready",
            "canonical_package_version": expected_package_version,
            "presentation_config": dict(session_state["presentation_config"]),
            "unsafe_allow_blocked_site_world": allow_blocked_site_world,
            "quality_flags": dict(session_state["quality_flags"]),
            "protected_region_violations": list(session_state["protected_region_violations"]),
            "debug_artifacts": dict(session_state["debug_artifacts"]),
            "runtime_capabilities": registration.get("runtime_capabilities", {}),
            "observation_cameras": list(robot_profile.get("observation_cameras") or []),
        }

    def load_session(self, session_id: str) -> Dict[str, Any]:
        path = self._session_state_path(session_id)
        if not path.is_file():
            raise FileNotFoundError(session_id)
        return _read_json(path)

    def _frame_for_session(self, session_state: Mapping[str, Any]) -> np.ndarray:
        registration = self.load_site_world(str(session_state.get("site_world_id") or ""))
        seed_frame = Path(str(registration.get("seed_frame_path") or "")).resolve()
        if not seed_frame.is_file():
            raise RuntimeError(f"seed frame missing for {registration.get('site_world_id')}")
        return _load_frame(seed_frame)

    def _runtime_layer_bundle_for_site_world(self, site_world_id: str) -> Dict[str, Any]:
        registration = self.load_site_world(site_world_id)
        snapshots = dict(registration.get("runtime_layer_policy_snapshots") or {})
        protected_path = Path(str(snapshots.get("protected_regions_manifest_path") or "")).resolve()
        render_policy_path = Path(str(snapshots.get("canonical_render_policy_path") or "")).resolve()
        variance_policy_path = Path(str(snapshots.get("presentation_variance_policy_path") or "")).resolve()
        return {
            "protected_regions_manifest": _read_json(protected_path),
            "canonical_render_policy": _read_json(render_policy_path),
            "presentation_variance_policy": _read_json(variance_policy_path),
        }

    def _observation_payload(self, session_state: Dict[str, Any], frame: np.ndarray) -> Dict[str, Any]:
        session_id = str(session_state["session_id"])
        session_dir = self._session_dir(session_id)
        step_index = int(session_state.get("step_index", 0))
        robot_profile = dict(session_state.get("robot_profile") or {})
        cameras = robot_profile.get("observation_cameras") or [{"id": "head_rgb", "role": "head"}]
        runtime_layer_bundle = self._runtime_layer_bundle_for_site_world(str(session_state.get("site_world_id") or ""))
        presentation_config = dict(session_state.get("presentation_config") or {})

        camera_summaries = []
        latest_render_paths: Dict[str, str] = {}
        primary_camera_id = ""
        quality_flags = {"presentation_quality": "normal", "editable_ratio": 0.0, "locked_ratio": 0.0}
        protected_region_violations: list[Dict[str, Any]] = []
        debug_artifacts: Dict[str, Any] = {}
        for camera in cameras:
            if not isinstance(camera, Mapping):
                continue
            camera_id = str(camera.get("id") or "").strip()
            if not camera_id:
                continue
            canonical_camera_frame = _coerce_camera_frame(frame, camera_id)
            composite = composite_runtime_layer(
                canonical_frame=canonical_camera_frame,
                protected_regions_manifest=runtime_layer_bundle["protected_regions_manifest"],
                canonical_render_policy=runtime_layer_bundle["canonical_render_policy"],
                presentation_config=presentation_config,
                presentation_variance_policy=runtime_layer_bundle["presentation_variance_policy"],
                session_dir=session_dir,
                step_index=step_index,
                camera_id=camera_id,
            )
            camera_frame = composite["frame"]
            output_path = session_dir / "renders" / camera_id / f"frame_{step_index:03d}.png"
            _save_frame(output_path, camera_frame)
            latest_render_paths[camera_id] = str(output_path)
            if not primary_camera_id:
                primary_camera_id = camera_id
                quality_flags = dict(composite.get("quality_flags") or quality_flags)
                debug_artifacts = dict(composite.get("debug_artifacts") or {})
            protected_region_violations.extend(
                [
                    {**dict(item), "camera_id": camera_id}
                    for item in composite.get("protected_region_violations", [])
                    if isinstance(item, Mapping)
                ]
            )
            camera_summaries.append(
                {
                    "cameraId": camera_id,
                    "role": str(camera.get("role") or ""),
                    "required": bool(camera.get("required", False)),
                    "available": True,
                    "framePath": f"{self.base_url}/v1/sessions/{session_id}/render?camera_id={camera_id}",
                }
            )

        session_state["latest_render_paths"] = latest_render_paths
        session_state["quality_flags"] = quality_flags
        session_state["protected_region_violations"] = protected_region_violations
        session_state["debug_artifacts"] = debug_artifacts
        update_presentation_session_manifest(
            session_dir=session_dir,
            payload={
                "quality_flags": quality_flags,
                "protected_region_violations": protected_region_violations,
                "latest_debug_artifacts": debug_artifacts,
            },
        )
        return {
            "frame_path": f"{self.base_url}/v1/sessions/{session_id}/render?camera_id={primary_camera_id}",
            "primaryCameraId": primary_camera_id,
            "cameraFrames": camera_summaries,
            "runtimeMetadata": {
                "site_world_id": session_state.get("site_world_id"),
                "build_id": session_state.get("build_id"),
                "step_index": step_index,
                "canonical_package_version": session_state.get("canonical_package_version"),
                "quality_flags": quality_flags,
                "protected_region_violations": protected_region_violations,
                "debug_artifacts": debug_artifacts,
            },
            "worldSnapshot": {
                "task_id": (session_state.get("task") or {}).get("id"),
                "scenario_id": (session_state.get("scenario") or {}).get("id"),
                "start_state_id": (session_state.get("start_state") or {}).get("id"),
                "status": session_state.get("status"),
            },
        }

    def _episode_payload(self, session_state: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "episodeId": str(session_state["session_id"]),
            "taskId": str((session_state.get("task") or {}).get("id") or ""),
            "task": str((session_state.get("task") or {}).get("task_text") or (session_state.get("task") or {}).get("name") or ""),
            "scenarioId": str((session_state.get("scenario") or {}).get("id") or ""),
            "scenario": str((session_state.get("scenario") or {}).get("name") or ""),
            "startStateId": str((session_state.get("start_state") or {}).get("id") or ""),
            "startState": str((session_state.get("start_state") or {}).get("name") or ""),
            "status": str(session_state.get("status") or "ready"),
            "stepIndex": int(session_state.get("step_index", 0)),
            "done": bool(session_state.get("done", False)),
            "reward": float(session_state.get("reward", 0.0) or 0.0),
            "success": session_state.get("success"),
            "failureReason": session_state.get("failure_reason"),
            "observation": observation,
            "observationCameras": observation.get("cameraFrames", []),
            "actionTrace": list(session_state.get("action_trace", [])),
            "artifactUris": {},
            "canonicalPackageVersion": session_state.get("canonical_package_version"),
            "presentationConfig": dict(session_state.get("presentation_config") or {}),
            "qualityFlags": dict(session_state.get("quality_flags") or {}),
            "protectedRegionViolations": list(session_state.get("protected_region_violations") or []),
            "debugArtifacts": dict(session_state.get("debug_artifacts") or {}),
        }

    def reset_session(
        self,
        session_id: str,
        *,
        task_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        start_state_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        session_state = self.load_session(session_id)
        bundle = self._load_site_world_bundle(str(session_state.get("site_world_id") or ""))
        site_world = bundle["resolved"]
        if task_id:
            session_state["task"] = self._catalog_entry(site_world.get("task_catalog", []), task_id, label="task")
        if scenario_id:
            session_state["scenario"] = self._catalog_entry(site_world.get("scenario_catalog", []), scenario_id, label="scenario")
        if start_state_id:
            session_state["start_state"] = self._catalog_entry(site_world.get("start_state_catalog", []), start_state_id, label="start state")
        session_state["step_index"] = 0
        session_state["done"] = False
        session_state["success"] = None
        session_state["failure_reason"] = None
        session_state["reward"] = 0.0
        session_state["action_trace"] = []
        session_state["status"] = "running"
        frame = self._frame_for_session(session_state)
        observation = self._observation_payload(session_state, frame)
        _write_json(self._session_state_path(session_id), session_state)
        return {
            "session_id": session_id,
            "episode": self._episode_payload(session_state, observation),
        }

    def step_session(self, session_id: str, *, action: Sequence[float]) -> Dict[str, Any]:
        session_state = self.load_session(session_id)
        if bool(session_state.get("done")):
            raise RuntimeError(f"session {session_id} is already complete")
        next_step = int(session_state.get("step_index", 0)) + 1
        base_frame = self._frame_for_session(session_state)
        frame = _apply_action(base_frame, action, next_step)
        session_state["step_index"] = next_step
        session_state["action_trace"] = list(session_state.get("action_trace", [])) + [list(action)]
        session_state["reward"] = round(min(float(next_step) / 6.0, 1.0), 4)
        session_state["done"] = next_step >= 6
        session_state["success"] = True if session_state["done"] else None
        session_state["failure_reason"] = None
        session_state["status"] = "completed" if session_state["done"] else "running"
        observation = self._observation_payload(session_state, frame)
        _write_json(self._session_state_path(session_id), session_state)
        return {
            "session_id": session_id,
            "episode": self._episode_payload(session_state, observation),
        }

    def session_state(self, session_id: str) -> Dict[str, Any]:
        session_state = self.load_session(session_id)
        frame = self._frame_for_session(session_state)
        observation = self._observation_payload(session_state, frame)
        return {
            "session_id": session_id,
            "site_world_id": session_state.get("site_world_id"),
            "build_id": session_state.get("build_id"),
            "status": session_state.get("status"),
            "step_index": session_state.get("step_index"),
            "done": session_state.get("done"),
            "reward": session_state.get("reward"),
            "observation": observation,
            "action_trace": session_state.get("action_trace", []),
            "canonical_package_version": session_state.get("canonical_package_version"),
            "presentation_config": dict(session_state.get("presentation_config") or {}),
            "quality_flags": dict(session_state.get("quality_flags") or {}),
            "protected_region_violations": list(session_state.get("protected_region_violations") or []),
            "debug_artifacts": dict(session_state.get("debug_artifacts") or {}),
        }

    def render_bytes(self, session_id: str, camera_id: str) -> bytes:
        session_state = self.load_session(session_id)
        latest = dict(session_state.get("latest_render_paths") or {})
        render_path = Path(str(latest.get(camera_id) or "")).resolve()
        if not render_path.is_file():
            frame = self._frame_for_session(session_state)
            observation = self._observation_payload(session_state, frame)
            latest = dict(session_state.get("latest_render_paths") or {})
            render_path = Path(str(latest.get(camera_id) or latest.get(observation.get("primaryCameraId")) or "")).resolve()
        payload = render_path.read_bytes()
        if not payload:
            raise RuntimeError(f"missing render bytes for {session_id}:{camera_id}")
        return payload
