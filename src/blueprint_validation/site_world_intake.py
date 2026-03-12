"""Helpers for loading and normalizing site-world handoff artifacts."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


class SiteWorldIntakeError(RuntimeError):
    """Raised when site-world handoff artifacts are incomplete or invalid."""


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _optional_path(value: Any) -> Optional[Path]:
    text = str(value or "").strip()
    if not text or text.startswith(("gs://", "http://", "https://")):
        return None
    return Path(text).resolve()


def adjacent_site_world_paths(registration_path: Path) -> tuple[Path, Path]:
    root = registration_path.parent
    return root / "site_world_health.json", root / "site_world_spec.json"


def normalize_trajectory_payload(trajectory: Mapping[str, Any] | str | None) -> Dict[str, Any]:
    if isinstance(trajectory, Mapping):
        payload = dict(trajectory)
        if "trajectory" not in payload:
            payload["trajectory"] = "static"
        return payload
    token = str(trajectory or "").strip()
    return {"trajectory": token or "static"}


def merge_site_world_definition(
    *,
    registration: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> Dict[str, Any]:
    merged = copy.deepcopy(dict(registration))
    if not spec:
        return merged
    for key in (
        "canonical_package_uri",
        "canonical_package_version",
        "task_catalog",
        "scenario_catalog",
        "start_state_catalog",
        "robot_profiles",
        "qualification_state",
        "downstream_evaluation_eligibility",
        "grounding_status",
        "ungrounded_reason",
        "capture_source",
        "conditioning",
        "geometry",
        "runtime_layer_policy",
        "task_anchor_manifest_path",
    ):
        if key in spec:
            merged[key] = copy.deepcopy(spec[key])
    return merged


def grounding_summary(spec: Mapping[str, Any]) -> Dict[str, Any]:
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


@dataclass(frozen=True)
class SiteWorldBundle:
    registration: Dict[str, Any]
    health: Dict[str, Any]
    spec: Dict[str, Any]
    resolved: Dict[str, Any]
    grounding: Dict[str, Any]
    registration_path: Path
    health_path: Path
    spec_path: Path


def load_site_world_bundle(registration_path: Path, *, require_spec: bool = False) -> SiteWorldBundle:
    registration_path = registration_path.resolve()
    if not registration_path.is_file():
        raise SiteWorldIntakeError(f"site-world registration not found: {registration_path}")
    registration = _read_json(registration_path)
    health_path, spec_path = adjacent_site_world_paths(registration_path)
    health = _read_json(health_path) if health_path.is_file() else {}
    if require_spec and not spec_path.is_file():
        raise SiteWorldIntakeError(f"adjacent site-world spec not found: {spec_path}")
    spec = _read_json(spec_path) if spec_path.is_file() else {}
    resolved = merge_site_world_definition(registration=registration, spec=spec)
    return SiteWorldBundle(
        registration=registration,
        health=health,
        spec=spec,
        resolved=resolved,
        grounding=grounding_summary(spec) if spec else {
            "checks": {},
            "missing_required": [],
            "missing_optional": [],
            "qualification_state": "",
            "downstream_evaluation_eligibility": None,
            "task_catalog_count": 0,
            "scenario_catalog_count": 0,
            "start_state_catalog_count": 0,
            "robot_profile_count": 0,
        },
        registration_path=registration_path,
        health_path=health_path,
        spec_path=spec_path,
    )
