"""Helpers for preferred intake lineage and scene-memory runtime resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .common import read_json
from .config import FacilityConfig, ValidationConfig
from .scene_memory_runtime import resolve_scene_memory_runtime_plan


def resolve_intake_lineage(facility: FacilityConfig) -> Dict[str, Any]:
    """Return normalized intake lineage for a facility.

    This keeps `intake_mode` as the source-contract marker while exposing the
    preferred downstream intake kind separately.
    """

    has_scene_memory_bundle = facility.scene_memory_bundle_path is not None
    has_geometry_bundle = (
        facility.geometry_bundle_path is not None or facility.ply_path is not None
    )
    has_scene_package = facility.scene_package_path is not None

    if has_scene_memory_bundle:
        preferred_intake_kind = "scene_memory_bundle"
    elif has_geometry_bundle:
        preferred_intake_kind = "geometry_bundle"
    elif has_scene_package:
        preferred_intake_kind = "scene_package"
    else:
        preferred_intake_kind = "legacy_direct"

    return {
        "intake_mode": str(getattr(facility, "intake_mode", "legacy_direct") or "legacy_direct"),
        "preferred_intake_kind": preferred_intake_kind,
        "has_scene_memory_bundle": has_scene_memory_bundle,
        "has_geometry_bundle": has_geometry_bundle,
        "has_scene_package": has_scene_package,
        "scene_memory_bundle_path": (
            str(facility.scene_memory_bundle_path)
            if facility.scene_memory_bundle_path is not None
            else None
        ),
        "geometry_bundle_path": (
            str(facility.geometry_bundle_path)
            if facility.geometry_bundle_path is not None
            else None
        ),
        "ply_path": str(facility.ply_path) if facility.ply_path is not None else None,
        "scene_package_path": (
            str(facility.scene_package_path)
            if facility.scene_package_path is not None
            else None
        ),
        "preview_simulation_path": (
            str(facility.preview_simulation_path)
            if facility.preview_simulation_path is not None
            else None
        ),
        "uses_qualified_handoff": bool(getattr(facility, "uses_qualified_handoff", False)),
    }


def _read_runtime_selection_file(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    return dict(payload) if isinstance(payload, dict) else None


def resolve_scene_memory_runtime_metadata(
    config: ValidationConfig,
    facility: FacilityConfig,
    *,
    work_dir: Path,
    previous_results: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Resolve stable scene-memory runtime metadata for stages and reports.

    Resolution order:
    1. `s0b_scene_memory_runtime` result path when present.
    2. Existing `scene_memory_runtime/runtime_selection.json` on disk.
    3. Recompute from config + facility without executing any runtime.
    """

    runtime_stage = (previous_results or {}).get("s0b_scene_memory_runtime")
    if runtime_stage is not None:
        runtime_selection_path = Path(
            str(runtime_stage.outputs.get("runtime_selection_path", "") or "").strip()
        )
        if runtime_selection_path:
            payload = _read_runtime_selection_file(runtime_selection_path)
            if payload is not None:
                return payload

    payload = _read_runtime_selection_file(
        work_dir / "scene_memory_runtime" / "runtime_selection.json"
    )
    if payload is not None:
        return payload

    if facility.scene_memory_bundle_path is not None:
        return resolve_scene_memory_runtime_plan(config, facility)

    return {
        "schema_version": "v1",
        "scene_memory_bundle_path": None,
        "preview_simulation_path": (
            str(facility.preview_simulation_path)
            if facility.preview_simulation_path is not None
            else None
        ),
        "preferred_backends": list(config.scene_memory_runtime.preferred_backends),
        "watchlist_backends": list(config.scene_memory_runtime.watchlist_backends),
        "preview_supported_backends": [],
        "available_backends": [],
        "selected_backend": None,
        "secondary_backend": None,
        "fallback_backend": None,
        "selection_reason": (
            "No scene-memory bundle configured for this evaluation target; "
            "legacy geometry and strict simulator adapters remain available."
        ),
        "skipped_watchlist_backends": [],
        "adapter_manifests": {},
        "default_runtime_policy": {
            "primary": "neoverse",
            "secondary": "gen3c",
            "fallback": "cosmos_transfer",
            "watchlist_only": ["3dsceneprompt"],
        },
    }


def summarize_scene_memory_runtime(runtime_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return the stable public runtime summary exposed in manifests and reports."""

    return {
        "selected_backend": runtime_payload.get("selected_backend"),
        "secondary_backend": runtime_payload.get("secondary_backend"),
        "fallback_backend": runtime_payload.get("fallback_backend"),
        "available_backends": list(runtime_payload.get("available_backends", []) or []),
        "selection_reason": runtime_payload.get("selection_reason"),
        "preferred_backends": list(runtime_payload.get("preferred_backends", []) or []),
        "watchlist_backends": list(runtime_payload.get("watchlist_backends", []) or []),
        "default_runtime_policy": dict(runtime_payload.get("default_runtime_policy", {}) or {}),
    }
