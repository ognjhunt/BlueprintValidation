"""Resolve the preferred world-model runtime for a scene-memory bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .common import read_json
from .config import FacilityConfig, ValidationConfig

_ACTIVE_RUNTIME_BACKENDS = ("neoverse", "gen3c", "cosmos_transfer")


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = read_json(path)
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _normalized_backend_list(values: List[str]) -> List[str]:
    normalized: List[str] = []
    for raw in values:
        backend = str(raw or "").strip().lower()
        if backend and backend not in normalized:
            normalized.append(backend)
    return normalized


def _scene_memory_backend_runtime_payload(
    config: ValidationConfig,
    backend_id: str,
) -> Dict[str, Any]:
    if backend_id == "neoverse":
        runtime = config.scene_memory_runtime.neoverse
        return {
            "allow_runtime_execution": bool(runtime.allow_runtime_execution),
            "repo_path": str(runtime.repo_path) if runtime.repo_path is not None else None,
            "python_executable": (
                str(runtime.python_executable)
                if runtime.python_executable is not None
                else None
            ),
            "inference_script": runtime.inference_script,
            "checkpoint_path": (
                str(runtime.checkpoint_path) if runtime.checkpoint_path is not None else None
            ),
            "execution_mode": (
                "runtime_configured"
                if runtime.enabled
                and runtime.allow_runtime_execution
                and runtime.repo_path is not None
                and bool(runtime.inference_script)
                else "manifest_only"
            ),
        }
    if backend_id == "gen3c":
        runtime = config.scene_memory_runtime.gen3c
        return {
            "allow_runtime_execution": bool(runtime.allow_runtime_execution),
            "repo_path": str(runtime.repo_path) if runtime.repo_path is not None else None,
            "python_executable": (
                str(runtime.python_executable)
                if runtime.python_executable is not None
                else None
            ),
            "inference_script": runtime.inference_script,
            "checkpoint_path": (
                str(runtime.checkpoint_path) if runtime.checkpoint_path is not None else None
            ),
            "execution_mode": (
                "runtime_configured"
                if runtime.enabled
                and runtime.allow_runtime_execution
                and runtime.repo_path is not None
                and bool(runtime.inference_script)
                else "manifest_only"
            ),
        }
    if backend_id == "cosmos_transfer":
        return {
            "allow_runtime_execution": True,
            "repo_path": str(config.enrich.cosmos_repo),
            "python_executable": None,
            "inference_script": "blueprint_validation.enrichment.cosmos_runner.enrich_clip",
            "checkpoint_path": str(config.enrich.cosmos_checkpoint),
            "execution_mode": "stage2_enrich",
        }
    return {
        "allow_runtime_execution": False,
        "repo_path": None,
        "python_executable": None,
        "inference_script": None,
        "checkpoint_path": None,
        "execution_mode": "watchlist_only",
    }


def resolve_scene_memory_runtime_plan(
    config: ValidationConfig,
    facility: FacilityConfig,
) -> Dict[str, Any]:
    preferred_backends = _normalized_backend_list(
        config.scene_memory_runtime.preferred_backends
    )
    watchlist_backends = set(
        _normalized_backend_list(config.scene_memory_runtime.watchlist_backends)
    )

    preview_manifest: Dict[str, Any] = {}
    preview_supported_backends: List[str] = []
    if facility.preview_simulation_path is not None:
        preview_manifest_path = facility.preview_simulation_path / "preview_simulation_manifest.json"
        if preview_manifest_path.exists():
            preview_manifest = _safe_read_json(preview_manifest_path)
            preview_supported_backends = _normalized_backend_list(
                [
                    str(v)
                    for v in preview_manifest.get("supported_backends", [])
                    if str(v).strip()
                ]
            )

    backend_plans: Dict[str, Dict[str, Any]] = {}
    available_backends: List[str] = []
    skipped_watchlist_backends: List[str] = []

    for backend_id, manifest_path in facility.scene_memory_adapter_manifests.items():
        normalized_backend = str(backend_id or "").strip().lower()
        if not normalized_backend:
            continue
        manifest_file = Path(manifest_path)
        manifest_payload = _safe_read_json(manifest_file) if manifest_file.exists() else {}
        runtime_payload = _scene_memory_backend_runtime_payload(config, normalized_backend)
        is_watchlist = normalized_backend in watchlist_backends
        if manifest_file.exists() and not is_watchlist:
            available_backends.append(normalized_backend)
        elif manifest_file.exists() and is_watchlist:
            skipped_watchlist_backends.append(normalized_backend)
        backend_plans[normalized_backend] = {
            "adapter_id": normalized_backend,
            "manifest_path": str(manifest_file),
            "manifest_exists": manifest_file.exists(),
            "family": manifest_payload.get("family"),
            "status": manifest_payload.get("status"),
            "preferred_conditioning": list(
                manifest_payload.get("preferred_conditioning", []) or []
            ),
            "watchlist_only": is_watchlist,
            **runtime_payload,
        }

    ordered_available = [
        backend for backend in preferred_backends if backend in available_backends
    ]
    ordered_available.extend(
        backend
        for backend in available_backends
        if backend not in ordered_available
    )

    selected_backend = ordered_available[0] if ordered_available else None
    secondary_backend = ordered_available[1] if len(ordered_available) > 1 else None
    fallback_backend = (
        ordered_available[2]
        if len(ordered_available) > 2 and config.scene_memory_runtime.allow_backend_fallback
        else None
    )

    if selected_backend:
        selection_reason = (
            f"Selected {selected_backend} as the primary scene-memory runtime based on "
            "configured backend priority."
        )
    elif skipped_watchlist_backends:
        selection_reason = (
            "Only watchlist backends were present in the scene-memory adapter bundle; "
            "no active runtime selected."
        )
    else:
        selection_reason = (
            "No active scene-memory runtime adapter manifests were available; "
            "legacy geometry and simulator adapters remain available."
        )

    for backend_id, backend_payload in backend_plans.items():
        backend_payload["selected"] = backend_id == selected_backend
        backend_payload["secondary"] = backend_id == secondary_backend
        backend_payload["fallback"] = backend_id == fallback_backend

    return {
        "schema_version": "v1",
        "scene_memory_bundle_path": (
            str(facility.scene_memory_bundle_path)
            if facility.scene_memory_bundle_path is not None
            else None
        ),
        "preview_simulation_path": (
            str(facility.preview_simulation_path)
            if facility.preview_simulation_path is not None
            else None
        ),
        "preferred_backends": preferred_backends,
        "watchlist_backends": sorted(watchlist_backends),
        "preview_supported_backends": preview_supported_backends,
        "available_backends": ordered_available,
        "selected_backend": selected_backend,
        "secondary_backend": secondary_backend,
        "fallback_backend": fallback_backend,
        "selection_reason": selection_reason,
        "skipped_watchlist_backends": sorted(skipped_watchlist_backends),
        "adapter_manifests": backend_plans,
        "default_runtime_policy": {
            "primary": "neoverse",
            "secondary": "gen3c",
            "fallback": "cosmos_transfer",
            "watchlist_only": ["3dsceneprompt"],
        },
    }

