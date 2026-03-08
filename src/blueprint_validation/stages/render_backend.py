"""Helpers for choosing between gsplat and Isaac-backed Stage-1 render paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from ..common import StageResult
from ..config import FacilityConfig, ValidationConfig
from ..teleop.contracts import TeleopManifestError, load_and_validate_scene_package
from .manifest_resolution import ManifestCandidate, ManifestSource, resolve_manifest_source


def active_render_backend(
    config: ValidationConfig,
    facility: FacilityConfig,
    previous_results: Dict[str, StageResult] | None = None,
) -> str:
    """Resolve the effective Stage-1 backend for a facility."""
    backend = str(getattr(config.render, "backend", "auto") or "auto").strip().lower()
    if backend in {"gsplat", "isaac_scene"}:
        return backend
    if not unsafe_scene_package_imports_enabled():
        return "gsplat"
    if resolved_scene_package_path(facility, previous_results) is not None:
        return "isaac_scene"
    if bool(getattr(config.scene_builder, "enabled", False)):
        return "isaac_scene"
    return "gsplat"


def unsafe_scene_package_imports_enabled() -> bool:
    """Return True when unsafe executable scene package imports are explicitly allowed."""
    return os.environ.get("BLUEPRINT_UNSAFE_ALLOW_SCENE_PACKAGE_IMPORT", "0") == "1"


def resolved_scene_package_path(
    facility: FacilityConfig,
    previous_results: Dict[str, StageResult] | None = None,
) -> Optional[Path]:
    """Return the validated scene package root if one is currently available."""
    stage_result = (previous_results or {}).get("s0a_scene_package")
    if stage_result is not None and stage_result.status == "success":
        scene_root_raw = str(stage_result.outputs.get("scene_package_path", "") or "").strip()
        if scene_root_raw:
            return Path(scene_root_raw)
    scene_root = facility.scene_package_path
    if scene_root is None:
        return None
    try:
        payload = load_and_validate_scene_package(scene_root)
    except (OSError, TeleopManifestError):
        return None
    if not bool(payload.get("has_runnable_env", False)):
        return None
    return scene_root.resolve()


def resolve_stage1_render_manifest_source(
    work_dir: Path,
    previous_results: Dict[str, StageResult] | None = None,
) -> Optional[ManifestSource]:
    candidates = [
        ManifestCandidate(
            stage_name="s1_isaac_render",
            manifest_relpath=Path("isaac_renders/render_manifest.json"),
        ),
        ManifestCandidate(
            stage_name="s1_render",
            manifest_relpath=Path("renders/render_manifest.json"),
        ),
    ]
    source = resolve_manifest_source(
        work_dir=work_dir,
        previous_results=previous_results or {},
        candidates=candidates,
    )
    if source is not None:
        return source
    present_stage_keys = {candidate.stage_name for candidate in candidates} & set(previous_results or {})
    if present_stage_keys:
        return None
    for candidate in candidates:
        manifest_path = work_dir / candidate.manifest_relpath
        if manifest_path.exists():
            return ManifestSource(
                source_stage=candidate.stage_name,
                source_manifest_path=manifest_path,
                source_mode="filesystem_fallback",
            )
    return None
