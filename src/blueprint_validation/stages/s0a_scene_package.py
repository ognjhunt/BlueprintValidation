"""Compatibility Stage 0a: resolve or build a simulator-ready fallback scene package."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional

from ..common import StageResult, sanitize_filename_component
from ..config import FacilityConfig, ValidationConfig
from ..scene_builder import SceneAssetManifestError, build_scene_package
from ..teleop.contracts import TeleopManifestError, load_and_validate_scene_package
from .base import PipelineStage


class ScenePackageStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s0a_scene_package"

    @property
    def description(self) -> str:
        return "Resolve or build a strict scene package for Isaac/PolaRiS compatibility fallback paths"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        del previous_results
        resolved = _validate_existing_scene_package(facility.scene_package_path)
        if resolved is not None:
            facility.scene_package_path = resolved
            payload = load_and_validate_scene_package(resolved)
            return StageResult(
                stage_name=self.name,
                status="success",
                elapsed_seconds=0.0,
                outputs={
                    "scene_package_path": str(resolved),
                    "scene_manifest_path": str(payload["scene_manifest_path"]),
                    "usd_scene_path": str(payload["usd_scene_path"]),
                    "has_isaac_lab": bool(payload["has_isaac_lab"]),
                    "has_geniesim_task_config": bool(payload["has_geniesim_task_config"]),
                    "source": "facility.scene_package_path",
                },
                metrics={
                    "render_backend_candidate": "isaac_scene",
                    "built_scene_package": False,
                },
                detail="Validated facility.scene_package_path for Isaac-backed stages.",
            )

        if not bool(config.scene_builder.enabled):
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0.0,
                metrics={"render_backend_candidate": "gsplat"},
                detail="No valid scene package configured and scene_builder.enabled=false.",
            )

        try:
            build_result = _build_scene_package_for_facility(config, facility, work_dir)
        except (RuntimeError, ValueError, SceneAssetManifestError, TeleopManifestError) as exc:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0.0,
                metrics={"render_backend_candidate": "gsplat"},
                detail=str(exc),
            )

        facility.scene_package_path = build_result.scene_root
        return StageResult(
            stage_name=self.name,
            status="success",
            elapsed_seconds=0.0,
            outputs={
                "scene_package_path": str(build_result.scene_root),
                "scene_manifest_path": str(build_result.scene_manifest_path),
                "usd_scene_path": str(build_result.usd_scene_path),
                "visual_usd_scene_path": str(getattr(build_result, "visual_usd_scene_path", "")),
                "physics_usd_scene_path": str(getattr(build_result, "physics_usd_scene_path", "")),
                "replacement_manifest_path": str(getattr(build_result, "replacement_manifest_path", "")),
                "support_surfaces_path": str(getattr(build_result, "support_surfaces_path", "")),
                "physics_qc_path": str(getattr(build_result, "physics_qc_path", "")),
                "task_config_path": str(build_result.task_config_path) if build_result.task_config_path else "",
                "isaac_lab_package_root": str(build_result.isaac_lab_package_root),
                "source": "scene_builder",
            },
            metrics={
                "render_backend_candidate": "isaac_scene",
                "built_scene_package": True,
            },
            detail="Built scene package for Isaac-backed stages.",
        )


def _validate_existing_scene_package(scene_root: Optional[Path]) -> Optional[Path]:
    if scene_root is None:
        return None
    try:
        payload = load_and_validate_scene_package(scene_root)
    except (OSError, TeleopManifestError):
        return None
    if not bool(payload.get("has_runnable_env", False)):
        return None
    return scene_root.resolve()


def _build_scene_package_for_facility(
    config: ValidationConfig,
    facility: FacilityConfig,
    work_dir: Path,
):
    if facility.ply_path is None:
        target_name = facility.opportunity_id or facility.name or work_dir.name
        raise ValueError(
            "Scene-package build requires a resolved geometry PLY for "
            f"'{target_name}'. Configure facility.geometry_bundle_path or facility.ply_path."
        )
    build_config = copy.deepcopy(config)
    build_config.scene_builder.source_ply_path = facility.ply_path.resolve()
    if facility.task_hints_path is not None:
        build_config.scene_builder.task_hints_path = facility.task_hints_path.resolve()
    if len(config.facilities) > 1:
        facility_id = sanitize_filename_component(work_dir.name, fallback="facility")
        build_config.scene_builder.output_scene_root = (
            build_config.scene_builder.output_scene_root / facility_id
        )
    return build_scene_package(build_config)
