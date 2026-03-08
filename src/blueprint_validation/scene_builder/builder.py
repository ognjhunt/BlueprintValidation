"""Direct PLY-to-scene-package builder for teleop and PolaRiS."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from ..common import sanitize_filename_component, write_json, write_text_atomic
from ..config import ValidationConfig
from ..rendering.scene_geometry import load_obbs_from_task_targets
from ..warmup import load_ply_means_numpy
from .manifest import (
    BoundingBoxSpec,
    ExternalArtifactSpec,
    ImportedAssetSpec,
    RemoveRegionSpec,
    SceneAssetManifest,
    SceneEditManifest,
    SupportSurfaceSpec,
    load_scene_asset_manifest,
    load_scene_edit_manifest,
)


@dataclass(frozen=True)
class SceneBuildResult:
    scene_root: Path
    scene_manifest_path: Path
    usd_scene_path: Path
    visual_usd_scene_path: Path
    physics_usd_scene_path: Path
    replacement_manifest_path: Path
    support_surfaces_path: Path
    physics_qc_path: Path
    task_config_path: Path
    isaac_lab_package_root: Path


@dataclass(frozen=True)
class _BackgroundAsset:
    source_ply_path: Path
    copied_ply_path: Path
    visual_usda_path: Path
    bbox_min: List[float]
    bbox_max: List[float]
    center: List[float]
    extents: List[float]
    collision_mode: str


@dataclass(frozen=True)
class _CopiedAsset:
    object_id: str
    label: str
    asset_type: str
    task_role: str
    copied_asset_path: Path
    relative_asset_path: str
    position: List[float]
    rotation_quaternion: List[float]
    scale: List[float]


@dataclass(frozen=True)
class _CopiedExternalArtifact:
    artifact_id: str
    artifact_type: str
    source_tool: str
    role: str
    copied_path: Path
    relative_path: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class _TaskHintSeed:
    instance_id: str
    label: str
    category: str
    confidence: float
    bounding_box: BoundingBoxSpec


@dataclass(frozen=True)
class _MaterializedRemoveRegion:
    spec: RemoveRegionSpec
    artifact_relative_path: str


@dataclass(frozen=True)
class _MaterializedSupportSurface:
    spec: SupportSurfaceSpec
    artifact_relative_path: str


def build_scene_package(config: ValidationConfig) -> SceneBuildResult:
    builder_cfg = config.scene_builder
    hybrid_scene_edit_enabled = bool(builder_cfg.emit_isaac_lab)
    if not bool(builder_cfg.enabled):
        raise RuntimeError("scene_builder.enabled=false")
    if builder_cfg.source_ply_path is None:
        raise RuntimeError("scene_builder.source_ply_path is not set")
    if builder_cfg.asset_manifest_path is None:
        raise RuntimeError("scene_builder.asset_manifest_path is not set")

    manifest = load_scene_asset_manifest(builder_cfg.asset_manifest_path)
    scene_edit_manifest = (
        load_scene_edit_manifest(builder_cfg.scene_edit_manifest_path)
        if hybrid_scene_edit_enabled
        else SceneEditManifest(external_artifacts=[], remove_regions=[], support_surfaces=[])
    )
    task_hint_seeds = (
        _load_task_hint_seeds(builder_cfg.task_hints_path) if hybrid_scene_edit_enabled else []
    )

    scene_root = builder_cfg.output_scene_root.resolve()
    if scene_root.exists():
        shutil.rmtree(scene_root)
    (scene_root / "assets").mkdir(parents=True, exist_ok=True)
    (scene_root / "usd").mkdir(parents=True, exist_ok=True)
    if bool(builder_cfg.emit_polaris_metadata):
        (scene_root / "geniesim").mkdir(parents=True, exist_ok=True)
    if bool(builder_cfg.emit_isaac_lab):
        (scene_root / "isaac_lab").mkdir(parents=True, exist_ok=True)

    background = _write_static_background_asset(
        source_ply_path=builder_cfg.source_ply_path,
        asset_root=scene_root / "assets" / "static_scene",
        collision_mode=builder_cfg.static_collision_mode,
    )
    copied_external_artifacts = [
        _copy_external_artifact(scene_root=scene_root, spec=artifact)
        for artifact in scene_edit_manifest.external_artifacts
    ]
    copied_external_by_id = {
        artifact.artifact_id: artifact for artifact in copied_external_artifacts
    }
    copied_assets = [
        _copy_imported_asset(scene_root=scene_root, spec=asset_spec) for asset_spec in manifest.assets
    ]
    copied_assets_by_id = {asset.object_id: asset for asset in copied_assets}
    materialized_remove_regions = _materialize_remove_regions(
        remove_regions=scene_edit_manifest.remove_regions,
        copied_external_by_id=copied_external_by_id,
    )
    materialized_support_surfaces = _materialize_support_surfaces(
        scene_root=scene_root,
        support_surfaces=scene_edit_manifest.support_surfaces,
        copied_external_by_id=copied_external_by_id,
    )
    physics_qc_payload = _build_physics_qc_payload(
        copied_assets=copied_assets_by_id,
        remove_regions=materialized_remove_regions,
        support_surfaces=materialized_support_surfaces,
        task_hint_seeds=task_hint_seeds,
        task=manifest.task,
        enabled=hybrid_scene_edit_enabled,
    )

    replacement_manifest_path = scene_root / "assets" / "replacement_manifest.json"
    replacement_manifest_payload = _build_replacement_manifest_payload(
        scene_id=manifest.scene_id,
        remove_regions=materialized_remove_regions,
        copied_external_by_id=copied_external_by_id,
        task_hint_seeds=task_hint_seeds,
        enabled=hybrid_scene_edit_enabled,
    )
    write_json(replacement_manifest_payload, replacement_manifest_path)

    support_surfaces_path = scene_root / "assets" / "support_surfaces.json"
    support_surfaces_payload = _build_support_surfaces_payload(
        scene_id=manifest.scene_id,
        support_surfaces=materialized_support_surfaces,
        copied_external_by_id=copied_external_by_id,
        task_hint_seeds=task_hint_seeds,
        enabled=hybrid_scene_edit_enabled,
    )
    write_json(support_surfaces_payload, support_surfaces_path)

    physics_qc_path = scene_root / "assets" / "physics_qc.json"
    write_json(physics_qc_payload, physics_qc_path)

    scene_manifest_path = scene_root / "assets" / "scene_manifest.json"
    scene_manifest_payload = _build_scene_manifest_payload(
        manifest=manifest,
        background=background,
        copied_assets=copied_assets,
        copied_external_artifacts=copied_external_artifacts,
        source_ply_path=builder_cfg.source_ply_path,
        replacement_manifest_path=replacement_manifest_path,
        support_surfaces_path=support_surfaces_path,
        physics_qc_path=physics_qc_path,
        visual_usd_scene_path=scene_root / "usd" / "scene_visual.usda",
        physics_usd_scene_path=scene_root / "usd" / "scene_physics.usda",
        task_hint_path=builder_cfg.task_hints_path,
        remove_regions=materialized_remove_regions,
        support_surfaces=materialized_support_surfaces,
        physics_qc_payload=physics_qc_payload,
        hybrid_scene_edit_enabled=hybrid_scene_edit_enabled,
    )
    write_json(scene_manifest_payload, scene_manifest_path)

    visual_usd_scene_path = scene_root / "usd" / "scene_visual.usda"
    _write_scene_visual_usda(
        path=visual_usd_scene_path,
        background=background,
        copied_assets=copied_assets,
        remove_regions=materialized_remove_regions,
    )

    physics_usd_scene_path = scene_root / "usd" / "scene_physics.usda"
    _write_scene_physics_usda(
        path=physics_usd_scene_path,
        support_surfaces=materialized_support_surfaces,
    )

    usd_scene_path = scene_root / "usd" / "scene.usda"
    _write_scene_root_usda(
        path=usd_scene_path,
        visual_path=visual_usd_scene_path,
        physics_path=physics_usd_scene_path,
    )

    task_config_path = scene_root / "geniesim" / "task_config.json"
    if bool(builder_cfg.emit_polaris_metadata):
        write_json(_build_task_config_payload(manifest), task_config_path)
    else:
        task_config_path = Path("")

    if hybrid_scene_edit_enabled and bool(builder_cfg.fail_on_physics_qc) and int(
        physics_qc_payload.get("summary", {}).get("blocking_failures", 0)
    ) > 0:
        raise RuntimeError(
            "Scene physics QC reported blocking failures; inspect assets/physics_qc.json"
        )

    isaac_lab_root = scene_root / "isaac_lab"
    if bool(builder_cfg.emit_isaac_lab):
        _write_isaac_lab_package(
            root=isaac_lab_root,
            scene_id=manifest.scene_id,
            usd_scene_path=usd_scene_path,
            task=manifest.task,
            copied_assets=copied_assets,
            robot_type=builder_cfg.robot_type,
        )

    return SceneBuildResult(
        scene_root=scene_root,
        scene_manifest_path=scene_manifest_path,
        usd_scene_path=usd_scene_path,
        visual_usd_scene_path=visual_usd_scene_path,
        physics_usd_scene_path=physics_usd_scene_path,
        replacement_manifest_path=replacement_manifest_path,
        support_surfaces_path=support_surfaces_path,
        physics_qc_path=physics_qc_path,
        task_config_path=task_config_path,
        isaac_lab_package_root=isaac_lab_root,
    )


def _write_static_background_asset(
    *,
    source_ply_path: Path,
    asset_root: Path,
    collision_mode: str,
) -> _BackgroundAsset:
    asset_root.mkdir(parents=True, exist_ok=True)
    means = load_ply_means_numpy(source_ply_path)
    if means.size == 0:
        raise RuntimeError(f"Source PLY contains no points: {source_ply_path}")
    bbox_min = means.min(axis=0).astype(float).tolist()
    bbox_max = means.max(axis=0).astype(float).tolist()
    center = [0.5 * (lo + hi) for lo, hi in zip(bbox_min, bbox_max)]
    extents = [max(0.05, hi - lo) for lo, hi in zip(bbox_min, bbox_max)]
    copied_ply_path = asset_root / "source_scene.ply"
    shutil.copy2(source_ply_path, copied_ply_path)
    visual_usda_path = asset_root / "background_visual.usda"
    write_text_atomic(
        visual_usda_path,
        _render_background_visual_usda(source_ply_path=copied_ply_path.name),
    )
    return _BackgroundAsset(
        source_ply_path=source_ply_path.resolve(),
        copied_ply_path=copied_ply_path,
        visual_usda_path=visual_usda_path,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        center=center,
        extents=extents,
        collision_mode=collision_mode,
    )


def _copy_imported_asset(scene_root: Path, spec: ImportedAssetSpec) -> _CopiedAsset:
    obj_dir = scene_root / "assets" / f"obj_{sanitize_filename_component(spec.object_id)}"
    obj_dir.mkdir(parents=True, exist_ok=True)
    copied_asset_path = obj_dir / spec.asset_path.name
    shutil.copy2(spec.asset_path, copied_asset_path)
    metadata = {
        "object_id": spec.object_id,
        "label": spec.label,
        "asset_type": spec.asset_type,
        "task_role": spec.task_role,
        "source_asset_path": str(spec.asset_path),
        "copied_asset_path": str(copied_asset_path),
        "pose": {
            "position": spec.position,
            "rotation_quaternion": spec.rotation_quaternion,
        },
        "scale": spec.scale,
        "articulation_hints": spec.articulation_hints,
        "collision_hints": spec.collision_hints,
    }
    write_json(metadata, obj_dir / "metadata.json")
    return _CopiedAsset(
        object_id=spec.object_id,
        label=spec.label,
        asset_type=spec.asset_type,
        task_role=spec.task_role,
        copied_asset_path=copied_asset_path,
        relative_asset_path=str(copied_asset_path.relative_to(scene_root)).replace("\\", "/"),
        position=list(spec.position),
        rotation_quaternion=list(spec.rotation_quaternion),
        scale=list(spec.scale),
    )


def _copy_external_artifact(scene_root: Path, spec: ExternalArtifactSpec) -> _CopiedExternalArtifact:
    artifact_dir = scene_root / "assets" / "external" / sanitize_filename_component(spec.artifact_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    copied_path = artifact_dir / spec.path.name
    shutil.copy2(spec.path, copied_path)
    metadata = {
        "artifact_id": spec.artifact_id,
        "artifact_type": spec.artifact_type,
        "source_tool": spec.source_tool,
        "role": spec.role,
        "source_path": str(spec.path),
        "copied_path": str(copied_path),
        "metadata": spec.metadata,
    }
    write_json(metadata, artifact_dir / "metadata.json")
    relative_path = str(copied_path.relative_to(scene_root)).replace("\\", "/")
    return _CopiedExternalArtifact(
        artifact_id=spec.artifact_id,
        artifact_type=spec.artifact_type,
        source_tool=spec.source_tool,
        role=spec.role,
        copied_path=copied_path,
        relative_path=relative_path,
        metadata=spec.metadata,
    )


def _materialize_remove_regions(
    *,
    remove_regions: List[RemoveRegionSpec],
    copied_external_by_id: Mapping[str, _CopiedExternalArtifact],
) -> List[_MaterializedRemoveRegion]:
    materialized: List[_MaterializedRemoveRegion] = []
    for region in remove_regions:
        artifact_relative_path = ""
        if region.source_artifact_id:
            artifact_relative_path = copied_external_by_id[region.source_artifact_id].relative_path
        materialized.append(
            _MaterializedRemoveRegion(
                spec=region,
                artifact_relative_path=artifact_relative_path,
            )
        )
    return materialized


def _materialize_support_surfaces(
    *,
    scene_root: Path,
    support_surfaces: List[SupportSurfaceSpec],
    copied_external_by_id: Mapping[str, _CopiedExternalArtifact],
) -> List[_MaterializedSupportSurface]:
    materialized: List[_MaterializedSupportSurface] = []
    for surface in support_surfaces:
        artifact_relative_path = ""
        if surface.asset_artifact_id:
            artifact_relative_path = copied_external_by_id[surface.asset_artifact_id].relative_path
        elif surface.asset_path is not None:
            artifact_dir = (
                scene_root / "assets" / "support_surfaces" / sanitize_filename_component(surface.surface_id)
            )
            artifact_dir.mkdir(parents=True, exist_ok=True)
            copied_path = artifact_dir / surface.asset_path.name
            shutil.copy2(surface.asset_path, copied_path)
            artifact_relative_path = str(copied_path.relative_to(scene_root)).replace("\\", "/")
        materialized.append(
            _MaterializedSupportSurface(
                spec=surface,
                artifact_relative_path=artifact_relative_path,
            )
        )
    return materialized


def _load_task_hint_seeds(path: Path | None) -> List[_TaskHintSeed]:
    if path is None or not path.exists():
        return []
    seeds: List[_TaskHintSeed] = []
    for obb in load_obbs_from_task_targets(path):
        axes = np.asarray(obb.axes, dtype=np.float64)
        if axes.shape != (3, 3):
            axes = np.eye(3, dtype=np.float64)
        seeds.append(
            _TaskHintSeed(
                instance_id=str(obb.instance_id),
                label=str(obb.label),
                category=str(obb.category),
                confidence=float(obb.confidence),
                bounding_box=BoundingBoxSpec(
                    center=np.asarray(obb.center, dtype=float).tolist(),
                    extents=np.asarray(obb.extents, dtype=float).tolist(),
                    axes=axes.tolist(),
                ),
            )
        )
    return seeds


def _build_scene_manifest_payload(
    *,
    manifest: SceneAssetManifest,
    background: _BackgroundAsset,
    copied_assets: List[_CopiedAsset],
    copied_external_artifacts: List[_CopiedExternalArtifact],
    source_ply_path: Path,
    replacement_manifest_path: Path,
    support_surfaces_path: Path,
    physics_qc_path: Path,
    visual_usd_scene_path: Path,
    physics_usd_scene_path: Path,
    task_hint_path: Path | None,
    remove_regions: List[_MaterializedRemoveRegion],
    support_surfaces: List[_MaterializedSupportSurface],
    physics_qc_payload: Dict[str, Any],
    hybrid_scene_edit_enabled: bool,
) -> Dict[str, Any]:
    return {
        "version": "v2",
        "scene_id": manifest.scene_id,
        "scene_family": "blueprint_validation_hybrid_scene_builder",
        "environment_type": "manipulation_scene",
        "scene": {
            "source_ply_path": str(source_ply_path.resolve()),
            "task_hints_path": str(task_hint_path.resolve()) if task_hint_path is not None else "",
            "background_asset": {
                "ply_path": str(
                    background.copied_ply_path.relative_to(background.copied_ply_path.parents[2])
                ).replace("\\", "/"),
                "visual_usda_path": str(
                    background.visual_usda_path.relative_to(background.visual_usda_path.parents[2])
                ).replace("\\", "/"),
                "collision_mode": background.collision_mode,
                "bbox_min": background.bbox_min,
                "bbox_max": background.bbox_max,
                "physics_authority": "visual_only",
            },
            "layers": {
                "root_usda_path": "usd/scene.usda",
                "visual_usda_path": str(visual_usd_scene_path.relative_to(visual_usd_scene_path.parents[1])).replace("\\", "/"),
                "physics_usda_path": str(physics_usd_scene_path.relative_to(physics_usd_scene_path.parents[1])).replace("\\", "/"),
            },
            "artifacts": {
                "replacement_manifest_path": str(
                    replacement_manifest_path.relative_to(replacement_manifest_path.parents[1])
                ).replace("\\", "/"),
                "support_surfaces_path": str(
                    support_surfaces_path.relative_to(support_surfaces_path.parents[1])
                ).replace("\\", "/"),
                "physics_qc_path": str(
                    physics_qc_path.relative_to(physics_qc_path.parents[1])
                ).replace("\\", "/"),
            },
        },
        "objects": [
            {
                "id": asset.object_id,
                "name": asset.label,
                "category": asset.label,
                "asset": {
                    "format": asset.copied_asset_path.suffix.lstrip("."),
                    "path": asset.relative_asset_path,
                },
                "sim_role": "interactive" if asset.task_role != "goal_region" else "static",
                "task_role": asset.task_role,
                "transform": {
                    "position": {
                        "x": asset.position[0],
                        "y": asset.position[1],
                        "z": asset.position[2],
                    },
                    "rotation_quaternion": {
                        "w": asset.rotation_quaternion[0],
                        "x": asset.rotation_quaternion[1],
                        "y": asset.rotation_quaternion[2],
                        "z": asset.rotation_quaternion[3],
                    },
                    "scale": {
                        "x": asset.scale[0],
                        "y": asset.scale[1],
                        "z": asset.scale[2],
                    },
                },
            }
            for asset in copied_assets
        ],
        "scene_edit": {
            "enabled": hybrid_scene_edit_enabled,
            "remove_regions_total": len(remove_regions),
            "support_surfaces_total": len(support_surfaces),
            "external_artifacts": [
                {
                    "artifact_id": artifact.artifact_id,
                    "artifact_type": artifact.artifact_type,
                    "source_tool": artifact.source_tool,
                    "role": artifact.role,
                    "path": artifact.relative_path,
                }
                for artifact in copied_external_artifacts
            ],
        },
        "physics_qc_summary": dict(physics_qc_payload.get("summary", {})),
        "metadata": {
            "builder": "blueprint_validation.scene_builder",
            "task_id": manifest.task.task_id,
            "task_text": manifest.task.task_text,
        },
    }


def _build_replacement_manifest_payload(
    *,
    scene_id: str,
    remove_regions: List[_MaterializedRemoveRegion],
    copied_external_by_id: Mapping[str, _CopiedExternalArtifact],
    task_hint_seeds: List[_TaskHintSeed],
    enabled: bool,
) -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "scene_id": scene_id,
        "enabled": enabled,
        "status": "active" if enabled else "skipped_teleop_disabled",
        "remove_regions": [
            {
                "region_id": region.spec.region_id,
                "label": region.spec.label,
                "source": region.spec.source,
                "source_instance_ids": list(region.spec.source_instance_ids),
                "source_artifact_id": region.spec.source_artifact_id,
                "source_artifact_path": region.artifact_relative_path,
                "replacement_scope": region.spec.replacement_scope,
                "physics_authority": region.spec.physics_authority,
                "replacement_object_id": region.spec.replacement_object_id,
                "pose_alignment_confidence": region.spec.pose_alignment_confidence,
                "approval_state": region.spec.approval_state,
                "bounding_box": _bbox_to_payload(region.spec.bounding_box),
                "metadata": region.spec.metadata,
            }
            for region in remove_regions
        ],
        "proposal_seeds": [
            {
                "instance_id": seed.instance_id,
                "label": seed.label,
                "category": seed.category,
                "confidence": seed.confidence,
                "bounding_box": _bbox_to_payload(seed.bounding_box),
            }
            for seed in task_hint_seeds
        ],
        "external_artifacts": [
            {
                "artifact_id": artifact.artifact_id,
                "artifact_type": artifact.artifact_type,
                "source_tool": artifact.source_tool,
                "role": artifact.role,
                "path": artifact.relative_path,
            }
            for artifact in copied_external_by_id.values()
        ],
    }


def _build_support_surfaces_payload(
    *,
    scene_id: str,
    support_surfaces: List[_MaterializedSupportSurface],
    copied_external_by_id: Mapping[str, _CopiedExternalArtifact],
    task_hint_seeds: List[_TaskHintSeed],
    enabled: bool,
) -> Dict[str, Any]:
    support_seed_labels = {"counter", "countertop", "island", "shelf", "table", "tray", "desk"}
    proposal_seeds = [
        {
            "instance_id": seed.instance_id,
            "label": seed.label,
            "category": seed.category,
            "confidence": seed.confidence,
            "bounding_box": _bbox_to_payload(seed.bounding_box),
        }
        for seed in task_hint_seeds
        if any(token in seed.label.lower() for token in support_seed_labels)
    ]
    return {
        "schema_version": "v1",
        "scene_id": scene_id,
        "enabled": enabled,
        "status": "active" if enabled else "skipped_teleop_disabled",
        "support_surfaces": [
            {
                "surface_id": surface.spec.surface_id,
                "label": surface.spec.label,
                "source": surface.spec.source,
                "support_role": surface.spec.support_role,
                "surface_class": surface.spec.surface_class,
                "physics_authority": surface.spec.physics_authority,
                "proxy_shape": surface.spec.proxy_shape,
                "pose_alignment_confidence": surface.spec.pose_alignment_confidence,
                "approval_state": surface.spec.approval_state,
                "thickness": surface.spec.thickness,
                "bounding_box": _bbox_to_payload(surface.spec.bounding_box),
                "source_region_id": surface.spec.source_region_id,
                "source_artifact_id": surface.spec.source_artifact_id,
                "asset_artifact_id": surface.spec.asset_artifact_id,
                "asset_path": surface.artifact_relative_path,
                "metadata": surface.spec.metadata,
            }
            for surface in support_surfaces
        ],
        "proposal_seeds": proposal_seeds,
        "external_artifacts": [
            {
                "artifact_id": artifact.artifact_id,
                "artifact_type": artifact.artifact_type,
                "source_tool": artifact.source_tool,
                "role": artifact.role,
                "path": artifact.relative_path,
            }
            for artifact in copied_external_by_id.values()
        ],
    }


def _build_physics_qc_payload(
    *,
    copied_assets: Mapping[str, _CopiedAsset],
    remove_regions: List[_MaterializedRemoveRegion],
    support_surfaces: List[_MaterializedSupportSurface],
    task_hint_seeds: List[_TaskHintSeed],
    task,
    enabled: bool,
) -> Dict[str, Any]:
    if not enabled:
        return {
            "schema_version": "v1",
            "enabled": False,
            "checks": [],
            "summary": {
                "total_checks": 0,
                "blocking_failures": 0,
                "warnings": 0,
                "passed": True,
                "status": "skipped_teleop_disabled",
            },
        }
    checks: List[Dict[str, Any]] = []

    for region in remove_regions:
        if region.spec.replacement_object_id:
            asset = copied_assets.get(region.spec.replacement_object_id)
            if asset is None:
                checks.append(
                    _qc_result(
                        "replacement_object_exists",
                        region.spec.region_id,
                        passed=False,
                        severity="blocking",
                        detail=f"replacement_object_id '{region.spec.replacement_object_id}' is missing",
                    )
                )
            else:
                inside = _point_inside_bbox(np.asarray(asset.position, dtype=np.float64), region.spec.bounding_box)
                checks.append(
                    _qc_result(
                        "replacement_asset_center_inside_region",
                        region.spec.region_id,
                        passed=inside,
                        severity="blocking",
                        detail=(
                            f"replacement asset '{asset.object_id}' center falls inside remove region"
                            if inside
                            else f"replacement asset '{asset.object_id}' center falls outside remove region"
                        ),
                    )
                )
        overlapping_unsuppressed = [
            seed.instance_id
            for seed in task_hint_seeds
            if _point_inside_bbox(np.asarray(seed.bounding_box.center, dtype=np.float64), region.spec.bounding_box)
            and seed.instance_id not in set(region.spec.source_instance_ids)
        ]
        checks.append(
            _qc_result(
                "unsuppressed_seed_overlap",
                region.spec.region_id,
                passed=not overlapping_unsuppressed,
                severity="blocking" if overlapping_unsuppressed else "info",
                detail=(
                    "no overlapping unsuppressed task-hint seeds remain inside remove region"
                    if not overlapping_unsuppressed
                    else "overlapping unsuppressed task-hint seeds: "
                    + ", ".join(sorted(overlapping_unsuppressed))
                ),
            )
        )

    for surface in support_surfaces:
        dims_valid = all(float(value) > 0.0 for value in surface.spec.bounding_box.extents)
        checks.append(
            _qc_result(
                "support_surface_extents_valid",
                surface.spec.surface_id,
                passed=dims_valid,
                severity="blocking",
                detail="support surface extents are positive"
                if dims_valid
                else "support surface extents must be positive",
            )
        )
        thickness = float(surface.spec.thickness)
        thickness_valid = thickness > 0.0
        checks.append(
            _qc_result(
                "support_surface_thickness_valid",
                surface.spec.surface_id,
                passed=thickness_valid,
                severity="blocking",
                detail="support surface thickness is positive"
                if thickness_valid
                else "support surface thickness must be positive",
            )
        )
        probe_ok = dims_valid and thickness_valid and surface.spec.physics_authority != "visual_only"
        checks.append(
            _qc_result(
                "support_surface_rest_probe",
                surface.spec.surface_id,
                passed=probe_ok,
                severity="blocking",
                detail="support proxy can host a static resting probe"
                if probe_ok
                else "support proxy cannot host a resting probe with current metadata",
            )
        )
        if surface.spec.physics_authority in {"static_mesh", "sdf_mesh", "authored_region"}:
            has_asset = bool(surface.artifact_relative_path)
            checks.append(
                _qc_result(
                    "support_surface_asset_present",
                    surface.spec.surface_id,
                    passed=has_asset,
                    severity="blocking",
                    detail="support surface mesh asset is present"
                    if has_asset
                    else "support surface mesh-backed authority requires an asset",
                )
            )

    target_surface_exists = any(
        surface.spec.support_role in {"support_surface", "goal_surface"} for surface in support_surfaces
    )
    checks.append(
        _qc_result(
            "task_has_support_surface",
            task.task_id,
            passed=target_surface_exists,
            severity="warning" if not target_surface_exists else "info",
            detail="support surface metadata exists for the scene"
            if target_surface_exists
            else "no support surface metadata was provided for contact-critical tasks",
        )
    )

    summary = {
        "total_checks": len(checks),
        "blocking_failures": sum(
            1 for check in checks if check["severity"] == "blocking" and not check["passed"]
        ),
        "warnings": sum(
            1 for check in checks if check["severity"] == "warning" and not check["passed"]
        ),
    }
    summary["passed"] = summary["blocking_failures"] == 0
    return {"schema_version": "v1", "enabled": True, "checks": checks, "summary": summary}


def _qc_result(name: str, subject_id: str, *, passed: bool, severity: str, detail: str) -> Dict[str, Any]:
    return {
        "check": name,
        "subject_id": subject_id,
        "passed": bool(passed),
        "severity": severity,
        "detail": detail,
    }


def _point_inside_bbox(point: np.ndarray, bbox: BoundingBoxSpec) -> bool:
    center = np.asarray(bbox.center, dtype=np.float64)
    extents = np.asarray(bbox.extents, dtype=np.float64)
    axes = np.asarray(bbox.axes, dtype=np.float64)
    if axes.shape != (3, 3):
        axes = np.eye(3, dtype=np.float64)
    rel = point - center
    local = axes.T @ rel
    return bool(np.all(np.abs(local) <= (extents * 0.5 + 1e-6)))


def _bbox_to_payload(bbox: BoundingBoxSpec) -> Dict[str, Any]:
    return {
        "center": list(bbox.center),
        "extents": list(bbox.extents),
        "axes": [list(row) for row in bbox.axes],
    }


def _build_task_config_payload(manifest: SceneAssetManifest) -> Dict[str, Any]:
    return {
        "schema_version": "3.0",
        "scene_id": manifest.scene_id,
        "environment_type": "manipulation_scene",
        "suggested_tasks": [
            {
                "task_type": manifest.task.task_type,
                "target_object": manifest.task.target_object_id,
                "goal_region": manifest.task.goal_object_id,
                "difficulty": "medium",
                "priority": 1,
                "description_hint": manifest.task.task_text,
            }
        ],
        "robot_config": {
            "type": "franka",
            "base_position": [0.0, 0.0, 0.0],
            "workspace_bounds": [[-1.0, -1.0, 0.0], [1.0, 1.0, 1.5]],
        },
        "metadata": {
            "task_template": "pick_place_v1",
            "tasks_total_after_reachability_filter": 1,
        },
    }


def _write_scene_visual_usda(
    *,
    path: Path,
    background: _BackgroundAsset,
    copied_assets: List[_CopiedAsset],
    remove_regions: List[_MaterializedRemoveRegion],
) -> None:
    lines: List[str] = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "World"',
        "{",
        '    def Xform "Scene"',
        "    {",
        '        def Xform "BackgroundVisual"',
        "        {",
        f'            prepend references = @../assets/static_scene/{background.visual_usda_path.name}@',
        "        }",
    ]
    for asset in copied_assets:
        prim_name = sanitize_filename_component(asset.object_id, fallback="asset")
        lines.extend(
            [
                f'        def Xform "{prim_name}"',
                "        {",
                f'            prepend references = @../{asset.relative_asset_path}@',
                f"            double3 xformOp:translate = ({asset.position[0]}, {asset.position[1]}, {asset.position[2]})",
                "            uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:orient\", \"xformOp:scale\"]",
                f"            quatd xformOp:orient = ({asset.rotation_quaternion[0]}, {asset.rotation_quaternion[1]}, {asset.rotation_quaternion[2]}, {asset.rotation_quaternion[3]})",
                f"            float3 xformOp:scale = ({asset.scale[0]}, {asset.scale[1]}, {asset.scale[2]})",
                "        }",
            ]
        )
    if remove_regions:
        lines.extend(
            [
                '        def Scope "SuppressedRegions"',
                "        {",
                f"            int suppressed_region_count = {len(remove_regions)}",
                "        }",
            ]
        )
    lines.extend(
        [
            "    }",
            "}",
            "",
        ]
    )
    write_text_atomic(path, "\n".join(lines))


def _write_scene_physics_usda(
    *,
    path: Path,
    support_surfaces: List[_MaterializedSupportSurface],
) -> None:
    lines: List[str] = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        '    upAxis = "Z"',
        ")",
        "",
        'over Xform "World"',
        "{",
        '    over Xform "Scene"',
        "    {",
        '        def Xform "PhysicsLayer"',
        "        {",
    ]
    for surface in support_surfaces:
        prim_name = sanitize_filename_component(surface.spec.surface_id, fallback="surface")
        if surface.spec.physics_authority == "primitive_proxy" or not surface.artifact_relative_path:
            transform = _bbox_transform_matrix(
                bbox=surface.spec.bounding_box,
                thickness=surface.spec.thickness,
                flatten_plane=surface.spec.proxy_shape == "plane",
            )
            lines.extend(
                [
                    f'            def Cube "{prim_name}"',
                    "            {",
                    f"                matrix4d xformOp:transform = {transform}",
                    "                uniform token[] xformOpOrder = [\"xformOp:transform\"]",
                    "                double size = 2",
                    f'                token physics_authority = "{surface.spec.physics_authority}"',
                    f'                token support_role = "{surface.spec.support_role}"',
                    "            }",
                ]
            )
        else:
            lines.extend(
                [
                    f'            def Xform "{prim_name}"',
                    "            {",
                    f'                prepend references = @../{surface.artifact_relative_path}@',
                    f'                token physics_authority = "{surface.spec.physics_authority}"',
                    f'                token support_role = "{surface.spec.support_role}"',
                    "            }",
                ]
            )
    lines.extend(
        [
            "        }",
            "    }",
            "}",
            "",
        ]
    )
    write_text_atomic(path, "\n".join(lines))


def _write_scene_root_usda(*, path: Path, visual_path: Path, physics_path: Path) -> None:
    write_text_atomic(
        path,
        "\n".join(
            [
                "#usda 1.0",
                "(",
                '    defaultPrim = "World"',
                '    upAxis = "Z"',
                "    subLayers = [",
                f'        @{visual_path.name}@,',
                f'        @{physics_path.name}@',
                "    ]",
                ")",
                "",
            ]
        ),
    )


def _render_background_visual_usda(*, source_ply_path: str) -> str:
    return "\n".join(
        [
            "#usda 1.0",
            "(",
            '    defaultPrim = "CapturedScene"',
            '    upAxis = "Z"',
            ")",
            "",
            'def Xform "CapturedScene"',
            "{",
            f'    string source_ply = "{source_ply_path}"',
            '    token layer_role = "visual_background"',
            "}",
            "",
        ]
    )


def _bbox_transform_matrix(
    *,
    bbox: BoundingBoxSpec,
    thickness: float,
    flatten_plane: bool,
) -> str:
    axes = np.asarray(bbox.axes, dtype=np.float64)
    if axes.shape != (3, 3):
        axes = np.eye(3, dtype=np.float64)
    extents = np.asarray(bbox.extents, dtype=np.float64) * 0.5
    if flatten_plane:
        extents[2] = max(float(thickness) * 0.5, 0.01)
    else:
        extents[2] = max(extents[2], float(thickness) * 0.5, 0.01)
    center = np.asarray(bbox.center, dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = axes[:, 0] * extents[0]
    matrix[:3, 1] = axes[:, 1] * extents[1]
    matrix[:3, 2] = axes[:, 2] * extents[2]
    matrix[:3, 3] = center
    rows = []
    for row in matrix:
        rows.append(
            "(" + ", ".join(f"{float(value):.6f}" for value in row.tolist()) + ")"
        )
    return "(" + ", ".join(rows) + ")"


def _write_isaac_lab_package(
    *,
    root: Path,
    scene_id: str,
    usd_scene_path: Path,
    task,
    copied_assets: List[_CopiedAsset],
    robot_type: str,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    package_name = "scene_task"
    package_root = root / package_name
    package_root.mkdir(parents=True, exist_ok=True)
    object_paths = {
        asset.object_id: f"/World/Scene/{sanitize_filename_component(asset.object_id, fallback='asset')}"
        for asset in copied_assets
    }
    target_path = object_paths.get(task.target_object_id, f"/World/Scene/{task.target_object_id}")
    goal_path = object_paths.get(task.goal_object_id, f"/World/Scene/{task.goal_object_id}")

    write_text_atomic(
        root / "__init__.py",
        (
            "from .scene_task import (\n"
            "    TeleopEnvCfg,\n"
            "    PickPlaceTask,\n"
            "    create_env,\n"
            "    get_reset_events,\n"
            "    get_interval_events,\n"
            ")\n"
            "__all__ = ['TeleopEnvCfg', 'PickPlaceTask', 'create_env', 'get_reset_events', 'get_interval_events']\n"
        ),
    )
    write_text_atomic(
        package_root / "__init__.py",
        (
            "from .env_cfg import TeleopEnvCfg\n"
            "from .blueprint_env import create_env\n"
            "from .task_pick_place import PickPlaceTask\n"
            "from .randomizations import get_reset_events, get_interval_events\n"
            "__all__ = ['TeleopEnvCfg', 'PickPlaceTask', 'create_env', 'get_reset_events', 'get_interval_events']\n"
        ),
    )
    write_text_atomic(
        package_root / "env_cfg.py",
        _render_env_cfg_py(
            scene_id=scene_id,
            usd_scene_path=usd_scene_path,
            target_object_id=task.target_object_id,
            target_object_path=target_path,
            goal_object_id=task.goal_object_id,
            goal_object_path=goal_path,
            robot_type=robot_type,
        ),
    )
    write_text_atomic(
        package_root / "task_pick_place.py",
        _render_task_py(
            task_id=task.task_id,
            task_text=task.task_text,
            target_object_id=task.target_object_id,
        ),
    )
    write_text_atomic(
        package_root / "randomizations.py",
        "def get_reset_events():\n    return []\n\n\ndef get_interval_events():\n    return []\n",
    )
    write_text_atomic(
        package_root / "reward_functions.py",
        (
            "def reward_pick_place(*args, **kwargs):\n"
            "    del args, kwargs\n"
            "    return 0.0\n"
        ),
    )
    write_text_atomic(
        package_root / "train_cfg.yaml",
        "seed: 0\nnum_envs: 1\nmax_iterations: 1\n",
    )
    write_text_atomic(
        package_root / "blueprint_env.py",
        _render_blueprint_env_py(
            scene_id=scene_id,
            task_id=task.task_id,
            task_text=task.task_text,
        ),
    )
    write_json(
        _build_blueprint_runtime_contract(
            task_package=package_name,
            scene_id=scene_id,
        ),
        package_root / "blueprint_runtime.json",
    )


def _render_env_cfg_py(
    *,
    scene_id: str,
    usd_scene_path: Path,
    target_object_id: str,
    target_object_path: str,
    goal_object_id: str,
    goal_object_path: str,
    robot_type: str,
) -> str:
    usd_path = json.dumps(str(usd_scene_path.resolve()))
    target_id = json.dumps(target_object_id)
    target_path_json = json.dumps(target_object_path)
    goal_id = json.dumps(goal_object_id)
    goal_path_json = json.dumps(goal_object_path)
    return f'''"""Generated Isaac Lab task package for {scene_id}."""

from __future__ import annotations

import os
import types

SCENE_ID = {json.dumps(scene_id)}
USD_SCENE_PATH = {usd_path}
TARGET_OBJECT_ID = {target_id}
TARGET_OBJECT_PATH = {target_path_json}
GOAL_OBJECT_ID = {goal_id}
GOAL_OBJECT_PATH = {goal_path_json}
ROBOT_TYPE = {json.dumps(robot_type)}

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
    from isaaclab.envs import ManagerBasedEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass
except Exception:
    def configclass(cls):
        return cls

    class _Stub:
        def __init__(self, *args, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    sim_utils = types.SimpleNamespace(
        GroundPlaneCfg=lambda *args, **kwargs: None,
        DomeLightCfg=lambda *args, **kwargs: None,
        UsdFileCfg=lambda *args, **kwargs: None,
    )
    AssetBaseCfg = ArticulationCfg = RigidObjectCfg = ManagerBasedEnvCfg = InteractiveSceneCfg = _Stub
    RigidObjectCfg.InitialStateCfg = _Stub
    ArticulationCfg.InitialStateCfg = _Stub


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    scene = AssetBaseCfg(
        prim_path="/World/Scene",
        spawn=sim_utils.UsdFileCfg(usd_path=USD_SCENE_PATH, scale=(1.0, 1.0, 1.0)),
    )
    robot = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.environ.get("BLUEPRINT_SCENE_BUILDER_ROBOT_USD", "robot/franka/franka.usd"),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={{}},
        ),
    )
    target = RigidObjectCfg(
        prim_path=TARGET_OBJECT_PATH,
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    goal = RigidObjectCfg(
        prim_path=GOAL_OBJECT_PATH,
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0)),
    )


@configclass
class TeleopEnvCfg(ManagerBasedEnvCfg):
    def __init__(self):
        super().__init__()
        self.scene = getattr(self, "scene", SceneCfg())
'''


def _render_task_py(*, task_id: str, task_text: str, target_object_id: str) -> str:
    return f'''"""Generated pick/place task wrapper."""

from __future__ import annotations

TASK_ID = {json.dumps(task_id)}
TASK_TEXT = {json.dumps(task_text)}
TARGET_OBJECT_ID = {json.dumps(target_object_id)}


class PickPlaceTask:
    def __init__(self, env=None, cfg=None):
        self.env = env
        self.cfg = cfg

    def reset(self, env_ids=None):
        del env_ids
        return
'''


def _build_blueprint_runtime_contract(*, task_package: str, scene_id: str) -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "runtime_kind": "blueprint_scene_env",
        "scene_id": scene_id,
        "task_package": task_package,
        "env_factory": f"{task_package}.create_env",
        "env_cfg_class": "TeleopEnvCfg",
        "action_dim": 7,
        "camera_keys": ["wrist_rgb", "front_rgb"],
        "state_keys": [
            "policy",
            "joint_positions",
            "joint_velocities",
            "end_effector_pose",
            "gripper_state",
        ],
    }


def _render_blueprint_env_py(*, scene_id: str, task_id: str, task_text: str) -> str:
    return f'''"""Runnable fallback env for generated Blueprint scene packages."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

SCENE_ID = {json.dumps(scene_id)}
TASK_ID = {json.dumps(task_id)}
TASK_TEXT = {json.dumps(task_text)}
ACTION_DIM = 7
IMAGE_HEIGHT = 96
IMAGE_WIDTH = 128


class _ActionSpec:
    def __init__(self, action_dim: int):
        self.shape = (int(action_dim),)


class _ActionManager:
    def __init__(self, action_dim: int):
        self.action_spec = _ActionSpec(action_dim)


class _FakeEnv:
    def __init__(self):
        self.action_manager = _ActionManager(ACTION_DIM)
        self._step = 0

    def reset(self):
        self._step = 0
        return self._obs()

    def step(self, action: Any):
        del action
        self._step += 1
        obs = self._obs()
        reward = float(math.sin(self._step))
        done = self._step >= 4
        info = {{"scene_id": SCENE_ID, "task_id": TASK_ID}}
        return obs, reward, done, info

    def close(self):
        return None

    def _obs(self):
        base = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        base[:, :, 1] = 64
        return {{
            "wrist_rgb": base.copy(),
            "front_rgb": base.copy(),
            "policy": np.zeros((ACTION_DIM,), dtype=np.float32),
            "joint_positions": np.zeros((7,), dtype=np.float32),
            "joint_velocities": np.zeros((7,), dtype=np.float32),
            "end_effector_pose": np.zeros((7,), dtype=np.float32),
            "gripper_state": np.zeros((1,), dtype=np.float32),
        }}


def create_env(*args, **kwargs):
    del args, kwargs
    return _FakeEnv()
'''
