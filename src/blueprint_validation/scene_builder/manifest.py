"""Config-driven manifest loaders for direct scene package building."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common import read_json


class SceneAssetManifestError(ValueError):
    """Raised when a scene-builder manifest is invalid."""


@dataclass(frozen=True)
class SceneTaskSpec:
    task_id: str
    task_text: str
    task_type: str
    target_object_id: str
    goal_object_id: str
    goal_region_label: str


@dataclass(frozen=True)
class ImportedAssetSpec:
    object_id: str
    label: str
    asset_type: str
    asset_path: Path
    position: List[float]
    rotation_quaternion: List[float]
    scale: List[float]
    task_role: str
    articulation_hints: Dict[str, Any] = field(default_factory=dict)
    collision_hints: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneAssetManifest:
    scene_id: str
    task: SceneTaskSpec
    assets: List[ImportedAssetSpec]


@dataclass(frozen=True)
class BoundingBoxSpec:
    center: List[float]
    extents: List[float]
    axes: List[List[float]]


@dataclass(frozen=True)
class ExternalArtifactSpec:
    artifact_id: str
    artifact_type: str
    source_tool: str
    path: Path
    role: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RemoveRegionSpec:
    region_id: str
    label: str
    source: str
    bounding_box: BoundingBoxSpec
    replacement_scope: str
    physics_authority: str
    source_instance_ids: List[str] = field(default_factory=list)
    source_artifact_id: str = ""
    replacement_object_id: str = ""
    pose_alignment_confidence: float = 1.0
    approval_state: str = "approved"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SupportSurfaceSpec:
    surface_id: str
    label: str
    source: str
    bounding_box: BoundingBoxSpec
    support_role: str
    surface_class: str
    physics_authority: str
    proxy_shape: str
    pose_alignment_confidence: float = 1.0
    thickness: float = 0.05
    source_region_id: str = ""
    source_artifact_id: str = ""
    asset_path: Optional[Path] = None
    asset_artifact_id: str = ""
    approval_state: str = "approved"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneEditManifest:
    external_artifacts: List[ExternalArtifactSpec]
    remove_regions: List[RemoveRegionSpec]
    support_surfaces: List[SupportSurfaceSpec]


def load_scene_asset_manifest(path: Path) -> SceneAssetManifest:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise SceneAssetManifestError(f"Asset manifest must be an object ({path})")
    schema_version = str(payload.get("schema_version", "") or "").strip()
    if schema_version != "v1":
        raise SceneAssetManifestError(
            f"Unsupported asset manifest schema_version '{schema_version}' ({path})"
        )
    scene_id = _require_text(payload, "scene_id", where=f" ({path})")
    task_raw = payload.get("task")
    if not isinstance(task_raw, dict):
        raise SceneAssetManifestError(f"Missing task block ({path})")
    task = SceneTaskSpec(
        task_id=_require_text(task_raw, "task_id", where=f" ({path})"),
        task_text=_require_text(task_raw, "task_text", where=f" ({path})"),
        task_type=_require_text(task_raw, "task_type", where=f" ({path})"),
        target_object_id=_require_text(task_raw, "target_object_id", where=f" ({path})"),
        goal_object_id=_require_text(task_raw, "goal_object_id", where=f" ({path})"),
        goal_region_label=_require_text(task_raw, "goal_region_label", where=f" ({path})"),
    )
    assets_raw = payload.get("assets")
    if not isinstance(assets_raw, list) or not assets_raw:
        raise SceneAssetManifestError(f"Asset manifest must contain non-empty 'assets' ({path})")
    assets = [_parse_asset(entry, path) for entry in assets_raw]
    object_ids = [asset.object_id for asset in assets]
    if len(object_ids) != len(set(object_ids)):
        raise SceneAssetManifestError(f"Duplicate object_id values in asset manifest ({path})")
    known_ids = set(object_ids)
    if task.target_object_id not in known_ids:
        raise SceneAssetManifestError(
            f"Task target_object_id '{task.target_object_id}' not found in assets ({path})"
        )
    if task.goal_object_id not in known_ids:
        raise SceneAssetManifestError(
            f"Task goal_object_id '{task.goal_object_id}' not found in assets ({path})"
        )
    return SceneAssetManifest(scene_id=scene_id, task=task, assets=assets)


def load_scene_edit_manifest(path: Optional[Path]) -> SceneEditManifest:
    if path is None:
        return SceneEditManifest(external_artifacts=[], remove_regions=[], support_surfaces=[])
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise SceneAssetManifestError(f"Scene edit manifest must be an object ({path})")
    schema_version = str(payload.get("schema_version", "") or "").strip()
    if schema_version != "v1":
        raise SceneAssetManifestError(
            f"Unsupported scene edit manifest schema_version '{schema_version}' ({path})"
        )
    external_raw = payload.get("external_artifacts", [])
    remove_raw = payload.get("remove_regions", [])
    support_raw = payload.get("support_surfaces", [])
    if not isinstance(external_raw, list):
        raise SceneAssetManifestError(f"'external_artifacts' must be a list ({path})")
    if not isinstance(remove_raw, list):
        raise SceneAssetManifestError(f"'remove_regions' must be a list ({path})")
    if not isinstance(support_raw, list):
        raise SceneAssetManifestError(f"'support_surfaces' must be a list ({path})")
    external_artifacts = [_parse_external_artifact(entry, path) for entry in external_raw]
    artifact_ids = [artifact.artifact_id for artifact in external_artifacts]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise SceneAssetManifestError(f"Duplicate external artifact ids in scene edit manifest ({path})")
    known_artifact_ids = set(artifact_ids)
    remove_regions = [
        _parse_remove_region(entry, path, known_artifact_ids=known_artifact_ids)
        for entry in remove_raw
    ]
    region_ids = [region.region_id for region in remove_regions]
    if len(region_ids) != len(set(region_ids)):
        raise SceneAssetManifestError(f"Duplicate region_id values in scene edit manifest ({path})")
    known_region_ids = set(region_ids)
    support_surfaces = [
        _parse_support_surface(
            entry,
            path,
            known_region_ids=known_region_ids,
            known_artifact_ids=known_artifact_ids,
        )
        for entry in support_raw
    ]
    surface_ids = [surface.surface_id for surface in support_surfaces]
    if len(surface_ids) != len(set(surface_ids)):
        raise SceneAssetManifestError(f"Duplicate surface_id values in scene edit manifest ({path})")
    return SceneEditManifest(
        external_artifacts=external_artifacts,
        remove_regions=remove_regions,
        support_surfaces=support_surfaces,
    )


def _parse_asset(raw: Any, source_path: Path) -> ImportedAssetSpec:
    if not isinstance(raw, dict):
        raise SceneAssetManifestError(f"Each asset entry must be an object ({source_path})")
    pose = raw.get("pose")
    if not isinstance(pose, dict):
        raise SceneAssetManifestError(f"Asset is missing pose block ({source_path})")
    asset_path = _resolve_optional_path(
        _require_text(raw, "asset_path", where=f" ({source_path})"),
        source_path,
    )
    if asset_path is None or not asset_path.exists():
        raise SceneAssetManifestError(f"Asset path does not exist: {asset_path}")
    ext = asset_path.suffix.lower()
    if ext not in {".usd", ".usda", ".usdc", ".usdz"}:
        raise SceneAssetManifestError(
            f"V1 scene builder requires a local USD/USDZ asset path, got: {asset_path}"
        )
    asset_type = _require_text(raw, "asset_type", where=f" ({source_path})")
    if asset_type not in {"rigid", "articulated"}:
        raise SceneAssetManifestError(
            f"asset_type must be 'rigid' or 'articulated' ({source_path})"
        )
    return ImportedAssetSpec(
        object_id=_require_text(raw, "object_id", where=f" ({source_path})"),
        label=_require_text(raw, "label", where=f" ({source_path})"),
        asset_type=asset_type,
        asset_path=asset_path,
        position=_require_float_list(pose, "position", expected_len=3, where=f" ({source_path})"),
        rotation_quaternion=_require_float_list(
            pose, "rotation_quaternion", expected_len=4, where=f" ({source_path})"
        ),
        scale=_require_float_list(raw, "scale", expected_len=3, where=f" ({source_path})"),
        task_role=_require_text(raw, "task_role", where=f" ({source_path})"),
        articulation_hints=dict(raw.get("articulation_hints") or {}),
        collision_hints=dict(raw.get("collision_hints") or {}),
    )


def _parse_external_artifact(raw: Any, source_path: Path) -> ExternalArtifactSpec:
    if not isinstance(raw, dict):
        raise SceneAssetManifestError(
            f"Each external_artifacts entry must be an object ({source_path})"
        )
    artifact_path = _resolve_optional_path(
        _require_text(raw, "path", where=f" ({source_path})"),
        source_path,
    )
    if artifact_path is None or not artifact_path.exists():
        raise SceneAssetManifestError(f"External artifact path does not exist: {artifact_path}")
    return ExternalArtifactSpec(
        artifact_id=_require_text(raw, "artifact_id", where=f" ({source_path})"),
        artifact_type=_require_text(raw, "artifact_type", where=f" ({source_path})"),
        source_tool=_require_text(raw, "source_tool", where=f" ({source_path})"),
        path=artifact_path,
        role=str(raw.get("role", "") or "").strip(),
        metadata=dict(raw.get("metadata") or {}),
    )


def _parse_remove_region(
    raw: Any,
    source_path: Path,
    *,
    known_artifact_ids: set[str],
) -> RemoveRegionSpec:
    if not isinstance(raw, dict):
        raise SceneAssetManifestError(
            f"Each remove_regions entry must be an object ({source_path})"
        )
    source_artifact_id = str(raw.get("source_artifact_id", "") or "").strip()
    if source_artifact_id and source_artifact_id not in known_artifact_ids:
        raise SceneAssetManifestError(
            f"remove_region references unknown source_artifact_id '{source_artifact_id}' ({source_path})"
        )
    replacement_scope = _require_text(raw, "replacement_scope", where=f" ({source_path})")
    if replacement_scope not in {"object_only", "support_region", "whole_module"}:
        raise SceneAssetManifestError(
            "replacement_scope must be one of: object_only, support_region, whole_module "
            f"({source_path})"
        )
    physics_authority = _require_text(raw, "physics_authority", where=f" ({source_path})")
    if physics_authority not in {
        "visual_only",
        "primitive_proxy",
        "static_mesh",
        "sdf_mesh",
        "authored_region",
    }:
        raise SceneAssetManifestError(
            "remove_region physics_authority must be one of: visual_only, primitive_proxy, "
            f"static_mesh, sdf_mesh, authored_region ({source_path})"
        )
    return RemoveRegionSpec(
        region_id=_require_text(raw, "region_id", where=f" ({source_path})"),
        label=_require_text(raw, "label", where=f" ({source_path})"),
        source=_require_text(raw, "source", where=f" ({source_path})"),
        bounding_box=_parse_bounding_box(raw.get("bounding_box"), source_path),
        replacement_scope=replacement_scope,
        physics_authority=physics_authority,
        source_instance_ids=_require_optional_text_list(raw.get("source_instance_ids")),
        source_artifact_id=source_artifact_id,
        replacement_object_id=str(raw.get("replacement_object_id", "") or "").strip(),
        pose_alignment_confidence=_require_float_range(
            raw,
            "pose_alignment_confidence",
            where=f" ({source_path})",
            default=1.0,
        ),
        approval_state=str(raw.get("approval_state", "approved") or "approved").strip(),
        metadata=dict(raw.get("metadata") or {}),
    )


def _parse_support_surface(
    raw: Any,
    source_path: Path,
    *,
    known_region_ids: set[str],
    known_artifact_ids: set[str],
) -> SupportSurfaceSpec:
    if not isinstance(raw, dict):
        raise SceneAssetManifestError(
            f"Each support_surfaces entry must be an object ({source_path})"
        )
    physics_authority = _require_text(raw, "physics_authority", where=f" ({source_path})")
    if physics_authority not in {"primitive_proxy", "static_mesh", "sdf_mesh", "authored_region"}:
        raise SceneAssetManifestError(
            "support_surface physics_authority must be one of: primitive_proxy, static_mesh, "
            f"sdf_mesh, authored_region ({source_path})"
        )
    proxy_shape = str(raw.get("proxy_shape", "box") or "box").strip()
    if proxy_shape not in {"box", "plane", "mesh"}:
        raise SceneAssetManifestError(
            f"support_surface proxy_shape must be one of: box, plane, mesh ({source_path})"
        )
    source_region_id = str(raw.get("source_region_id", "") or "").strip()
    if source_region_id and source_region_id not in known_region_ids:
        raise SceneAssetManifestError(
            f"support_surface references unknown source_region_id '{source_region_id}' ({source_path})"
        )
    source_artifact_id = str(raw.get("source_artifact_id", "") or "").strip()
    if source_artifact_id and source_artifact_id not in known_artifact_ids:
        raise SceneAssetManifestError(
            f"support_surface references unknown source_artifact_id '{source_artifact_id}' ({source_path})"
        )
    asset_artifact_id = str(raw.get("asset_artifact_id", "") or "").strip()
    if asset_artifact_id and asset_artifact_id not in known_artifact_ids:
        raise SceneAssetManifestError(
            f"support_surface references unknown asset_artifact_id '{asset_artifact_id}' ({source_path})"
        )
    asset_path_value = raw.get("asset_path")
    asset_path = _resolve_optional_path(asset_path_value, source_path)
    if asset_path is not None and not asset_path.exists():
        raise SceneAssetManifestError(f"support_surface asset_path does not exist: {asset_path}")
    return SupportSurfaceSpec(
        surface_id=_require_text(raw, "surface_id", where=f" ({source_path})"),
        label=_require_text(raw, "label", where=f" ({source_path})"),
        source=_require_text(raw, "source", where=f" ({source_path})"),
        bounding_box=_parse_bounding_box(raw.get("bounding_box"), source_path),
        support_role=_require_text(raw, "support_role", where=f" ({source_path})"),
        surface_class=_require_text(raw, "surface_class", where=f" ({source_path})"),
        physics_authority=physics_authority,
        proxy_shape=proxy_shape,
        pose_alignment_confidence=_require_float_range(
            raw,
            "pose_alignment_confidence",
            where=f" ({source_path})",
            default=1.0,
        ),
        thickness=float(raw.get("thickness", 0.05) or 0.05),
        source_region_id=source_region_id,
        source_artifact_id=source_artifact_id,
        asset_path=asset_path,
        asset_artifact_id=asset_artifact_id,
        approval_state=str(raw.get("approval_state", "approved") or "approved").strip(),
        metadata=dict(raw.get("metadata") or {}),
    )


def _parse_bounding_box(raw: Any, source_path: Path) -> BoundingBoxSpec:
    if not isinstance(raw, dict):
        raise SceneAssetManifestError(f"Missing bounding_box object ({source_path})")
    axes = raw.get("axes")
    if axes is None:
        axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    if not isinstance(axes, list) or len(axes) != 3:
        raise SceneAssetManifestError(f"bounding_box.axes must be a 3x3 list ({source_path})")
    parsed_axes = [_require_float_list({"axes": row}, "axes", expected_len=3, where=f" ({source_path})") for row in axes]
    return BoundingBoxSpec(
        center=_require_float_list(raw, "center", expected_len=3, where=f" ({source_path})"),
        extents=_require_float_list(raw, "extents", expected_len=3, where=f" ({source_path})"),
        axes=parsed_axes,
    )


def _require_text(payload: Dict[str, Any], key: str, *, where: str) -> str:
    value = str(payload.get(key, "") or "").strip()
    if not value:
        raise SceneAssetManifestError(f"Missing non-empty '{key}'{where}")
    return value


def _require_float_list(
    payload: Dict[str, Any],
    key: str,
    *,
    expected_len: int,
    where: str,
) -> List[float]:
    value = payload.get(key)
    if not isinstance(value, list) or len(value) != int(expected_len):
        raise SceneAssetManifestError(f"Expected '{key}' to be a list of {expected_len}{where}")
    try:
        return [float(item) for item in value]
    except Exception as exc:  # pragma: no cover - defensive
        raise SceneAssetManifestError(f"Invalid numeric values for '{key}'{where}") from exc


def _require_optional_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise SceneAssetManifestError("Expected a list of strings")
    return [str(item).strip() for item in value if str(item).strip()]


def _require_float_range(
    payload: Dict[str, Any],
    key: str,
    *,
    where: str,
    default: float,
) -> float:
    if key not in payload:
        return float(default)
    try:
        value = float(payload.get(key))
    except Exception as exc:  # pragma: no cover - defensive
        raise SceneAssetManifestError(f"Invalid numeric value for '{key}'{where}") from exc
    if not 0.0 <= value <= 1.0:
        raise SceneAssetManifestError(f"'{key}' must be in [0, 1]{where}")
    return value


def _resolve_optional_path(value: Any, source_path: Path) -> Optional[Path]:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (source_path.parent / path).resolve()
    return path
