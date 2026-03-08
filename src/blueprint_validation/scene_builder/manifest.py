"""Config-driven asset manifest loader for direct scene package building."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from ..common import read_json


class SceneAssetManifestError(ValueError):
    """Raised when a scene-builder asset manifest is invalid."""


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


def _parse_asset(raw: Any, source_path: Path) -> ImportedAssetSpec:
    if not isinstance(raw, dict):
        raise SceneAssetManifestError(f"Each asset entry must be an object ({source_path})")
    pose = raw.get("pose")
    if not isinstance(pose, dict):
        raise SceneAssetManifestError(f"Asset is missing pose block ({source_path})")
    asset_path = Path(_require_text(raw, "asset_path", where=f" ({source_path})")).expanduser()
    if not asset_path.is_absolute():
        asset_path = (source_path.parent / asset_path).resolve()
    if not asset_path.exists():
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
