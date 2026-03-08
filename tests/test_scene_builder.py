"""Tests for the direct scene package builder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def _write_usda_asset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#usda 1.0\n"
        "(\n"
        '    defaultPrim = "Asset"\n'
        ")\n"
        'def Xform "Asset"\n'
        "{\n"
        '    def Cube "Geometry"\n'
        "    {\n"
        "        double size = 0.1\n"
        "    }\n"
        "}\n"
    )


def _write_asset_manifest(path: Path, asset_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "demo_scene",
                "task": {
                    "task_id": "pick_place_demo",
                    "task_text": "Pick up the mug and place it on the tray",
                    "task_type": "pick_place",
                    "target_object_id": "mug_001",
                    "goal_object_id": "tray_001",
                    "goal_region_label": "tray_surface",
                },
                "assets": [
                    {
                        "object_id": "mug_001",
                        "label": "mug",
                        "asset_type": "rigid",
                        "asset_path": str(asset_path),
                        "pose": {
                            "position": [0.1, 0.2, 0.3],
                            "rotation_quaternion": [1.0, 0.0, 0.0, 0.0],
                        },
                        "scale": [1.0, 1.0, 1.0],
                        "task_role": "target_object",
                    },
                    {
                        "object_id": "tray_001",
                        "label": "tray",
                        "asset_type": "rigid",
                        "asset_path": str(asset_path),
                        "pose": {
                            "position": [0.5, 0.2, 0.3],
                            "rotation_quaternion": [1.0, 0.0, 0.0, 0.0],
                        },
                        "scale": [1.0, 1.0, 1.0],
                        "task_role": "goal_region",
                    },
                ],
            }
        )
    )


def test_scene_asset_manifest_rejects_missing_fields(tmp_path: Path) -> None:
    from blueprint_validation.scene_builder import SceneAssetManifestError, load_scene_asset_manifest

    manifest_path = tmp_path / "bad_assets.json"
    manifest_path.write_text(json.dumps({"schema_version": "v1", "scene_id": "a", "assets": []}))
    with pytest.raises(SceneAssetManifestError, match="task block"):
        load_scene_asset_manifest(manifest_path)


def test_build_scene_package_emits_expected_contracts(sample_config, sample_ply, tmp_path: Path) -> None:
    from blueprint_validation.polaris.runtime import resolve_polaris_scene_spec
    from blueprint_validation.scene_builder import build_scene_package
    from blueprint_validation.teleop import load_and_validate_scene_package
    from blueprint_validation.teleop.runtime import _default_task_package_name

    asset_path = tmp_path / "asset.usda"
    _write_usda_asset(asset_path)
    manifest_path = tmp_path / "assets.json"
    _write_asset_manifest(manifest_path, asset_path)

    sample_config.scene_builder.enabled = True
    sample_config.scene_builder.source_ply_path = sample_ply
    sample_config.scene_builder.output_scene_root = tmp_path / "built_scene"
    sample_config.scene_builder.asset_manifest_path = manifest_path

    result = build_scene_package(sample_config)
    assert result.scene_manifest_path.exists()
    assert result.usd_scene_path.exists()
    assert (result.scene_root / "geniesim" / "task_config.json").exists()
    payload = load_and_validate_scene_package(result.scene_root)
    assert payload["has_isaac_lab"] is True
    assert payload["has_geniesim_task_config"] is True
    assert _default_task_package_name(result.scene_root) == "scene_task"

    sys.path.insert(0, str(result.scene_root / "isaac_lab"))
    try:
        pkg = __import__("scene_task")
        assert hasattr(pkg, "TeleopEnvCfg")
    finally:
        sys.path.pop(0)

    sample_config.facilities["test_facility"].scene_package_path = result.scene_root
    spec = resolve_polaris_scene_spec(sample_config, sample_config.facilities["test_facility"])
    assert spec.primary_eligible is True
    assert spec.task_metadata_path is not None


def test_build_scene_package_scene_manifest_matches_task_metadata(
    sample_config, sample_ply, tmp_path: Path
) -> None:
    from blueprint_validation.scene_builder import build_scene_package

    asset_path = tmp_path / "asset.usda"
    _write_usda_asset(asset_path)
    manifest_path = tmp_path / "assets.json"
    _write_asset_manifest(manifest_path, asset_path)

    sample_config.scene_builder.enabled = True
    sample_config.scene_builder.source_ply_path = sample_ply
    sample_config.scene_builder.output_scene_root = tmp_path / "scene_pkg"
    sample_config.scene_builder.asset_manifest_path = manifest_path

    result = build_scene_package(sample_config)
    scene_manifest = json.loads(result.scene_manifest_path.read_text())
    task_config = json.loads((result.scene_root / "geniesim" / "task_config.json").read_text())
    object_ids = {item["id"] for item in scene_manifest["objects"]}
    assert "mug_001" in object_ids
    assert "tray_001" in object_ids
    assert task_config["suggested_tasks"][0]["target_object"] == "mug_001"
    assert task_config["suggested_tasks"][0]["goal_region"] == "tray_001"
