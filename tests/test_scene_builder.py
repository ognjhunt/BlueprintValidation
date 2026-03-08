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


def _write_scene_edit_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "external_artifacts": [
                    {
                        "artifact_id": "book_mask",
                        "artifact_type": "segmentation_mask",
                        "source_tool": "saga",
                        "path": str(path.parent / "book_mask.json"),
                        "role": "remove_region_mask",
                    }
                ],
                "remove_regions": [
                    {
                        "region_id": "books_cluster",
                        "label": "books_cluster",
                        "source": "task_hints",
                        "source_instance_ids": ["book_a", "book_b"],
                        "source_artifact_id": "book_mask",
                        "replacement_scope": "object_only",
                        "physics_authority": "visual_only",
                        "replacement_object_id": "mug_001",
                        "pose_alignment_confidence": 0.9,
                        "bounding_box": {
                            "center": [0.1, 0.2, 0.3],
                            "extents": [0.4, 0.3, 0.3],
                            "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        },
                    }
                ],
                "support_surfaces": [
                    {
                        "surface_id": "counter_surface",
                        "label": "counter_surface",
                        "source": "task_hints",
                        "support_role": "support_surface",
                        "surface_class": "countertop",
                        "physics_authority": "primitive_proxy",
                        "proxy_shape": "box",
                        "pose_alignment_confidence": 0.95,
                        "thickness": 0.05,
                        "bounding_box": {
                            "center": [0.5, 0.2, 0.25],
                            "extents": [0.8, 0.4, 0.05],
                            "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        },
                    }
                ],
            }
        )
    )


def _write_task_hints(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "manipulation_candidates": [
                    {
                        "instance_id": "book_a",
                        "label": "book",
                        "confidence": 0.9,
                        "boundingBox": {
                            "center": [0.1, 0.2, 0.3],
                            "extents": [0.1, 0.05, 0.2],
                            "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        },
                    },
                    {
                        "instance_id": "book_b",
                        "label": "book",
                        "confidence": 0.9,
                        "boundingBox": {
                            "center": [0.18, 0.2, 0.3],
                            "extents": [0.1, 0.05, 0.2],
                            "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        },
                    },
                    {
                        "instance_id": "cup_ignored",
                        "label": "cup",
                        "confidence": 0.7,
                        "boundingBox": {
                            "center": [1.2, 0.2, 0.3],
                            "extents": [0.1, 0.1, 0.15],
                            "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        },
                    },
                ],
                "articulation_hints": [],
                "navigation_hints": [],
            }
        )
    )


def test_scene_asset_manifest_rejects_missing_fields(tmp_path: Path) -> None:
    from blueprint_validation.scene_builder import SceneAssetManifestError, load_scene_asset_manifest

    manifest_path = tmp_path / "bad_assets.json"
    manifest_path.write_text(json.dumps({"schema_version": "v1", "scene_id": "a", "assets": []}))
    with pytest.raises(SceneAssetManifestError, match="task block"):
        load_scene_asset_manifest(manifest_path)


def test_scene_edit_manifest_rejects_unknown_artifact(tmp_path: Path) -> None:
    from blueprint_validation.scene_builder import SceneAssetManifestError, load_scene_edit_manifest

    manifest_path = tmp_path / "scene_edit.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "remove_regions": [
                    {
                        "region_id": "books_cluster",
                        "label": "books",
                        "source": "task_hints",
                        "source_artifact_id": "missing_mask",
                        "replacement_scope": "object_only",
                        "physics_authority": "visual_only",
                        "bounding_box": {
                            "center": [0.0, 0.0, 0.0],
                            "extents": [0.2, 0.2, 0.2],
                            "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        },
                    }
                ],
            }
        )
    )
    with pytest.raises(SceneAssetManifestError, match="unknown source_artifact_id"):
        load_scene_edit_manifest(manifest_path)


def test_build_scene_package_emits_expected_contracts(sample_config, sample_ply, tmp_path: Path) -> None:
    from blueprint_validation.polaris.runtime import resolve_polaris_scene_spec
    from blueprint_validation.scene_builder import build_scene_package
    from blueprint_validation.teleop import load_and_validate_scene_package
    from blueprint_validation.teleop.runtime import _default_task_package_name, _resolve_env_action_dim, load_scene_env

    asset_path = tmp_path / "asset.usda"
    _write_usda_asset(asset_path)
    manifest_path = tmp_path / "assets.json"
    _write_asset_manifest(manifest_path, asset_path)
    scene_edit_path = tmp_path / "scene_edit.json"
    (tmp_path / "book_mask.json").write_text('{"mask": "demo"}')
    _write_scene_edit_manifest(scene_edit_path)
    task_hints_path = tmp_path / "task_hints.json"
    _write_task_hints(task_hints_path)

    sample_config.scene_builder.enabled = True
    sample_config.scene_builder.source_ply_path = sample_ply
    sample_config.scene_builder.output_scene_root = tmp_path / "built_scene"
    sample_config.scene_builder.asset_manifest_path = manifest_path
    sample_config.scene_builder.scene_edit_manifest_path = scene_edit_path
    sample_config.scene_builder.task_hints_path = task_hints_path

    result = build_scene_package(sample_config)
    assert result.scene_manifest_path.exists()
    assert result.usd_scene_path.exists()
    assert result.visual_usd_scene_path.exists()
    assert result.physics_usd_scene_path.exists()
    assert result.replacement_manifest_path.exists()
    assert result.support_surfaces_path.exists()
    assert result.physics_qc_path.exists()
    assert (result.scene_root / "geniesim" / "task_config.json").exists()
    payload = load_and_validate_scene_package(result.scene_root)
    assert payload["has_isaac_lab"] is True
    assert payload["has_geniesim_task_config"] is True
    assert payload["has_runnable_env"] is True
    assert _default_task_package_name(result.scene_root) == "scene_task"

    sys.path.insert(0, str(result.scene_root / "isaac_lab"))
    try:
        pkg = __import__("scene_task")
        assert hasattr(pkg, "TeleopEnvCfg")
    finally:
        sys.path.pop(0)

    loaded = load_scene_env(scene_root=result.scene_root, headless=True)
    try:
        obs = loaded.env.reset()
        assert "wrist_rgb" in obs
        assert _resolve_env_action_dim(loaded.env) == 7
    finally:
        loaded.close()

    replacement_manifest = json.loads(result.replacement_manifest_path.read_text())
    assert replacement_manifest["enabled"] is True
    assert replacement_manifest["remove_regions"][0]["replacement_object_id"] == "mug_001"

    support_surfaces = json.loads(result.support_surfaces_path.read_text())
    assert support_surfaces["enabled"] is True
    assert support_surfaces["support_surfaces"][0]["surface_id"] == "counter_surface"

    physics_qc = json.loads(result.physics_qc_path.read_text())
    assert physics_qc["enabled"] is True
    assert physics_qc["summary"]["blocking_failures"] == 0

    sample_config.facilities["test_facility"].scene_package_path = result.scene_root
    spec = resolve_polaris_scene_spec(sample_config, sample_config.facilities["test_facility"])
    assert spec.primary_eligible is True
    assert spec.runnable_scene_env is True
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
    assert scene_manifest["scene"]["layers"]["visual_usda_path"] == "usd/scene_visual.usda"
    assert scene_manifest["scene"]["layers"]["physics_usda_path"] == "usd/scene_physics.usda"


def test_build_scene_package_skips_hybrid_editing_when_isaac_disabled(
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
    sample_config.scene_builder.scene_edit_manifest_path = tmp_path / "does_not_need_to_exist.json"
    sample_config.scene_builder.task_hints_path = tmp_path / "does_not_need_to_exist_task_hints.json"
    sample_config.scene_builder.emit_isaac_lab = False

    result = build_scene_package(sample_config)
    replacement_manifest = json.loads(result.replacement_manifest_path.read_text())
    support_surfaces = json.loads(result.support_surfaces_path.read_text())
    physics_qc = json.loads(result.physics_qc_path.read_text())

    assert replacement_manifest["enabled"] is False
    assert replacement_manifest["status"] == "skipped_teleop_disabled"
    assert support_surfaces["enabled"] is False
    assert support_surfaces["status"] == "skipped_teleop_disabled"
    assert physics_qc["enabled"] is False
    assert physics_qc["summary"]["status"] == "skipped_teleop_disabled"
