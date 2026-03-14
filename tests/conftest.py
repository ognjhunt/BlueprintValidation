from __future__ import annotations

import json
from pathlib import Path

import pytest
@pytest.fixture()
def sample_site_world_bundle(tmp_path: Path) -> dict[str, Path]:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    registration_path = bundle_dir / "site_world_registration.json"
    spec_path = bundle_dir / "site_world_spec.json"
    health_path = bundle_dir / "site_world_health.json"

    registration = {
        "schema_version": "v1",
        "site_world_id": "site-world-1",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "build_id": "build-1",
        "runtime_base_url": "http://runtime.local",
        "status": "ready",
    }
    conditioning_frame_path = bundle_dir / "conditioning_frame.png"
    conditioning_frame_path.write_bytes(b"placeholder")
    presentation_manifest_path = bundle_dir / "presentation_world_manifest.json"
    runtime_demo_manifest_path = bundle_dir / "runtime_demo_manifest.json"
    presentation_primary_asset_path = bundle_dir / "gaussian_splat.ply"
    presentation_primary_asset_path.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    spec = {
        "schema_version": "v1",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "canonical_package_version": "pkg-1",
        "qualification_state": "ready",
        "downstream_evaluation_eligibility": True,
        "robot_profiles": [
            {
                "id": "mobile_manipulator_rgb_v1",
                "observation_cameras": [
                    {"id": "head_rgb", "role": "head", "required": True},
                ],
                "action_space": {"dim": 7},
            }
        ],
        "conditioning": {
            "capture_orientation": {
                "encoded_width": 1920,
                "encoded_height": 1080,
                "declared_capture_width": 1080,
                "declared_capture_height": 1920,
                "display_rotation_degrees": 90,
                "display_orientation": "portrait",
                "normalization_applied": True,
                "source": "capture_context",
                "rotation_degrees": 90,
                "display_size": {"width": 1080, "height": 1920},
                "encoded_size": {"width": 1920, "height": 1080},
            },
            "sensor_availability": {
                "arkit_poses": True,
                "arkit_intrinsics": True,
            },
            "local_paths": {
                "arkit_poses_path": str(bundle_dir / "arkit_poses.json"),
                "arkit_intrinsics_path": str(bundle_dir / "arkit_intrinsics.json"),
                "keyframe_path": str(conditioning_frame_path),
                "occupancy_path": str(bundle_dir / "occupancy.json"),
                "object_index_path": str(bundle_dir / "object_index.json"),
            },
        },
        "presentation": {
            "presentation_world_manifest_path": str(presentation_manifest_path),
            "runtime_demo_manifest_path": str(runtime_demo_manifest_path),
            "bundle_type": "gsplat_scene_v1",
            "renderer_backend": "gsplat",
            "bundle_status": "ready",
            "primary_asset_path": str(presentation_primary_asset_path),
            "orientation": {
                "encoded_width": 1920,
                "encoded_height": 1080,
                "declared_capture_width": 1080,
                "declared_capture_height": 1920,
                "display_rotation_degrees": 90,
                "display_orientation": "portrait",
                "normalization_applied": True,
                "source": "capture_context",
                "rotation_degrees": 90,
                "display_size": {"width": 1080, "height": 1920},
                "encoded_size": {"width": 1920, "height": 1080},
            },
            "fallback_policy": "canonical_only",
        },
        "task_catalog": [{"id": "task-1", "task_text": "Pick item"}],
        "scenario_catalog": [{"id": "scenario-default", "name": "Default scenario"}],
        "start_state_catalog": [{"id": "start-default", "name": "Start default"}],
        "runtime_layer_policy": {
            "protected_regions_manifest_path": str(bundle_dir / "protected_regions_manifest.json"),
            "canonical_render_policy_path": str(bundle_dir / "canonical_render_policy.json"),
            "presentation_variance_policy_path": str(bundle_dir / "presentation_variance_policy.json"),
        },
        "local_paths": {
            "conditioning_bundle_path": str(bundle_dir / "conditioning_bundle.json"),
            "task_anchor_manifest_path": str(bundle_dir / "task_anchor_manifest.json"),
            "object_geometry_manifest_path": str(bundle_dir / "object_geometry_manifest.json"),
            "scene_memory_manifest_path": str(bundle_dir / "scene_memory_manifest.json"),
        },
    }
    health = {
        "schema_version": "v1",
        "site_world_id": "site-world-1",
        "healthy": True,
        "launchable": True,
        "status": "healthy",
        "blockers": [],
        "warnings": [],
    }

    aux_payloads = {
        "protected_regions_manifest.json": {"schema_version": "v1", "grounding_status": "grounded", "regions": []},
        "canonical_render_policy.json": {"schema_version": "v1", "fallback_behavior": {"retry_budget": 1}},
        "presentation_variance_policy.json": {"schema_version": "v1", "editable_overlay": {"enabled": True}},
        "conditioning_bundle.json": {"schema_version": "v1"},
        "task_anchor_manifest.json": {"schema_version": "v1"},
        "object_geometry_manifest.json": {"schema_version": "v1", "objects": []},
        "scene_memory_manifest.json": {"schema_version": "v1"},
        "presentation_world_manifest.json": {
            "schema_version": "v1",
            "bundle_type": "gsplat_scene_v1",
            "renderer_backend": "gsplat",
            "status": "ready",
            "primary_asset_path": str(presentation_primary_asset_path),
            "orientation": spec["presentation"]["orientation"],
            "fallback_policy": "canonical_only",
            "readiness": {"bundle_status": "ready"},
        },
        "runtime_demo_manifest.json": {
            "schema_version": "v1",
            "bundle_type": "gsplat_scene_v1",
            "renderer_backend": "gsplat",
            "bundle_status": "ready",
            "presentation_world_manifest_uri": "gs://bundle/presentation_world_manifest.json",
            "primary_asset_path": str(presentation_primary_asset_path),
            "orientation": spec["presentation"]["orientation"],
            "fallback_policy": "canonical_only",
        },
        "arkit_poses.json": {"schema_version": "v1", "poses": []},
        "arkit_intrinsics.json": {"schema_version": "v1", "intrinsics": []},
        "occupancy.json": {"schema_version": "v1", "voxels": []},
        "object_index.json": {"schema_version": "v1", "objects": []},
    }

    registration_path.write_text(json.dumps(registration), encoding="utf-8")
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    health_path.write_text(json.dumps(health), encoding="utf-8")
    for name, payload in aux_payloads.items():
        (bundle_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    return {
        "bundle_dir": bundle_dir,
        "registration_path": registration_path,
        "spec_path": spec_path,
        "health_path": health_path,
    }
