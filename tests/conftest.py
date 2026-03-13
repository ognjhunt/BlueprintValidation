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
    spec = {
        "schema_version": "v1",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "canonical_package_version": "pkg-1",
        "robot_profiles": [
            {
                "id": "mobile_manipulator_rgb_v1",
                "observation_cameras": [
                    {"id": "head_rgb", "role": "head", "required": True},
                ],
                "action_space": {"dim": 7},
            }
        ],
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
