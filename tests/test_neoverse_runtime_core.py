from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from blueprint_validation.neoverse_runtime_core import NeoVerseRuntimeStore
from blueprint_validation.runtime_layer_grounding import compute_canonical_package_version


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_runtime_ready_spec(tmp_path: Path) -> dict:
    raw_dir = tmp_path / "capture" / "raw" / "arkit"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "poses.jsonl").write_text("{}\n", encoding="utf-8")
    (raw_dir / "intrinsics.json").write_text("{}", encoding="utf-8")
    walkthrough = tmp_path / "capture" / "raw" / "walkthrough.mov"
    frame_path = tmp_path / "capture" / "raw" / "keyframe.png"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frame_path), np.full((48, 64, 3), 96, dtype=np.uint8))
    walkthrough.write_bytes(b"mov")

    scene_memory_manifest_path = tmp_path / "capture" / "pipeline" / "scene_memory" / "scene_memory_manifest.json"
    conditioning_bundle_path = tmp_path / "capture" / "pipeline" / "scene_memory" / "conditioning_bundle.json"
    object_geometry_manifest_path = tmp_path / "capture" / "pipeline" / "evaluation_prep" / "object_geometry_manifest.json"
    task_anchor_manifest_path = tmp_path / "capture" / "pipeline" / "evaluation_prep" / "task_anchor_manifest.json"
    protected_regions_manifest_path = tmp_path / "capture" / "pipeline" / "evaluation_prep" / "protected_regions_manifest.json"
    canonical_render_policy_path = tmp_path / "capture" / "pipeline" / "evaluation_prep" / "canonical_render_policy.json"
    presentation_variance_policy_path = tmp_path / "capture" / "pipeline" / "evaluation_prep" / "presentation_variance_policy.json"

    scene_memory_manifest = {
        "schema_version": "v1",
        "scene_id": "scene-a",
        "capture_id": "cap-a",
        "grounding_level": "observed",
        "confidence": 0.92,
        "canonical_truth": True,
        "presentation_only": False,
        "evidence_sources": ["walkthrough.mov"],
        "observation_coverage": {"scene_memory_status": "ready"},
    }
    conditioning_bundle = {
        "schema_version": "v1",
        "scene_id": "scene-a",
        "capture_id": "cap-a",
        "grounding_level": "observed",
        "confidence": 0.92,
        "canonical_truth": True,
        "presentation_only": False,
        "evidence_sources": ["walkthrough.mov"],
        "observation_coverage": {"capture_modality": "iphone"},
    }
    object_geometry_manifest = {
        "schema_version": "v1",
        "scene_id": "scene-a",
        "capture_id": "cap-a",
        "objects": [
            {
                "object_id": "obj-1",
                "label": "fridge",
                "task_critical": True,
                "grounding_level": "observed",
                "confidence": 0.91,
                "canonical_truth": True,
                "presentation_only": False,
                "evidence_sources": ["keyframe.png"],
                "observation_coverage": {"selected_view_count": 1},
                "placement_bbox": {"center": [0.0, 0.0, 0.0], "extents": [1.2, 0.9, 2.0]},
            }
        ],
    }
    task_anchor_manifest = {
        "schema_version": "v1",
        "scene_id": "scene-a",
        "capture_id": "cap-a",
        "tasks": [
            {
                "task_id": "task-1",
                "task_text": "Open the fridge",
                "task_category": "open_close",
                "target_object_ids": ["obj-1"],
                "articulation_required_ids": ["obj-1"],
                "task_critical": True,
                "grounding_level": "reconstructed",
                "confidence": 1.0,
                "canonical_truth": True,
                "presentation_only": False,
                "evidence_sources": ["task_scope_record.json"],
                "observation_coverage": {"target_object_count": 1},
            }
        ],
    }
    protected_regions_manifest = {
        "schema_version": "v1",
        "scene_id": "scene-a",
        "capture_id": "cap-a",
        "regions": [
            {
                "region_id": "object:obj-1",
                "region_type": "object",
                "object_id": "obj-1",
                "label": "fridge",
                "classification": "locked",
                "task_critical": True,
                "grounding_level": "observed",
                "confidence": 0.91,
                "canonical_truth": True,
                "presentation_only": False,
                "evidence_sources": ["keyframe.png"],
                "observation_coverage": {"selected_view_count": 1},
                "geometry_refs": {
                    "placement_bbox": {"center": [0.0, 0.0, 0.0], "extents": [1.2, 0.9, 2.0]},
                    "mesh_glb_path": None,
                    "collision_hulls": [],
                },
                "mask_refs": [],
                "coverage_refs": {"selected_views": []},
            }
        ],
    }
    canonical_render_policy = {
        "schema_version": "v1",
        "compositing_mode": "runtime_layer_grounded",
        "fallback_behavior": {"retry_budget": 1, "on_locked_region_violation": "canonical_only"},
    }
    presentation_variance_policy = {
        "schema_version": "v1",
        "allowed_variable_inputs": ["trajectory", "prompt", "presentation_model"],
        "allowed_editable_region_classes": ["generated"],
        "forbidden_changes": ["protected_object_placement"],
    }

    for path, payload in (
        (scene_memory_manifest_path, scene_memory_manifest),
        (conditioning_bundle_path, conditioning_bundle),
        (object_geometry_manifest_path, object_geometry_manifest),
        (task_anchor_manifest_path, task_anchor_manifest),
        (protected_regions_manifest_path, protected_regions_manifest),
        (canonical_render_policy_path, canonical_render_policy),
        (presentation_variance_policy_path, presentation_variance_policy),
    ):
        _write_json(path, payload)

    spec = {
        "schema_version": "v1",
        "scene_id": "scene-a",
        "capture_id": "cap-a",
        "site_submission_id": "site-sub-1",
        "qualification_state": "ready",
        "downstream_evaluation_eligibility": True,
        "capture_source": "iphone",
        "processing_profile": "pose_assisted",
        "canonical_package_uri": "gs://bucket/scenes/scene-a/captures/cap-a/pipeline/evaluation_prep/site_world_spec.json",
        "conditioning": {
            "scene_memory_manifest_path": str(scene_memory_manifest_path),
            "conditioning_bundle_path": str(conditioning_bundle_path),
            "sensor_availability": {
                "arkit_poses": True,
                "arkit_intrinsics": True,
            },
            "local_paths": {
                "raw_video_path": str(walkthrough),
                "keyframe_path": str(frame_path),
                "arkit_poses_path": str(raw_dir / "poses.jsonl"),
                "arkit_intrinsics_path": str(raw_dir / "intrinsics.json"),
                "object_index_path": str(tmp_path / "capture" / "raw" / "object_index.json"),
                "scene_memory_manifest_path": str(scene_memory_manifest_path),
                "conditioning_bundle_path": str(conditioning_bundle_path),
                "object_geometry_manifest_path": str(object_geometry_manifest_path),
            },
        },
        "geometry": {
            "object_geometry_manifest_path": str(object_geometry_manifest_path),
        },
        "task_anchor_manifest_path": str(task_anchor_manifest_path),
        "task_catalog": [{"id": "task-1", "task_id": "task-1", "task_text": "Open the fridge", "task_critical": True}],
        "scenario_catalog": [{"id": "scenario-default", "name": "default"}],
        "start_state_catalog": [{"id": "start-default", "name": "default_start_state"}],
        "robot_profiles": [
            {
                "id": "mobile_manipulator_rgb_v1",
                "observation_cameras": [
                    {"id": "head_rgb", "role": "head", "required": True},
                    {"id": "wrist_rgb", "role": "wrist", "required": False},
                ],
            }
        ],
        "runtime_layer_policy": {
            "protected_regions_manifest_uri": "gs://bucket/protected_regions_manifest.json",
            "canonical_render_policy_uri": "gs://bucket/canonical_render_policy.json",
            "presentation_variance_policy_uri": "gs://bucket/presentation_variance_policy.json",
            "protected_regions_manifest_path": str(protected_regions_manifest_path),
            "canonical_render_policy_path": str(canonical_render_policy_path),
            "presentation_variance_policy_path": str(presentation_variance_policy_path),
        },
    }
    spec["canonical_package_version"] = compute_canonical_package_version(
        scene_memory_manifest=scene_memory_manifest,
        conditioning_bundle=conditioning_bundle,
        object_geometry_manifest=object_geometry_manifest,
        task_anchor_manifest=task_anchor_manifest,
        site_world_spec=spec,
        protected_regions_manifest=protected_regions_manifest,
        canonical_render_policy=canonical_render_policy,
        presentation_variance_policy=presentation_variance_policy,
    )
    return spec


def test_runtime_store_builds_and_steps_site_world(tmp_path: Path) -> None:
    spec = _build_runtime_ready_spec(tmp_path)

    store = NeoVerseRuntimeStore(tmp_path / "runtime", base_url="http://runtime.local")
    registration = store.build_site_world(spec)
    assert registration["status"] == "ready"
    assert registration["canonical_package_version"] == spec["canonical_package_version"]
    assert registration["runtime_capabilities"]["protected_region_locking"] is True

    session = store.create_session(
        str(registration["site_world_id"]),
        session_id="session-1",
        robot_profile_id="mobile_manipulator_rgb_v1",
        task_id="task-1",
        scenario_id="scenario-default",
        start_state_id="start-default",
        canonical_package_uri=spec["canonical_package_uri"],
        canonical_package_version=spec["canonical_package_version"],
        prompt="force_locked_violation",
        trajectory={"kind": "arc"},
        presentation_model="demo-model",
        debug_mode=True,
    )
    assert session["session_id"] == "session-1"
    assert session["canonical_package_version"] == spec["canonical_package_version"]
    assert session["presentation_config"]["debug_mode"] is True

    reset = store.reset_session("session-1")
    assert reset["episode"]["stepIndex"] == 0
    assert reset["episode"]["observation"]["frame_path"].startswith("http://runtime.local")
    assert reset["episode"]["qualityFlags"]["presentation_quality"] in {"normal", "degraded"}
    assert reset["episode"]["protectedRegionViolations"]
    for path in reset["episode"]["debugArtifacts"].values():
        assert Path(path).is_file()

    step = store.step_session("session-1", action=[0.2, 0.0, 0.1, 0, 0, 0, 1])
    assert step["episode"]["stepIndex"] == 1
    assert step["episode"]["status"] == "running"
    assert step["episode"]["canonicalPackageVersion"] == spec["canonical_package_version"]

    state = store.session_state("session-1")
    assert state["canonical_package_version"] == spec["canonical_package_version"]
    assert "quality_flags" in state

    render = store.render_bytes("session-1", "head_rgb")
    assert render.startswith(b"\x89PNG")


def test_runtime_store_rejects_canonical_package_mismatch(tmp_path: Path) -> None:
    spec = _build_runtime_ready_spec(tmp_path)
    store = NeoVerseRuntimeStore(tmp_path / "runtime", base_url="http://runtime.local")
    registration = store.build_site_world(spec)

    with pytest.raises(RuntimeError, match="canonical_package_version_mismatch"):
        store.create_session(
            str(registration["site_world_id"]),
            session_id="session-2",
            robot_profile_id="mobile_manipulator_rgb_v1",
            task_id="task-1",
            scenario_id="scenario-default",
            start_state_id="start-default",
            canonical_package_version="wrong-version",
        )
