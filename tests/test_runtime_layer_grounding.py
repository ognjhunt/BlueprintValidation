from __future__ import annotations

from pathlib import Path

import numpy as np

from blueprint_validation.runtime_layer_grounding import composite_runtime_layer


def test_ungrounded_manifest_defaults_to_canonical_only(tmp_path: Path) -> None:
    frame = np.full((12, 16, 3), 96, dtype=np.uint8)
    result = composite_runtime_layer(
        canonical_frame=frame,
        protected_regions_manifest={
            "schema_version": "v1",
            "grounding_status": "ungrounded",
            "ungrounded_reason": "empty_object_index",
            "regions": [],
        },
        canonical_render_policy={},
        presentation_config={},
        presentation_variance_policy={},
        session_dir=tmp_path,
        step_index=0,
        camera_id="head_rgb",
    )

    assert np.array_equal(result["frame"], frame)
    assert bool(np.any(result["locked_mask"])) is False
    assert bool(np.any(result["editable_mask"])) is False
    assert bool(np.all(result["uncertain_mask"])) is True
    assert result["quality_flags"]["presentation_quality"] == "ungrounded"
    assert result["quality_flags"]["fallback_mode"] == "ungrounded_canonical_only"


def test_ungrounded_manifest_becomes_editable_only_with_unsafe_override(tmp_path: Path) -> None:
    frame = np.full((12, 16, 3), 96, dtype=np.uint8)
    result = composite_runtime_layer(
        canonical_frame=frame,
        protected_regions_manifest={
            "schema_version": "v1",
            "grounding_status": "ungrounded",
            "ungrounded_reason": "empty_object_index",
            "regions": [],
        },
        canonical_render_policy={},
        presentation_config={"unsafe_allow_blocked_site_world": True},
        presentation_variance_policy={},
        session_dir=tmp_path,
        step_index=0,
        camera_id="head_rgb",
    )

    assert bool(np.any(result["editable_mask"])) is True
    assert result["quality_flags"]["grounding_status"] == "ungrounded"
    assert result["quality_flags"]["unsafe_editable_override"] is True
    assert bool(np.any(result["frame"] != frame)) is True


def test_unprojectable_locked_region_blocks_editability(tmp_path: Path) -> None:
    frame = np.full((12, 16, 3), 96, dtype=np.uint8)
    result = composite_runtime_layer(
        canonical_frame=frame,
        protected_regions_manifest={
            "schema_version": "v1",
            "grounding_status": "grounded",
            "regions": [
                {
                    "classification": "locked",
                    "task_critical": True,
                    "geometry_refs": {},
                }
            ],
        },
        canonical_render_policy={"fallback_behavior": {"retry_budget": 0, "on_locked_region_violation": "canonical_only"}},
        presentation_config={},
        presentation_variance_policy={},
        session_dir=tmp_path,
        step_index=0,
        camera_id="head_rgb",
    )

    assert bool(np.all(result["blocked_mask"])) is True
    assert bool(np.any(result["editable_mask"])) is False
    assert np.array_equal(result["frame"], frame)
    assert result["quality_flags"]["unprojectable_region_count"] == 1
    assert result["protected_region_violations"][0]["reason"] == "unprojectable_region_blocked"


def test_runtime_layer_honors_policy_thresholds_and_allowed_inputs(tmp_path: Path) -> None:
    frame = np.full((12, 16, 3), 96, dtype=np.uint8)
    result = composite_runtime_layer(
        canonical_frame=frame,
        protected_regions_manifest={
            "schema_version": "v1",
            "grounding_status": "grounded",
            "regions": [],
        },
        canonical_render_policy={"degraded_quality_threshold": 0.0},
        presentation_config={
            "prompt": "should_be_ignored",
            "presentation_model": "demo-model",
            "trajectory": {"trajectory": "arc"},
        },
        presentation_variance_policy={"allowed_variable_inputs": ["presentation_model"]},
        session_dir=tmp_path,
        step_index=0,
        camera_id="head_rgb",
    )

    assert result["quality_flags"]["presentation_quality"] == "degraded"
    assert bool(np.any(result["frame"] != frame)) is True
