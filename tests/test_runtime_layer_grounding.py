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
        presentation_config={},
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
        presentation_config={"unsafe_allow_blocked_site_world": True},
        session_dir=tmp_path,
        step_index=0,
        camera_id="head_rgb",
    )

    assert bool(np.any(result["editable_mask"])) is True
    assert result["quality_flags"]["grounding_status"] == "ungrounded"
    assert result["quality_flags"]["unsafe_editable_override"] is True
    assert bool(np.any(result["frame"] != frame)) is True
