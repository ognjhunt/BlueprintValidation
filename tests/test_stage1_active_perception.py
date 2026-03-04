"""Unit tests for Stage-1 active-perception deterministic helpers."""

from __future__ import annotations

from blueprint_validation.config import CameraPathSpec
from blueprint_validation.rendering.stage1_active_perception import (
    apply_issue_tag_corrections,
    resolve_probe_budget,
    should_probe_clip,
)


def test_resolve_probe_budget_medium_defaults():
    budget = resolve_probe_budget(
        candidate_budget="medium",
        max_loops_cap=2,
        probe_frames_override=0,
        probe_resolution_scale_override=0.0,
    )
    assert budget.top_k == 2
    assert budget.probe_frames == 13
    assert budget.max_loops == 2
    assert budget.probe_resolution_scale == 0.67


def test_resolve_probe_budget_respects_overrides_and_cap():
    budget = resolve_probe_budget(
        candidate_budget="high",
        max_loops_cap=1,
        probe_frames_override=21,
        probe_resolution_scale_override=0.55,
    )
    assert budget.top_k == 3
    assert budget.probe_frames == 21
    assert budget.max_loops == 1
    assert budget.probe_resolution_scale == 0.55


def test_should_probe_clip_targeted_scope_uses_metadata():
    assert should_probe_clip(scope="targeted", path_type="orbit", path_context={}) is False
    assert (
        should_probe_clip(
            scope="targeted",
            path_type="orbit",
            path_context={"target_label": "bowl_101"},
        )
        is True
    )


def test_apply_issue_corrections_camera_too_far_and_off_center():
    spec = CameraPathSpec(
        type="orbit",
        radius_m=2.0,
        look_down_override_deg=20.0,
    )
    updated = apply_issue_tag_corrections(
        spec=spec,
        issue_tags=["camera_too_far", "target_off_center"],
        default_camera_height=1.0,
        default_look_down_deg=15.0,
    )
    assert updated.type == "orbit"
    # 2.0 * 0.88 * 0.94 = 1.6544
    assert abs(float(updated.radius_m) - 1.6544) < 1e-6
    assert abs(float(updated.look_down_override_deg) - 23.0) < 1e-6


def test_apply_issue_corrections_target_missing_converts_to_manipulation():
    spec = CameraPathSpec(
        type="sweep",
        length_m=5.0,
        approach_point=[0.1, -0.2, 0.3],
        target_label="bowl_101",
    )
    updated = apply_issue_tag_corrections(
        spec=spec,
        issue_tags=["target_missing"],
        default_camera_height=1.0,
        default_look_down_deg=20.0,
    )
    assert updated.type == "manipulation"
    assert updated.approach_point == [0.1, -0.2, 0.3]
    assert updated.target_label == "bowl_101"
    assert float(updated.arc_radius_m) > 0.0


def test_apply_issue_corrections_motion_reduction_changes_manipulation_arc():
    spec = CameraPathSpec(
        type="manipulation",
        approach_point=[0.0, 0.0, 0.4],
        arc_radius_m=0.5,
        arc_span_deg=180.0,
    )
    updated = apply_issue_tag_corrections(
        spec=spec,
        issue_tags=["camera_motion_too_fast"],
        default_camera_height=1.0,
        default_look_down_deg=20.0,
    )
    assert float(updated.arc_radius_m) < float(spec.arc_radius_m)
    assert float(updated.arc_span_deg) < float(spec.arc_span_deg)


def test_apply_issue_corrections_target_missing_uses_fallback_target_point():
    spec = CameraPathSpec(
        type="orbit",
        radius_m=2.0,
        approach_point=None,
        target_label="bowl_101",
    )
    updated = apply_issue_tag_corrections(
        spec=spec,
        issue_tags=["target_missing"],
        default_camera_height=1.0,
        default_look_down_deg=20.0,
        fallback_target_point=[0.2, -0.1, 0.6],
    )
    assert updated.type == "manipulation"
    assert updated.approach_point == [0.2, -0.1, 0.6]


def test_apply_issue_corrections_motion_reduction_escalates_by_loop_idx():
    spec = CameraPathSpec(
        type="manipulation",
        approach_point=[0.0, 0.0, 0.4],
        arc_radius_m=0.6,
        arc_span_deg=180.0,
    )
    l0 = apply_issue_tag_corrections(
        spec=spec,
        issue_tags=["camera_motion_too_fast"],
        default_camera_height=1.0,
        default_look_down_deg=20.0,
        loop_idx=0,
    )
    l2 = apply_issue_tag_corrections(
        spec=spec,
        issue_tags=["camera_motion_too_fast"],
        default_camera_height=1.0,
        default_look_down_deg=20.0,
        loop_idx=2,
    )
    assert float(l2.arc_radius_m) < float(l0.arc_radius_m)
    assert float(l2.arc_span_deg) < float(l0.arc_span_deg)
