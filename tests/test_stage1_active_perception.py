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


# ---------------------------------------------------------------------------
# Fix B regression: target_missing guard based on best_task_score
# ---------------------------------------------------------------------------


def test_target_missing_orbit_stays_orbit_when_task_partial():
    """Orbit with task≥1 and target_missing must tighten zoom, NOT convert to manipulation.

    This is the primary regression guard for the canary finding where orbit→manipulation
    conversion caused task to drop from 1 → 0 when the orbit was already focused on target.
    """
    spec = CameraPathSpec(
        type="orbit",
        radius_m=1.0,
        approach_point=[0.5, 0.5, 0.85],
        target_label="bowl_101",
        look_down_override_deg=15.0,
    )
    updated = apply_issue_tag_corrections(
        spec=spec,
        issue_tags=["target_missing"],
        default_camera_height=1.0,
        default_look_down_deg=15.0,
        best_task_score=1.0,  # task≥1 → must stay orbit
    )
    assert updated.type == "orbit", (
        f"Expected orbit to stay orbit when task≥1, got {updated.type!r}"
    )
    # Radius should be tightened by 0.72× scale
    assert float(updated.radius_m) < float(spec.radius_m), (
        "Orbit radius should decrease (zoom tighten) on target_missing+task≥1"
    )
    assert abs(float(updated.radius_m) - 0.72) < 1e-6, (
        f"Expected radius 0.72 (1.0 × 0.72), got {updated.radius_m}"
    )
    # look_down should increase by 5°
    assert abs(float(updated.look_down_override_deg) - 20.0) < 1e-6, (
        f"Expected look_down 20.0 (15.0 + 5.0), got {updated.look_down_override_deg}"
    )


def test_target_missing_orbit_converts_to_manipulation_when_task_zero():
    """Orbit with task=0 and target_missing must still convert to manipulation (unchanged behavior)."""
    spec = CameraPathSpec(
        type="orbit",
        radius_m=1.0,
        approach_point=[0.5, 0.5, 0.85],
        target_label="bowl_101",
    )
    # Default best_task_score=0.0 → should convert
    updated = apply_issue_tag_corrections(
        spec=spec,
        issue_tags=["target_missing"],
        default_camera_height=1.0,
        default_look_down_deg=20.0,
        best_task_score=0.0,
    )
    assert updated.type == "manipulation", (
        f"Expected orbit to convert to manipulation when task=0, got {updated.type!r}"
    )


# ---------------------------------------------------------------------------
# Fix C regression: arc radius floor raised to 0.45 m in _convert_to_manipulation
# ---------------------------------------------------------------------------


def test_convert_to_manipulation_arc_radius_floor_small_orbit():
    """When a small-radius orbit (e.g. 0.3 m after scale-aware tuning) converts to
    manipulation, arc_radius_m must be ≥ 0.45 m (not the old 0.20 m floor).

    This guards against the camera_too_close regression introduced in patch2 where
    scale-aware tuning reduced orbit radius to ~0.3 m, collapsing arc_radius to 0.24 m.
    """
    spec = CameraPathSpec(
        type="orbit",
        radius_m=0.30,  # Small radius from scale-aware tuning
        approach_point=[0.5, 0.5, 0.85],
        target_label="small_object",
    )
    updated = apply_issue_tag_corrections(
        spec=spec,
        issue_tags=["target_missing"],
        default_camera_height=1.0,
        default_look_down_deg=20.0,
        best_task_score=0.0,  # task=0 → convert to manipulation
    )
    assert updated.type == "manipulation"
    assert float(updated.arc_radius_m) >= 0.45, (
        f"arc_radius_m must be ≥ 0.45 m (Fix C floor), got {updated.arc_radius_m:.3f}"
    )
