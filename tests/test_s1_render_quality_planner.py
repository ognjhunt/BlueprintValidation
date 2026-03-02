"""Tests for Stage-1 quality planner and retry behavior."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def test_plan_best_camera_spec_medium_budget_evaluates_multiple_candidates():
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.rendering.camera_quality_planner import plan_best_camera_spec

    spec = CameraPathSpec(
        type="manipulation",
        approach_point=[0.0, 0.0, 0.5],
        arc_radius_m=0.45,
        height_override_m=0.75,
        look_down_override_deg=45.0,
    )
    selected, candidate_count, metrics = plan_best_camera_spec(
        base_spec=spec,
        scene_center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        num_frames=9,
        camera_height=0.8,
        look_down_deg=35.0,
        resolution=(64, 80),
        start_offset=np.zeros(3, dtype=np.float64),
        manipulation_target_z_bias_m=0.0,
        budget="medium",
        min_visible_frame_ratio=0.2,
        min_center_band_ratio=0.2,
        min_approach_angle_bins=2,
        angle_bin_deg=45.0,
        center_band_x=[0.2, 0.8],
        center_band_y=[0.2, 0.8],
    )
    assert selected.type == "manipulation"
    assert candidate_count >= 5
    assert "planner_best_score" in metrics


def test_rank_camera_spec_candidates_sorted_and_stable():
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.rendering.camera_quality_planner import (
        rank_camera_spec_candidates,
    )

    spec = CameraPathSpec(
        type="manipulation",
        approach_point=[0.0, 0.0, 0.5],
        arc_radius_m=0.45,
        height_override_m=0.75,
        look_down_override_deg=45.0,
    )
    ranked = rank_camera_spec_candidates(
        base_spec=spec,
        scene_center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        num_frames=9,
        camera_height=0.8,
        look_down_deg=35.0,
        resolution=(64, 80),
        start_offset=np.zeros(3, dtype=np.float64),
        manipulation_target_z_bias_m=0.0,
        budget="medium",
        min_visible_frame_ratio=0.2,
        min_center_band_ratio=0.2,
        min_approach_angle_bins=2,
        angle_bin_deg=45.0,
        center_band_x=[0.2, 0.8],
        center_band_y=[0.2, 0.8],
    )
    assert len(ranked) >= 5
    assert float(ranked[0].score) >= float(ranked[-1].score)
    # Deterministic tie-break: candidate index is non-decreasing for equal scores.
    for i in range(1, len(ranked)):
        if abs(float(ranked[i - 1].score) - float(ranked[i].score)) <= 1e-9:
            assert int(ranked[i - 1].candidate_index) < int(ranked[i].candidate_index)


def test_generate_and_render_retries_then_recovers(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.rendering.camera_paths import CameraPose
    from blueprint_validation.stages.s1_render import RenderStage

    sample_config.render.stage1_quality_planner_enabled = True
    sample_config.render.stage1_quality_autoretry_enabled = True
    sample_config.render.stage1_quality_max_regen_attempts = 2
    sample_config.render.stage1_quality_candidate_budget = "medium"
    sample_config.render.num_clips_per_path = 1

    def _pose(x: float) -> CameraPose:
        c2w = np.eye(4, dtype=np.float64)
        c2w[0, 3] = x
        return CameraPose(
            c2w=c2w,
            fx=50.0,
            fy=50.0,
            cx=32.0,
            cy=24.0,
            width=64,
            height=48,
        )

    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.plan_best_camera_spec",
        lambda **kwargs: (kwargs["base_spec"], 5, {"planner_best_score": 0.75}),
    )
    monkeypatch.setattr(
        RenderStage,
        "_build_render_poses",
        lambda self, **kwargs: ([_pose(0.0), _pose(0.1), _pose(0.2)], 3, 3, 3, 0),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.save_path_to_json",
        lambda poses, out_path: None,
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.render_video",
        lambda **kwargs: SimpleNamespace(
            video_path=tmp_path / "clip.mp4",
            depth_video_path=tmp_path / "clip_depth.mp4",
        ),
    )

    call_count = {"n": 0}

    def _fake_annotate(self, *, clip_entry, quality_retries_used, **kwargs):
        call_count["n"] += 1
        passed = call_count["n"] >= 2
        clip_entry["quality_gate_passed"] = passed
        clip_entry["quality_reject_reasons"] = [] if passed else ["blur_laplacian_low"]
        clip_entry["clip_quality_score"] = 0.8 if passed else 0.2
        clip_entry["quality_retries_used"] = quality_retries_used
        clip_entry["target_visibility_ratio"] = 0.7
        clip_entry["target_center_band_ratio"] = 0.7
        clip_entry["target_approach_angle_bins"] = 3
        clip_entry["blur_laplacian_score"] = 30.0
        return passed

    monkeypatch.setattr(RenderStage, "_annotate_entry_quality", _fake_annotate)

    stage = RenderStage()
    entries, _, _, quality_summary = stage._generate_and_render(
        config=sample_config,
        splat=object(),
        all_path_specs=[
            CameraPathSpec(
                type="manipulation",
                approach_point=[0.0, 0.0, 0.5],
                arc_radius_m=0.4,
                source_tag="task_scoped",
            )
        ],
        scene_center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        occupancy=None,
        render_dir=tmp_path,
        num_frames=3,
        camera_height=0.8,
        look_down_deg=35.0,
        resolution=(48, 64),
        fps=5,
        scene_T=None,
    )
    assert len(entries) == 1
    assert entries[0]["quality_retries_used"] == 1
    assert entries[0]["candidate_count_evaluated"] == 5
    assert entries[0]["quality_gate_passed"] is True
    assert quality_summary["num_quality_retries"] == 1
    assert quality_summary["num_quality_recovered"] == 1
    assert quality_summary["num_quality_failures"] == 0


def test_path_context_serializes_target_metadata():
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.stages.s1_render import _path_context_from_spec

    spec = CameraPathSpec(
        type="manipulation",
        approach_point=[0.0, 0.0, 0.5],
        target_instance_id="101",
        target_label="bowl",
        target_category="manipulation",
        target_role="targets",
    )
    ctx = _path_context_from_spec(spec)
    assert ctx["target_instance_id"] == "101"
    assert ctx["target_label"] == "bowl"
    assert ctx["target_category"] == "manipulation"
    assert ctx["target_role"] == "targets"
