"""Tests for Stage-1 active-perception probe selection and fail-closed behavior."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _pose():
    from blueprint_validation.rendering.camera_paths import CameraPose

    c2w = np.eye(4, dtype=np.float64)
    return CameraPose(
        c2w=c2w,
        fx=50.0,
        fy=50.0,
        cx=32.0,
        cy=24.0,
        width=64,
        height=48,
    )


def _pose_at(x: float, y: float, z: float):
    from blueprint_validation.rendering.camera_paths import CameraPose

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return CameraPose(
        c2w=c2w,
        fx=50.0,
        fy=50.0,
        cx=32.0,
        cy=24.0,
        width=64,
        height=48,
    )


def test_active_probe_selects_candidate_that_passes_threshold(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.evaluation.vlm_judge import Stage1ProbeScore
    from blueprint_validation.stages.s1_render import RenderStage

    sample_config.render.stage1_active_perception_enabled = True
    sample_config.render.stage1_active_perception_scope = "all"
    sample_config.render.stage1_active_perception_max_loops = 1
    sample_config.render.stage1_active_perception_fail_closed = True
    sample_config.render.stage1_quality_candidate_budget = "medium"
    sample_config.render.stage1_probe_min_viable_pose_ratio = 0.1
    sample_config.render.stage1_probe_min_unique_positions = 1
    sample_config.render.stage1_probe_dedupe_enabled = False

    spec_a = CameraPathSpec(type="orbit", radius_m=2.0)
    spec_b = CameraPathSpec(type="orbit", radius_m=1.6)
    ranked = [
        SimpleNamespace(spec=spec_a, score=0.9, metrics={"planner_visible_ratio": 0.5}),
        SimpleNamespace(spec=spec_b, score=0.8, metrics={"planner_visible_ratio": 0.6}),
    ]

    monkeypatch.setattr(
        RenderStage,
        "_build_render_poses",
        lambda self, **kwargs: ([_pose(), _pose()], 21, 21, 21, 0),
    )

    def _fake_render_video(**kwargs):
        clip_name = kwargs["clip_name"]
        video = tmp_path / f"{clip_name}.mp4"
        depth = tmp_path / f"{clip_name}_depth.mp4"
        video.write_bytes(b"x")
        depth.write_bytes(b"x")
        return SimpleNamespace(video_path=video, depth_video_path=depth)

    monkeypatch.setattr("blueprint_validation.stages.s1_render.render_video", _fake_render_video)
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render._ensure_probe_h264_for_scoring",
        lambda video_path, min_frames: SimpleNamespace(
            path=video_path,
            codec_name="h264",
            decoded_frames=int(min_frames),
            width=128,
            height=96,
            content_monochrome_warning=False,
        ),
    )

    def _fake_probe_score(*, video_path, **kwargs):
        if "_c00" in str(video_path):
            return Stage1ProbeScore(
                task_score=5.0,
                visual_score=6.0,
                spatial_score=5.0,
                issue_tags=["target_off_center"],
                reasoning="off-center",
                raw_response="{}",
            )
        return Stage1ProbeScore(
            task_score=8.0,
            visual_score=8.0,
            spatial_score=7.0,
            issue_tags=[],
            reasoning="good framing",
            raw_response="{}",
        )

    monkeypatch.setattr("blueprint_validation.stages.s1_render.score_stage1_probe", _fake_probe_score)

    stage = RenderStage()
    selected, probe_meta = stage._run_active_perception_probe(
        config=sample_config,
        splat=object(),
        clip_name="clip_000_orbit",
        initial_spec=spec_a,
        ranked_candidates=ranked,
        scene_center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        occupancy=None,
        render_dir=tmp_path,
        camera_height=1.0,
        look_down_deg=20.0,
        resolution=(96, 128),
        start_offset=np.zeros(3, dtype=np.float64),
        fps=8,
        scene_T=None,
        facility_description="",
    )

    assert selected.radius_m == spec_b.radius_m
    assert probe_meta["vlm_probe_passed"] is True
    assert probe_meta["vlm_probe_attempts"] >= 2
    assert probe_meta["selected_probe_render_video_path"] is not None
    assert probe_meta["selected_probe_scoring_video_path"] is not None
    assert Path(str(probe_meta["selected_probe_render_video_path"])).exists()
    assert Path(str(probe_meta["selected_probe_scoring_video_path"])).exists()


def test_active_probe_fail_closed_on_scoring_error(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.stages.s1_render import RenderStage

    sample_config.render.stage1_active_perception_enabled = True
    sample_config.render.stage1_active_perception_scope = "all"
    sample_config.render.stage1_active_perception_max_loops = 1
    sample_config.render.stage1_active_perception_fail_closed = True
    sample_config.render.stage1_probe_min_viable_pose_ratio = 0.1
    sample_config.render.stage1_probe_min_unique_positions = 1

    spec = CameraPathSpec(type="orbit", radius_m=2.0)
    ranked = [SimpleNamespace(spec=spec, score=0.9, metrics={})]

    monkeypatch.setattr(
        RenderStage,
        "_build_render_poses",
        lambda self, **kwargs: ([_pose(), _pose()], 21, 21, 21, 0),
    )

    def _fake_render_video(**kwargs):
        clip_name = kwargs["clip_name"]
        video = tmp_path / f"{clip_name}.mp4"
        depth = tmp_path / f"{clip_name}_depth.mp4"
        video.write_bytes(b"x")
        depth.write_bytes(b"x")
        return SimpleNamespace(video_path=video, depth_video_path=depth)

    monkeypatch.setattr("blueprint_validation.stages.s1_render.render_video", _fake_render_video)
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render._ensure_probe_h264_for_scoring",
        lambda video_path, min_frames: SimpleNamespace(
            path=video_path,
            codec_name="h264",
            decoded_frames=int(min_frames),
            width=128,
            height=96,
            content_monochrome_warning=False,
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.score_stage1_probe",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("api unavailable")),
    )

    stage = RenderStage()
    selected, probe_meta = stage._run_active_perception_probe(
        config=sample_config,
        splat=object(),
        clip_name="clip_000_orbit",
        initial_spec=spec,
        ranked_candidates=ranked,
        scene_center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        occupancy=None,
        render_dir=tmp_path,
        camera_height=1.0,
        look_down_deg=20.0,
        resolution=(96, 128),
        start_offset=np.zeros(3, dtype=np.float64),
        fps=8,
        scene_T=None,
        facility_description="",
    )

    # Degenerate fallback target points no longer force type conversion.
    assert selected.type == "orbit"
    assert selected.radius_m != spec.radius_m
    assert probe_meta["vlm_probe_passed"] is False
    assert "probe_threshold_not_met" in str(probe_meta["vlm_probe_fail_reason"])
    assert int(probe_meta["num_vlm_probe_api_failures"]) >= 1


def test_active_probe_rejects_target_presence_before_vlm(sample_config, tmp_path, monkeypatch):
    import json

    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.stages.s1_render import RenderStage

    sample_config.render.stage1_active_perception_max_loops = 0
    sample_config.render.stage1_probe_min_viable_pose_ratio = 0.1
    sample_config.render.stage1_probe_min_unique_positions = 1
    sample_config.render.stage1_probe_dedupe_enabled = False

    spec = CameraPathSpec(
        type="orbit",
        radius_m=2.0,
        approach_point=[1.0, 0.0, 0.0],
        target_label="bowl_101",
        target_role="targets",
    )
    ranked = [SimpleNamespace(spec=spec, score=0.9, metrics={})]

    monkeypatch.setattr(
        RenderStage,
        "_build_render_poses",
        lambda self, **kwargs: ([_pose(), _pose()], 21, 21, 21, 0),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.render_video",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("render_video should not be called")),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.score_stage1_probe",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("VLM scoring should be skipped")),
    )

    scores_path = tmp_path / "probe_scores.jsonl"
    stage = RenderStage()
    _selected, probe_meta = stage._run_active_perception_probe(
        config=sample_config,
        splat=object(),
        clip_name="clip_000_orbit",
        initial_spec=spec,
        ranked_candidates=ranked,
        scene_center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        occupancy=None,
        render_dir=tmp_path,
        camera_height=1.0,
        look_down_deg=20.0,
        resolution=(96, 128),
        start_offset=np.zeros(3, dtype=np.float64),
        fps=8,
        scene_T=None,
        facility_description="",
        target_presence_enforced=True,
        probe_scores_path=scores_path,
    )

    assert probe_meta["vlm_probe_passed"] is False
    rows = [json.loads(line) for line in scores_path.read_text().splitlines() if line.strip()]
    assert any(row.get("status") == "target_presence_reject" for row in rows)


def test_active_probe_rejects_target_too_small_before_vlm(sample_config, tmp_path, monkeypatch):
    import json

    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.stages.s1_render import RenderStage

    sample_config.render.stage1_active_perception_max_loops = 0
    sample_config.render.stage1_probe_min_viable_pose_ratio = 0.1
    sample_config.render.stage1_probe_min_unique_positions = 1
    sample_config.render.stage1_probe_dedupe_enabled = False

    spec = CameraPathSpec(
        type="file",
        approach_point=[0.0, 0.0, -1.0],
        target_label="bowl_101",
        target_role="targets",
        target_extents_m=[0.05, 0.05, 0.05],
    )
    ranked = [SimpleNamespace(spec=spec, score=0.9, metrics={})]

    monkeypatch.setattr(
        RenderStage,
        "_build_render_poses",
        lambda self, **kwargs: ([_pose(), _pose()], 13, 13, 13, 0),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.render_video",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("render_video should not run when target is too small")
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.score_stage1_probe",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("VLM scoring should be skipped when target is too small")
        ),
    )

    scores_path = tmp_path / "probe_scores.jsonl"
    stage = RenderStage()
    _selected, probe_meta = stage._run_active_perception_probe(
        config=sample_config,
        splat=object(),
        clip_name="clip_000_locked",
        initial_spec=spec,
        ranked_candidates=ranked,
        scene_center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        occupancy=None,
        render_dir=tmp_path,
        camera_height=1.0,
        look_down_deg=20.0,
        resolution=(96, 128),
        start_offset=np.zeros(3, dtype=np.float64),
        fps=8,
        scene_T=None,
        facility_description="",
        target_presence_enforced=True,
        probe_scores_path=scores_path,
    )

    assert probe_meta["vlm_probe_passed"] is False
    rows = [json.loads(line) for line in scores_path.read_text().splitlines() if line.strip()]
    assert any(
        row.get("status") == "target_presence_reject" and row.get("reason") == "target_too_small"
        for row in rows
    )


def test_estimate_target_projected_size_ratio_positive_for_visible_target():
    from blueprint_validation.rendering.camera_paths import CameraPose, _look_at
    from blueprint_validation.stages.s1_render import _estimate_target_projected_size_ratio

    width = 128
    height = 96
    fx = width / (2.0 * np.tan(np.deg2rad(60.0 / 2.0)))
    pose = CameraPose(
        c2w=_look_at(
            np.asarray([1.0, 0.0, 0.2], dtype=np.float64),
            np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        ),
        fx=float(fx),
        fy=float(fx),
        cx=float(width) / 2.0,
        cy=float(height) / 2.0,
        width=width,
        height=height,
    )
    ratio = _estimate_target_projected_size_ratio(
        poses=[pose],
        target_xyz=[0.0, 0.0, 0.0],
        target_extents_m=[0.35, 0.25, 0.20],
    )
    assert ratio > 0.05


def test_locked_mode_allows_near_static_probe_with_high_unique_threshold(
    sample_config, tmp_path, monkeypatch
):
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.evaluation.vlm_judge import Stage1ProbeScore
    from blueprint_validation.stages.s1_render import RenderStage

    sample_config.render.stage1_active_perception_max_loops = 0
    sample_config.render.stage1_probe_min_viable_pose_ratio = 0.1
    sample_config.render.stage1_probe_min_unique_positions = 8
    sample_config.render.stage1_probe_dedupe_enabled = False

    spec = CameraPathSpec(
        type="file",
        source_tag="kitchen_0787_locked",
        approach_point=[0.0, 0.0, -1.0],
        target_label="bowl_101",
        target_role="targets",
        target_extents_m=[0.3, 0.2, 0.2],
    )

    monkeypatch.setattr(
        RenderStage,
        "_build_render_poses",
        lambda self, **kwargs: ([_pose(), _pose(), _pose()], 3, 3, 3, 0),
    )

    def _fake_render_video(**kwargs):
        clip_name = kwargs["clip_name"]
        video = tmp_path / f"{clip_name}.mp4"
        depth = tmp_path / f"{clip_name}_depth.mp4"
        video.write_bytes(b"x")
        depth.write_bytes(b"x")
        return SimpleNamespace(video_path=video, depth_video_path=depth)

    monkeypatch.setattr("blueprint_validation.stages.s1_render.render_video", _fake_render_video)
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render._ensure_probe_h264_for_scoring",
        lambda video_path, min_frames: SimpleNamespace(
            path=video_path,
            codec_name="h264",
            decoded_frames=int(min_frames),
            width=128,
            height=96,
            content_monochrome_warning=False,
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.score_stage1_probe",
        lambda **kwargs: Stage1ProbeScore(
            task_score=8.0,
            visual_score=8.0,
            spatial_score=8.0,
            issue_tags=[],
            reasoning="good",
            raw_response="{}",
        ),
    )

    selected, probe_meta = RenderStage()._run_active_perception_probe(
        config=sample_config,
        splat=object(),
        clip_name="clip_000_file",
        initial_spec=spec,
        ranked_candidates=[],
        scene_center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        occupancy=None,
        render_dir=tmp_path,
        camera_height=1.0,
        look_down_deg=20.0,
        resolution=(96, 128),
        start_offset=np.zeros(3, dtype=np.float64),
        fps=8,
        scene_T=None,
        facility_description="",
        target_presence_enforced=True,
        locked_mode=True,
    )

    assert selected.type == "file"
    assert probe_meta["vlm_probe_passed"] is True


def test_locked_mode_bypasses_los_hard_reject(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.evaluation.vlm_judge import Stage1ProbeScore
    from blueprint_validation.stages.s1_render import RenderStage

    class _Occ:
        def is_free(self, point, min_clearance_m=0.0):
            return True

        def has_line_of_sight(self, start, end, **kwargs):
            return False

    sample_config.render.stage1_active_perception_max_loops = 0
    sample_config.render.stage1_probe_min_viable_pose_ratio = 0.1
    sample_config.render.stage1_probe_min_unique_positions = 1
    sample_config.render.stage1_probe_dedupe_enabled = False

    spec = CameraPathSpec(
        type="file",
        source_tag="kitchen_0787_locked",
        approach_point=[0.0, 0.0, -1.0],
        target_label="bowl_101",
        target_role="targets",
        target_extents_m=[0.3, 0.2, 0.2],
    )

    monkeypatch.setattr(
        RenderStage,
        "_build_render_poses",
        lambda self, **kwargs: ([_pose(), _pose(), _pose()], 3, 3, 3, 0),
    )

    def _fake_render_video(**kwargs):
        clip_name = kwargs["clip_name"]
        video = tmp_path / f"{clip_name}.mp4"
        depth = tmp_path / f"{clip_name}_depth.mp4"
        video.write_bytes(b"x")
        depth.write_bytes(b"x")
        return SimpleNamespace(video_path=video, depth_video_path=depth)

    monkeypatch.setattr("blueprint_validation.stages.s1_render.render_video", _fake_render_video)
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render._ensure_probe_h264_for_scoring",
        lambda video_path, min_frames: SimpleNamespace(
            path=video_path,
            codec_name="h264",
            decoded_frames=int(min_frames),
            width=128,
            height=96,
            content_monochrome_warning=False,
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.score_stage1_probe",
        lambda **kwargs: Stage1ProbeScore(
            task_score=7.0,
            visual_score=7.0,
            spatial_score=7.0,
            issue_tags=[],
            reasoning="good",
            raw_response="{}",
        ),
    )

    selected, probe_meta = RenderStage()._run_active_perception_probe(
        config=sample_config,
        splat=object(),
        clip_name="clip_000_file",
        initial_spec=spec,
        ranked_candidates=[],
        scene_center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        occupancy=_Occ(),
        render_dir=tmp_path,
        camera_height=1.0,
        look_down_deg=20.0,
        resolution=(96, 128),
        start_offset=np.zeros(3, dtype=np.float64),
        fps=8,
        scene_T=None,
        facility_description="",
        target_presence_enforced=True,
        locked_mode=True,
    )

    assert selected.type == "file"
    assert probe_meta["vlm_probe_passed"] is True


def test_target_los_uses_camera_facing_surface_endpoint():
    from blueprint_validation.stages.s1_render import _compute_target_line_of_sight_ratio

    class _FakeOccupancy:
        def __init__(self):
            self.endpoints = []

        def is_free(self, point, min_clearance_m=0.0):
            # Target center is "inside" occupied geometry; +X is free.
            return float(np.asarray(point, dtype=np.float64)[0]) >= 0.30

        def has_line_of_sight(self, start, end, **kwargs):
            e = np.asarray(end, dtype=np.float64)
            self.endpoints.append(e.copy())
            return float(e[0]) >= 0.30

    occ = _FakeOccupancy()
    ratio = _compute_target_line_of_sight_ratio(
        poses=[_pose_at(1.0, 0.0, 0.0)],
        target_xyz=[0.0, 0.0, 0.0],
        occupancy=occ,  # type: ignore[arg-type]
        min_clearance_m=0.12,
    )
    assert ratio == 1.0
    assert occ.endpoints, "Expected LOS to evaluate at least one endpoint"
    assert float(occ.endpoints[0][0]) >= 0.30


def test_active_probe_flags_monochrome_probe_media_invalid(sample_config, tmp_path, monkeypatch):
    import json

    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.stages.s1_render import RenderStage

    sample_config.render.stage1_active_perception_max_loops = 0
    sample_config.render.stage1_probe_min_viable_pose_ratio = 0.1
    sample_config.render.stage1_probe_min_unique_positions = 1
    sample_config.render.stage1_probe_dedupe_enabled = False

    spec = CameraPathSpec(
        type="orbit",
        radius_m=2.0,
        approach_point=[0.0, 0.0, -1.0],
        target_label="bowl_101",
        target_role="targets",
    )
    ranked = [SimpleNamespace(spec=spec, score=0.9, metrics={})]

    monkeypatch.setattr(
        RenderStage,
        "_build_render_poses",
        lambda self, **kwargs: ([_pose(), _pose()], 21, 21, 21, 0),
    )

    def _fake_render_video(**kwargs):
        clip_name = kwargs["clip_name"]
        video = tmp_path / f"{clip_name}.mp4"
        depth = tmp_path / f"{clip_name}_depth.mp4"
        video.write_bytes(b"x")
        depth.write_bytes(b"x")
        return SimpleNamespace(video_path=video, depth_video_path=depth)

    monkeypatch.setattr("blueprint_validation.stages.s1_render.render_video", _fake_render_video)
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render._ensure_probe_h264_for_scoring",
        lambda video_path, min_frames: SimpleNamespace(
            path=video_path,
            codec_name="h264",
            decoded_frames=int(min_frames),
            width=128,
            height=96,
            content_monochrome_warning=True,
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.score_stage1_probe",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("VLM scoring should be skipped on monochrome probe")
        ),
    )

    scores_path = tmp_path / "probe_scores.jsonl"
    stage = RenderStage()
    _selected, probe_meta = stage._run_active_perception_probe(
        config=sample_config,
        splat=object(),
        clip_name="clip_000_orbit",
        initial_spec=spec,
        ranked_candidates=ranked,
        scene_center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        occupancy=None,
        render_dir=tmp_path,
        camera_height=1.0,
        look_down_deg=20.0,
        resolution=(96, 128),
        start_offset=np.zeros(3, dtype=np.float64),
        fps=8,
        scene_T=None,
        facility_description="",
        target_presence_enforced=True,
        probe_scores_path=scores_path,
    )

    assert probe_meta["vlm_probe_passed"] is False
    rows = [json.loads(line) for line in scores_path.read_text().splitlines() if line.strip()]
    assert any(row.get("status") == "probe_media_invalid" for row in rows)


def test_active_probe_rejects_target_missing_vlm_tags_for_target_grounded(
    sample_config, tmp_path, monkeypatch
):
    import json

    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.evaluation.vlm_judge import Stage1ProbeScore
    from blueprint_validation.stages.s1_render import RenderStage

    sample_config.render.stage1_active_perception_max_loops = 0
    sample_config.render.stage1_probe_min_viable_pose_ratio = 0.1
    sample_config.render.stage1_probe_min_unique_positions = 1
    sample_config.render.stage1_probe_dedupe_enabled = False

    spec = CameraPathSpec(
        type="orbit",
        radius_m=1.2,
        approach_point=[0.0, 0.0, -1.0],
        target_label="bowl_101",
        target_role="targets",
    )
    ranked = [SimpleNamespace(spec=spec, score=0.9, metrics={})]

    monkeypatch.setattr(
        RenderStage,
        "_build_render_poses",
        lambda self, **kwargs: ([_pose(), _pose()], 21, 21, 21, 0),
    )

    def _fake_render_video(**kwargs):
        clip_name = kwargs["clip_name"]
        video = tmp_path / f"{clip_name}.mp4"
        depth = tmp_path / f"{clip_name}_depth.mp4"
        video.write_bytes(b"x")
        depth.write_bytes(b"x")
        return SimpleNamespace(video_path=video, depth_video_path=depth)

    monkeypatch.setattr("blueprint_validation.stages.s1_render.render_video", _fake_render_video)
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render._ensure_probe_h264_for_scoring",
        lambda video_path, min_frames: SimpleNamespace(
            path=video_path,
            codec_name="h264",
            decoded_frames=int(min_frames),
            width=128,
            height=96,
            content_monochrome_warning=False,
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.score_stage1_probe",
        lambda **kwargs: Stage1ProbeScore(
            task_score=9.0,
            visual_score=8.0,
            spatial_score=8.0,
            issue_tags=["target_missing", "target_off_center"],
            reasoning="target not visible enough",
            raw_response="{}",
        ),
    )

    scores_path = tmp_path / "probe_scores.jsonl"
    stage = RenderStage()
    selected, probe_meta = stage._run_active_perception_probe(
        config=sample_config,
        splat=object(),
        clip_name="clip_000_orbit",
        initial_spec=spec,
        ranked_candidates=ranked,
        scene_center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        occupancy=None,
        render_dir=tmp_path,
        camera_height=1.0,
        look_down_deg=20.0,
        resolution=(96, 128),
        start_offset=np.zeros(3, dtype=np.float64),
        fps=8,
        scene_T=None,
        facility_description="",
        target_presence_enforced=True,
        probe_scores_path=scores_path,
    )

    assert selected.type == "orbit"
    assert probe_meta["vlm_probe_passed"] is False
    rows = [json.loads(line) for line in scores_path.read_text().splitlines() if line.strip()]
    assert any(
        row.get("status") == "target_presence_reject" and row.get("reason") == "vlm_target_missing"
        for row in rows
    )


def test_active_probe_trusts_geometry_for_scene_locked_facility_a_target_presence(
    sample_config, tmp_path, monkeypatch
):
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.evaluation.vlm_judge import Stage1ProbeScore
    from blueprint_validation.stages.s1_render import RenderStage

    sample_config.render.stage1_active_perception_max_loops = 0
    sample_config.render.stage1_probe_min_viable_pose_ratio = 0.1
    sample_config.render.stage1_probe_min_unique_positions = 1
    sample_config.render.stage1_probe_dedupe_enabled = False

    spec = CameraPathSpec(
        type="file",
        radius_m=1.0,
        approach_point=[0.0, 0.0, -1.0],
        source_tag="scene_locked:facility_a",
        target_label="bookshelf_right",
        target_role="targets",
        target_extents_m=[0.2, 0.4, 0.6],
        locked_eye_point=[0.0, 0.0, 0.0],
        locked_look_at_point=[0.0, 0.0, -1.0],
        locked_probe_motion_radius_m=0.0,
    )
    ranked = [SimpleNamespace(spec=spec, score=0.9, metrics={})]

    monkeypatch.setattr(
        RenderStage,
        "_build_render_poses",
        lambda self, **kwargs: ([_pose(), _pose()], 13, 13, 13, 0),
    )

    def _fake_render_video(**kwargs):
        clip_name = kwargs["clip_name"]
        video = tmp_path / f"{clip_name}.mp4"
        depth = tmp_path / f"{clip_name}_depth.mp4"
        video.write_bytes(b"x")
        depth.write_bytes(b"x")
        return SimpleNamespace(video_path=video, depth_video_path=depth)

    monkeypatch.setattr("blueprint_validation.stages.s1_render.render_video", _fake_render_video)
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render._ensure_probe_h264_for_scoring",
        lambda video_path, min_frames: SimpleNamespace(
            path=video_path,
            codec_name="h264",
            decoded_frames=int(min_frames),
            width=128,
            height=96,
            content_monochrome_warning=False,
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.score_stage1_probe",
        lambda **kwargs: Stage1ProbeScore(
            task_score=9.0,
            visual_score=8.0,
            spatial_score=8.0,
            issue_tags=["target_missing", "target_off_center"],
            reasoning="vlm thought the shelf target was unclear",
            raw_response="{}",
        ),
    )

    stage = RenderStage()
    selected, probe_meta = stage._run_active_perception_probe(
        config=sample_config,
        splat=object(),
        clip_name="clip_000_file",
        initial_spec=spec,
        ranked_candidates=ranked,
        scene_center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        occupancy=None,
        render_dir=tmp_path,
        camera_height=1.0,
        look_down_deg=20.0,
        resolution=(96, 128),
        start_offset=np.zeros(3, dtype=np.float64),
        fps=8,
        scene_T=None,
        facility_description="",
        target_presence_enforced=True,
        probe_scores_path=tmp_path / "probe_scores.jsonl",
        locked_mode=True,
    )

    assert selected.type == "file"
    assert probe_meta["vlm_probe_passed"] is True


def test_active_probe_scene_locked_facility_a_geometry_can_override_low_vlm_scores(
    sample_config, tmp_path, monkeypatch
):
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.evaluation.vlm_judge import Stage1ProbeScore
    from blueprint_validation.stages.s1_render import RenderStage

    sample_config.render.stage1_active_perception_max_loops = 0
    sample_config.render.stage1_probe_min_viable_pose_ratio = 0.1
    sample_config.render.stage1_probe_min_unique_positions = 1
    sample_config.render.stage1_probe_dedupe_enabled = False

    spec = CameraPathSpec(
        type="file",
        radius_m=1.0,
        approach_point=[0.0, 0.0, -1.0],
        source_tag="scene_locked:facility_a",
        target_label="bookshelf_right",
        target_role="targets",
        target_extents_m=[0.2, 0.4, 0.6],
        locked_eye_point=[0.0, 0.0, 0.0],
        locked_look_at_point=[0.0, 0.0, -1.0],
        locked_probe_motion_radius_m=0.0,
    )
    ranked = [SimpleNamespace(spec=spec, score=0.9, metrics={})]

    monkeypatch.setattr(
        RenderStage,
        "_build_render_poses",
        lambda self, **kwargs: ([_pose(), _pose()], 13, 13, 13, 0),
    )

    def _fake_render_video(**kwargs):
        clip_name = kwargs["clip_name"]
        video = tmp_path / f"{clip_name}.mp4"
        depth = tmp_path / f"{clip_name}_depth.mp4"
        video.write_bytes(b"x")
        depth.write_bytes(b"x")
        return SimpleNamespace(video_path=video, depth_video_path=depth)

    monkeypatch.setattr("blueprint_validation.stages.s1_render.render_video", _fake_render_video)
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render._ensure_probe_h264_for_scoring",
        lambda video_path, min_frames: SimpleNamespace(
            path=video_path,
            codec_name="h264",
            decoded_frames=int(min_frames),
            width=128,
            height=96,
            content_monochrome_warning=False,
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.score_stage1_probe",
        lambda **kwargs: Stage1ProbeScore(
            task_score=2.0,
            visual_score=3.0,
            spatial_score=2.0,
            issue_tags=["blur_or_soft_focus", "camera_too_far", "target_off_center"],
            reasoning="shelf clip still looks a bit far",
            raw_response="{}",
        ),
    )

    stage = RenderStage()
    selected, probe_meta = stage._run_active_perception_probe(
        config=sample_config,
        splat=object(),
        clip_name="clip_000_file",
        initial_spec=spec,
        ranked_candidates=ranked,
        scene_center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        occupancy=None,
        render_dir=tmp_path,
        camera_height=1.0,
        look_down_deg=20.0,
        resolution=(96, 128),
        start_offset=np.zeros(3, dtype=np.float64),
        fps=8,
        scene_T=None,
        facility_description="",
        target_presence_enforced=True,
        probe_scores_path=tmp_path / "probe_scores.jsonl",
        locked_mode=True,
    )

    assert selected.type == "file"
    assert probe_meta["vlm_probe_passed"] is False


def test_active_probe_scene_locked_facility_a_geometry_override_requires_min_semantic_floor(
    sample_config, tmp_path, monkeypatch
):
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.evaluation.vlm_judge import Stage1ProbeScore
    from blueprint_validation.stages.s1_render import RenderStage

    sample_config.render.stage1_active_perception_max_loops = 0
    sample_config.render.stage1_probe_min_viable_pose_ratio = 0.1
    sample_config.render.stage1_probe_min_unique_positions = 1
    sample_config.render.stage1_probe_dedupe_enabled = False

    spec = CameraPathSpec(
        type="file",
        radius_m=1.0,
        approach_point=[0.0, 0.0, -1.0],
        source_tag="scene_locked:facility_a",
        target_label="bookshelf_right",
        target_role="targets",
        target_extents_m=[0.2, 0.4, 0.6],
        locked_eye_point=[0.0, 0.0, 0.0],
        locked_look_at_point=[0.0, 0.0, -1.0],
        locked_probe_motion_radius_m=0.0,
    )
    ranked = [SimpleNamespace(spec=spec, score=0.9, metrics={})]

    monkeypatch.setattr(
        RenderStage,
        "_build_render_poses",
        lambda self, **kwargs: ([_pose(), _pose()], 13, 13, 13, 0),
    )

    def _fake_render_video(**kwargs):
        clip_name = kwargs["clip_name"]
        video = tmp_path / f"{clip_name}.mp4"
        depth = tmp_path / f"{clip_name}_depth.mp4"
        video.write_bytes(b"x")
        depth.write_bytes(b"x")
        return SimpleNamespace(video_path=video, depth_video_path=depth)

    monkeypatch.setattr("blueprint_validation.stages.s1_render.render_video", _fake_render_video)
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render._ensure_probe_h264_for_scoring",
        lambda video_path, min_frames: SimpleNamespace(
            path=video_path,
            codec_name="h264",
            decoded_frames=int(min_frames),
            width=128,
            height=96,
            content_monochrome_warning=False,
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.score_stage1_probe",
        lambda **kwargs: Stage1ProbeScore(
            task_score=7.0,
            visual_score=4.0,
            spatial_score=7.0,
            issue_tags=["blur_or_soft_focus", "camera_too_far"],
            reasoning="shelf clip is soft but semantically usable",
            raw_response="{}",
        ),
    )

    selected, probe_meta = RenderStage()._run_active_perception_probe(
        config=sample_config,
        splat=object(),
        clip_name="clip_000_file",
        initial_spec=spec,
        ranked_candidates=ranked,
        scene_center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        occupancy=None,
        render_dir=tmp_path,
        camera_height=1.0,
        look_down_deg=20.0,
        resolution=(96, 128),
        start_offset=np.zeros(3, dtype=np.float64),
        fps=8,
        scene_T=None,
        facility_description="",
        target_presence_enforced=True,
        probe_scores_path=tmp_path / "probe_scores.jsonl",
        locked_mode=True,
    )

    assert selected.type == "file"
    assert probe_meta["vlm_probe_passed"] is True


def test_stage1_bypasses_warmup_cache_when_active_perception_enabled(
    sample_config, tmp_path, monkeypatch
):
    from blueprint_validation.stages.s1_render import RenderStage, _empty_quality_summary

    sample_config.render.stage1_active_perception_enabled = True
    fac = sample_config.facilities["test_facility"]
    fac.ply_path.write_bytes(b"ply")

    class _FakeMeans:
        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((8, 3), dtype=np.float32)

    class _FakeSplat:
        means = _FakeMeans()

    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.load_cached_clips",
        lambda work_dir: [{"clip_name": "cached_clip"}],
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.load_warmup_cache",
        lambda work_dir: {"resolved_up_axis": "auto", "quality_cache_key": "x"},
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.load_splat",
        lambda *args, **kwargs: _FakeSplat(),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.resolve_facility_orientation",
        lambda **kwargs: (kwargs["facility"], {"orientation_candidates": []}),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.compute_scene_transform",
        lambda facility: np.eye(4, dtype=np.float64),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.is_identity_transform",
        lambda T: True,
    )
    monkeypatch.setattr(
        RenderStage,
        "_build_scene_aware_specs",
        lambda self, *args, **kwargs: ([], None),
    )

    called = {"render_from_cache": False, "generate": False}

    def _fail_if_cache_used(self, *args, **kwargs):
        called["render_from_cache"] = True
        raise AssertionError("warmup cache path should be bypassed")

    def _fake_generate(self, *args, **kwargs):
        called["generate"] = True
        return [], 0, 0, _empty_quality_summary()

    monkeypatch.setattr(RenderStage, "_render_from_cache", _fail_if_cache_used)
    monkeypatch.setattr(RenderStage, "_generate_and_render", _fake_generate)

    result = RenderStage().run(
        config=sample_config,
        facility=fac,
        work_dir=tmp_path / "out",
        previous_results={},
    )
    assert result.status == "success"
    assert called["generate"] is True
    assert called["render_from_cache"] is False


def test_probe_consensus_uses_median_scores_and_majority_tags(sample_config, monkeypatch, tmp_path):
    from blueprint_validation.evaluation.vlm_judge import Stage1ProbeScore
    from blueprint_validation.stages.s1_render import _score_stage1_probe_consensus

    rows = [
        Stage1ProbeScore(
            task_score=6.0,
            visual_score=7.0,
            spatial_score=6.0,
            issue_tags=["camera_motion_too_fast", "target_off_center"],
            reasoning="r1",
            raw_response="{}",
            model_used="gemini-3-flash-preview",
        ),
        Stage1ProbeScore(
            task_score=8.0,
            visual_score=7.0,
            spatial_score=7.0,
            issue_tags=["camera_motion_too_fast"],
            reasoning="r2",
            raw_response="{}",
            model_used="gemini-3-flash-preview",
        ),
        Stage1ProbeScore(
            task_score=7.0,
            visual_score=9.0,
            spatial_score=6.0,
            issue_tags=["target_off_center"],
            reasoning="r3",
            raw_response="{}",
            model_used="gemini-3-flash-preview",
        ),
    ]
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.score_stage1_probe",
        lambda **kwargs: rows.pop(0),
    )
    out = _score_stage1_probe_consensus(
        video_path=tmp_path / "probe.mp4",
        expected_focus_text="target",
        config=sample_config.eval_policy.vlm_judge,
        facility_description="",
        votes=3,
        primary_model_only=True,
    )
    score = out["score"]
    assert score is not None
    assert score.task_score == 7.0
    assert score.visual_score == 7.0
    assert score.spatial_score == 6.0
    assert score.issue_tags == ["camera_motion_too_fast", "target_off_center"]
    assert out["votes_effective"] == 3
    assert out["active_model_used"] == "gemini-3-flash-preview"


def test_probe_consensus_uses_fallback_only_after_primary_hard_failure(
    sample_config, monkeypatch, tmp_path
):
    from blueprint_validation.evaluation.vlm_judge import Stage1ProbeScore
    from blueprint_validation.stages.s1_render import _score_stage1_probe_consensus

    calls = {"primary": 0, "fallback": 0}

    def _fake_score(**kwargs):
        cfg = kwargs["config"]
        if not list(getattr(cfg, "fallback_models", []) or []):
            calls["primary"] += 1
            raise RuntimeError("503 service unavailable")
        calls["fallback"] += 1
        return Stage1ProbeScore(
            task_score=8.0,
            visual_score=8.0,
            spatial_score=7.0,
            issue_tags=[],
            reasoning="fallback ok",
            raw_response="{}",
            model_used="gemini-2.5-flash",
        )

    monkeypatch.setattr("blueprint_validation.stages.s1_render.score_stage1_probe", _fake_score)
    cfg = replace(
        sample_config.eval_policy.vlm_judge,
        fallback_models=["gemini-2.5-flash"],
    )
    out = _score_stage1_probe_consensus(
        video_path=tmp_path / "probe.mp4",
        expected_focus_text="target",
        config=cfg,
        facility_description="",
        votes=2,
        primary_model_only=True,
    )
    assert calls["primary"] == 2
    assert calls["fallback"] == 2
    assert out["votes_effective"] == 2
    assert out["active_model_used"] == "gemini-2.5-flash"


def test_probe_consensus_adds_tiebreak_votes_when_spread_is_high(sample_config, monkeypatch, tmp_path):
    from blueprint_validation.evaluation.vlm_judge import Stage1ProbeScore
    from blueprint_validation.stages.s1_render import _score_stage1_probe_consensus

    rows = [
        Stage1ProbeScore(
            task_score=1.0,
            visual_score=1.0,
            spatial_score=1.0,
            issue_tags=["target_missing"],
            reasoning="bad",
            raw_response="{}",
            model_used="gemini-3-flash-preview",
        ),
        Stage1ProbeScore(
            task_score=9.0,
            visual_score=9.0,
            spatial_score=9.0,
            issue_tags=[],
            reasoning="great",
            raw_response="{}",
            model_used="gemini-3-flash-preview",
        ),
        Stage1ProbeScore(
            task_score=5.0,
            visual_score=5.0,
            spatial_score=5.0,
            issue_tags=["camera_motion_too_fast"],
            reasoning="mid",
            raw_response="{}",
            model_used="gemini-3-flash-preview",
        ),
        Stage1ProbeScore(
            task_score=6.0,
            visual_score=6.0,
            spatial_score=6.0,
            issue_tags=["camera_motion_too_fast"],
            reasoning="extra1",
            raw_response="{}",
            model_used="gemini-3-flash-preview",
        ),
        Stage1ProbeScore(
            task_score=4.0,
            visual_score=4.0,
            spatial_score=4.0,
            issue_tags=["camera_motion_too_fast"],
            reasoning="extra2",
            raw_response="{}",
            model_used="gemini-3-flash-preview",
        ),
    ]

    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.score_stage1_probe",
        lambda **kwargs: rows.pop(0),
    )

    out = _score_stage1_probe_consensus(
        video_path=tmp_path / "probe.mp4",
        expected_focus_text="target",
        config=sample_config.eval_policy.vlm_judge,
        facility_description="",
        votes=3,
        primary_model_only=True,
        tiebreak_extra_votes=2,
        tiebreak_spread_threshold=3.0,
    )
    assert out["votes_effective"] == 5
    assert len(out["vote_rows"]) == 5
    assert any(str(v.get("phase")) == "tiebreak" for v in out["vote_rows"])


def test_detect_duplicate_clip_flags_sha_match(monkeypatch, tmp_path):
    from blueprint_validation.stages.s1_render import _detect_duplicate_clip

    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render._clip_fingerprint",
        lambda _video_path: {"sha256": "same_hash", "samples": []},
    )
    out = _detect_duplicate_clip(
        video_path=tmp_path / "clip.mp4",
        seen_fingerprints=[{"sha256": "same_hash", "samples": []}],
        similarity_threshold=0.995,
    )
    assert out["is_duplicate"] is True
    assert out["reason"] == "sha256"
