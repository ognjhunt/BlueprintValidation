"""Tests for Stage-1 active-perception probe selection and fail-closed behavior."""

from __future__ import annotations

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


def test_active_probe_selects_candidate_that_passes_threshold(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.evaluation.vlm_judge import Stage1ProbeScore
    from blueprint_validation.stages.s1_render import RenderStage

    sample_config.render.stage1_active_perception_enabled = True
    sample_config.render.stage1_active_perception_scope = "all"
    sample_config.render.stage1_active_perception_max_loops = 1
    sample_config.render.stage1_active_perception_fail_closed = True
    sample_config.render.stage1_quality_candidate_budget = "medium"

    spec_a = CameraPathSpec(type="orbit", radius_m=2.0)
    spec_b = CameraPathSpec(type="orbit", radius_m=1.6)
    ranked = [
        SimpleNamespace(spec=spec_a, score=0.9, metrics={"planner_visible_ratio": 0.5}),
        SimpleNamespace(spec=spec_b, score=0.8, metrics={"planner_visible_ratio": 0.6}),
    ]

    monkeypatch.setattr(
        RenderStage,
        "_build_render_poses",
        lambda self, **kwargs: ([_pose(), _pose()], 2, 2, 2, 0),
    )

    def _fake_render_video(**kwargs):
        clip_name = kwargs["clip_name"]
        video = tmp_path / f"{clip_name}.mp4"
        depth = tmp_path / f"{clip_name}_depth.mp4"
        video.write_bytes(b"x")
        depth.write_bytes(b"x")
        return SimpleNamespace(video_path=video, depth_video_path=depth)

    monkeypatch.setattr("blueprint_validation.stages.s1_render.render_video", _fake_render_video)

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


def test_active_probe_fail_closed_on_scoring_error(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.stages.s1_render import RenderStage

    sample_config.render.stage1_active_perception_enabled = True
    sample_config.render.stage1_active_perception_scope = "all"
    sample_config.render.stage1_active_perception_max_loops = 1
    sample_config.render.stage1_active_perception_fail_closed = True

    spec = CameraPathSpec(type="orbit", radius_m=2.0)
    ranked = [SimpleNamespace(spec=spec, score=0.9, metrics={})]

    monkeypatch.setattr(
        RenderStage,
        "_build_render_poses",
        lambda self, **kwargs: ([_pose(), _pose()], 2, 2, 2, 0),
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

    assert selected.radius_m == spec.radius_m
    assert probe_meta["vlm_probe_passed"] is False
    assert "probe_scoring_failed" in str(probe_meta["vlm_probe_fail_reason"])
    assert int(probe_meta["num_vlm_probe_api_failures"]) == 1


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
