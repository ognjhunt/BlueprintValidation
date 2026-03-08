"""Tests for task-scoped scene-aware OBB selection in Stage 1 render."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest


def _obb(
    instance_id: str,
    label: str,
    center: tuple[float, float, float],
    category: str,
    confidence: float = 1.0,
):
    from blueprint_validation.rendering.scene_geometry import OrientedBoundingBox

    return OrientedBoundingBox(
        instance_id=instance_id,
        label=label,
        center=np.asarray(center, dtype=np.float64),
        extents=np.asarray([0.3, 0.3, 0.3], dtype=np.float64),
        axes=np.eye(3, dtype=np.float64),
        confidence=confidence,
        category=category,
    )


def test_task_scoped_selection_targets_context_and_overview():
    from blueprint_validation.stages.s1_render import _select_task_scoped_obbs

    obbs = [
        _obb("1", "mug", (0.0, 0.0, 0.0), "manipulation"),
        _obb("2", "bowl", (0.2, 0.0, 0.0), "manipulation"),
        _obb("3", "plate", (0.4, 0.0, 0.0), "manipulation"),
        _obb("4", "fridge", (5.0, 0.0, 0.0), "articulation"),
        _obb("5", "hallway", (8.0, 0.0, 0.0), "navigation"),
        _obb("6", "cabinet", (6.0, 0.0, 0.0), "articulation"),
    ]
    selected, stats, role_by_instance = _select_task_scoped_obbs(
        obbs=obbs,
        tasks=["Pick up mug_1 and place it on the counter"],
        max_specs=4,
        context_per_target=2,
        overview_specs=2,
        fallback_specs=3,
    )

    assert len(selected) == 4
    assert selected[0].instance_id == "1"  # primary task target first
    assert stats["targets"] == 1
    assert stats["context"] == 2
    assert stats["overview"] == 1
    assert stats["fallback"] == 0
    assert role_by_instance["1"] == "targets"
    selected_ids = {o.instance_id for o in selected}
    assert {"1", "2", "3"}.issubset(selected_ids)


def test_task_scoped_selection_fallback_when_no_task_targets_match():
    from blueprint_validation.stages.s1_render import _select_task_scoped_obbs

    obbs = [
        _obb("10", "hallway", (8.0, 0.0, 0.0), "navigation", confidence=1.0),
        _obb("11", "mug", (0.0, 0.0, 0.0), "manipulation", confidence=0.8),
        _obb("12", "bowl", (1.0, 0.0, 0.0), "manipulation", confidence=0.9),
        _obb("13", "cabinet", (2.0, 0.0, 0.0), "articulation", confidence=1.0),
    ]
    selected, stats, role_by_instance = _select_task_scoped_obbs(
        obbs=obbs,
        tasks=["Do something unrelated"],
        max_specs=3,
        context_per_target=1,
        overview_specs=1,
        fallback_specs=3,
    )

    assert len(selected) == 3
    assert stats["fallback"] == 3
    assert stats["targets"] == 0
    assert all(role_by_instance.get(o.instance_id) == "fallback" for o in selected)
    # Fallback prioritizes manipulation/articulation before navigation.
    assert selected[0].category == "manipulation"
    assert selected[1].category in {"manipulation", "articulation"}


def test_task_scoped_selection_resolves_label_instance_token():
    from blueprint_validation.stages.s1_render import _select_task_scoped_obbs

    obbs = [
        _obb("101", "bowl", (0.0, 0.0, 0.0), "manipulation"),
        _obb("102", "cup", (1.0, 0.0, 0.0), "manipulation"),
        _obb("103", "cabinet", (2.0, 0.0, 0.0), "articulation"),
    ]
    selected, stats, role_by_instance = _select_task_scoped_obbs(
        obbs=obbs,
        tasks=["Pick up bowl_101 and place it in the sink"],
        max_specs=2,
        context_per_target=0,
        overview_specs=0,
        fallback_specs=2,
    )

    assert len(selected) == 1
    assert selected[0].instance_id == "101"
    assert stats["targets"] == 1
    assert role_by_instance["101"] == "targets"


def test_task_scoped_selection_dedupes_near_identical_label_centers():
    from blueprint_validation.stages.s1_render import _select_task_scoped_obbs

    obbs = [
        _obb("200", "bowl", (0.00, 0.00, 0.50), "manipulation", confidence=0.8),
        _obb("201", "bowl", (0.03, 0.00, 0.50), "manipulation", confidence=0.9),
        _obb("202", "cup", (1.00, 0.00, 0.50), "manipulation", confidence=0.8),
    ]
    selected, stats, _ = _select_task_scoped_obbs(
        obbs=obbs,
        tasks=["pick up bowl_201"],
        max_specs=3,
        context_per_target=1,
        overview_specs=0,
        fallback_specs=3,
        center_dedupe_dist_m=0.08,
    )
    ids = [o.instance_id for o in selected]
    assert "200" not in ids
    assert "201" in ids
    assert stats["targets"] == 1


def test_sample_start_offset_defaults_zero_for_manipulation(sample_config):
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.stages.s1_render import _sample_start_offset

    rng = np.random.default_rng(42)
    sample_config.render.manipulation_random_xy_offset_m = 0.0
    offset = _sample_start_offset(sample_config, CameraPathSpec(type="manipulation"), rng)
    np.testing.assert_allclose(offset, np.zeros(3, dtype=np.float64), atol=1e-10)


def test_sample_start_offset_non_manipulation_retains_jitter(sample_config):
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.stages.s1_render import _sample_start_offset

    rng = np.random.default_rng(7)
    sample_config.render.non_manipulation_random_xy_offset_m = 1.0
    offset = _sample_start_offset(sample_config, CameraPathSpec(type="orbit"), rng)
    assert offset.shape == (3,)
    assert float(abs(offset[0])) > 0 or float(abs(offset[1])) > 0
    assert float(offset[2]) == 0.0


def test_sample_start_offset_target_grounded_forces_zero_for_probe(sample_config):
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.stages.s1_render import _sample_start_offset

    rng = np.random.default_rng(7)
    sample_config.render.non_manipulation_random_xy_offset_m = 1.0
    offset = _sample_start_offset(
        sample_config,
        CameraPathSpec(type="orbit"),
        rng,
        target_grounded=True,
    )
    np.testing.assert_allclose(offset, np.zeros(3, dtype=np.float64), atol=1e-10)


def test_sample_start_offset_task_scoped_repeat_gets_deterministic_jitter(sample_config):
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.stages.s1_render import _sample_start_offset

    rng = np.random.default_rng(3)
    sample_config.render.manipulation_random_xy_offset_m = 0.0
    sample_config.render.stage1_repeat_min_xy_jitter_m = 0.06
    base = _sample_start_offset(
        sample_config,
        CameraPathSpec(type="manipulation"),
        rng,
        clip_repeat_index=0,
        is_task_scoped=True,
    )
    repeated = _sample_start_offset(
        sample_config,
        CameraPathSpec(type="manipulation"),
        rng,
        clip_repeat_index=1,
        is_task_scoped=True,
    )
    np.testing.assert_allclose(base, np.zeros(3, dtype=np.float64), atol=1e-10)
    assert float(np.linalg.norm(repeated[:2])) > 0.0


def test_generate_and_render_preserves_requested_frames_after_collision_filter(
    sample_config,
    tmp_path,
    monkeypatch,
):
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.rendering.camera_paths import CameraPose
    from blueprint_validation.stages.s1_render import RenderStage

    sample_config.render.preserve_num_frames_after_collision_filter = True
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

    generated = [_pose(float(i)) for i in range(5)]

    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.generate_path_from_spec",
        lambda **kwargs: list(generated),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.filter_and_fix_poses",
        lambda poses, *_args, **_kwargs: list(poses[:3]),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.correct_upside_down_camera_poses",
        lambda poses: (poses, 0),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.save_path_to_json",
        lambda poses, out_path: None,
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.render_video",
        lambda **kwargs: (
            (tmp_path / "clip.mp4").write_bytes(b"x"),
            (tmp_path / "clip_depth.mp4").write_bytes(b"x"),
            SimpleNamespace(
                video_path=tmp_path / "clip.mp4",
                depth_video_path=tmp_path / "clip_depth.mp4",
            ),
        )[-1],
    )

    stage = RenderStage()
    manifest_entries, _, _, _ = stage._generate_and_render(
        config=sample_config,
        splat=object(),
        all_path_specs=[CameraPathSpec(type="orbit", radius_m=2.0)],
        scene_center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        occupancy=object(),
        render_dir=tmp_path,
        num_frames=5,
        camera_height=1.0,
        look_down_deg=20.0,
        resolution=(48, 64),
        fps=5,
        scene_T=None,
    )

    assert len(manifest_entries) == 1
    entry = manifest_entries[0]
    assert entry["requested_num_frames"] == 5
    assert entry["pre_filter_num_frames"] == 5
    assert entry["post_filter_num_frames"] == 3
    assert entry["post_resample_num_frames"] == 5
    assert entry["num_frames"] == 5
    assert isinstance(entry.get("expected_focus_text"), str)
    assert entry["expected_focus_text"].strip()
    assert "vlm_probe_attempts" in entry
    assert "vlm_probe_passed" in entry
    assert "vlm_probe_retries_used" in entry
    assert "vlm_probe_issue_tags_final" in entry
    assert "vlm_probe_candidate_count" in entry
    assert "vlm_probe_selected_fps" in entry
    assert "vlm_probe_fail_reason" in entry


def test_expected_focus_text_uses_role_and_target():
    from blueprint_validation.stages.s1_render import _build_expected_focus_text

    text = _build_expected_focus_text(
        path_type="manipulation",
        path_context={"target_role": "targets", "target_label": "bowl_101"},
    )
    assert "Primary target focus" in text
    assert "bowl_101" in text


def test_expected_focus_text_falls_back_to_path_type():
    from blueprint_validation.stages.s1_render import _build_expected_focus_text

    text = _build_expected_focus_text(path_type="sweep", path_context={})
    assert "Sweep focus" in text


def test_is_kitchen_0787_scene_detects_known_facility_tokens(tmp_path):
    from blueprint_validation.config import FacilityConfig
    from blueprint_validation.stages.s1_render import _is_kitchen_0787_scene

    fac = FacilityConfig(
        name="Kitchen Scene 0787 (InteriorGS)",
        ply_path=tmp_path / "0787_841244" / "3dgs_compressed.ply",
        task_hints_path=tmp_path / "0787_841244" / "task_targets.synthetic.json",
    )
    assert _is_kitchen_0787_scene(fac) is True


def test_resolve_scene_locked_profile_honors_explicit_facility_a(sample_config, tmp_path):
    from blueprint_validation.config import FacilityConfig
    from blueprint_validation.stages.s1_render import _resolve_scene_locked_profile

    sample_config.render.scene_locked_profile = "facility_a"
    fac = FacilityConfig(
        name="Facility A",
        ply_path=tmp_path / "facility_a" / "splat.ply",
        task_hints_path=tmp_path / "facility_a" / "task_targets.manual.json",
    )
    assert _resolve_scene_locked_profile(sample_config, fac) == "facility_a"


def test_build_kitchen_0787_locked_specs_is_target_grounded_and_deterministic(
    sample_config, tmp_path, monkeypatch
):
    from blueprint_validation.config import FacilityConfig
    from blueprint_validation.rendering.scene_geometry import OrientedBoundingBox
    from blueprint_validation.stages.s1_render import _build_kitchen_0787_locked_specs

    hints = tmp_path / "task_targets.synthetic.json"
    hints.write_text("{}")
    fac = FacilityConfig(
        name="Kitchen Scene 0787 (InteriorGS)",
        ply_path=tmp_path / "3dgs_compressed.ply",
        task_hints_path=hints,
    )

    obbs = [
        OrientedBoundingBox(
            instance_id="101",
            label="bowl",
            center=np.asarray([0.0, 0.0, 0.8], dtype=np.float64),
            extents=np.asarray([0.5, 0.4, 0.3], dtype=np.float64),
            axes=np.eye(3, dtype=np.float64),
            confidence=0.9,
            category="manipulation",
        ),
        OrientedBoundingBox(
            instance_id="202",
            label="cabinet",
            center=np.asarray([1.2, -0.4, 1.1], dtype=np.float64),
            extents=np.asarray([0.8, 0.2, 1.8], dtype=np.float64),
            axes=np.eye(3, dtype=np.float64),
            confidence=0.8,
            category="articulation",
        ),
    ]

    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.load_obbs_from_task_targets",
        lambda _path: list(obbs),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render._build_task_prompt_pool",
        lambda **_kwargs: ["pick up bowl_101"],
    )

    specs = _build_kitchen_0787_locked_specs(
        config=sample_config,
        facility=fac,
        scene_transform=None,
    )
    assert specs
    assert all(spec.source_tag == "kitchen_0787_locked" for spec in specs)
    assert all(spec.approach_point is not None for spec in specs)
    assert specs[0].target_instance_id == "101"
    assert specs[0].target_label == "bowl"
    assert specs[0].target_role == "targets"
    assert all(spec.type == "file" for spec in specs)
    assert all(isinstance(spec.target_extents_m, list) and len(spec.target_extents_m) == 3 for spec in specs)
    assert all(isinstance(spec.locked_eye_point, list) and len(spec.locked_eye_point) == 3 for spec in specs)
    assert all(
        isinstance(spec.locked_look_at_point, list) and len(spec.locked_look_at_point) == 3
        for spec in specs
    )
    assert all(
        spec.locked_probe_motion_radius_m is not None and spec.locked_probe_motion_radius_m >= 0.0
        for spec in specs
    )


def test_build_scene_locked_specs_facility_a_builds_target_grounded_locked_pose(
    sample_config, tmp_path, monkeypatch
):
    from blueprint_validation.config import FacilityConfig
    from blueprint_validation.rendering.scene_geometry import OrientedBoundingBox
    from blueprint_validation.stages.s1_render import _build_scene_locked_specs

    hints = tmp_path / "task_targets.synthetic.json"
    hints.write_text(
        json.dumps(
            {
                "tasks": [{"task_id": "Pick up the book from the shelf"}],
                "manipulation_candidates": [{"instance_id": "101", "label": "book", "category": "manipulation"}],
            }
        )
    )
    fac = FacilityConfig(
        name="Facility A",
        ply_path=tmp_path / "facility_a.ply",
        task_hints_path=hints,
    )

    obbs = [
        OrientedBoundingBox(
            instance_id="101",
            label="book",
            center=np.asarray([0.0, 0.0, 0.8], dtype=np.float64),
            extents=np.asarray([0.3, 0.3, 0.2], dtype=np.float64),
            axes=np.eye(3, dtype=np.float64),
            confidence=0.9,
            category="manipulation",
        ),
    ]

    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.load_obbs_from_task_targets",
        lambda _path: list(obbs),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render._build_task_prompt_pool",
        lambda **_kwargs: ["pick up book_101"],
    )

    specs = _build_scene_locked_specs(
        config=sample_config,
        facility=fac,
        scene_transform=None,
        profile="facility_a",
    )

    assert specs
    assert specs[0].source_tag == "scene_locked:facility_a"
    assert specs[0].type == "file"
    assert specs[0].target_instance_id == "101"
    assert isinstance(specs[0].locked_eye_point, list) and len(specs[0].locked_eye_point) == 3
    assert isinstance(specs[0].locked_look_at_point, list) and len(specs[0].locked_look_at_point) == 3
    assert np.all(np.isfinite(np.asarray(specs[0].locked_eye_point, dtype=np.float64)))
    assert np.all(np.isfinite(np.asarray(specs[0].locked_look_at_point, dtype=np.float64)))
    assert specs[0].locked_look_at_point[0] == pytest.approx(0.0)
    assert specs[0].locked_look_at_point[1] == pytest.approx(0.0)
    assert 0.70 <= float(specs[0].locked_look_at_point[2]) <= 0.80
    assert float(np.linalg.norm(np.asarray(specs[0].locked_eye_point) - np.asarray(specs[0].locked_look_at_point))) > 0.5
    assert specs[0].locked_probe_motion_radius_m == pytest.approx(0.12)


def test_build_scene_locked_specs_facility_a_rejects_non_bookshelf_semantics(
    sample_config, tmp_path, monkeypatch
):
    from blueprint_validation.config import FacilityConfig
    from blueprint_validation.rendering.scene_geometry import OrientedBoundingBox
    from blueprint_validation.stages.s1_render import _build_scene_locked_specs

    hints = tmp_path / "facility_a" / "task_targets.manual.json"
    hints.parent.mkdir(parents=True, exist_ok=True)
    hints.write_text(
        json.dumps(
            {
                "tasks": [{"task_id": "Pick up the bottle from the shelf and place it in the target zone"}],
                "manipulation_candidates": [{"instance_id": "102", "label": "bottle", "category": "manipulation"}],
                "navigation_hints": [{"instance_id": "region::pantry_shelf", "label": "pantry_shelf", "category": "navigation"}],
            }
        )
    )
    fac = FacilityConfig(
        name="Facility A",
        ply_path=tmp_path / "dummy.ply",
        task_hints_path=hints,
        description="facility_a",
    )
    obbs = [
        OrientedBoundingBox(
            instance_id="102",
            label="bottle",
            center=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            extents=np.asarray([0.2, 0.2, 0.4], dtype=np.float64),
            axes=np.eye(3, dtype=np.float64),
            confidence=0.9,
            category="manipulation",
        )
    ]

    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.load_obbs_from_task_targets",
        lambda _path: list(obbs),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render._build_task_prompt_pool",
        lambda **_kwargs: ["pick up bottle_102"],
    )

    specs = _build_scene_locked_specs(
        config=sample_config,
        facility=fac,
        scene_transform=None,
        profile="facility_a",
    )

    assert specs == []


def test_build_render_poses_scene_locked_uses_fixed_eye_lookat_without_collision_nudge(
    sample_config, monkeypatch
):
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.stages.s1_render import RenderStage

    locked_spec = CameraPathSpec(
        type="file",
        source_tag="kitchen_0787_locked",
        target_instance_id="999",
        target_label="bowl_190",
        target_role="targets",
        approach_point=[0.0, 0.0, 0.8],
        target_extents_m=[0.25, 0.20, 0.18],
        locked_eye_point=[0.5, -0.2, 1.1],
        locked_look_at_point=[0.0, 0.0, 0.85],
    )

    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.generate_path_from_spec",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("locked mode should not call generate_path_from_spec")
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.filter_and_fix_poses",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("locked mode should bypass collision nudge")
        ),
    )

    poses, pre, post_filter, post_resample, corrected = RenderStage()._build_render_poses(
        config=sample_config,
        path_spec=locked_spec,
        scene_center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        occupancy=object(),  # non-None to prove filter bypass
        num_frames=11,
        camera_height=1.2,
        look_down_deg=20.0,
        resolution=(96, 128),
        start_offset=np.array([0.2, -0.1, 0.0], dtype=np.float64),
    )

    assert len(poses) == 11
    assert pre == 11
    assert post_filter == 11
    assert post_resample == 11
    assert corrected >= 0
    target = np.asarray([0.0, 0.0, 0.85], dtype=np.float64)
    for pose in poses:
        to_target = target - pose.position
        to_target = to_target / max(float(np.linalg.norm(to_target)), 1e-8)
        forward = pose.forward / max(float(np.linalg.norm(pose.forward)), 1e-8)
        assert float(np.dot(forward, to_target)) > 0.995


def test_generate_and_render_scene_locked_forces_single_clip_and_locked_probe(
    sample_config, tmp_path, monkeypatch
):
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.rendering.camera_paths import CameraPose
    from blueprint_validation.stages.s1_render import (
        RenderStage,
        _KITCHEN_0787_LOCKED_SOURCE_TAG,
    )

    sample_config.render.num_clips_per_path = 3
    sample_config.render.stage1_quality_planner_enabled = True
    sample_config.render.stage1_active_perception_enabled = True
    sample_config.render.stage1_active_perception_fail_closed = True

    spec = CameraPathSpec(
        type="orbit",
        radius_m=1.0,
        approach_point=[0.0, 0.0, 0.8],
        source_tag=_KITCHEN_0787_LOCKED_SOURCE_TAG,
        target_label="bowl_101",
        target_role="targets",
    )

    pose = CameraPose(
        c2w=np.eye(4, dtype=np.float64),
        fx=50.0,
        fy=50.0,
        cx=32.0,
        cy=24.0,
        width=64,
        height=48,
    )

    captured = {}

    monkeypatch.setattr(
        RenderStage,
        "_build_render_poses",
        lambda self, **kwargs: ([pose, pose], 2, 2, 2, 0),
    )
    monkeypatch.setattr(
        RenderStage,
        "_annotate_entry_quality",
        lambda self, **kwargs: True,
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.save_path_to_json",
        lambda poses, out_path: out_path.write_text("{}"),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.render_video",
        lambda **kwargs: (
            (tmp_path / f"{kwargs['clip_name']}.mp4").write_bytes(b"x"),
            (tmp_path / f"{kwargs['clip_name']}_depth.mp4").write_bytes(b"x"),
            SimpleNamespace(
                video_path=tmp_path / f"{kwargs['clip_name']}.mp4",
                depth_video_path=tmp_path / f"{kwargs['clip_name']}_depth.mp4",
            ),
        )[-1],
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render._detect_duplicate_clip",
        lambda **kwargs: {"is_duplicate": False, "sha256": "abc", "samples": [], "max_similarity": None},
    )

    def _fake_probe(self, **kwargs):
        captured["locked_mode"] = bool(kwargs.get("locked_mode"))
        captured["start_offset"] = np.asarray(kwargs.get("start_offset"), dtype=np.float64).copy()
        return kwargs["initial_spec"], {
            "vlm_probe_attempts": 1,
            "vlm_probe_evaluated": True,
            "vlm_probe_passed": True,
            "vlm_probe_retries_used": 0,
            "vlm_probe_issue_tags_final": [],
            "vlm_probe_candidate_count": 1,
            "vlm_probe_selected_fps": float(sample_config.eval_policy.vlm_judge.video_metadata_fps),
            "vlm_probe_fail_reason": "",
            "num_vlm_probe_api_failures": 0,
            "num_vlm_probe_parse_failures": 0,
            "num_vlm_probe_high_variance": 0,
            "num_probe_duplicate_detected": 0,
            "num_probe_duplicate_regenerated": 0,
            "num_probe_duplicate_unresolved": 0,
            "num_probe_viability_rejects": 0,
            "num_probe_monochrome_warnings": 0,
            "active_model_used": "gemini-3-flash-preview",
            "vlm_probe_consensus_votes_configured": 1,
            "vlm_probe_consensus_votes_effective": 1,
            "vlm_probe_score_spread": None,
            "probe_codec": "h264",
            "probe_resolution": [64, 48],
            "probe_decoded_frames": 2,
        }

    monkeypatch.setattr(RenderStage, "_run_active_perception_probe", _fake_probe)

    stage = RenderStage()
    entries, _, _, _ = stage._generate_and_render(
        config=sample_config,
        splat=object(),
        all_path_specs=[spec],
        scene_center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        occupancy=None,
        render_dir=tmp_path,
        num_frames=4,
        camera_height=1.0,
        look_down_deg=20.0,
        resolution=(48, 64),
        fps=5,
        scene_T=None,
    )

    assert len(entries) == 1
    assert captured["locked_mode"] is True
    np.testing.assert_allclose(captured["start_offset"], np.zeros(3, dtype=np.float64), atol=1e-10)


def test_annotate_entry_quality_ignores_approach_bins_for_scene_locked_facility_a(
    sample_config, monkeypatch
):
    from blueprint_validation.stages.s1_render import RenderStage

    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.evaluate_clip_quality",
        lambda **kwargs: {
            "target_visibility_ratio": 1.0,
            "target_center_band_ratio": 1.0,
            "target_approach_angle_bins": 1,
            "target_visible_frames": 25,
            "target_total_frames": 25,
            "target_center_band_frames": 25,
            "blur_laplacian_score": 300.0,
            "clip_quality_score": 0.92,
            "quality_gate_passed": False,
            "quality_reject_reasons": ["approach_angle_bins_low"],
        },
    )

    entry = {
        "clip_name": "clip_001_file",
        "path_type": "file",
        "path_context": {
            "source_tag": "scene_locked:facility_a",
            "approach_point": [0.0, 0.0, -1.0],
            "target_label": "bookshelf_right",
            "target_role": "targets",
        },
        "video_path": "unused.mp4",
        "camera_path": "unused.json",
        "resolution": [64, 80],
    }

    passed = RenderStage()._annotate_entry_quality(
        config=sample_config,
        clip_entry=entry,
        quality_retries_used=0,
        candidate_count_evaluated=1,
    )

    assert passed is True
    assert entry["quality_gate_passed"] is True
    assert entry["quality_reject_reasons"] == []


def test_run_geometry_canary_reports_target_presence(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.rendering.camera_paths import CameraPose, _look_at
    from blueprint_validation.stages.s1_render import RenderStage

    class _FakeTensor:
        def __init__(self, arr: np.ndarray):
            self._arr = arr

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    class _FakeSplat:
        def __init__(self, arr: np.ndarray):
            self.means = _FakeTensor(arr)

    sample_config.render.scene_aware = False
    sample_config.render.collision_check = False
    sample_config.render.stage1_probe_frames_override = 4
    sample_config.render.stage1_probe_min_unique_positions = 1
    sample_config.render.camera_paths = [
        CameraPathSpec(
            type="orbit",
            radius_m=1.0,
            approach_point=[0.0, 0.0, 0.0],
            target_role="targets",
            target_label="bowl",
            target_extents_m=[0.40, 0.35, 0.30],
        )
    ]

    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.load_splat",
        lambda *_args, **_kwargs: _FakeSplat(np.array([[0.0, 0.0, 0.0]], dtype=np.float32)),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.resolve_facility_orientation",
        lambda **kwargs: (kwargs["facility"], {}),
    )

    def _fake_build_render_poses(
        self,
        *,
        config,
        path_spec,
        scene_center,
        occupancy,
        num_frames,
        camera_height,
        look_down_deg,
        resolution,
        start_offset,
    ):
        height, width = int(resolution[0]), int(resolution[1])
        fx = fy = float(width) / float(2.0 * np.tan(np.deg2rad(60.0 / 2.0)))
        cx = float(width) / 2.0
        cy = float(height) / 2.0
        poses = []
        for i in range(int(num_frames)):
            eye = np.array([0.0, -0.95 + 0.01 * float(i), 0.45], dtype=np.float64)
            c2w = _look_at(eye, np.array([0.0, 0.0, 0.0], dtype=np.float64))
            poses.append(
                CameraPose(
                    c2w=c2w,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    width=width,
                    height=height,
                )
            )
        n = len(poses)
        return poses, n, n, n, 0

    monkeypatch.setattr(
        RenderStage,
        "_build_render_poses",
        _fake_build_render_poses,
    )

    stage = RenderStage()
    summary = stage.run_geometry_canary(
        config=sample_config,
        facility=sample_config.facilities["test_facility"],
        work_dir=tmp_path,
        max_specs=1,
    )

    assert int(summary["num_rows"]) == 1
    assert int(summary["num_target_grounded_rows"]) == 1
    assert int(summary["target_missing_count"]) == 0
    rows_path = tmp_path / "renders" / "s1_geometry_canary_rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"


def test_run_post_s1_audit_combines_geometry_integrity_quality_and_vlm(
    sample_config,
    tmp_path,
    monkeypatch,
):
    from types import SimpleNamespace

    from blueprint_validation.stages.s1_render import RenderStage
    from blueprint_validation.stages.s2_enrich import CoverageGateResult
    from blueprint_validation.video_io import VideoValidationResult

    fac = sample_config.facilities["test_facility"]
    fac_dir = tmp_path / "test_facility"
    render_dir = fac_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    video_path = render_dir / "clip_000.mp4"
    depth_path = render_dir / "clip_000_depth.mp4"
    camera_path = render_dir / "clip_000_camera_path.json"
    video_path.write_bytes(b"video")
    depth_path.write_bytes(b"depth")
    camera_path.write_text("{}")
    (render_dir / "render_manifest.json").write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "clip_name": "clip_000",
                        "clip_index": 0,
                        "path_type": "manipulation",
                        "video_path": str(video_path),
                        "depth_video_path": str(depth_path),
                        "camera_path": str(camera_path),
                        "num_frames": 4,
                        "expected_focus_text": "keep bowl centered",
                        "path_context": {
                            "approach_point": [0.0, 0.0, 0.0],
                            "target_label": "bowl",
                            "target_role": "targets",
                            "target_extents_m": [0.4, 0.3, 0.2],
                        },
                    }
                ]
            }
        )
    )

    monkeypatch.setattr(
        RenderStage,
        "run_geometry_canary",
        lambda self, **kwargs: {
            "num_rows": 1,
            "num_target_grounded_rows": 1,
            "first6_target_missing_count": 0,
        },
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.ensure_h264_video",
        lambda **kwargs: VideoValidationResult(
            path=video_path,
            codec_name="h264",
            decoded_frames=4,
            duration_seconds=0.4,
            transcoded=False,
            width=64,
            height=48,
            content_monochrome_warning=False,
            content_max_std_dev=12.0,
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render.evaluate_clip_quality",
        lambda **kwargs: {
            "target_visibility_ratio": 1.0,
            "target_center_band_ratio": 1.0,
            "target_approach_angle_bins": 3,
            "target_visible_frames": 4,
            "target_total_frames": 4,
            "target_center_band_frames": 4,
            "blur_laplacian_score": 80.0,
            "clip_quality_score": 0.95,
            "quality_gate_passed": True,
            "quality_reject_reasons": [],
        },
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s2_enrich._evaluate_stage1_coverage_gate",
        lambda render_manifest, config: CoverageGateResult(
            passed=True,
            detail="ok",
            metrics={"coverage_gate_passed": True},
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s1_render._score_stage1_probe_consensus",
        lambda **kwargs: {
            "score": SimpleNamespace(
                task_score=2.0,
                visual_score=4.0,
                spatial_score=3.0,
                issue_tags=[],
                reasoning="ok",
            ),
            "votes_effective": 1,
            "score_spread": 0.0,
            "active_model_used": "gemini-3-flash-preview",
        },
    )

    stage = RenderStage()
    summary = stage.run_post_s1_audit(
        config=sample_config,
        facility=fac,
        work_dir=fac_dir,
        geometry_max_specs=1,
        vlm_rescore_first=1,
    )
    assert summary["status"] == "success"
    assert int(summary["num_clips_in_manifest"]) == 1
    assert int(summary["num_videos_missing"]) == 0
    assert int(summary["num_videos_invalid"]) == 0
    assert int(summary["num_quality_gate_failed"]) == 0
    assert int(summary["vlm_rows_scored"]) == 1
