"""Tests for task-scoped scene-aware OBB selection in Stage 1 render."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


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
    selected, stats = _select_task_scoped_obbs(
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
    selected, stats = _select_task_scoped_obbs(
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
    selected, stats = _select_task_scoped_obbs(
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
        lambda **kwargs: SimpleNamespace(
            video_path=tmp_path / "clip.mp4",
            depth_video_path=tmp_path / "clip_depth.mp4",
        ),
    )

    stage = RenderStage()
    manifest_entries, _, _ = stage._generate_and_render(
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
