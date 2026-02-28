"""Tests for warmup pre-computation module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blueprint_validation.config import (
    CameraPathSpec,
    FacilityConfig,
    RenderConfig,
    ValidationConfig,
)
from blueprint_validation.warmup import (
    _deserialize_camera_poses,
    _load_ply_means_numpy,
    _serialize_camera_poses,
    load_cached_clips,
    load_cached_variants,
    load_ply_means_and_colors_numpy,
    load_warmup_cache,
    warmup_facility,
)


@pytest.fixture
def warmup_config(sample_ply, tmp_path) -> tuple[ValidationConfig, FacilityConfig, Path]:
    fac = FacilityConfig(
        name="Test Warehouse",
        ply_path=sample_ply,
        description="A test facility for warmup",
        landmarks=["door", "shelf"],
    )
    config = ValidationConfig(
        project_name="Warmup Test",
        facilities={"test": fac},
        render=RenderConfig(
            resolution=(120, 160),
            fps=5,
            num_frames=4,
            camera_paths=[CameraPathSpec(type="orbit", radius_m=2.0)],
            num_clips_per_path=2,
            scene_aware=True,
            collision_check=True,
        ),
    )
    work_dir = tmp_path / "outputs" / "test"
    work_dir.mkdir(parents=True)
    return config, fac, work_dir


def test_load_ply_means_numpy(sample_ply):
    means = _load_ply_means_numpy(sample_ply)
    assert means.shape == (100, 3)
    assert means.dtype == np.float32


def test_warmup_facility_creates_cache(warmup_config):
    config, fac, work_dir = warmup_config
    summary = warmup_facility(config, fac, work_dir)

    assert summary["warmup_complete"] is True
    assert summary["ply_loaded"] is True
    assert summary["num_gaussians"] == 100
    assert summary["num_clips"] > 0
    assert summary["elapsed_seconds"] >= 0

    # Cache manifest should exist
    cache = load_warmup_cache(work_dir)
    assert cache is not None
    assert cache["warmup_complete"] is True


def test_warmup_cache_clips_loadable(warmup_config):
    config, fac, work_dir = warmup_config
    warmup_facility(config, fac, work_dir)

    clips = load_cached_clips(work_dir)
    assert clips is not None
    assert len(clips) > 0

    clip = clips[0]
    assert "clip_name" in clip
    assert "poses" in clip
    assert "path_type" in clip

    # Poses should be deserializable
    poses = _deserialize_camera_poses(clip["poses"])
    assert len(poses) == config.render.num_frames
    assert poses[0].c2w.shape == (4, 4)


def test_warmup_occupancy_grid_cached(warmup_config):
    config, fac, work_dir = warmup_config
    summary = warmup_facility(config, fac, work_dir)

    grid_path = summary.get("occupancy_grid_path")
    assert grid_path is not None
    assert Path(grid_path).exists()

    # Should be loadable
    from blueprint_validation.warmup import load_cached_occupancy_grid

    grid = load_cached_occupancy_grid(Path(grid_path))
    assert grid.voxels is not None
    assert grid.voxel_size > 0


def test_warmup_missing_ply(tmp_path):
    fac = FacilityConfig(
        name="Missing PLY",
        ply_path=tmp_path / "nonexistent.ply",
    )
    config = ValidationConfig(
        project_name="Test",
        facilities={"test": fac},
    )
    work_dir = tmp_path / "out"
    work_dir.mkdir()
    summary = warmup_facility(config, fac, work_dir)
    assert summary["ply_loaded"] is False


def test_load_warmup_cache_none_when_missing(tmp_path):
    assert load_warmup_cache(tmp_path) is None


def test_load_cached_clips_none_when_missing(tmp_path):
    assert load_cached_clips(tmp_path) is None


def test_load_cached_variants_none_when_missing(tmp_path):
    assert load_cached_variants(tmp_path) is None


def test_camera_pose_roundtrip():
    from blueprint_validation.rendering.camera_paths import CameraPose

    pose = CameraPose(
        c2w=np.eye(4, dtype=np.float64),
        fx=320.0,
        fy=320.0,
        cx=80.0,
        cy=60.0,
        width=160,
        height=120,
    )
    serialized = _serialize_camera_poses([pose])
    roundtripped = _deserialize_camera_poses(serialized)
    assert len(roundtripped) == 1
    np.testing.assert_allclose(roundtripped[0].c2w, pose.c2w)
    assert roundtripped[0].fx == pose.fx
    assert roundtripped[0].width == pose.width


def test_warmup_scene_center_saved(warmup_config):
    config, fac, work_dir = warmup_config
    warmup_facility(config, fac, work_dir)

    center_path = work_dir / "warmup_cache" / "scene_center.npy"
    assert center_path.exists()
    center = np.load(str(center_path))
    assert center.shape == (3,)


def test_warmup_with_y_up_axis(sample_ply, tmp_path):
    """Warmup with up_axis='y' should produce a transformed scene center."""
    fac_z = FacilityConfig(
        name="Z-up",
        ply_path=sample_ply,
        up_axis="z",
    )
    fac_y = FacilityConfig(
        name="Y-up",
        ply_path=sample_ply,
        up_axis="y",
    )
    config = ValidationConfig(
        project_name="Transform Test",
        facilities={"test": fac_z},
        render=RenderConfig(
            resolution=(120, 160),
            fps=5,
            num_frames=4,
            camera_paths=[CameraPathSpec(type="orbit", radius_m=2.0)],
            num_clips_per_path=1,
        ),
    )

    work_z = tmp_path / "z_up"
    work_z.mkdir()
    summary_z = warmup_facility(config, fac_z, work_z)

    work_y = tmp_path / "y_up"
    work_y.mkdir()
    summary_y = warmup_facility(config, fac_y, work_y)

    center_z = np.array(summary_z["scene_center"])
    center_y = np.array(summary_y["scene_center"])

    # Y-up transform sends Y→+Z: center_y[2] should equal center_z[1]
    np.testing.assert_allclose(center_y[2], center_z[1], atol=1e-5)
    np.testing.assert_allclose(center_y[0], center_z[0], atol=1e-5)
    assert "scene_transform" in summary_z
    assert "scene_transform" in summary_y

    # Both should produce clips
    assert summary_y["num_clips"] > 0
    assert summary_y["warmup_complete"] is True


def test_load_ply_means_and_colors_sh_dc(sample_ply):
    """sample_ply has f_dc_0/1/2; colors should be extracted via sigmoid."""
    means, colors = load_ply_means_and_colors_numpy(sample_ply)
    assert means.shape == (100, 3)
    assert means.dtype == np.float32
    assert colors is not None
    assert colors.shape == (100, 3)
    assert colors.dtype == np.uint8
    # All values should be valid RGB (0-255)
    assert colors.min() >= 0
    assert colors.max() <= 255


def test_load_ply_means_and_colors_direct_rgb(sample_ply_with_rgb):
    """sample_ply_with_rgb has direct uint8 red/green/blue."""
    means, colors = load_ply_means_and_colors_numpy(sample_ply_with_rgb)
    assert means.shape == (32, 3)
    assert colors is not None
    assert colors.shape == (32, 3)
    assert colors.dtype == np.uint8


def test_warmup_auto_up_axis(sample_ply, tmp_path):
    """Warmup with up_axis='auto' should detect Z-up for the sample PLY (10x10x3 box)."""
    fac = FacilityConfig(
        name="Auto Detect",
        ply_path=sample_ply,
        up_axis="auto",
    )
    config = ValidationConfig(
        project_name="Auto Test",
        facilities={"test": fac},
        render=RenderConfig(
            resolution=(120, 160),
            fps=5,
            num_frames=4,
            camera_paths=[CameraPathSpec(type="orbit", radius_m=2.0)],
            num_clips_per_path=1,
        ),
    )
    work_dir = tmp_path / "auto_up"
    work_dir.mkdir()
    summary = warmup_facility(config, fac, work_dir)

    # Sample PLY is 10x10x3 → auto should detect Z as up
    assert summary["detected_up_axis"] == "z"
    assert summary["resolved_up_axis"] == "z"
    assert summary["warmup_complete"] is True
    assert summary["num_clips"] > 0
