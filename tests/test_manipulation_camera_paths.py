"""Tests for manipulation camera path generation."""

import numpy as np
import pytest


def test_generate_manipulation_arc_basic():
    from blueprint_validation.rendering.camera_paths import generate_manipulation_arc

    approach = np.array([2.0, 1.0, 0.8])
    poses = generate_manipulation_arc(
        approach_point=approach,
        arc_radius=0.5,
        height=0.6,
        num_frames=8,
    )
    assert len(poses) == 8


def test_manipulation_arc_height():
    from blueprint_validation.rendering.camera_paths import generate_manipulation_arc

    approach = np.array([0.0, 0.0, 0.0])
    height = 0.6
    poses = generate_manipulation_arc(
        approach_point=approach,
        arc_radius=0.4,
        height=height,
        num_frames=10,
    )
    for pose in poses:
        assert abs(pose.position[2] - height) < 0.01, (
            f"Camera z={pose.position[2]}, expected {height}"
        )


def test_manipulation_arc_radius():
    from blueprint_validation.rendering.camera_paths import generate_manipulation_arc

    approach = np.array([0.0, 0.0, 0.0])
    radius = 0.5
    poses = generate_manipulation_arc(
        approach_point=approach,
        arc_radius=radius,
        height=0.6,
        num_frames=10,
    )
    for pose in poses:
        dx = pose.position[0] - approach[0]
        dy = pose.position[1] - approach[1]
        dist_xy = np.sqrt(dx**2 + dy**2)
        assert abs(dist_xy - radius) < 0.01, (
            f"XY distance={dist_xy}, expected {radius}"
        )


def test_manipulation_arc_resolution():
    from blueprint_validation.rendering.camera_paths import generate_manipulation_arc

    approach = np.array([0.0, 0.0, 0.0])
    poses = generate_manipulation_arc(
        approach_point=approach,
        arc_radius=0.4,
        height=0.6,
        num_frames=5,
        resolution=(240, 320),
    )
    for pose in poses:
        assert pose.width == 320
        assert pose.height == 240


def test_generate_path_from_spec_manipulation():
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.rendering.camera_paths import generate_path_from_spec

    spec = CameraPathSpec(
        type="manipulation",
        height_override_m=0.6,
        look_down_override_deg=45.0,
        approach_point=[1.0, 2.0, 0.5],
        arc_radius_m=0.3,
    )
    poses = generate_path_from_spec(
        spec=spec,
        scene_center=np.array([0.0, 0.0, 0.0]),
        num_frames=6,
        camera_height=1.2,
        look_down_deg=15.0,
        resolution=(120, 160),
    )
    assert len(poses) == 6
    # Should use the override height, not the default
    for pose in poses:
        assert abs(pose.position[2] - 0.6) < 0.01
