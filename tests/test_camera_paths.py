"""Tests for camera path generation."""

import json

import numpy as np


def test_generate_orbit():
    from blueprint_validation.rendering.camera_paths import generate_orbit

    poses = generate_orbit(
        center=np.array([0.0, 0.0, 0.0]),
        radius=3.0,
        height=1.2,
        num_frames=10,
    )
    assert len(poses) == 10
    # All positions should be at the specified height
    for pose in poses:
        assert abs(pose.position[2] - 1.2) < 0.01
    # First and last positions should be close to radius distance from center
    for pose in poses:
        dist = np.sqrt(pose.position[0] ** 2 + pose.position[1] ** 2)
        assert abs(dist - 3.0) < 0.01


def test_generate_sweep():
    from blueprint_validation.rendering.camera_paths import generate_sweep

    poses = generate_sweep(
        start=np.array([0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        length=5.0,
        height=1.0,
        num_frames=10,
    )
    assert len(poses) == 10
    # All at specified height
    for pose in poses:
        assert abs(pose.position[2] - 1.0) < 0.01
    # Should progress along x-axis
    assert poses[-1].position[0] > poses[0].position[0]


def test_save_and_load_path(tmp_path):
    from blueprint_validation.rendering.camera_paths import (
        generate_orbit,
        load_path_from_json,
        save_path_to_json,
    )

    original = generate_orbit(
        center=np.array([0.0, 0.0, 0.0]),
        radius=2.0,
        height=1.0,
        num_frames=5,
    )

    path_file = tmp_path / "test_path.json"
    save_path_to_json(original, path_file)

    assert path_file.exists()
    data = json.loads(path_file.read_text())
    assert len(data["camera_path"]) == 5

    loaded = load_path_from_json(path_file)
    assert len(loaded) == 5


def test_camera_pose_viewmat():
    from blueprint_validation.rendering.camera_paths import CameraPose

    c2w = np.eye(4)
    pose = CameraPose(c2w=c2w, fx=500, fy=500, cx=320, cy=240, width=640, height=480)

    viewmat = pose.viewmat()
    assert viewmat.shape == (4, 4)

    K = pose.K()
    assert K.shape == (3, 3)
    assert float(K[0, 0]) == 500


def test_generate_path_from_spec():
    from blueprint_validation.config import CameraPathSpec
    from blueprint_validation.rendering.camera_paths import generate_path_from_spec

    spec = CameraPathSpec(type="orbit", radius_m=2.0, num_orbits=1)
    poses = generate_path_from_spec(
        spec=spec,
        scene_center=np.array([0.0, 0.0, 0.0]),
        num_frames=10,
        camera_height=1.0,
        look_down_deg=10.0,
        resolution=(120, 160),
    )
    assert len(poses) == 10
    assert poses[0].width == 160
    assert poses[0].height == 120
