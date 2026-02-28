"""Tests for scene orientation transform (up-axis correction)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from blueprint_validation.config import CameraPathSpec, FacilityConfig
from blueprint_validation.rendering.scene_geometry import (
    OrientedBoundingBox,
    compute_scene_transform,
    detect_up_axis,
    is_identity_transform,
    transform_camera_path_specs,
    transform_c2w,
    transform_means,
    transform_obbs,
)


def _make_facility(up_axis: str = "z", rotation_deg: list | None = None) -> FacilityConfig:
    return FacilityConfig(
        name="Test",
        ply_path=Path("/tmp/test.ply"),
        up_axis=up_axis,
        scene_rotation_deg=rotation_deg or [0.0, 0.0, 0.0],
    )


class TestComputeSceneTransform:
    def test_identity_for_z_up(self):
        T = compute_scene_transform(_make_facility("z"))
        assert is_identity_transform(T)

    def test_identity_for_plus_z(self):
        T = compute_scene_transform(_make_facility("+z"))
        assert is_identity_transform(T)

    def test_y_up_sends_y_to_z(self):
        T = compute_scene_transform(_make_facility("y"))
        R = T[:3, :3]
        y_axis = np.array([0.0, 1.0, 0.0])
        result = R @ y_axis
        np.testing.assert_allclose(result, [0, 0, 1], atol=1e-10)

    def test_y_up_preserves_x(self):
        T = compute_scene_transform(_make_facility("y"))
        R = T[:3, :3]
        x_axis = np.array([1.0, 0.0, 0.0])
        result = R @ x_axis
        np.testing.assert_allclose(result, [1, 0, 0], atol=1e-10)

    def test_neg_y_up_sends_neg_y_to_z(self):
        T = compute_scene_transform(_make_facility("-y"))
        R = T[:3, :3]
        neg_y = np.array([0.0, -1.0, 0.0])
        result = R @ neg_y
        np.testing.assert_allclose(result, [0, 0, 1], atol=1e-10)

    def test_neg_z_flips_upside_down(self):
        T = compute_scene_transform(_make_facility("-z"))
        R = T[:3, :3]
        neg_z = np.array([0.0, 0.0, -1.0])
        result = R @ neg_z
        np.testing.assert_allclose(result, [0, 0, 1], atol=1e-10)

    def test_x_up_sends_x_to_z(self):
        T = compute_scene_transform(_make_facility("x"))
        R = T[:3, :3]
        x_axis = np.array([1.0, 0.0, 0.0])
        result = R @ x_axis
        np.testing.assert_allclose(result, [0, 0, 1], atol=1e-10)

    def test_neg_x_up_sends_neg_x_to_z(self):
        T = compute_scene_transform(_make_facility("-x"))
        R = T[:3, :3]
        neg_x = np.array([-1.0, 0.0, 0.0])
        result = R @ neg_x
        np.testing.assert_allclose(result, [0, 0, 1], atol=1e-10)

    def test_invalid_up_axis_raises(self):
        with pytest.raises(ValueError, match="Unknown up_axis"):
            compute_scene_transform(_make_facility("q"))

    def test_scene_rotation_deg_applied(self):
        # 90° around Z should rotate X→Y
        T = compute_scene_transform(_make_facility("z", [0, 0, 90]))
        R = T[:3, :3]
        x_axis = np.array([1.0, 0.0, 0.0])
        result = R @ x_axis
        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-10)

    def test_rotation_composes_with_up_axis(self):
        # Y-up + 180° around Z
        T = compute_scene_transform(_make_facility("y", [0, 0, 180]))
        R = T[:3, :3]
        # Y-up maps (0,1,0)→(0,0,1); then 180° Z sends X→-X, Y→-Y
        # So (0,1,0) with Y-up gives (0,0,1), then 180° Z leaves Z alone
        y_axis = np.array([0.0, 1.0, 0.0])
        result = R @ y_axis
        np.testing.assert_allclose(result, [0, 0, 1], atol=1e-10)

    def test_is_not_identity_for_y_up(self):
        T = compute_scene_transform(_make_facility("y"))
        assert not is_identity_transform(T)

    def test_rotation_matrix_is_orthogonal(self):
        for up in ["z", "y", "-y", "-z", "x", "-x"]:
            T = compute_scene_transform(_make_facility(up))
            R = T[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


class TestTransformMeans:
    def test_identity_preserves_means(self):
        means = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        T = np.eye(4)
        result = transform_means(means, T)
        np.testing.assert_allclose(result, means, atol=1e-6)

    def test_y_up_transforms_positions(self):
        # Point at (0, 5, 0) should go to (0, 0, 5) with Y-up
        means = np.array([[0.0, 5.0, 0.0]], dtype=np.float32)
        T = compute_scene_transform(_make_facility("y"))
        result = transform_means(means, T)
        np.testing.assert_allclose(result, [[0, 0, 5]], atol=1e-6)

    def test_roundtrip_with_inverse(self):
        means = np.random.default_rng(42).uniform(-5, 5, (50, 3)).astype(np.float32)
        T = compute_scene_transform(_make_facility("y", [10, 20, 30]))
        transformed = transform_means(means, T)
        # Inverse: use transpose of rotation
        T_inv = np.eye(4)
        T_inv[:3, :3] = T[:3, :3].T
        restored = transform_means(transformed, T_inv)
        np.testing.assert_allclose(restored, means, atol=1e-5)


class TestTransformC2W:
    def test_identity_preserves_c2w(self):
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 3] = [1, 2, 3]
        T = np.eye(4)
        result = transform_c2w(c2w, T)
        np.testing.assert_allclose(result, c2w, atol=1e-10)

    def test_y_up_converts_camera_back(self):
        # Camera at Z=1.2 in corrected (Z-up) frame
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 3] = [2.0, 3.0, 1.2]

        T = compute_scene_transform(_make_facility("y"))
        c2w_orig = transform_c2w(c2w, T)

        # In original Y-up frame, the camera should be at Y=1.2
        np.testing.assert_allclose(c2w_orig[1, 3], 1.2, atol=1e-10)

    def test_c2w_transform_roundtrip(self):
        # Generate a random c2w in corrected frame, convert to original, convert back
        T = compute_scene_transform(_make_facility("y", [15, 0, 0]))
        T_inv = np.eye(4)
        T_inv[:3, :3] = T[:3, :3].T

        c2w_corrected = np.eye(4, dtype=np.float64)
        c2w_corrected[:3, 3] = [1, 2, 3]
        # Add some rotation to the camera
        angle = math.radians(30)
        c2w_corrected[0, 0] = math.cos(angle)
        c2w_corrected[0, 1] = -math.sin(angle)
        c2w_corrected[1, 0] = math.sin(angle)
        c2w_corrected[1, 1] = math.cos(angle)

        c2w_orig = transform_c2w(c2w_corrected, T)
        # Convert back: c2w_corrected = T @ c2w_orig
        c2w_back = T @ c2w_orig
        np.testing.assert_allclose(c2w_back, c2w_corrected, atol=1e-10)

    def test_position_transform_consistency(self):
        """Camera position in original frame should match transformed mean position."""
        T = compute_scene_transform(_make_facility("y"))

        # A point at (0, 5, 0) in original frame → (0, 0, 5) in corrected frame
        means_orig = np.array([[0.0, 5.0, 0.0]])
        means_corr = transform_means(means_orig, T)
        np.testing.assert_allclose(means_corr, [[0, 0, 5]], atol=1e-10)

        # A camera at (0, 0, 5) in corrected frame → should be near (0, 5, 0) in original
        c2w_corr = np.eye(4, dtype=np.float64)
        c2w_corr[:3, 3] = [0, 0, 5]
        c2w_orig = transform_c2w(c2w_corr, T)
        np.testing.assert_allclose(c2w_orig[:3, 3], [0, 5, 0], atol=1e-10)


def test_transform_camera_path_specs_rotates_approach_points():
    T = compute_scene_transform(_make_facility("y"))
    specs = [
        CameraPathSpec(type="manipulation", approach_point=[0.0, 2.0, 0.0], arc_radius_m=0.5),
        CameraPathSpec(type="orbit", radius_m=2.0),
    ]
    transformed = transform_camera_path_specs(specs, T)
    np.testing.assert_allclose(transformed[0].approach_point, [0.0, 0.0, 2.0], atol=1e-10)
    assert transformed[1].approach_point is None


def test_transform_obbs_rotates_center_and_axes():
    T = compute_scene_transform(_make_facility("y"))
    obb = OrientedBoundingBox(
        instance_id="box_1",
        label="box",
        center=np.array([0.0, 3.0, 0.0], dtype=np.float64),
        extents=np.array([0.2, 0.2, 0.2], dtype=np.float64),
        axes=np.eye(3, dtype=np.float64),
    )
    transformed = transform_obbs([obb], T)[0]
    np.testing.assert_allclose(transformed.center, [0.0, 0.0, 3.0], atol=1e-10)
    # Local +Y axis should now align with global +Z after Y-up correction.
    np.testing.assert_allclose(transformed.axes[:, 1], [0.0, 0.0, 1.0], atol=1e-10)


class TestDetectUpAxis:
    """Tests for automatic up-axis detection from point cloud extents."""

    def test_y_up_room(self):
        """Room-like cloud: wide X/Z, narrow Y → detects Y as up."""
        rng = np.random.default_rng(0)
        # X: 0-10, Y: 0-3 (ceiling height), Z: 0-8
        means = rng.uniform([0, 0, 0], [10, 3, 8], size=(500, 3)).astype(np.float32)
        assert detect_up_axis(means) == "y"

    def test_z_up_room(self):
        """Room-like cloud: wide X/Y, narrow Z → detects Z as up."""
        rng = np.random.default_rng(1)
        means = rng.uniform([0, 0, 0], [10, 8, 3], size=(500, 3)).astype(np.float32)
        assert detect_up_axis(means) == "z"

    def test_x_up_room(self):
        """Room-like cloud: narrow X, wide Y/Z → detects X as up."""
        rng = np.random.default_rng(2)
        means = rng.uniform([0, 0, 0], [3, 10, 8], size=(500, 3)).astype(np.float32)
        assert detect_up_axis(means) == "x"

    def test_cubical_defaults_to_z(self):
        """Roughly equal extents → ambiguous, defaults to Z."""
        rng = np.random.default_rng(3)
        means = rng.uniform([0, 0, 0], [5, 5, 5], size=(500, 3)).astype(np.float32)
        assert detect_up_axis(means) == "z"

    def test_too_few_points_defaults_to_z(self):
        means = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        assert detect_up_axis(means) == "z"

    def test_flat_scene_defaults_to_z(self):
        """All points on a plane (zero extent on one axis)."""
        rng = np.random.default_rng(4)
        means = rng.uniform([0, 0, 0], [10, 8, 0], size=(100, 3)).astype(np.float32)
        means[:, 2] = 0.0  # perfectly flat
        assert detect_up_axis(means) == "z"

    def test_custom_confidence_ratio(self):
        """With a strict ratio, borderline cases default to Z."""
        rng = np.random.default_rng(5)
        # Y extent = 4, X = 5, Z = 6 → ratio = 5/4 = 1.25
        means = rng.uniform([0, 0, 0], [5, 4, 6], size=(500, 3)).astype(np.float32)
        # Default ratio 1.5 → ambiguous, returns "z"
        assert detect_up_axis(means, confidence_ratio=1.5) == "z"
        # Lower ratio → confident enough to pick "y"
        assert detect_up_axis(means, confidence_ratio=1.1) == "y"

    def test_outlier_robustness(self):
        """A few extreme outlier Gaussians should not change the detected axis."""
        rng = np.random.default_rng(10)
        # Room: X=10, Y=3, Z=8 → Y is up
        means = rng.uniform([0, 0, 0], [10, 3, 8], size=(500, 3)).astype(np.float32)
        # Add outliers that stretch Y to 100
        outliers = np.array([[5, 100, 4], [5, -80, 4]], dtype=np.float32)
        noisy = np.vstack([means, outliers])
        # Without robust extents, raw Y extent = 180 → would pick X or Z.
        # With percentile trimming, Y is still ~3.
        assert detect_up_axis(noisy) == "y"

    def test_negative_sign_detection(self):
        """When density is higher near the top of the up axis, returns negative."""
        rng = np.random.default_rng(20)
        # Build a room where the "floor" is at Y=3 (top of Y range).
        # Normal room points: sparse bottom, dense floor at top
        sparse_body = rng.uniform([0, 0, 0], [10, 3, 8], size=(200, 3)).astype(np.float32)
        # Dense floor slab at Y ~ 2.5-3.0 (top of Y range)
        floor_slab = rng.uniform([0, 2.5, 0], [10, 3.0, 8], size=(500, 3)).astype(np.float32)
        means = np.vstack([sparse_body, floor_slab])
        result = detect_up_axis(means)
        assert result == "-y"

    def test_positive_sign_normal_room(self):
        """Floor at bottom of axis → positive direction."""
        rng = np.random.default_rng(30)
        sparse_body = rng.uniform([0, 0, 0], [10, 3, 8], size=(200, 3)).astype(np.float32)
        # Dense floor slab at Y ~ 0-0.5 (bottom of Y range)
        floor_slab = rng.uniform([0, 0, 0], [10, 0.5, 8], size=(500, 3)).astype(np.float32)
        means = np.vstack([sparse_body, floor_slab])
        result = detect_up_axis(means)
        assert result == "y"

    def test_auto_facility_warns_on_direct_transform(self):
        """compute_scene_transform with up_axis='auto' warns and defaults to Z."""
        fac = _make_facility("auto")
        T = compute_scene_transform(fac)
        assert is_identity_transform(T)
