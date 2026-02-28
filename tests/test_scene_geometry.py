"""Tests for scene-aware camera placement: scene_geometry.py."""

from __future__ import annotations

import json

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# OBB loading from task_targets.json
# ---------------------------------------------------------------------------


def _make_task_targets(
    manipulation_candidates: list | None = None,
    articulation_hints: list | None = None,
    navigation_hints: list | None = None,
) -> dict:
    """Build a minimal task_targets.json-style dict."""
    data: dict = {}
    if manipulation_candidates is not None:
        data["manipulation_candidates"] = manipulation_candidates
    if articulation_hints is not None:
        data["articulation_hints"] = articulation_hints
    if navigation_hints is not None:
        data["navigation_hints"] = navigation_hints
    return data


def _make_candidate(label: str, center, extents, axes=None, **kwargs) -> dict:
    entry: dict = {"label": label, "instance_id": f"id_{label}"}
    bbox: dict = {"center": list(center), "extents": list(extents)}
    if axes is not None:
        bbox["axes"] = [list(row) for row in axes]
    entry["boundingBox"] = bbox
    entry.update(kwargs)
    return entry


def test_load_obbs_from_task_targets(tmp_path):
    from blueprint_validation.rendering.scene_geometry import load_obbs_from_task_targets

    data = _make_task_targets(
        manipulation_candidates=[
            _make_candidate("tote", [1.0, 2.0, 0.5], [0.3, 0.4, 0.2]),
            _make_candidate("bin", [3.0, 0.0, 0.8], [0.5, 0.5, 0.3]),
        ],
        articulation_hints=[
            _make_candidate("door", [0.0, 5.0, 1.0], [0.8, 0.1, 2.0]),
        ],
    )
    path = tmp_path / "task_targets.json"
    path.write_text(json.dumps(data))

    obbs = load_obbs_from_task_targets(path)
    assert len(obbs) == 3

    labels = [o.label for o in obbs]
    assert "tote" in labels
    assert "bin" in labels
    assert "door" in labels

    manip = [o for o in obbs if o.category == "manipulation"]
    artic = [o for o in obbs if o.category == "articulation"]
    assert len(manip) == 2
    assert len(artic) == 1

    tote = next(o for o in obbs if o.label == "tote")
    np.testing.assert_allclose(tote.center, [1.0, 2.0, 0.5])
    np.testing.assert_allclose(tote.extents, [0.3, 0.4, 0.2])


def test_load_obbs_includes_navigation_hints(tmp_path):
    from blueprint_validation.rendering.scene_geometry import load_obbs_from_task_targets

    data = _make_task_targets(
        navigation_hints=[
            _make_candidate("charging_station", [2.0, 1.0, 0.0], [1.0, 1.0, 0.2]),
        ]
    )
    path = tmp_path / "task_targets.json"
    path.write_text(json.dumps(data))

    obbs = load_obbs_from_task_targets(path)
    assert len(obbs) == 1
    assert obbs[0].category == "navigation"
    assert obbs[0].label == "charging_station"


def test_load_obbs_missing_bbox(tmp_path):
    from blueprint_validation.rendering.scene_geometry import load_obbs_from_task_targets

    data = _make_task_targets(
        manipulation_candidates=[
            {"label": "broken_entry"},  # no boundingBox
            _make_candidate("valid", [1.0, 0.0, 0.5], [0.2, 0.2, 0.2]),
        ],
    )
    path = tmp_path / "task_targets.json"
    path.write_text(json.dumps(data))

    obbs = load_obbs_from_task_targets(path)
    assert len(obbs) == 1
    assert obbs[0].label == "valid"


def test_load_obbs_axes_identity_default(tmp_path):
    from blueprint_validation.rendering.scene_geometry import load_obbs_from_task_targets

    data = _make_task_targets(
        manipulation_candidates=[
            _make_candidate("box", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ],
    )
    path = tmp_path / "task_targets.json"
    path.write_text(json.dumps(data))

    obbs = load_obbs_from_task_targets(path)
    np.testing.assert_allclose(obbs[0].axes, np.eye(3))


# ---------------------------------------------------------------------------
# Occupancy grid
# ---------------------------------------------------------------------------


def test_build_occupancy_grid():
    from blueprint_validation.rendering.scene_geometry import build_occupancy_grid

    # Dense cluster of points at (5, 5, 1)
    rng = np.random.default_rng(0)
    cluster = rng.normal(loc=[5.0, 5.0, 1.0], scale=0.02, size=(50, 3))
    # Sparse scatter elsewhere
    sparse = rng.uniform(low=-2, high=2, size=(10, 3))

    means = np.vstack([cluster, sparse])
    grid = build_occupancy_grid(means, voxel_size=0.1, density_threshold=3)

    # Cluster region should be occupied
    assert grid.is_occupied(np.array([5.0, 5.0, 1.0]))
    # Far away should be free
    assert not grid.is_occupied(np.array([-10.0, -10.0, -10.0]))


def test_occupancy_density_threshold():
    from blueprint_validation.rendering.scene_geometry import build_occupancy_grid

    # Only 2 points per voxel — should be below threshold=3
    means = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    grid = build_occupancy_grid(means, voxel_size=0.5, density_threshold=3)
    assert not grid.is_occupied(np.array([0.0, 0.0, 0.0]))

    # With threshold=1, same points should be occupied
    grid2 = build_occupancy_grid(means, voxel_size=0.5, density_threshold=1)
    assert grid2.is_occupied(np.array([0.0, 0.0, 0.0]))


def test_is_free_with_clearance():
    from blueprint_validation.rendering.scene_geometry import build_occupancy_grid

    # Create a dense wall at x=0 (many points to exceed density threshold)
    rng = np.random.default_rng(42)
    wall = np.column_stack(
        [
            rng.uniform(-0.05, 0.05, 5000),
            rng.uniform(-1, 1, 5000),
            rng.uniform(0, 2, 5000),
        ]
    )
    grid = build_occupancy_grid(wall, voxel_size=0.1, density_threshold=2)

    # Right at the wall should not be free with clearance
    assert not grid.is_free(np.array([0.0, 0.0, 1.0]), min_clearance_m=0.15)
    # Far from the wall should be free
    assert grid.is_free(np.array([5.0, 0.0, 1.0]), min_clearance_m=0.15)


# ---------------------------------------------------------------------------
# Standoff / height computation
# ---------------------------------------------------------------------------


def test_compute_standoff_distance_small():
    from blueprint_validation.rendering.scene_geometry import (
        OrientedBoundingBox,
        compute_standoff_distance,
    )

    obb = OrientedBoundingBox(
        instance_id="small",
        label="cup",
        center=np.array([0.0, 0.0, 0.5]),
        extents=np.array([0.1, 0.1, 0.15]),
        axes=np.eye(3),
    )
    d = compute_standoff_distance(obb)
    assert d == pytest.approx(0.6, abs=0.01)  # base minimum


def test_compute_standoff_distance_large():
    from blueprint_validation.rendering.scene_geometry import (
        OrientedBoundingBox,
        compute_standoff_distance,
    )

    obb = OrientedBoundingBox(
        instance_id="shelf",
        label="shelf",
        center=np.array([0.0, 0.0, 1.0]),
        extents=np.array([2.0, 1.5, 2.5]),
        axes=np.eye(3),
    )
    d = compute_standoff_distance(obb)
    assert d == pytest.approx(3.0, abs=0.01)  # clamped to max


def test_compute_camera_height():
    from blueprint_validation.rendering.scene_geometry import (
        OrientedBoundingBox,
        compute_camera_height,
    )

    # Floor object
    floor_obb = OrientedBoundingBox(
        instance_id="floor",
        label="tote",
        center=np.array([0.0, 0.0, 0.0]),
        extents=np.array([0.3, 0.3, 0.2]),
        axes=np.eye(3),
    )
    h = compute_camera_height(floor_obb)
    assert h == pytest.approx(0.4, abs=0.01)  # min clamp

    # Shelf-level object
    shelf_obb = OrientedBoundingBox(
        instance_id="shelf",
        label="box",
        center=np.array([0.0, 0.0, 1.2]),
        extents=np.array([0.3, 0.3, 0.3]),
        axes=np.eye(3),
    )
    h2 = compute_camera_height(shelf_obb)
    assert h2 == pytest.approx(1.5, abs=0.01)


# ---------------------------------------------------------------------------
# Scene-aware spec generation
# ---------------------------------------------------------------------------


def test_generate_scene_aware_specs():
    from blueprint_validation.rendering.scene_geometry import (
        OrientedBoundingBox,
        generate_scene_aware_specs,
    )

    obbs = [
        OrientedBoundingBox(
            instance_id="tote_1",
            label="tote",
            center=np.array([1.0, 0.0, 0.5]),
            extents=np.array([0.3, 0.3, 0.2]),
            axes=np.eye(3),
            category="manipulation",
        ),
        OrientedBoundingBox(
            instance_id="bin_1",
            label="bin",
            center=np.array([2.0, 1.0, 0.8]),
            extents=np.array([0.4, 0.4, 0.3]),
            axes=np.eye(3),
            category="manipulation",
        ),
        OrientedBoundingBox(
            instance_id="door_1",
            label="door",
            center=np.array([0.0, 5.0, 1.0]),
            extents=np.array([0.8, 0.1, 2.0]),
            axes=np.eye(3),
            category="articulation",
        ),
    ]

    specs = generate_scene_aware_specs(obbs)
    assert len(specs) == 3

    manip_specs = [s for s in specs if s.type == "manipulation"]
    orbit_specs = [s for s in specs if s.type == "orbit"]
    assert len(manip_specs) == 2
    assert len(orbit_specs) == 1

    # Manipulation specs should have approach_point set
    for s in manip_specs:
        assert s.approach_point is not None
        assert len(s.approach_point) == 3

    # Articulation spec should use orbit
    assert orbit_specs[0].radius_m > 0


# ---------------------------------------------------------------------------
# Collision avoidance
# ---------------------------------------------------------------------------


def test_nudge_to_free_space():
    from blueprint_validation.rendering.scene_geometry import (
        build_occupancy_grid,
        nudge_to_free_space,
    )

    # Dense wall at x=0
    rng = np.random.default_rng(7)
    wall = np.column_stack(
        [
            rng.uniform(-0.05, 0.05, 200),
            rng.uniform(-3, 3, 200),
            rng.uniform(0, 3, 200),
        ]
    )
    grid = build_occupancy_grid(wall, voxel_size=0.1, density_threshold=2)

    # Camera inside the wall, target behind it
    pos = np.array([0.0, 0.0, 1.0])
    target = np.array([-2.0, 0.0, 1.0])

    nudged = nudge_to_free_space(pos, target, grid, min_clearance_m=0.15)
    assert nudged is not None
    # Should have been pushed away from target (positive x direction)
    assert nudged[0] > pos[0]


def test_filter_and_fix_poses():
    from blueprint_validation.rendering.camera_paths import CameraPose
    from blueprint_validation.rendering.scene_geometry import (
        build_occupancy_grid,
        filter_and_fix_poses,
    )

    # Wall at x=0
    rng = np.random.default_rng(99)
    wall = np.column_stack(
        [
            rng.uniform(-0.05, 0.05, 200),
            rng.uniform(-3, 3, 200),
            rng.uniform(0, 3, 200),
        ]
    )
    grid = build_occupancy_grid(wall, voxel_size=0.1, density_threshold=2)

    # One pose in free space, one inside wall
    free_c2w = np.eye(4)
    free_c2w[:3, 3] = [5.0, 0.0, 1.0]
    wall_c2w = np.eye(4)
    wall_c2w[:3, 3] = [0.0, 0.0, 1.0]

    poses = [
        CameraPose(c2w=free_c2w, fx=500, fy=500, cx=320, cy=240, width=640, height=480),
        CameraPose(c2w=wall_c2w, fx=500, fy=500, cx=320, cy=240, width=640, height=480),
    ]

    target = np.array([0.0, 0.0, 0.0])
    result = filter_and_fix_poses(poses, grid, target, min_clearance_m=0.15)

    # The free pose should survive; the wall pose may be nudged or dropped
    assert len(result) >= 1
    # First pose should be unchanged (it was free)
    np.testing.assert_allclose(result[0].position, [5.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Auto-populate manipulation zones
# ---------------------------------------------------------------------------


def test_auto_populate_zones_from_obbs():
    from blueprint_validation.rendering.scene_geometry import (
        OrientedBoundingBox,
        auto_populate_manipulation_zones,
    )

    obbs = [
        OrientedBoundingBox(
            instance_id="tote_1",
            label="tote",
            center=np.array([1.0, 2.0, 0.5]),
            extents=np.array([0.3, 0.3, 0.2]),
            axes=np.eye(3),
            category="manipulation",
        ),
        OrientedBoundingBox(
            instance_id="door_1",
            label="door",
            center=np.array([0.0, 5.0, 1.0]),
            extents=np.array([0.8, 0.1, 2.0]),
            axes=np.eye(3),
            category="articulation",
        ),
    ]

    # Empty zones -> should auto-populate from manipulation OBBs only
    zones = auto_populate_manipulation_zones([], obbs)
    assert len(zones) == 1
    assert "tote" in zones[0].name


def test_auto_populate_preserves_existing_zones():
    from blueprint_validation.config import ManipulationZoneConfig
    from blueprint_validation.rendering.scene_geometry import (
        OrientedBoundingBox,
        auto_populate_manipulation_zones,
    )

    existing = [
        ManipulationZoneConfig(
            name="my_zone",
            approach_point=[1.0, 0.0, 0.8],
            target_point=[1.0, 0.5, 0.8],
        )
    ]
    obbs = [
        OrientedBoundingBox(
            instance_id="box",
            label="box",
            center=np.array([5.0, 5.0, 0.5]),
            extents=np.array([0.3, 0.3, 0.3]),
            axes=np.eye(3),
            category="manipulation",
        ),
    ]

    result = auto_populate_manipulation_zones(existing, obbs)
    # Should return existing zones unchanged
    assert len(result) == 1
    assert result[0].name == "my_zone"
