"""Scene-aware camera placement using OBBs from task_targets.json and occupancy grids."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..common import get_logger
from ..config import CameraPathSpec, ManipulationZoneConfig

logger = get_logger("rendering.scene_geometry")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OrientedBoundingBox:
    """An oriented bounding box parsed from task_targets.json."""

    instance_id: str
    label: str
    center: np.ndarray  # (3,)
    extents: np.ndarray  # (3,) full widths (not half-widths)
    axes: np.ndarray  # (3, 3) — columns are local-frame axes
    confidence: float = 1.0
    category: str = "manipulation"  # "manipulation" or "articulation"


@dataclass
class OccupancyGrid:
    """Voxelized occupancy grid built from Gaussian splat point positions."""

    voxels: np.ndarray  # (Nx, Ny, Nz) uint8, 1 = occupied
    origin: np.ndarray  # (3,) world-space corner of voxel (0,0,0)
    voxel_size: float
    shape: Tuple[int, int, int]

    def _to_index(self, point: np.ndarray) -> np.ndarray:
        return ((point - self.origin) / self.voxel_size).astype(int)

    def _in_bounds(self, idx: np.ndarray) -> bool:
        return bool(np.all(idx >= 0) and np.all(idx < np.array(self.shape)))

    def is_occupied(self, point: np.ndarray) -> bool:
        idx = self._to_index(point)
        if not self._in_bounds(idx):
            return False
        return bool(self.voxels[idx[0], idx[1], idx[2]])

    def is_free(self, point: np.ndarray, min_clearance_m: float = 0.15) -> bool:
        """Check that the point and its neighborhood are unoccupied."""
        idx = self._to_index(point)
        steps = max(1, int(np.ceil(min_clearance_m / self.voxel_size)))
        lo = np.maximum(idx - steps, 0)
        hi = np.minimum(idx + steps + 1, np.array(self.shape))
        if np.any(lo >= hi):
            return True  # outside grid = free
        region = self.voxels[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        return not bool(np.any(region))


# ---------------------------------------------------------------------------
# OBB loading from task_targets.json
# ---------------------------------------------------------------------------

_IDENTITY_AXES = np.eye(3, dtype=np.float64)


def load_obbs_from_task_targets(path: Path) -> List[OrientedBoundingBox]:
    """Parse task_targets.json and extract OBBs from manipulation_candidates + articulation_hints."""
    with open(path) as f:
        data = json.load(f)

    obbs: List[OrientedBoundingBox] = []

    for category, key in [
        ("manipulation", "manipulation_candidates"),
        ("articulation", "articulation_hints"),
    ]:
        for entry in data.get(key, []):
            bbox = entry.get("boundingBox") or entry.get("obb")
            if bbox is None:
                logger.warning(
                    "Skipping %s entry '%s' — no boundingBox/obb field",
                    category,
                    entry.get("label", entry.get("instance_id", "?")),
                )
                continue

            center = np.array(bbox["center"], dtype=np.float64)
            extents = np.array(bbox["extents"], dtype=np.float64)

            axes_raw = bbox.get("axes")
            if axes_raw is not None:
                axes = np.array(axes_raw, dtype=np.float64)
            else:
                axes = _IDENTITY_AXES.copy()

            obbs.append(
                OrientedBoundingBox(
                    instance_id=entry.get("instance_id", entry.get("label", "unknown")),
                    label=entry.get("label", "unknown"),
                    center=center,
                    extents=extents,
                    axes=axes,
                    confidence=float(entry.get("confidence", 1.0)),
                    category=category,
                )
            )

    logger.info(
        "Loaded %d OBBs from %s (manipulation=%d, articulation=%d)",
        len(obbs),
        path,
        sum(1 for o in obbs if o.category == "manipulation"),
        sum(1 for o in obbs if o.category == "articulation"),
    )
    return obbs


# ---------------------------------------------------------------------------
# Occupancy grid construction
# ---------------------------------------------------------------------------

_MAX_GRID_CELLS = 200_000_000


def build_occupancy_grid(
    means: np.ndarray,
    voxel_size: float = 0.1,
    padding: float = 1.0,
    density_threshold: int = 3,
) -> OccupancyGrid:
    """Voxelize Gaussian splat point positions into an occupancy grid.

    Args:
        means: (N, 3) array of Gaussian center positions.
        voxel_size: Edge length of each voxel in meters.
        padding: Meters of padding around the scene AABB.
        density_threshold: Minimum point count per voxel to mark it occupied.
    """
    mins = means.min(axis=0) - padding
    maxs = means.max(axis=0) + padding

    grid_shape = np.ceil((maxs - mins) / voxel_size).astype(int)
    grid_shape = np.maximum(grid_shape, 1)

    # Auto-coarsen if grid would be too large
    if int(np.prod(grid_shape)) > _MAX_GRID_CELLS:
        extent = (maxs - mins).max()
        voxel_size = float(extent / 500.0)
        grid_shape = np.ceil((maxs - mins) / voxel_size).astype(int)
        grid_shape = np.maximum(grid_shape, 1)
        logger.warning("Auto-coarsened voxel size to %.3f m (grid %s)", voxel_size, grid_shape)

    # Scatter-count points into voxels
    indices = ((means - mins) / voxel_size).astype(int)
    indices = np.clip(indices, 0, grid_shape - 1)

    counts = np.zeros(tuple(grid_shape), dtype=np.int32)
    np.add.at(counts, (indices[:, 0], indices[:, 1], indices[:, 2]), 1)

    occupied = (counts >= density_threshold).astype(np.uint8)

    logger.info(
        "Occupancy grid: shape=%s, voxel=%.2fm, occupied=%d/%d voxels (%.1f%%)",
        tuple(grid_shape),
        voxel_size,
        int(occupied.sum()),
        int(np.prod(grid_shape)),
        100.0 * occupied.sum() / max(np.prod(grid_shape), 1),
    )
    return OccupancyGrid(
        voxels=occupied,
        origin=mins,
        voxel_size=voxel_size,
        shape=tuple(int(s) for s in grid_shape),
    )


# ---------------------------------------------------------------------------
# Standoff / height computation
# ---------------------------------------------------------------------------


def compute_standoff_distance(obb: OrientedBoundingBox, base_standoff: float = 0.6) -> float:
    """Camera standoff distance that scales with object size.

    Larger objects need the camera further back for proper framing.
    """
    max_xy = float(np.max(np.abs(obb.extents[:2])))
    return min(max(base_standoff, max_xy * 1.5), 3.0)


def compute_camera_height(obb: OrientedBoundingBox, offset_m: float = 0.3) -> float:
    """Camera height derived from object center z-coordinate."""
    return max(0.4, float(obb.center[2]) + offset_m)


# ---------------------------------------------------------------------------
# Scene-aware camera spec generation
# ---------------------------------------------------------------------------


def generate_scene_aware_specs(
    obbs: List[OrientedBoundingBox],
    occupancy: Optional[OccupancyGrid] = None,
) -> List[CameraPathSpec]:
    """Generate CameraPathSpecs from OBBs — one per detected object.

    Manipulation candidates get tight arcs; articulation objects get orbits.
    """
    specs: List[CameraPathSpec] = []

    for obb in obbs:
        standoff = compute_standoff_distance(obb)
        cam_height = compute_camera_height(obb)
        approach = obb.center.tolist()

        if obb.category == "manipulation":
            specs.append(
                CameraPathSpec(
                    type="manipulation",
                    approach_point=approach,
                    arc_radius_m=standoff,
                    height_override_m=cam_height,
                    look_down_override_deg=45.0,
                )
            )
        else:
            # Articulation objects (doors, drawers) — orbit around them
            specs.append(
                CameraPathSpec(
                    type="orbit",
                    radius_m=standoff,
                    num_orbits=1,
                    height_override_m=cam_height,
                    look_down_override_deg=25.0,
                )
            )

    logger.info("Generated %d scene-aware camera specs from %d OBBs", len(specs), len(obbs))
    return specs


# ---------------------------------------------------------------------------
# Collision avoidance
# ---------------------------------------------------------------------------


def nudge_to_free_space(
    position: np.ndarray,
    target: np.ndarray,
    occupancy: OccupancyGrid,
    max_nudge_m: float = 0.5,
    step_m: float = 0.05,
    min_clearance_m: float = 0.15,
) -> Optional[np.ndarray]:
    """If position is inside geometry, push it away from target until free."""
    direction = position - target
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = direction / norm

    for d in np.arange(step_m, max_nudge_m + step_m, step_m):
        candidate = position + direction * d
        if occupancy.is_free(candidate, min_clearance_m):
            return candidate
    return None


def filter_and_fix_poses(
    poses: list,
    occupancy: OccupancyGrid,
    target_center: np.ndarray,
    min_clearance_m: float = 0.15,
) -> list:
    """Post-process camera poses: nudge colliding ones or drop them."""
    from .camera_paths import CameraPose

    valid: List[CameraPose] = []
    nudged = 0
    dropped = 0

    for pose in poses:
        pos = pose.position
        if occupancy.is_free(pos, min_clearance_m):
            valid.append(pose)
            continue

        new_pos = nudge_to_free_space(
            pos, target_center, occupancy, min_clearance_m=min_clearance_m
        )
        if new_pos is not None:
            # Rebuild pose with nudged position, keeping orientation
            new_c2w = pose.c2w.copy()
            new_c2w[:3, 3] = new_pos
            valid.append(
                CameraPose(
                    c2w=new_c2w,
                    fx=pose.fx,
                    fy=pose.fy,
                    cx=pose.cx,
                    cy=pose.cy,
                    width=pose.width,
                    height=pose.height,
                )
            )
            nudged += 1
        else:
            dropped += 1

    if nudged or dropped:
        logger.info(
            "Collision filter: %d ok, %d nudged, %d dropped (of %d)",
            len(valid) - nudged,
            nudged,
            dropped,
            len(poses),
        )
    return valid


# ---------------------------------------------------------------------------
# Auto-populate manipulation zones from OBBs
# ---------------------------------------------------------------------------


def auto_populate_manipulation_zones(
    zones: List[ManipulationZoneConfig],
    obbs: List[OrientedBoundingBox],
) -> List[ManipulationZoneConfig]:
    """If zones is empty, populate from OBBs. Otherwise return as-is."""
    if zones:
        logger.info(
            "Facility already has %d manipulation zones; skipping auto-populate", len(zones)
        )
        return zones

    new_zones: List[ManipulationZoneConfig] = []
    for obb in obbs:
        if obb.category != "manipulation":
            continue
        new_zones.append(
            ManipulationZoneConfig(
                name=f"auto_{obb.instance_id}",
                approach_point=obb.center.tolist(),
                target_point=obb.center.tolist(),
                camera_height_m=compute_camera_height(obb),
                camera_look_down_deg=45.0,
                arc_radius_m=compute_standoff_distance(obb),
            )
        )

    logger.info("Auto-populated %d manipulation zones from task_targets.json", len(new_zones))
    return new_zones


# ---------------------------------------------------------------------------
# Gaussian subset utilities for RoboSplat-style edits
# ---------------------------------------------------------------------------


def select_gaussians_in_sphere(
    means: np.ndarray,
    center: np.ndarray,
    radius_m: float,
    max_points: int = 12000,
) -> np.ndarray:
    """Return indices of Gaussian centers inside a spherical region."""
    if means.size == 0:
        return np.asarray([], dtype=np.int64)
    center = np.asarray(center, dtype=np.float32).reshape(1, 3)
    dist = np.linalg.norm(means - center, axis=1)
    idx = np.flatnonzero(dist <= max(0.01, float(radius_m)))
    if len(idx) > max_points:
        idx = idx[:max_points]
    return idx.astype(np.int64)


def cluster_scene_points(
    means: np.ndarray,
    num_clusters: int = 8,
    max_points: int = 30000,
    seed: int = 13,
) -> np.ndarray:
    """Cluster scene points and return approximate object anchor centers."""
    if means.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = means
    if len(pts) > max_points:
        rng = np.random.default_rng(seed)
        choice = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[choice]
    k = max(1, min(int(num_clusters), len(pts)))
    # Lightweight k-means (fixed iterations) to avoid external deps.
    rng = np.random.default_rng(seed)
    centers = pts[rng.choice(len(pts), size=k, replace=False)].astype(np.float32)
    for _ in range(12):
        d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
        assign = np.argmin(d, axis=1)
        for ci in range(k):
            members = pts[assign == ci]
            if len(members) > 0:
                centers[ci] = members.mean(axis=0)
    return centers.astype(np.float32)
