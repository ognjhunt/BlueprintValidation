"""Scene-aware camera placement using OBBs from task_targets.json and occupancy grids."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..common import get_logger
from ..config import CameraPathSpec, FacilityConfig, ManipulationZoneConfig

logger = get_logger("rendering.scene_geometry")


# ---------------------------------------------------------------------------
# Scene orientation transform
# ---------------------------------------------------------------------------

_UP_AXIS_ROTATIONS = {
    "z": np.eye(3, dtype=np.float64),
    "+z": np.eye(3, dtype=np.float64),
    "y": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64),
    "+y": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64),
    "-y": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64),
    "-z": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64),
    "x": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
    "+x": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
    "-x": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),
}


def _euler_xyz_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from extrinsic XYZ Euler angles in degrees."""
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def detect_up_axis(means: np.ndarray, confidence_ratio: float = 1.5) -> str:
    """Infer the up axis from point cloud variance analysis.

    Uses a two-stage approach:

    1. **Robust extents** — Trim the outer 1% of points per axis to ignore
       noise/outliers, then pick the axis with the smallest trimmed range.
    2. **Floor-density sign check** — Split the candidate axis at the median.
       If the lower slab has more points than the upper slab the direction is
       positive (normal: floor at min, ceiling at max).  Otherwise negative.

    Returns one of ``"x"``, ``"y"``, ``"z"``, ``"-x"``, ``"-y"``, ``"-z"``.
    Falls back to ``"z"`` when the scene is roughly cubical or there are
    too few points.

    Args:
        means: (N, 3) point positions.
        confidence_ratio: The second-smallest extent must be at least this
            many times larger than the smallest for the detection to be
            confident.  Below this threshold the function falls back to "z".
    """
    if means.shape[0] < 10:
        logger.warning(
            "Too few points (%d) for up-axis detection; defaulting to Z-up",
            means.shape[0],
        )
        return "z"

    # Robust extents: trim 1st-99th percentile to ignore outlier Gaussians
    lo = np.percentile(means, 1, axis=0)
    hi = np.percentile(means, 99, axis=0)
    extents = hi - lo

    if extents.min() < 1e-6:
        logger.warning(
            "Near-zero extent detected (%.4f); defaulting to Z-up",
            extents.min(),
        )
        return "z"

    min_axis = int(np.argmin(extents))
    sorted_extents = np.sort(extents)
    ratio = sorted_extents[1] / sorted_extents[0]

    if ratio < confidence_ratio:
        logger.info(
            "Scene extents roughly equal (X=%.2f Y=%.2f Z=%.2f, ratio=%.1f); "
            "defaulting to Z-up. Set up_axis manually if needed.",
            extents[0],
            extents[1],
            extents[2],
            ratio,
        )
        return "z"

    axis_name = {0: "x", 1: "y", 2: "z"}[min_axis]

    # Sign detection: count points in the bottom vs top 15% of the physical
    # extent along the candidate axis.  The floor is a large flat surface so
    # typically has more Gaussians than the ceiling.
    vals = means[:, min_axis]
    axis_lo = lo[min_axis]
    axis_extent = extents[min_axis]
    slab = 0.15 * axis_extent
    n_lo = int(np.sum(vals <= axis_lo + slab))
    n_hi = int(np.sum(vals >= axis_lo + axis_extent - slab))

    if n_hi > n_lo * 1.3:
        # More density near the top → scene is flipped
        axis_name = f"-{axis_name}"

    logger.info(
        "Auto-detected up_axis='%s' (extents: X=%.2f Y=%.2f Z=%.2f, "
        "ratio=%.1f, lo_density=%d, hi_density=%d)",
        axis_name,
        extents[0],
        extents[1],
        extents[2],
        ratio,
        n_lo,
        n_hi,
    )
    return axis_name


def compute_scene_transform(facility: FacilityConfig) -> np.ndarray:
    """Compute a 4x4 transform that maps PLY-native coords to pipeline Z-up coords.

    Composes the up-axis preset rotation with optional additional Euler rotation.
    Returns identity when up_axis="z" and scene_rotation_deg=[0,0,0].

    Note: ``up_axis="auto"`` must be resolved to a concrete axis *before*
    calling this function (see :func:`detect_up_axis`).  If "auto" is passed
    directly a warning is emitted and Z-up is assumed.
    """
    up_key = facility.up_axis.lower().strip()

    if up_key == "auto":
        logger.warning(
            "compute_scene_transform called with up_axis='auto' without prior "
            "resolution; falling back to 'z'. Call detect_up_axis() first."
        )
        up_key = "z"

    if up_key not in _UP_AXIS_ROTATIONS:
        raise ValueError(
            f"Unknown up_axis '{facility.up_axis}'. "
            f"Valid values: {sorted(_UP_AXIS_ROTATIONS.keys())}"
        )

    R_up = _UP_AXIS_ROTATIONS[up_key]

    # Additional Euler rotation (applied after up-axis correction)
    rx, ry, rz = facility.scene_rotation_deg
    if rx != 0.0 or ry != 0.0 or rz != 0.0:
        R_extra = _euler_xyz_to_matrix(rx, ry, rz)
        R = R_extra @ R_up
    else:
        R = R_up

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def is_identity_transform(T: np.ndarray) -> bool:
    """Check if a 4x4 transform is (close to) identity."""
    return bool(np.allclose(T, np.eye(4), atol=1e-10))


def transform_means(means: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rotation transform to (N, 3) positions."""
    R = T[:3, :3]
    return (R @ means.T).T


def transform_c2w(c2w: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Convert a c2w matrix from the corrected (Z-up) frame to the original PLY frame.

    If T maps original→corrected, then T_inv maps corrected→original.
    c2w_original = T_inv @ c2w_corrected
    """
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = T[:3, :3].T  # orthogonal rotation: inverse = transpose
    return T_inv @ c2w


def transform_obbs(
    obbs: List[OrientedBoundingBox],
    T: np.ndarray,
) -> List[OrientedBoundingBox]:
    """Rotate OBB centers/axes from PLY-native frame into corrected frame."""
    R = T[:3, :3]
    transformed: List[OrientedBoundingBox] = []
    for obb in obbs:
        transformed.append(
            OrientedBoundingBox(
                instance_id=obb.instance_id,
                label=obb.label,
                center=R @ obb.center,
                extents=obb.extents.copy(),
                axes=R @ obb.axes,
                confidence=obb.confidence,
                category=obb.category,
            )
        )
    return transformed


def transform_camera_path_specs(
    specs: List[CameraPathSpec],
    T: np.ndarray,
) -> List[CameraPathSpec]:
    """Rotate any explicit approach point into the corrected frame."""
    transformed: List[CameraPathSpec] = []
    for spec in specs:
        if spec.approach_point is None:
            transformed.append(spec)
            continue
        point = np.asarray(spec.approach_point, dtype=np.float64).reshape(1, 3)
        rotated = transform_means(point, T)[0].tolist()
        transformed.append(replace(spec, approach_point=rotated))
    return transformed


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


_UP_AXIS_CANONICAL = {
    "z": "z",
    "+z": "z",
    "-z": "-z",
    "y": "y",
    "+y": "y",
    "-y": "-y",
    "x": "x",
    "+x": "x",
    "-x": "-x",
}

_UP_AXIS_INVERSE = {
    "z": "-z",
    "-z": "z",
    "y": "-y",
    "-y": "y",
    "x": "-x",
    "-x": "x",
}


def _canonical_up_axis(axis: str) -> str:
    key = str(axis).strip().lower()
    return _UP_AXIS_CANONICAL.get(key, key)


def inverse_up_axis(axis: str) -> str:
    """Return the opposite up-axis direction (e.g. z -> -z)."""
    canonical = _canonical_up_axis(axis)
    return _UP_AXIS_INVERSE.get(canonical, canonical)


def _orientation_floor_ceiling_score(
    means_z: np.ndarray,
    floor_height_m: float,
    ceiling_height_m: float,
) -> tuple[float, float, float]:
    """Score transformed scene Z alignment to expected floor/ceiling heights."""
    z_lo = float(np.percentile(means_z, 1))
    z_hi = float(np.percentile(means_z, 99))
    z_span = max(1e-3, z_hi - z_lo)

    expected_floor = float(floor_height_m)
    expected_ceiling = float(ceiling_height_m)
    if expected_ceiling <= expected_floor:
        expected_ceiling = expected_floor + z_span

    floor_err = abs(z_lo - expected_floor) / max(z_span, 1.0)
    ceiling_err = abs(z_hi - expected_ceiling) / max(z_span, 1.0)
    score = max(0.0, 1.0 - 0.5 * (floor_err + ceiling_err))
    return score, z_lo, z_hi


def _orientation_obb_height_score(
    transformed_obbs: Optional[List["OrientedBoundingBox"]],
    floor_height_m: float,
    ceiling_height_m: float,
) -> tuple[float, int]:
    """Score whether OBB centers land in plausible vertical range."""
    if not transformed_obbs:
        return 0.5, 0

    floor = float(floor_height_m) - 0.2
    ceil = float(ceiling_height_m) + 0.2
    if ceil <= floor:
        floor = min(floor, 0.0)
        ceil = max(ceil, floor + 3.0)

    centers = np.asarray([obb.center[2] for obb in transformed_obbs], dtype=np.float64)
    within = np.logical_and(centers >= floor, centers <= ceil)
    return float(within.mean()), int(len(centers))


def _orientation_camera_up_score(
    transformed_means: np.ndarray,
    transformed_obbs: Optional[List["OrientedBoundingBox"]],
    camera_look_down_deg: float,
) -> tuple[float, float]:
    """Probe manipulation-like poses and score camera-up alignment to world-up."""
    try:
        from .camera_paths import generate_manipulation_arc

        target = transformed_means.mean(axis=0)
        if transformed_obbs:
            manip = [obb for obb in transformed_obbs if obb.category == "manipulation"]
            if manip:
                target = manip[0].center
        probe_height = max(0.4, float(target[2]) + 0.3)
        poses = generate_manipulation_arc(
            approach_point=np.asarray(target, dtype=np.float64),
            arc_radius=0.6,
            height=probe_height,
            num_frames=9,
            look_down_deg=float(camera_look_down_deg),
            target_z_bias_m=0.0,
            resolution=(120, 160),
        )
        up_dots = [float(np.dot(p.c2w[:3, 1], np.array([0.0, 0.0, 1.0]))) for p in poses]
        median_dot = float(np.median(np.asarray(up_dots, dtype=np.float64)))
        return float((median_dot + 1.0) * 0.5), median_dot
    except Exception:
        return 0.5, 0.0


def resolve_facility_orientation(
    *,
    facility: FacilityConfig,
    means_raw: np.ndarray,
    obbs_raw: Optional[List["OrientedBoundingBox"]] = None,
    camera_look_down_deg: float = 20.0,
    orientation_autocorrect_enabled: bool = True,
    orientation_autocorrect_mode: str = "auto",
) -> tuple[FacilityConfig, Dict[str, Any]]:
    """Resolve orientation with optional deterministic auto-correction."""
    detected_up_axis: Optional[str] = None
    if facility.up_axis.lower().strip() == "auto":
        detected_up_axis = detect_up_axis(means_raw)
        primary_axis = _canonical_up_axis(detected_up_axis)
    else:
        primary_axis = _canonical_up_axis(facility.up_axis)

    inverse_axis = inverse_up_axis(primary_axis)
    candidate_axes: List[str] = [primary_axis]
    if inverse_axis != primary_axis:
        candidate_axes.append(inverse_axis)

    candidate_rows: List[Dict[str, Any]] = []
    for axis in candidate_axes:
        candidate_fac = replace(facility, up_axis=axis)
        T = compute_scene_transform(candidate_fac)
        transformed_means = transform_means(means_raw, T)
        transformed_obbs = transform_obbs(obbs_raw, T) if obbs_raw else None
        floor_score, z_lo, z_hi = _orientation_floor_ceiling_score(
            transformed_means[:, 2],
            floor_height_m=facility.floor_height_m,
            ceiling_height_m=facility.ceiling_height_m,
        )
        obb_score, obb_count = _orientation_obb_height_score(
            transformed_obbs,
            floor_height_m=facility.floor_height_m,
            ceiling_height_m=facility.ceiling_height_m,
        )
        cam_score, cam_up_median = _orientation_camera_up_score(
            transformed_means,
            transformed_obbs,
            camera_look_down_deg=camera_look_down_deg,
        )
        total = 0.6 * floor_score + 0.25 * obb_score + 0.15 * cam_score
        candidate_rows.append(
            {
                "axis": axis,
                "score_total": round(float(total), 6),
                "score_floor_ceiling": round(float(floor_score), 6),
                "score_obb_height": round(float(obb_score), 6),
                "score_camera_up": round(float(cam_score), 6),
                "camera_up_median_dot": round(float(cam_up_median), 6),
                "z_percentile_1": round(float(z_lo), 6),
                "z_percentile_99": round(float(z_hi), 6),
                "obb_count": int(obb_count),
            }
        )

    ranked = sorted(candidate_rows, key=lambda row: row["score_total"], reverse=True)
    best = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None
    score_margin = float(best["score_total"] - runner["score_total"]) if runner else 0.0

    mode = str(orientation_autocorrect_mode or "auto").strip().lower()
    if mode not in {"auto", "fail_fast", "warn_only"}:
        mode = "auto"

    selected_axis = primary_axis
    if orientation_autocorrect_enabled:
        if mode == "auto":
            selected_axis = str(best["axis"])
        elif mode == "fail_fast":
            if str(best["axis"]) != primary_axis:
                raise RuntimeError(
                    "Orientation auto-check failed in fail_fast mode: "
                    f"primary={primary_axis}, best={best['axis']}, margin={score_margin:.4f}"
                )
            selected_axis = primary_axis
        else:  # warn_only
            selected_axis = primary_axis

    resolved_facility = replace(facility, up_axis=selected_axis)
    metadata: Dict[str, Any] = {
        "detected_up_axis": detected_up_axis,
        "resolved_up_axis": selected_axis,
        "orientation_primary_axis": primary_axis,
        "orientation_autocorrect_enabled": bool(orientation_autocorrect_enabled),
        "orientation_autocorrect_mode": mode,
        "orientation_candidates": ranked,
        "orientation_score_selected": float(best["score_total"]),
        "orientation_score_runner_up": float(runner["score_total"]) if runner else None,
        "orientation_score_margin": round(score_margin, 6),
    }
    return resolved_facility, metadata


def correct_upside_down_camera_poses(poses: list) -> tuple[list, int]:
    """Apply 180° roll around forward axis when camera up points downward."""
    from .camera_paths import CameraPose

    corrected: list = []
    num_corrected = 0
    for pose in poses:
        up_dot = float(np.dot(pose.c2w[:3, 1], np.array([0.0, 0.0, 1.0])))
        if up_dot >= 0.0:
            corrected.append(pose)
            continue
        new_c2w = pose.c2w.copy()
        # 180° roll about forward axis flips right/up and preserves forward.
        new_c2w[:3, 0] *= -1.0
        new_c2w[:3, 1] *= -1.0
        corrected.append(
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
        num_corrected += 1
    return corrected, num_corrected


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
    """Parse task_targets.json and extract OBBs from hint candidate groups."""
    with open(path) as f:
        data = json.load(f)

    obbs: List[OrientedBoundingBox] = []

    for category, key in [
        ("manipulation", "manipulation_candidates"),
        ("articulation", "articulation_hints"),
        ("navigation", "navigation_hints"),
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
        "Loaded %d OBBs from %s (manipulation=%d, articulation=%d, navigation=%d)",
        len(obbs),
        path,
        sum(1 for o in obbs if o.category == "manipulation"),
        sum(1 for o in obbs if o.category == "articulation"),
        sum(1 for o in obbs if o.category == "navigation"),
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

    Manipulation candidates get tight arcs; articulation/navigation objects get orbits.
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
        elif obb.category == "articulation":
            # Articulation objects (doors, drawers) — orbit around them.
            specs.append(
                CameraPathSpec(
                    type="orbit",
                    radius_m=standoff,
                    num_orbits=1,
                    height_override_m=cam_height,
                    look_down_override_deg=25.0,
                )
            )
        else:
            # Navigation regions are coarse waypoints; use a gentle overview orbit.
            specs.append(
                CameraPathSpec(
                    type="orbit",
                    radius_m=standoff,
                    num_orbits=1,
                    height_override_m=cam_height,
                    look_down_override_deg=35.0,
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


@dataclass
class ClusterResult:
    """Result of scene point clustering with per-cluster bounding info."""

    centers: np.ndarray  # (K, 3) float32 cluster centers
    extents: np.ndarray  # (K, 3) float32 per-cluster bounding box extents
    point_counts: np.ndarray  # (K,) int32 number of points in each cluster


def cluster_scene_points_with_extents(
    means: np.ndarray,
    num_clusters: int = 8,
    max_points: int = 30000,
    seed: int = 13,
    min_extent: float = 0.1,
    max_extent: float = 2.0,
) -> ClusterResult:
    """Cluster scene points and return centers with per-cluster bounding extents.

    Same lightweight k-means as :func:`cluster_scene_points`, but also computes
    the 5th-95th percentile axis-aligned bounding box of each cluster, clamped
    to *[min_extent, max_extent]* per axis.
    """
    if means.size == 0:
        return ClusterResult(
            centers=np.zeros((0, 3), dtype=np.float32),
            extents=np.zeros((0, 3), dtype=np.float32),
            point_counts=np.zeros(0, dtype=np.int32),
        )
    pts = means.astype(np.float32)
    if len(pts) > max_points:
        rng = np.random.default_rng(seed)
        choice = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[choice]
    k = max(1, min(int(num_clusters), len(pts)))

    rng = np.random.default_rng(seed)
    centers = pts[rng.choice(len(pts), size=k, replace=False)].astype(np.float32)
    for _ in range(12):
        d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
        assign = np.argmin(d, axis=1)
        for ci in range(k):
            members = pts[assign == ci]
            if len(members) > 0:
                centers[ci] = members.mean(axis=0)

    # Compute per-cluster extents from 5th-95th percentile
    ext = np.full((k, 3), min_extent, dtype=np.float32)
    counts = np.zeros(k, dtype=np.int32)
    d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
    assign = np.argmin(d, axis=1)
    for ci in range(k):
        members = pts[assign == ci]
        counts[ci] = len(members)
        if len(members) >= 2:
            lo = np.percentile(members, 5, axis=0)
            hi = np.percentile(members, 95, axis=0)
            ext[ci] = np.clip(hi - lo, min_extent, max_extent)

    logger.info(
        "Clustered %d points into %d clusters (sizes: %s)",
        len(pts),
        k,
        counts.tolist(),
    )
    return ClusterResult(centers=centers, extents=ext, point_counts=counts)
