"""Pre-compute CPU-only artifacts per facility to speed up GPU pipeline runs.

Caches occupancy grids, camera paths, OBBs, and dynamic variant prompts so
that Stage 1 (render) and Stage 2 (enrich) can skip expensive CPU prep when
running on a GPU machine.
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .common import get_logger, write_json
from .config import (
    CameraPathSpec,
    FacilityConfig,
    ValidationConfig,
    VariantSpec,
)
from .rendering.camera_paths import (
    CameraPose,
    generate_path_from_spec,
    save_path_to_json,
)
from .rendering.scene_geometry import (
    OccupancyGrid,
    auto_populate_manipulation_zones,
    build_occupancy_grid,
    compute_scene_transform,
    detect_up_axis,
    filter_and_fix_poses,
    generate_scene_aware_specs,
    is_identity_transform,
    load_obbs_from_task_targets,
    transform_camera_path_specs,
    transform_means,
    transform_obbs,
)

logger = get_logger("warmup")

CACHE_DIR_NAME = "warmup_cache"
CACHE_MANIFEST = "warmup_manifest.json"


_PLY_STRUCT_TYPES = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def _parse_vertex_header(lines: List[str]) -> tuple[str, int, List[tuple[str, str]]]:
    """Parse PLY header lines and return (format, vertex_count, vertex_properties)."""
    fmt: Optional[str] = None
    vertex_count: Optional[int] = None
    in_vertex = False
    properties: List[tuple[str, str]] = []

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        key = parts[0].lower()
        if key == "format" and len(parts) >= 2:
            fmt = parts[1].lower()
        elif key == "element" and len(parts) >= 3:
            in_vertex = parts[1].lower() == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
                properties = []
        elif key == "property" and in_vertex and len(parts) >= 3:
            if parts[1].lower() == "list":
                raise ValueError("PLY list properties are not supported by warmup fallback parser")
            properties.append((parts[2], parts[1].lower()))

    if fmt is None or vertex_count is None:
        raise ValueError("PLY header missing format or vertex element")
    if not properties:
        raise ValueError("PLY header has no vertex properties")
    return fmt, vertex_count, properties


def _load_ply_means_fallback(ply_path: Path) -> np.ndarray:
    """Load XYZ from a PLY file without external dependencies."""
    with ply_path.open("rb") as f:
        first = f.readline().decode("ascii", errors="strict").strip().lower()
        if first != "ply":
            raise ValueError(f"Invalid PLY file: {ply_path}")

        header_lines: List[str] = []
        while True:
            raw_line = f.readline()
            if not raw_line:
                raise ValueError(f"Invalid PLY header (missing end_header): {ply_path}")
            line = raw_line.decode("ascii", errors="strict").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        fmt, vertex_count, properties = _parse_vertex_header(header_lines)
        name_to_idx = {name: idx for idx, (name, _typ) in enumerate(properties)}
        for required in ("x", "y", "z"):
            if required not in name_to_idx:
                raise ValueError(f"PLY vertex properties missing '{required}': {ply_path}")
        x_idx, y_idx, z_idx = name_to_idx["x"], name_to_idx["y"], name_to_idx["z"]

        means = np.empty((vertex_count, 3), dtype=np.float32)
        if fmt == "binary_little_endian":
            try:
                row_fmt = "<" + "".join(_PLY_STRUCT_TYPES[typ] for _, typ in properties)
            except KeyError as exc:
                raise ValueError(f"Unsupported PLY property type: {exc.args[0]}") from exc
            row_struct = struct.Struct(row_fmt)
            raw = f.read(row_struct.size * vertex_count)
            expected = row_struct.size * vertex_count
            if len(raw) != expected:
                raise ValueError(
                    f"PLY payload truncated for {ply_path} (expected {expected} bytes, got {len(raw)})"
                )
            for i, row in enumerate(row_struct.iter_unpack(raw)):
                means[i] = (float(row[x_idx]), float(row[y_idx]), float(row[z_idx]))
            return means

        if fmt == "ascii":
            for i in range(vertex_count):
                raw_line = f.readline()
                if not raw_line:
                    raise ValueError(f"PLY payload truncated for {ply_path} at vertex {i}")
                parts = raw_line.decode("ascii", errors="strict").strip().split()
                if len(parts) < len(properties):
                    raise ValueError(
                        f"Malformed ASCII PLY row {i} for {ply_path}: expected {len(properties)} columns"
                    )
                means[i] = (float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx]))
            return means

        raise ValueError(f"Unsupported PLY format '{fmt}' in {ply_path}")


def load_ply_means_numpy(ply_path: Path) -> np.ndarray:
    """Load only positions from PLY as numpy — no torch/GPU needed."""
    logger.info("Loading PLY (CPU-only): %s", ply_path)
    try:
        from plyfile import PlyData

        plydata = PlyData.read(str(ply_path))
        vertex = plydata["vertex"]
        x = np.array(vertex["x"], dtype=np.float32)
        y = np.array(vertex["y"], dtype=np.float32)
        z = np.array(vertex["z"], dtype=np.float32)
        means = np.stack([x, y, z], axis=-1)
    except ModuleNotFoundError:
        logger.warning("plyfile is not installed; using fallback PLY parser for warmup")
        means = _load_ply_means_fallback(ply_path)
    logger.info("Loaded %d Gaussian points (CPU numpy)", len(means))
    return means


def _load_ply_means_numpy(ply_path: Path) -> np.ndarray:
    """Backward-compatible alias for tests and legacy callers."""
    return load_ply_means_numpy(ply_path)


def _sh_dc_to_rgb(dc0: np.ndarray, dc1: np.ndarray, dc2: np.ndarray) -> np.ndarray:
    """Convert spherical harmonic DC coefficients to uint8 RGB.

    3DGS convention: color ≈ sigmoid(f_dc).
    """
    raw = np.stack([dc0, dc1, dc2], axis=-1).astype(np.float32)
    rgb_float = 1.0 / (1.0 + np.exp(-raw))
    return np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8)


def _extract_colors_plyfile(vertex) -> Optional[np.ndarray]:
    """Try to extract RGB colors from a plyfile vertex element."""
    names = {p.name for p in vertex.properties}
    if {"red", "green", "blue"}.issubset(names):
        r = np.array(vertex["red"], dtype=np.uint8)
        g = np.array(vertex["green"], dtype=np.uint8)
        b = np.array(vertex["blue"], dtype=np.uint8)
        return np.stack([r, g, b], axis=-1)
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        dc0 = np.array(vertex["f_dc_0"], dtype=np.float32)
        dc1 = np.array(vertex["f_dc_1"], dtype=np.float32)
        dc2 = np.array(vertex["f_dc_2"], dtype=np.float32)
        return _sh_dc_to_rgb(dc0, dc1, dc2)
    return None


def _load_ply_means_and_colors_fallback(ply_path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Load XYZ + optional RGB from a PLY file without external dependencies."""
    with ply_path.open("rb") as f:
        first = f.readline().decode("ascii", errors="strict").strip().lower()
        if first != "ply":
            raise ValueError(f"Invalid PLY file: {ply_path}")

        header_lines: List[str] = []
        while True:
            raw_line = f.readline()
            if not raw_line:
                raise ValueError(f"Invalid PLY header (missing end_header): {ply_path}")
            line = raw_line.decode("ascii", errors="strict").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        fmt, vertex_count, properties = _parse_vertex_header(header_lines)
        name_to_idx = {name: idx for idx, (name, _typ) in enumerate(properties)}
        for required in ("x", "y", "z"):
            if required not in name_to_idx:
                raise ValueError(f"PLY vertex properties missing '{required}': {ply_path}")
        x_idx, y_idx, z_idx = name_to_idx["x"], name_to_idx["y"], name_to_idx["z"]

        # Detect color columns
        has_rgb = all(c in name_to_idx for c in ("red", "green", "blue"))
        has_sh_dc = all(c in name_to_idx for c in ("f_dc_0", "f_dc_1", "f_dc_2"))
        color_mode: Optional[str] = "rgb" if has_rgb else ("sh_dc" if has_sh_dc else None)

        means = np.empty((vertex_count, 3), dtype=np.float32)
        color_raw: Optional[np.ndarray] = None
        if color_mode == "rgb":
            color_raw = np.empty((vertex_count, 3), dtype=np.uint8)
            r_idx = name_to_idx["red"]
            g_idx = name_to_idx["green"]
            b_idx = name_to_idx["blue"]
        elif color_mode == "sh_dc":
            color_raw = np.empty((vertex_count, 3), dtype=np.float32)
            r_idx = name_to_idx["f_dc_0"]
            g_idx = name_to_idx["f_dc_1"]
            b_idx = name_to_idx["f_dc_2"]
        else:
            r_idx = g_idx = b_idx = 0  # unused

        if fmt == "binary_little_endian":
            try:
                row_fmt = "<" + "".join(_PLY_STRUCT_TYPES[typ] for _, typ in properties)
            except KeyError as exc:
                raise ValueError(f"Unsupported PLY property type: {exc.args[0]}") from exc
            row_struct = struct.Struct(row_fmt)
            raw = f.read(row_struct.size * vertex_count)
            expected = row_struct.size * vertex_count
            if len(raw) != expected:
                raise ValueError(
                    f"PLY payload truncated for {ply_path} (expected {expected} bytes, got {len(raw)})"
                )
            for i, row in enumerate(row_struct.iter_unpack(raw)):
                means[i] = (float(row[x_idx]), float(row[y_idx]), float(row[z_idx]))
                if color_raw is not None:
                    color_raw[i] = (row[r_idx], row[g_idx], row[b_idx])
        elif fmt == "ascii":
            for i in range(vertex_count):
                raw_line = f.readline()
                if not raw_line:
                    raise ValueError(f"PLY payload truncated for {ply_path} at vertex {i}")
                parts = raw_line.decode("ascii", errors="strict").strip().split()
                if len(parts) < len(properties):
                    raise ValueError(
                        f"Malformed ASCII PLY row {i} for {ply_path}: expected {len(properties)} columns"
                    )
                means[i] = (float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx]))
                if color_raw is not None:
                    color_raw[i] = (float(parts[r_idx]), float(parts[g_idx]), float(parts[b_idx]))
        else:
            raise ValueError(f"Unsupported PLY format '{fmt}' in {ply_path}")

        colors: Optional[np.ndarray] = None
        if color_mode == "rgb":
            colors = color_raw
        elif color_mode == "sh_dc" and color_raw is not None:
            colors = _sh_dc_to_rgb(color_raw[:, 0], color_raw[:, 1], color_raw[:, 2])

        return means, colors


def load_ply_means_and_colors_numpy(
    ply_path: Path,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Load positions and RGB colors from PLY. No torch/GPU needed.

    Color extraction priority:
        1. Direct RGB (red/green/blue uint8)
        2. SH DC coefficients (f_dc_0/1/2 -> sigmoid -> 0-255 uint8)
        3. None if neither is present

    Returns:
        (means, colors): means is (N,3) float32, colors is (N,3) uint8 or None.
    """
    logger.info("Loading PLY with colors (CPU-only): %s", ply_path)
    try:
        from plyfile import PlyData

        plydata = PlyData.read(str(ply_path))
        vertex = plydata["vertex"]
        x = np.array(vertex["x"], dtype=np.float32)
        y = np.array(vertex["y"], dtype=np.float32)
        z = np.array(vertex["z"], dtype=np.float32)
        means = np.stack([x, y, z], axis=-1)
        colors = _extract_colors_plyfile(vertex)
    except ModuleNotFoundError:
        logger.warning("plyfile is not installed; using fallback PLY parser")
        means, colors = _load_ply_means_and_colors_fallback(ply_path)

    logger.info(
        "Loaded %d Gaussian points (CPU numpy), colors=%s",
        len(means),
        "yes" if colors is not None else "no",
    )
    return means, colors


def _save_occupancy_grid(grid: OccupancyGrid, path: Path) -> None:
    np.savez_compressed(
        str(path),
        voxels=grid.voxels,
        origin=grid.origin,
        voxel_size=np.array([grid.voxel_size]),
        shape=np.array(grid.shape),
    )
    logger.info("Saved occupancy grid to %s", path)


def load_cached_occupancy_grid(path: Path) -> OccupancyGrid:
    """Load a cached occupancy grid from .npz."""
    data = np.load(str(path))
    shape = tuple(int(s) for s in data["shape"])
    return OccupancyGrid(
        voxels=data["voxels"],
        origin=data["origin"],
        voxel_size=float(data["voxel_size"][0]),
        shape=shape,
    )


def _serialize_camera_poses(poses: List[CameraPose]) -> List[dict]:
    return [
        {
            "c2w": pose.c2w.tolist(),
            "fx": pose.fx,
            "fy": pose.fy,
            "cx": pose.cx,
            "cy": pose.cy,
            "width": pose.width,
            "height": pose.height,
        }
        for pose in poses
    ]


def _deserialize_camera_poses(data: List[dict]) -> List[CameraPose]:
    return [
        CameraPose(
            c2w=np.array(d["c2w"], dtype=np.float64),
            fx=d["fx"],
            fy=d["fy"],
            cx=d["cx"],
            cy=d["cy"],
            width=d["width"],
            height=d["height"],
        )
        for d in data
    ]


def _warmup_dynamic_variants(
    config: ValidationConfig,
    facility: FacilityConfig,
) -> List[dict]:
    """Call Gemini API for dynamic variant prompts (no GPU needed)."""
    if not config.enrich.dynamic_variants:
        return []
    if config.enrich.variants:
        return [asdict(v) for v in config.enrich.variants]

    from .enrichment.variant_specs import generate_dynamic_variants

    variants = generate_dynamic_variants(
        num_variants=config.enrich.num_variants_per_render,
        model=config.enrich.dynamic_variants_model,
        facility_description=facility.description,
    )
    return [{"name": v.name, "prompt": v.prompt} for v in variants]


def warmup_facility(
    config: ValidationConfig,
    facility: FacilityConfig,
    work_dir: Path,
) -> Dict:
    """Run all CPU-only pre-computation for a single facility.

    Returns a summary dict with paths to cached artifacts.
    """
    t0 = time.time()
    cache_dir = work_dir / CACHE_DIR_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict = {
        "facility": facility.name,
        "ply_path": str(facility.ply_path),
    }

    # 1. Load PLY as numpy (CPU-only, no torch)
    if not facility.ply_path.exists():
        logger.warning("PLY file not found: %s — skipping geometry warmup", facility.ply_path)
        summary["ply_loaded"] = False
        summary["elapsed_seconds"] = round(time.time() - t0, 2)
        write_json(summary, cache_dir / CACHE_MANIFEST)
        return summary

    means_raw = load_ply_means_numpy(facility.ply_path)
    summary["ply_loaded"] = True
    summary["num_gaussians"] = int(len(means_raw))

    # Resolve auto up-axis from point cloud extents
    if facility.up_axis.lower().strip() == "auto":
        detected = detect_up_axis(means_raw)
        summary["detected_up_axis"] = detected
        facility = replace(facility, up_axis=detected)

    # Apply scene orientation transform (e.g. Y-up → Z-up)
    scene_T = compute_scene_transform(facility)
    has_transform = not is_identity_transform(scene_T)
    summary["scene_transform"] = scene_T.tolist()
    summary["resolved_up_axis"] = facility.up_axis
    if has_transform:
        logger.info("Applying scene transform (up_axis=%s)", facility.up_axis)
        means = transform_means(means_raw, scene_T)
    else:
        means = means_raw

    scene_center = means.mean(axis=0)
    summary["scene_center"] = scene_center.tolist()

    # Save scene center for later use
    np.save(str(cache_dir / "scene_center.npy"), scene_center)

    # 2. Build occupancy grid
    occupancy: Optional[OccupancyGrid] = None
    if config.render.collision_check and config.render.scene_aware:
        occupancy = build_occupancy_grid(
            means,
            voxel_size=config.render.voxel_size_m,
            density_threshold=config.render.density_threshold,
        )
        grid_path = cache_dir / "occupancy_grid.npz"
        _save_occupancy_grid(occupancy, grid_path)
        summary["occupancy_grid_path"] = str(grid_path)
        summary["occupancy_grid_shape"] = list(occupancy.shape)

    # 3. Load OBBs from task_targets.json
    extra_specs: List[CameraPathSpec] = []
    if config.render.scene_aware:
        if facility.task_hints_path and facility.task_hints_path.exists():
            obbs = load_obbs_from_task_targets(facility.task_hints_path)
            if obbs:
                if has_transform:
                    obbs = transform_obbs(obbs, scene_T)
                extra_specs = generate_scene_aware_specs(obbs, occupancy)
                facility.manipulation_zones = auto_populate_manipulation_zones(
                    facility.manipulation_zones, obbs
                )
                summary["obbs_loaded"] = len(obbs)
                summary["scene_aware_specs"] = len(extra_specs)
        else:
            summary["obbs_loaded"] = 0

    # 4. Generate all camera paths
    base_specs = list(config.render.camera_paths)
    if has_transform:
        base_specs = transform_camera_path_specs(base_specs, scene_T)
    all_specs = base_specs + extra_specs
    all_clips: List[dict] = []
    clip_index = 0

    for path_spec in all_specs:
        for clip_num in range(config.render.num_clips_per_path):
            rng = np.random.default_rng(seed=clip_index * 42)
            offset = rng.uniform(-1.0, 1.0, size=3)

            poses = generate_path_from_spec(
                spec=path_spec,
                scene_center=scene_center,
                num_frames=config.render.num_frames,
                camera_height=config.render.camera_height_m,
                look_down_deg=config.render.camera_look_down_deg,
                resolution=config.render.resolution,
                start_offset=offset,
            )

            # Collision filter
            if occupancy is not None:
                target = np.array(path_spec.approach_point or scene_center[:3])
                poses = filter_and_fix_poses(
                    poses, occupancy, target, config.render.min_clearance_m
                )
                if not poses:
                    clip_index += 1
                    continue

            clip_name = f"clip_{clip_index:03d}_{path_spec.type}"

            # Save individual camera path JSON
            path_dir = cache_dir / "camera_paths"
            path_dir.mkdir(parents=True, exist_ok=True)
            save_path_to_json(poses, path_dir / f"{clip_name}_camera_path.json")

            all_clips.append(
                {
                    "clip_name": clip_name,
                    "path_type": path_spec.type,
                    "clip_index": clip_index,
                    "num_frames": len(poses),
                    "camera_path_file": str(path_dir / f"{clip_name}_camera_path.json"),
                    "poses": _serialize_camera_poses(poses),
                }
            )
            clip_index += 1

    # Save clips manifest
    clips_path = cache_dir / "precomputed_clips.json"
    write_json({"clips": all_clips}, clips_path)
    summary["num_clips"] = len(all_clips)
    summary["clips_path"] = str(clips_path)

    # 5. Dynamic variant prompts via Gemini API (CPU + network, no GPU)
    try:
        variants = _warmup_dynamic_variants(config, facility)
        if variants:
            variants_path = cache_dir / "dynamic_variants.json"
            write_json({"variants": variants}, variants_path)
            summary["dynamic_variants_path"] = str(variants_path)
            summary["dynamic_variants_count"] = len(variants)
            logger.info("Cached %d dynamic variant prompts", len(variants))
    except Exception:
        logger.warning("Dynamic variant warmup failed (non-fatal)", exc_info=True)

    elapsed = round(time.time() - t0, 2)
    summary["elapsed_seconds"] = elapsed
    summary["warmup_complete"] = True

    write_json(summary, cache_dir / CACHE_MANIFEST)
    logger.info(
        "Warmup complete for %s: %d clips, %.1fs",
        facility.name,
        len(all_clips),
        elapsed,
    )
    return summary


def load_warmup_cache(work_dir: Path) -> Optional[Dict]:
    """Load warmup cache manifest if it exists and is complete."""
    manifest_path = work_dir / CACHE_DIR_NAME / CACHE_MANIFEST
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text())
        if data.get("warmup_complete"):
            return data
    except Exception:
        logger.debug("Failed to load warmup cache", exc_info=True)
    return None


def load_cached_clips(work_dir: Path) -> Optional[List[dict]]:
    """Load pre-computed camera paths and clip metadata from warmup cache."""
    cache = load_warmup_cache(work_dir)
    if cache is None:
        return None
    clips_path = cache.get("clips_path")
    if not clips_path or not Path(clips_path).exists():
        return None
    try:
        data = json.loads(Path(clips_path).read_text())
        return data.get("clips")
    except Exception:
        logger.debug("Failed to load cached clips", exc_info=True)
        return None


def load_cached_variants(work_dir: Path) -> Optional[List[VariantSpec]]:
    """Load pre-computed dynamic variants from warmup cache."""
    cache = load_warmup_cache(work_dir)
    if cache is None:
        return None
    variants_path = cache.get("dynamic_variants_path")
    if not variants_path or not Path(variants_path).exists():
        return None
    try:
        data = json.loads(Path(variants_path).read_text())
        return [VariantSpec(name=v["name"], prompt=v["prompt"]) for v in data.get("variants", [])]
    except Exception:
        logger.debug("Failed to load cached variants", exc_info=True)
        return None
