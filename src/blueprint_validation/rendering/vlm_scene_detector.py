"""VLM-based object detection fallback for scene-aware camera placement.

Used when no task_targets.json is available. Renders a few overview frames
of the Gaussian splat, sends them to Gemini for object detection, and
unprojects the 2D detections to approximate 3D positions using depth maps.
"""

from __future__ import annotations

import json
import os
import time as _time
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..common import get_logger
from ..config import CameraPathSpec

logger = get_logger("rendering.vlm_scene_detector")

_MAX_RENDER_POINTS = 200_000
_POINT_RADIUS_PX = 2


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DetectedRegion:
    """A 3D region of interest detected via VLM + depth unprojection."""

    label: str
    center_3d: np.ndarray  # (3,) approximate world position
    extents_3d: np.ndarray = field(
        default_factory=lambda: np.array([0.3, 0.3, 0.3])
    )  # (3,) estimated 3D bounding extents
    confidence: float = 0.8
    frame_index: int = 0
    category: str = "manipulation"  # manipulation | articulation | navigation


@dataclass
class SceneDetectionResult:
    """Full result of VLM scene detection including metadata."""

    specs: List[CameraPathSpec]
    detections: List[DetectedRegion]
    scene_type: str = "unknown"
    suggested_tasks: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _unproject_pixel_to_3d(
    u: float,
    v: float,
    depth: float,
    c2w: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Unproject a 2D pixel + depth to 3D world coordinates."""
    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = depth
    point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
    point_world = c2w @ point_cam
    return point_world[:3]


def _estimate_3d_extents_from_bbox(
    bbox: list,
    depth: np.ndarray,
    fx: float,
    fy: float,
) -> np.ndarray:
    """Estimate 3D object extents from a 2D bbox + depth map + intrinsics."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = depth.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return np.array([0.3, 0.3, 0.3])

    patch = depth[y1:y2, x1:x2]
    valid = patch[(patch > 0) & (patch < 100)]
    if len(valid) == 0:
        return np.array([0.3, 0.3, 0.3])

    median_d = float(np.median(valid))
    extent_x = float((x2 - x1) * median_d / fx)
    extent_y = float((y2 - y1) * median_d / fy)
    extent_z = max(float(valid.max() - valid.min()), 0.1)

    return np.clip([extent_x, extent_y, extent_z], 0.05, 3.0)


# ---------------------------------------------------------------------------
# Detection clustering / merging
# ---------------------------------------------------------------------------


def _cluster_detections(
    detections: List[DetectedRegion],
    radius: Optional[float] = None,
) -> List[DetectedRegion]:
    """Merge detections within *radius* meters of each other.

    If *radius* is None, uses an adaptive radius based on the median max-extent
    of all detections, floored at 0.3 m.
    """
    if not detections:
        return []

    if radius is None:
        max_exts = [float(np.max(d.extents_3d)) for d in detections]
        radius = max(0.3, float(np.median(max_exts)) * 0.5) if max_exts else 0.5

    used = [False] * len(detections)
    merged: List[DetectedRegion] = []

    for i, det in enumerate(detections):
        if used[i]:
            continue
        cluster_points = [det.center_3d]
        cluster_extents = [det.extents_3d]
        cluster_labels = [det.label]
        cluster_categories = [det.category]
        best_conf = det.confidence
        best_label = det.label
        used[i] = True

        for j in range(i + 1, len(detections)):
            if used[j]:
                continue
            if np.linalg.norm(det.center_3d - detections[j].center_3d) < radius:
                cluster_points.append(detections[j].center_3d)
                cluster_extents.append(detections[j].extents_3d)
                cluster_labels.append(detections[j].label)
                cluster_categories.append(detections[j].category)
                if detections[j].confidence > best_conf:
                    best_conf = detections[j].confidence
                    best_label = detections[j].label
                used[j] = True

        center = np.mean(cluster_points, axis=0)
        avg_ext = np.mean(cluster_extents, axis=0)
        # Most common category in cluster
        cat_counts = Counter(cluster_categories)
        best_cat = cat_counts.most_common(1)[0][0]

        merged.append(
            DetectedRegion(
                label=best_label,
                center_3d=center,
                extents_3d=avg_ext,
                confidence=best_conf,
                category=best_cat,
            )
        )

    return merged


# ---------------------------------------------------------------------------
# VLM prompt and API calls
# ---------------------------------------------------------------------------


_VLM_SCENE_PROMPT = """\
Analyze this image of a 3D environment rendered from a point cloud. \
Identify regions of interest for a robot operating in this space.

1. Classify the scene type (e.g. "warehouse", "kitchen", "office", "laboratory", \
"factory_floor", "outdoor", "retail", "medical", "residential", or "unknown").

2. Identify distinct objects or regions a robot could interact with. For each, provide:
   - label: short descriptive name (e.g. "cardboard_box", "coffee_mug", "cabinet_door")
   - bbox: pixel bounding box as [x1, y1, x2, y2]
   - category: one of "manipulation" (pick/place/push), "articulation" (open/close/turn), \
or "navigation" (approach/avoid)
   - suggested_task: a short imperative sentence for the robot (e.g. "Pick up the box")

Return ONLY valid JSON:
{
  "scene_type": "warehouse",
  "objects": [
    {"label": "cardboard_box", "bbox": [120, 200, 250, 340], "category": "manipulation", \
"suggested_task": "Pick up the box"},
    {"label": "cabinet_door", "bbox": [300, 100, 400, 350], "category": "articulation", \
"suggested_task": "Open the cabinet door"}
  ]
}

Return {"scene_type": "unknown", "objects": []} if nothing interactable is visible.
"""


def _generate_with_retry(client, *, model, contents, max_retries: int = 3):
    """Call generate_content with exponential backoff on transient errors."""
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except Exception as exc:
            exc_text = str(exc).lower()
            transient = any(
                kw in exc_text
                for kw in ("rate limit", "429", "500", "503", "timeout", "unavailable", "deadline")
            )
            if not transient or attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning(
                "Transient VLM API error (attempt %d/%d), retrying in %ds: %s",
                attempt + 1, max_retries, wait, exc,
            )
            _time.sleep(wait)


def _extract_json(text: str) -> Optional[object]:
    """Extract JSON from a response that may be wrapped in markdown fences."""
    text = text.strip()
    if "```" in text:
        start = text.index("```") + 3
        if text[start:].startswith("json"):
            start += 4
        try:
            end = text.index("```", start)
        except ValueError:
            end = len(text)
        text = text[start:end].strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


_EMPTY_DETECTION = {"scene_type": "unknown", "objects": []}


def _call_gemini_detect(
    rgb_frame: np.ndarray,
    model: str = "gemini-3-flash-preview",
    max_retries: int = 3,
) -> dict:
    """Send a single frame to Gemini and parse structured scene analysis.

    Returns ``{"scene_type": str, "objects": list}``.
    """
    api_key = os.environ.get("GOOGLE_GENAI_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_GENAI_API_KEY not set; skipping VLM detection")
        return dict(_EMPTY_DETECTION)

    try:
        from google import genai
        from PIL import Image
    except ImportError:
        logger.warning("google-genai or Pillow not installed; skipping VLM detection")
        return dict(_EMPTY_DETECTION)

    client = genai.Client(api_key=api_key)
    image = Image.fromarray(rgb_frame)

    try:
        response = _generate_with_retry(
            client, model=model, contents=[_VLM_SCENE_PROMPT, image], max_retries=max_retries,
        )
    except Exception:
        logger.warning("VLM detection API call failed", exc_info=True)
        return dict(_EMPTY_DETECTION)

    response_text = str(getattr(response, "text", "") or "")
    parsed = _extract_json(response_text)
    if parsed is None:
        logger.warning("Failed to parse VLM detection response: %s", response_text[:200])
        return dict(_EMPTY_DETECTION)

    if isinstance(parsed, list):
        # Backward compat: old-style bare list response
        return {"scene_type": "unknown", "objects": parsed}
    if not isinstance(parsed, dict):
        return dict(_EMPTY_DETECTION)
    if "objects" not in parsed:
        # Backward compat: old-style bare list response
        return dict(_EMPTY_DETECTION)
    return parsed


# ---------------------------------------------------------------------------
# Point-cloud rendering (CPU-only)
# ---------------------------------------------------------------------------


def _render_colored_point_cloud(
    splat_means_np: np.ndarray,
    splat_colors: Optional[np.ndarray],
    pose,
    resolution: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project colored points to a camera view.

    Returns (rgb_image, depth_image).
    """
    h, w = resolution
    w2c = np.linalg.inv(pose.c2w)
    pts_cam = (w2c[:3, :3] @ splat_means_np.T).T + w2c[:3, 3]

    # Only keep points in front of camera
    mask = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[mask]
    colors_filtered = splat_colors[mask] if splat_colors is not None else None

    if len(pts_cam) == 0:
        return np.zeros((h, w, 3), dtype=np.uint8), np.zeros((h, w), dtype=np.float32)

    # Subsample for performance
    if len(pts_cam) > _MAX_RENDER_POINTS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts_cam), size=_MAX_RENDER_POINTS, replace=False)
        pts_cam = pts_cam[idx]
        if colors_filtered is not None:
            colors_filtered = colors_filtered[idx]

    # Project to pixel coordinates
    depths = pts_cam[:, 2].astype(np.float32)
    us = (pts_cam[:, 0] * pose.fx / depths + pose.cx).astype(np.int32)
    vs = (pts_cam[:, 1] * pose.fy / depths + pose.cy).astype(np.int32)

    # Filter to in-bounds
    in_bounds = (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
    us, vs, depths = us[in_bounds], vs[in_bounds], depths[in_bounds]
    if colors_filtered is not None:
        colors_filtered = colors_filtered[in_bounds]

    if len(us) == 0:
        return np.zeros((h, w, 3), dtype=np.uint8), np.zeros((h, w), dtype=np.float32)

    # Sort front-to-back for proper occlusion
    order = np.argsort(depths)
    us, vs, depths = us[order], vs[order], depths[order]
    if colors_filtered is not None:
        colors_filtered = colors_filtered[order]

    # Build depth image (nearest depth per pixel)
    depth_img = np.full((h, w), fill_value=np.inf, dtype=np.float32)
    np.minimum.at(depth_img, (vs, us), depths)
    depth_img[depth_img == np.inf] = 0

    # Build RGB image
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

    if colors_filtered is not None:
        # Splat colored disks — iterate front-to-back, write only first hit per pixel
        written = np.zeros((h, w), dtype=bool)
        r = _POINT_RADIUS_PX
        for i in range(len(us)):
            u, v = int(us[i]), int(vs[i])
            y_lo, y_hi = max(0, v - r), min(h, v + r + 1)
            x_lo, x_hi = max(0, u - r), min(w, u + r + 1)
            patch = written[y_lo:y_hi, x_lo:x_hi]
            color = colors_filtered[i]
            for dy in range(y_hi - y_lo):
                for dx in range(x_hi - x_lo):
                    if not patch[dy, dx]:
                        rgb_img[y_lo + dy, x_lo + dx] = color
                        patch[dy, dx] = True
    else:
        # Grayscale depth fallback (current behavior)
        valid = depth_img > 0
        if valid.any():
            d_min, d_max = depth_img[valid].min(), depth_img[valid].max()
            if d_max > d_min:
                norm = ((d_max - depth_img) / (d_max - d_min) * 200 + 55).astype(np.uint8)
                norm[~valid] = 0
                rgb_img[:, :, 0] = rgb_img[:, :, 1] = rgb_img[:, :, 2] = norm

    return rgb_img, depth_img


# ---------------------------------------------------------------------------
# Multi-view detection orchestrator
# ---------------------------------------------------------------------------


def detect_objects_in_scene(
    rgb_frames: List[np.ndarray],
    depth_frames: List[np.ndarray],
    poses: list,
    model: str = "gemini-3-flash-preview",
    max_retries: int = 3,
) -> tuple[List[DetectedRegion], str, List[dict]]:
    """Detect objects across multiple rendered views and unproject to 3D.

    Returns (merged_detections, consensus_scene_type, suggested_tasks).
    """
    all_detections: List[DetectedRegion] = []
    scene_type_votes: List[str] = []
    suggested_tasks: List[dict] = []
    seen_tasks: set = set()

    for i, (rgb, depth, pose) in enumerate(zip(rgb_frames, depth_frames, poses)):
        result = _call_gemini_detect(rgb, model=model, max_retries=max_retries)

        scene_type_votes.append(result.get("scene_type", "unknown"))

        for det in result.get("objects", []):
            bbox = det.get("bbox")
            label = det.get("label", "object")
            category = det.get("category", "manipulation")
            if category not in ("manipulation", "articulation", "navigation"):
                category = "manipulation"

            suggested = det.get("suggested_task")
            if isinstance(suggested, str):
                suggested_clean = suggested.strip()
                if suggested_clean:
                    key = suggested_clean.lower()
                    if key not in seen_tasks:
                        suggested_tasks.append(
                            {
                                "suggested_task": suggested_clean,
                                "label": label,
                                "category": category,
                            }
                        )
                        seen_tasks.add(key)

            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            cu = (x1 + x2) / 2.0
            cv = (y1 + y2) / 2.0

            h, w = depth.shape[:2]
            cu = max(0, min(cu, w - 1))
            cv = max(0, min(cv, h - 1))

            # Use median valid depth in the bbox region instead of a single
            # center pixel — sparse splat projections can leave the center
            # pixel empty while the rest of the bbox has valid depth.
            bx1, bx2 = max(0, int(x1)), min(w, int(x2))
            by1, by2 = max(0, int(y1)), min(h, int(y2))
            if bx2 > bx1 and by2 > by1:
                patch = depth[by1:by2, bx1:bx2]
                valid = patch[(patch > 0) & (patch < 100)]
                d = float(np.median(valid)) if len(valid) > 0 else 0.0
            else:
                d = float(depth[int(cv), int(cu)])
            if d <= 0 or d > 100:
                continue

            center_3d = _unproject_pixel_to_3d(
                cu, cv, d, pose.c2w, pose.fx, pose.fy, pose.cx, pose.cy
            )
            extents_3d = _estimate_3d_extents_from_bbox(bbox, depth, pose.fx, pose.fy)

            all_detections.append(
                DetectedRegion(
                    label=label,
                    center_3d=center_3d,
                    extents_3d=extents_3d,
                    frame_index=i,
                    category=category,
                )
            )

    # Consensus scene type by majority vote (ignore "unknown")
    real_votes = [v for v in scene_type_votes if v != "unknown"]
    if real_votes:
        scene_type = Counter(real_votes).most_common(1)[0][0]
    else:
        scene_type = "unknown"

    merged = _cluster_detections(all_detections)
    logger.info(
        "VLM detected %d objects across %d views (merged to %d), scene_type=%s",
        len(all_detections),
        len(rgb_frames),
        len(merged),
        scene_type,
    )
    return merged, scene_type, suggested_tasks


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def detect_and_generate_specs(
    splat_means_np: np.ndarray,
    scene_center: np.ndarray,
    num_views: int = 4,
    model: str = "gemini-3-flash-preview",
    resolution: tuple[int, int] = (480, 640),
    splat_colors: Optional[np.ndarray] = None,
    max_retries: int = 3,
) -> SceneDetectionResult:
    """Full fallback pipeline: render overview frames, detect objects, generate camera specs.

    CPU-only — uses point-cloud projection (colored if ``splat_colors`` is provided,
    otherwise grayscale depth-coded). Returns a :class:`SceneDetectionResult` with
    camera specs, 3D detections, scene classification, and suggested tasks.
    """
    from .camera_paths import generate_orbit
    from .scene_geometry import OrientedBoundingBox, compute_camera_height, compute_standoff_distance

    orbit_poses = generate_orbit(
        center=scene_center,
        radius=3.0,
        height=1.5,
        num_frames=num_views,
        num_orbits=1,
        look_down_deg=20.0,
        resolution=resolution,
    )

    rgb_frames: List[np.ndarray] = []
    depth_frames: List[np.ndarray] = []

    for pose in orbit_poses:
        rgb, depth = _render_colored_point_cloud(
            splat_means_np, splat_colors, pose, resolution,
        )
        rgb_frames.append(rgb)
        depth_frames.append(depth)

    if not rgb_frames:
        return SceneDetectionResult(specs=[], detections=[])

    detections, scene_type, suggested_tasks = detect_objects_in_scene(
        rgb_frames, depth_frames, orbit_poses, model=model, max_retries=max_retries,
    )

    # Convert detections to camera path specs
    specs: List[CameraPathSpec] = []
    for det in detections:
        obb = OrientedBoundingBox(
            instance_id=f"vlm_{det.label}",
            label=det.label,
            center=det.center_3d,
            extents=det.extents_3d,
            axes=np.eye(3),
            category=det.category,
        )
        standoff = compute_standoff_distance(obb)
        cam_height = compute_camera_height(obb)

        specs.append(
            CameraPathSpec(
                type="manipulation",
                approach_point=det.center_3d.tolist(),
                arc_radius_m=standoff,
                height_override_m=cam_height,
                look_down_override_deg=45.0,
            )
        )

    logger.info("VLM fallback generated %d camera specs (scene_type=%s)", len(specs), scene_type)
    return SceneDetectionResult(
        specs=specs,
        detections=detections,
        scene_type=scene_type,
        suggested_tasks=suggested_tasks,
    )
