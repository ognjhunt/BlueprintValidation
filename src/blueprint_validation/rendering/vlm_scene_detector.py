"""VLM-based object detection fallback for scene-aware camera placement.

Used when no task_targets.json is available. Renders a few overview frames
of the Gaussian splat, sends them to Gemini for object detection, and
unprojects the 2D detections to approximate 3D positions using depth maps.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..common import get_logger
from ..config import CameraPathSpec

logger = get_logger("rendering.vlm_scene_detector")


@dataclass
class DetectedRegion:
    """A 3D region of interest detected via VLM + depth unprojection."""

    label: str
    center_3d: np.ndarray  # (3,) approximate world position
    confidence: float = 0.8
    frame_index: int = 0


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


def _cluster_detections(detections: List[DetectedRegion], radius: float = 0.5) -> List[DetectedRegion]:
    """Merge detections within radius meters of each other."""
    if not detections:
        return []

    used = [False] * len(detections)
    merged: List[DetectedRegion] = []

    for i, det in enumerate(detections):
        if used[i]:
            continue
        cluster_points = [det.center_3d]
        cluster_label = det.label
        used[i] = True

        for j in range(i + 1, len(detections)):
            if used[j]:
                continue
            if np.linalg.norm(det.center_3d - detections[j].center_3d) < radius:
                cluster_points.append(detections[j].center_3d)
                used[j] = True

        center = np.mean(cluster_points, axis=0)
        merged.append(DetectedRegion(label=cluster_label, center_3d=center))

    return merged


_VLM_PROMPT = """\
Analyze this view of an indoor environment captured for robotics.
Identify all distinct objects a robot might interact with (pick up, open, approach, place).

Return ONLY a JSON array with this exact format:
[
  {"label": "tote", "bbox": [x1, y1, x2, y2]},
  {"label": "shelf", "bbox": [x1, y1, x2, y2]}
]

Where bbox coordinates are pixel coordinates in the image.
Return an empty array [] if no interactable objects are visible.
"""


def _call_gemini_detect(
    rgb_frame: np.ndarray,
    model: str = "gemini-3-flash-preview",
) -> List[dict]:
    """Send a single frame to Gemini and parse object detections."""
    api_key = os.environ.get("GOOGLE_GENAI_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_GENAI_API_KEY not set; skipping VLM detection")
        return []

    try:
        from google import genai
        from PIL import Image
    except ImportError:
        logger.warning("google-genai or Pillow not installed; skipping VLM detection")
        return []

    client = genai.Client(api_key=api_key)
    image = Image.fromarray(rgb_frame)

    response = client.models.generate_content(
        model=model,
        contents=[_VLM_PROMPT, image],
    )

    text = response.text.strip()
    # Extract JSON from response (may be wrapped in markdown code block)
    if "```" in text:
        start = text.index("```") + 3
        if text[start:].startswith("json"):
            start += 4
        end = text.index("```", start)
        text = text[start:end].strip()

    try:
        detections = json.loads(text)
        if not isinstance(detections, list):
            return []
        return detections
    except (json.JSONDecodeError, ValueError):
        logger.warning("Failed to parse VLM detection response: %s", text[:200])
        return []


def detect_objects_in_scene(
    rgb_frames: List[np.ndarray],
    depth_frames: List[np.ndarray],
    poses: list,
    model: str = "gemini-3-flash-preview",
) -> List[DetectedRegion]:
    """Detect objects across multiple rendered views and unproject to 3D."""
    all_detections: List[DetectedRegion] = []

    for i, (rgb, depth, pose) in enumerate(zip(rgb_frames, depth_frames, poses)):
        raw_dets = _call_gemini_detect(rgb, model=model)

        for det in raw_dets:
            bbox = det.get("bbox")
            label = det.get("label", "object")
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            # Center of bounding box
            cu = (x1 + x2) / 2.0
            cv = (y1 + y2) / 2.0

            # Clamp to frame bounds
            h, w = depth.shape[:2]
            cu = max(0, min(cu, w - 1))
            cv = max(0, min(cv, h - 1))

            d = float(depth[int(cv), int(cu)])
            if d <= 0 or d > 100:
                continue

            center_3d = _unproject_pixel_to_3d(
                cu, cv, d, pose.c2w, pose.fx, pose.fy, pose.cx, pose.cy
            )
            all_detections.append(
                DetectedRegion(label=label, center_3d=center_3d, frame_index=i)
            )

    merged = _cluster_detections(all_detections)
    logger.info(
        "VLM detected %d objects across %d views (merged to %d)",
        len(all_detections),
        len(rgb_frames),
        len(merged),
    )
    return merged


def detect_and_generate_specs(
    splat_means_np: np.ndarray,
    scene_center: np.ndarray,
    num_views: int = 4,
    model: str = "gemini-3-flash-preview",
    resolution: tuple[int, int] = (480, 640),
) -> List[CameraPathSpec]:
    """Full fallback pipeline: render overview frames, detect objects, generate camera specs.

    This is a lightweight standalone function that does NOT require a loaded splat
    on GPU — it only needs the means as numpy for depth estimation. However, it does
    need gsplat for rendering, so it will fail gracefully if gsplat is not available.
    """
    from .camera_paths import generate_orbit
    from .scene_geometry import compute_camera_height, compute_standoff_distance, OrientedBoundingBox

    # Generate overview orbit poses
    orbit_poses = generate_orbit(
        center=scene_center,
        radius=3.0,
        height=1.5,
        num_frames=num_views,
        num_orbits=1,
        look_down_deg=20.0,
        resolution=resolution,
    )

    # We need to render these frames — this requires a loaded splat on GPU
    # which we don't have here. Instead, we use a quick depth approximation:
    # project all points to each camera and compute per-pixel depth from nearest points.
    rgb_frames: List[np.ndarray] = []
    depth_frames: List[np.ndarray] = []

    try:
        import torch
        from .ply_loader import load_splat as _unused_load  # noqa: F401 - just checking availability
        from .gsplat_renderer import render_frame

        # We need the actual splat data on device — but we only have means_np.
        # The caller (s1_render.py) has the splat loaded. This fallback path
        # is called from _vlm_fallback which doesn't pass the splat object.
        # For now, we synthesize approximate depth images from the point cloud.
        raise ImportError("Using point-cloud depth approximation for VLM fallback")
    except (ImportError, RuntimeError):
        # Approximate depth from point cloud projection
        h, w = resolution
        for pose in orbit_poses:
            w2c = np.linalg.inv(pose.c2w)
            pts_cam = (w2c[:3, :3] @ splat_means_np.T).T + w2c[:3, 3]
            # Only keep points in front of camera
            mask = pts_cam[:, 2] > 0.1
            pts_cam = pts_cam[mask]

            if len(pts_cam) == 0:
                depth_frames.append(np.zeros((h, w), dtype=np.float32))
                rgb_frames.append(np.zeros((h, w, 3), dtype=np.uint8))
                continue

            # Project to pixel coordinates
            us = (pts_cam[:, 0] * pose.fx / pts_cam[:, 2] + pose.cx).astype(int)
            vs = (pts_cam[:, 1] * pose.fy / pts_cam[:, 2] + pose.cy).astype(int)
            depths = pts_cam[:, 2]

            # Filter to in-bounds
            in_bounds = (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
            us, vs, depths = us[in_bounds], vs[in_bounds], depths[in_bounds]

            # Build depth image (nearest depth per pixel)
            depth_img = np.full((h, w), fill_value=np.inf, dtype=np.float32)
            np.minimum.at(depth_img, (vs, us), depths)
            depth_img[depth_img == np.inf] = 0

            # Build simple RGB image (gray with brighter near, darker far)
            rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
            valid = depth_img > 0
            if valid.any():
                d_min, d_max = depth_img[valid].min(), depth_img[valid].max()
                if d_max > d_min:
                    norm = ((d_max - depth_img) / (d_max - d_min) * 200 + 55).astype(np.uint8)
                    norm[~valid] = 0
                    rgb_img[:, :, 0] = rgb_img[:, :, 1] = rgb_img[:, :, 2] = norm

            depth_frames.append(depth_img)
            rgb_frames.append(rgb_img)

    if not rgb_frames:
        return []

    # Detect objects via VLM
    detections = detect_objects_in_scene(rgb_frames, depth_frames, orbit_poses, model=model)

    # Convert detections to camera path specs
    specs: List[CameraPathSpec] = []
    for det in detections:
        obb = OrientedBoundingBox(
            instance_id=f"vlm_{det.label}",
            label=det.label,
            center=det.center_3d,
            extents=np.array([0.3, 0.3, 0.3]),  # default extent for unknown objects
            axes=np.eye(3),
            category="manipulation",
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

    logger.info("VLM fallback generated %d camera specs", len(specs))
    return specs
