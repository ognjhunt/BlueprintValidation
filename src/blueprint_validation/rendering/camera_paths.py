"""Camera path generation and loading for Gaussian splat rendering."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from ..common import get_logger
from ..config import CameraPathSpec

logger = get_logger("rendering.camera_paths")


@dataclass
class CameraPose:
    """A single camera pose with extrinsics and intrinsics."""

    # Camera-to-world transform (4x4)
    c2w: np.ndarray  # (4, 4)
    # Intrinsics
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def position(self) -> np.ndarray:
        return self.c2w[:3, 3]

    @property
    def forward(self) -> np.ndarray:
        return -self.c2w[:3, 2]

    def viewmat(self) -> Any:
        """Return world-to-camera (view) matrix as a (4,4) tensor."""
        w2c = np.linalg.inv(self.c2w)
        try:
            import torch
        except ImportError:
            return w2c.astype(np.float32)
        return torch.from_numpy(w2c.astype(np.float32))

    def K(self) -> Any:
        """Return 3x3 intrinsics matrix."""
        mat = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
            dtype=np.float32,
        )
        try:
            import torch
        except ImportError:
            return mat
        return torch.tensor(mat, dtype=torch.float32)


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """Construct a camera-to-world matrix looking from eye toward target."""
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    new_up = np.cross(right, forward)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = new_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye
    return c2w


def generate_orbit(
    center: np.ndarray,
    radius: float,
    height: float,
    num_frames: int,
    num_orbits: int = 1,
    look_down_deg: float = 15.0,
    target_point: np.ndarray | None = None,
    resolution: tuple[int, int] = (480, 640),
    fov_deg: float = 60.0,
) -> List[CameraPose]:
    """Generate a circular orbit camera path around a center point."""
    h, w = resolution
    fx = fy = w / (2.0 * math.tan(math.radians(fov_deg / 2)))
    cx, cy = w / 2.0, h / 2.0

    poses = []
    for i in range(num_frames):
        angle = 2.0 * math.pi * num_orbits * i / num_frames
        eye = np.array(
            [
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
                height,
            ]
        )
        # Look toward explicit target when provided; otherwise keep legacy
        # look-down behavior around orbit center.
        if target_point is not None:
            target = np.asarray(target_point, dtype=np.float64)
        else:
            look_down_rad = math.radians(look_down_deg)
            target = np.array(
                [
                    center[0],
                    center[1],
                    height - radius * math.tan(look_down_rad),
                ]
            )
        c2w = _look_at(eye, target)
        poses.append(CameraPose(c2w=c2w, fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h))

    return poses


def generate_sweep(
    start: np.ndarray,
    direction: np.ndarray,
    length: float,
    height: float,
    num_frames: int,
    look_down_deg: float = 15.0,
    focus_point: np.ndarray | None = None,
    resolution: tuple[int, int] = (480, 640),
    fov_deg: float = 60.0,
) -> List[CameraPose]:
    """Generate a linear sweep camera path."""
    h, w = resolution
    fx = fy = w / (2.0 * math.tan(math.radians(fov_deg / 2)))
    cx, cy = w / 2.0, h / 2.0

    direction = direction / (np.linalg.norm(direction) + 1e-8)

    poses = []
    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        eye = start + direction * length * t
        eye[2] = height
        if focus_point is not None:
            target = np.asarray(focus_point, dtype=np.float64)
        else:
            target = eye + direction * 2.0
            look_down_rad = math.radians(look_down_deg)
            target[2] = height - 2.0 * math.tan(look_down_rad)
        c2w = _look_at(eye, target)
        poses.append(CameraPose(c2w=c2w, fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h))

    return poses


def generate_manipulation_arc(
    approach_point: np.ndarray,
    arc_radius: float,
    height: float,
    num_frames: int,
    look_down_deg: float = 45.0,
    target_z_bias_m: float = 0.0,
    arc_span_deg: float = 150.0,
    arc_phase_offset_deg: float = 0.0,
    resolution: tuple[int, int] = (480, 640),
    fov_deg: float = 60.0,
) -> List[CameraPose]:
    """Generate a tight arc camera path around a manipulation zone.

    The camera orbits at gripper height around the approach point and points
    directly at the interaction target by default.
    """
    h, w = resolution
    fx = fy = w / (2.0 * math.tan(math.radians(fov_deg / 2)))
    cx, cy = w / 2.0, h / 2.0

    approach = np.asarray(approach_point, dtype=np.float64)
    arc_span_rad = math.radians(float(max(20.0, min(330.0, arc_span_deg))))
    phase_offset_rad = math.radians(float(arc_phase_offset_deg))
    start_angle = phase_offset_rad - arc_span_rad / 2.0

    poses = []
    # Keep manipulation look-down meaningful but bounded: using the full
    # arc radius for pitch-derived height can push cameras to ceiling level
    # on large-radius paths and produce unusable captures.
    target_z = float(approach[2]) + float(target_z_bias_m)
    desired_pitch = math.radians(float(max(5.0, min(80.0, look_down_deg))))
    pitch_radius = min(float(arc_radius), 0.9)
    min_height_for_pitch = target_z + pitch_radius * math.tan(desired_pitch)
    effective_height = max(float(height), float(min_height_for_pitch))
    effective_height = min(effective_height, target_z + 1.8)
    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        angle = start_angle + arc_span_rad * t
        eye = np.array(
            [
                approach[0] + arc_radius * math.cos(angle),
                approach[1] + arc_radius * math.sin(angle),
                effective_height,
            ]
        )
        target = np.array(
            [
                approach[0],
                approach[1],
                target_z,
            ]
        )
        c2w = _look_at(eye, target)
        poses.append(CameraPose(c2w=c2w, fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h))

    return poses


def load_path_from_json(
    json_path: Path, resolution: tuple[int, int] = (480, 640)
) -> List[CameraPose]:
    """Load a camera path from a JSON file.

    Expected format:
    {
        "camera_path": [
            {
                "camera_to_world": [16 floats, row-major 4x4],
                "fov": 60.0  (optional, degrees)
            },
            ...
        ]
    }
    """
    with open(json_path) as f:
        data = json.load(f)

    h, w = resolution
    poses = []
    for frame in data["camera_path"]:
        c2w = np.array(frame["camera_to_world"], dtype=np.float64).reshape(4, 4)
        fov = frame.get("fov", 60.0)
        fx = fy = w / (2.0 * math.tan(math.radians(fov / 2)))
        cx, cy = w / 2.0, h / 2.0
        poses.append(CameraPose(c2w=c2w, fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h))

    logger.info("Loaded %d camera poses from %s", len(poses), json_path)
    return poses


def generate_path_from_spec(
    spec: CameraPathSpec,
    scene_center: np.ndarray,
    num_frames: int,
    camera_height: float,
    look_down_deg: float,
    resolution: tuple[int, int],
    start_offset: Optional[np.ndarray] = None,
    manipulation_target_z_bias_m: float = 0.0,
) -> List[CameraPose]:
    """Generate a camera path from a CameraPathSpec."""

    def _valid_approach_point3() -> np.ndarray | None:
        """Return a sanitized 3D approach point or None when malformed."""
        if not isinstance(spec.approach_point, list) or len(spec.approach_point) < 3:
            return None
        try:
            point3 = np.asarray(spec.approach_point[:3], dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if point3.shape != (3,) or not np.all(np.isfinite(point3)):
            return None
        return point3

    approach_point3 = _valid_approach_point3()

    if spec.type == "orbit":
        target_point = (
            approach_point3.copy()
            if approach_point3 is not None
            else np.asarray(scene_center[:3], dtype=np.float64)
        )
        center_xy = target_point[:2].copy()
        if start_offset is not None:
            center_xy = center_xy + start_offset[:2]
            target_point = target_point.copy()
            target_point[:2] = target_point[:2] + start_offset[:2]
        center_3d = np.array([center_xy[0], center_xy[1], target_point[2]], dtype=np.float64)
        effective_height = (
            float(spec.height_override_m) if spec.height_override_m is not None else float(camera_height)
        )
        effective_look_down = (
            float(spec.look_down_override_deg)
            if spec.look_down_override_deg is not None
            else float(look_down_deg)
        )
        # Keep camera slightly above the explicit target plane for stable views.
        if approach_point3 is not None:
            effective_height = max(effective_height, float(target_point[2]) + 0.25)
        return generate_orbit(
            center=center_3d,
            radius=spec.radius_m,
            height=effective_height,
            num_frames=num_frames,
            num_orbits=spec.num_orbits,
            look_down_deg=effective_look_down,
            target_point=target_point,
            resolution=resolution,
        )
    elif spec.type == "sweep":
        focus_point = (
            approach_point3.copy()
            if approach_point3 is not None
            else np.asarray(scene_center[:3], dtype=np.float64)
        )
        if start_offset is not None:
            focus_point = focus_point.copy()
            focus_point[:2] += start_offset[:2]
        effective_height = (
            float(spec.height_override_m) if spec.height_override_m is not None else float(camera_height)
        )
        effective_look_down = (
            float(spec.look_down_override_deg)
            if spec.look_down_override_deg is not None
            else float(look_down_deg)
        )
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        start = focus_point.copy()
        start[:2] = start[:2] - direction[:2] * (float(spec.length_m) * 0.5)
        start[2] = effective_height
        return generate_sweep(
            start=start,
            direction=direction,
            length=spec.length_m,
            height=effective_height,
            num_frames=num_frames,
            look_down_deg=effective_look_down,
            focus_point=focus_point,
            resolution=resolution,
        )
    elif spec.type == "manipulation":
        center = np.array(spec.approach_point or [0.0, 0.0, 0.0], dtype=np.float64)
        if start_offset is not None:
            center[:2] += start_offset[:2]
        effective_height = (
            spec.height_override_m if spec.height_override_m is not None else camera_height
        )
        effective_look_down = (
            spec.look_down_override_deg
            if spec.look_down_override_deg is not None
            else look_down_deg
        )
        return generate_manipulation_arc(
            approach_point=center,
            arc_radius=spec.arc_radius_m,
            height=effective_height,
            look_down_deg=effective_look_down,
            target_z_bias_m=float(manipulation_target_z_bias_m),
            arc_span_deg=float(spec.arc_span_deg),
            arc_phase_offset_deg=float(spec.arc_phase_offset_deg),
            num_frames=num_frames,
            resolution=resolution,
        )
    elif spec.type == "file":
        if spec.path is None:
            raise ValueError("Camera path spec type='file' requires a 'path' field")
        return load_path_from_json(Path(spec.path), resolution=resolution)
    else:
        raise ValueError(f"Unknown camera path type: {spec.type}")


def save_path_to_json(poses: List[CameraPose], output_path: Path) -> None:
    """Save a camera path to JSON."""
    frames = []
    for pose in poses:
        frames.append(
            {
                "camera_to_world": pose.c2w.flatten().tolist(),
                "fov": 2.0 * math.degrees(math.atan2(pose.width / 2.0, pose.fx)),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"camera_path": frames}, f, indent=2)
