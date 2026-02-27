"""Camera path generation and loading for Gaussian splat rendering."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

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

    def viewmat(self) -> torch.Tensor:
        """Return world-to-camera (view) matrix as a (4,4) tensor."""
        w2c = np.linalg.inv(self.c2w)
        return torch.from_numpy(w2c.astype(np.float32))

    def K(self) -> torch.Tensor:
        """Return 3x3 intrinsics matrix."""
        return torch.tensor(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
            dtype=torch.float32,
        )


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
        eye = np.array([
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
            height,
        ])
        # Look toward center, slightly downward
        look_down_rad = math.radians(look_down_deg)
        target = np.array([
            center[0],
            center[1],
            height - radius * math.tan(look_down_rad),
        ])
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
        target = eye + direction * 2.0
        look_down_rad = math.radians(look_down_deg)
        target[2] = height - 2.0 * math.tan(look_down_rad)
        c2w = _look_at(eye, target)
        poses.append(CameraPose(c2w=c2w, fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h))

    return poses


def load_path_from_json(json_path: Path, resolution: tuple[int, int] = (480, 640)) -> List[CameraPose]:
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
) -> List[CameraPose]:
    """Generate a camera path from a CameraPathSpec."""
    if spec.type == "orbit":
        center = scene_center[:2] if start_offset is None else scene_center[:2] + start_offset[:2]
        center_3d = np.array([center[0], center[1], 0.0])
        return generate_orbit(
            center=center_3d,
            radius=spec.radius_m,
            height=camera_height,
            num_frames=num_frames,
            num_orbits=spec.num_orbits,
            look_down_deg=look_down_deg,
            resolution=resolution,
        )
    elif spec.type == "sweep":
        start = scene_center.copy()
        if start_offset is not None:
            start[:2] += start_offset[:2]
        start[2] = camera_height
        direction = np.array([1.0, 0.0, 0.0])
        return generate_sweep(
            start=start,
            direction=direction,
            length=spec.length_m,
            height=camera_height,
            num_frames=num_frames,
            look_down_deg=look_down_deg,
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
        frames.append({
            "camera_to_world": pose.c2w.flatten().tolist(),
            "fov": 2.0 * math.degrees(math.atan2(pose.width / 2.0, pose.fx)),
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"camera_path": frames}, f, indent=2)
