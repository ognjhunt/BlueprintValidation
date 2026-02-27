"""Robot arm compositing with URDF kinematics and camera extrinsics."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..common import get_logger

logger = get_logger("synthetic.robot_compositor")


@dataclass
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray


@dataclass
class CameraPose:
    c2w: np.ndarray
    fov_deg: float


@dataclass
class CompositeMetrics:
    clip_name: str
    mean_visible_joint_ratio: float
    mean_segment_length_px: float
    geometry_consistency_score: float
    passed: bool

    def to_dict(self) -> dict:
        return {
            "clip_name": self.clip_name,
            "mean_visible_joint_ratio": round(self.mean_visible_joint_ratio, 4),
            "mean_segment_length_px": round(self.mean_segment_length_px, 3),
            "geometry_consistency_score": round(self.geometry_consistency_score, 4),
            "passed": self.passed,
        }


def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    rx, ry, rz = rpy
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def _homogeneous_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = _rotation_from_rpy(rpy)
    mat[:3, 3] = xyz
    return mat


def _axis_angle_transform(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    rot = np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    return out


def load_urdf_chain(urdf_path: Path, end_effector_link: str | None = None) -> List[JointSpec]:
    """Load a serial chain of joints from URDF."""
    if not urdf_path.exists():
        raise RuntimeError(f"URDF not found: {urdf_path}")
    root = ET.fromstring(urdf_path.read_text())
    joint_nodes = root.findall("joint")
    joints: List[JointSpec] = []
    for node in joint_nodes:
        parent = node.find("parent")
        child = node.find("child")
        if parent is None or child is None:
            continue
        origin = node.find("origin")
        axis_node = node.find("axis")
        xyz = np.zeros(3, dtype=np.float64)
        rpy = np.zeros(3, dtype=np.float64)
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if origin is not None:
            xyz_txt = origin.attrib.get("xyz")
            rpy_txt = origin.attrib.get("rpy")
            if xyz_txt:
                xyz = np.array([float(v) for v in xyz_txt.split()], dtype=np.float64)
            if rpy_txt:
                rpy = np.array([float(v) for v in rpy_txt.split()], dtype=np.float64)
        if axis_node is not None and axis_node.attrib.get("xyz"):
            axis = np.array([float(v) for v in axis_node.attrib["xyz"].split()], dtype=np.float64)

        joints.append(
            JointSpec(
                name=node.attrib.get("name", child.attrib["link"]),
                joint_type=node.attrib.get("type", "fixed"),
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                origin_xyz=xyz,
                origin_rpy=rpy,
                axis=axis,
            )
        )

    if not joints:
        raise RuntimeError(f"No joints parsed from URDF: {urdf_path}")

    by_parent: Dict[str, List[JointSpec]] = {}
    child_links = set()
    for j in joints:
        by_parent.setdefault(j.parent, []).append(j)
        child_links.add(j.child)
    root_candidates = [j.parent for j in joints if j.parent not in child_links]
    current_link = root_candidates[0] if root_candidates else joints[0].parent

    chain: List[JointSpec] = []
    visited = set()
    while current_link in by_parent:
        candidates = by_parent[current_link]
        picked = None
        if end_effector_link:
            for c in candidates:
                if c.child == end_effector_link:
                    picked = c
                    break
        if picked is None:
            picked = candidates[0]
        if picked.name in visited:
            break
        visited.add(picked.name)
        chain.append(picked)
        current_link = picked.child
        if end_effector_link and current_link == end_effector_link:
            break

    if not chain:
        raise RuntimeError(f"Could not build a serial chain from URDF: {urdf_path}")
    return chain


def load_camera_path(camera_path_json: Path) -> List[CameraPose]:
    raw = json.loads(camera_path_json.read_text())
    poses = []
    for p in raw.get("camera_path", []):
        mat = np.array(p["camera_to_world"], dtype=np.float64).reshape(4, 4)
        poses.append(CameraPose(c2w=mat, fov_deg=float(p.get("fov", 60.0))))
    return poses


def _forward_kinematics_world_points(
    chain: List[JointSpec],
    joint_values: np.ndarray,
    base_xyz: np.ndarray,
    base_rpy: np.ndarray,
) -> np.ndarray:
    t = _homogeneous_transform(base_xyz, base_rpy)
    points = [t[:3, 3].copy()]
    for idx, joint in enumerate(chain):
        t = t @ _homogeneous_transform(joint.origin_xyz, joint.origin_rpy)
        if joint.joint_type in {"revolute", "continuous"}:
            angle = float(joint_values[min(idx, len(joint_values) - 1)])
            t = t @ _axis_angle_transform(joint.axis, angle)
        points.append(t[:3, 3].copy())
    return np.asarray(points)


def _project_points(
    points_world: np.ndarray,
    camera_pose: CameraPose,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    w2c = np.linalg.inv(camera_pose.c2w)
    points_h = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1)
    cam = (w2c @ points_h.T).T
    z = cam[:, 2]
    focal = width / (2.0 * math.tan(math.radians(camera_pose.fov_deg / 2.0)))
    x = (cam[:, 0] * focal / (z + 1e-8)) + (width / 2.0)
    y = (cam[:, 1] * focal / (z + 1e-8)) + (height / 2.0)
    pix = np.stack([x, y], axis=1)
    visible = (z > 1e-4) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return pix, visible


def _generate_joint_trajectory(
    num_frames: int,
    num_joints: int,
    start: List[float],
    end: List[float],
) -> np.ndarray:
    s = np.asarray(start, dtype=np.float64)
    e = np.asarray(end, dtype=np.float64)
    if len(s) < num_joints:
        s = np.pad(s, (0, num_joints - len(s)))
    if len(e) < num_joints:
        e = np.pad(e, (0, num_joints - len(e)))
    t = np.linspace(0.0, 1.0, num_frames)
    return s[None, :] * (1.0 - t[:, None]) + e[None, :] * t[:, None]


def composite_robot_arm_into_clip(
    input_video: Path,
    output_video: Path,
    camera_path_json: Path,
    urdf_path: Path,
    base_xyz: List[float],
    base_rpy: List[float],
    start_joints: List[float],
    end_joints: List[float],
    line_color_bgr: tuple[int, int, int] = (50, 180, 255),
    line_thickness: int = 3,
    min_visible_joint_ratio: float = 0.6,
    min_consistency_score: float = 0.6,
    end_effector_link: str | None = None,
) -> CompositeMetrics:
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required for robot compositing. Install dependency 'opencv-python'."
        ) from e

    if not input_video.exists():
        raise RuntimeError(f"Input video not found: {input_video}")
    camera_poses = load_camera_path(camera_path_json)
    chain = load_urdf_chain(urdf_path, end_effector_link=end_effector_link)

    cap = cv2.VideoCapture(str(input_video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    joint_traj = _generate_joint_trajectory(
        num_frames=max(num_frames, 1),
        num_joints=len(chain),
        start=start_joints,
        end=end_joints,
    )
    visible_ratios: List[float] = []
    seg_lengths: List[float] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pose = camera_poses[min(frame_idx, len(camera_poses) - 1)] if camera_poses else CameraPose(
            c2w=np.eye(4),
            fov_deg=60.0,
        )
        joints_world = _forward_kinematics_world_points(
            chain=chain,
            joint_values=joint_traj[min(frame_idx, len(joint_traj) - 1)],
            base_xyz=np.asarray(base_xyz, dtype=np.float64),
            base_rpy=np.asarray(base_rpy, dtype=np.float64),
        )
        pix, visible = _project_points(
            points_world=joints_world,
            camera_pose=pose,
            width=width,
            height=height,
        )
        visible_ratio = float(np.mean(visible.astype(np.float64)))
        visible_ratios.append(visible_ratio)

        for i in range(len(pix) - 1):
            p0 = tuple(int(v) for v in pix[i])
            p1 = tuple(int(v) for v in pix[i + 1])
            if visible[i] or visible[i + 1]:
                cv2.line(frame, p0, p1, line_color_bgr, line_thickness, cv2.LINE_AA)
                cv2.circle(frame, p0, max(2, line_thickness // 2), (0, 255, 255), -1)
                seg_lengths.append(float(np.linalg.norm(pix[i + 1] - pix[i])))

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    mean_visible = float(np.mean(visible_ratios)) if visible_ratios else 0.0
    mean_seg_len = float(np.mean(seg_lengths)) if seg_lengths else 0.0
    # Reward frames with enough visible joints and non-degenerate projected geometry.
    consistency = min(
        1.0,
        (mean_visible / max(min_visible_joint_ratio, 1e-6))
        * min(1.0, mean_seg_len / 12.0),
    )
    passed = (mean_visible >= min_visible_joint_ratio) and (consistency >= min_consistency_score)
    return CompositeMetrics(
        clip_name=input_video.stem,
        mean_visible_joint_ratio=mean_visible,
        mean_segment_length_px=mean_seg_len,
        geometry_consistency_score=consistency,
        passed=passed,
    )
