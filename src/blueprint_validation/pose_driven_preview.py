from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import json
import math
import threading
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .optional_dependencies import require_optional_dependency


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _resolve_local_path(*values: Any) -> Optional[Path]:
    for raw in values:
        value = str(raw or "").strip()
        if not value or value.startswith(("gs://", "http://", "https://")):
            continue
        path = Path(value).expanduser().resolve()
        if path.exists():
            return path
    return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if isinstance(record, Mapping):
                rows.append(dict(record))
    return rows


def _rotation_y(angle_radians: float) -> np.ndarray:
    c = math.cos(angle_radians)
    s = math.sin(angle_radians)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def _rotation_x(angle_radians: float) -> np.ndarray:
    c = math.cos(angle_radians)
    s = math.sin(angle_radians)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )


def _transform_matrix(
    translation: Sequence[float],
    *,
    yaw_radians: float = 0.0,
    pitch_radians: float = 0.0,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = _rotation_y(yaw_radians) @ _rotation_x(pitch_radians)
    transform[:3, 3] = np.asarray(list(translation)[:3], dtype=np.float32)
    return transform


@dataclass(frozen=True)
class PoseSample:
    frame_id: str
    t_device_sec: float
    world_from_camera: np.ndarray
    camera_from_world: np.ndarray
    camera_center_world: np.ndarray
    forward_world: np.ndarray
    video_frame_index: int
    depth_path: Path
    confidence_path: Optional[Path]


@dataclass
class SpatialPreviewScene:
    site_world_id: str
    video_path: Path
    intrinsics_width: int
    intrinsics_height: int
    fx: float
    fy: float
    cx: float
    cy: float
    video_fps: float
    video_frame_count: int
    pose_samples: list[PoseSample]
    anchor_pose: PoseSample
    preview_width: int = 640
    preview_height: int = 480
    _capture_lock: threading.Lock = field(default_factory=threading.Lock)
    _frame_cache: OrderedDict[int, np.ndarray] = field(default_factory=OrderedDict)
    _capture: Any = None

    def _cv2(self):
        return require_optional_dependency("cv2", extra="vision", purpose="pose-driven preview rendering")

    def _ensure_capture(self):
        cv2 = self._cv2()
        if self._capture is None:
            capture = cv2.VideoCapture(str(self.video_path))
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"failed to open preview video: {self.video_path}")
            self._capture = capture
        return self._capture

    def _normalize_rgb_frame(self, frame: np.ndarray) -> np.ndarray:
        cv2 = self._cv2()
        height, width = frame.shape[:2]
        if width == self.intrinsics_width and height == self.intrinsics_height:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if width == self.intrinsics_height and height == self.intrinsics_width:
            rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            return cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame, (self.intrinsics_width, self.intrinsics_height), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def rgb_frame(self, frame_index: int) -> np.ndarray:
        with self._capture_lock:
            cached = self._frame_cache.get(frame_index)
            if cached is not None:
                return cached.copy()
            capture = self._ensure_capture()
            capture.set(self._cv2().CAP_PROP_POS_FRAMES, float(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                fresh = self._cv2().VideoCapture(str(self.video_path))
                try:
                    fresh.set(self._cv2().CAP_PROP_POS_FRAMES, float(frame_index))
                    ok, frame = fresh.read()
                finally:
                    fresh.release()
            if not ok or frame is None:
                raise RuntimeError(f"failed to decode preview frame {frame_index} from {self.video_path}")
            normalized = self._normalize_rgb_frame(frame)
            self._frame_cache[frame_index] = normalized
            while len(self._frame_cache) > 8:
                self._frame_cache.popitem(last=False)
            return normalized.copy()

    def _load_depth(self, sample: PoseSample) -> tuple[np.ndarray, Optional[np.ndarray]]:
        cv2 = self._cv2()
        depth = cv2.imread(str(sample.depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"failed to load preview depth map: {sample.depth_path}")
        confidence = None
        if sample.confidence_path is not None and sample.confidence_path.is_file():
            confidence = cv2.imread(str(sample.confidence_path), cv2.IMREAD_UNCHANGED)
        if depth.ndim != 2:
            depth = depth[..., 0]
        if confidence is not None and confidence.ndim != 2:
            confidence = confidence[..., 0]
        return depth.astype(np.float32), confidence.astype(np.float32) if confidence is not None else None

    def _target_pose(self, world_state: Mapping[str, Any], camera_id: str) -> np.ndarray:
        robot_state = dict(world_state.get("robot_state") or {})
        base_pose = list(robot_state.get("base_pose") or [0.0, 0.0, 0.0])
        joint_targets = list(robot_state.get("joint_targets") or [])
        while len(base_pose) < 3:
            base_pose.append(0.0)
        while len(joint_targets) < 7:
            joint_targets.append(0.0)

        forward = float(base_pose[0]) * 2.2 + float(joint_targets[3]) * 0.55
        lateral = float(base_pose[1]) * 1.6 + float(joint_targets[4]) * 0.40
        lift = float(joint_targets[5]) * 0.28
        yaw = float(base_pose[2]) * 10.0 + float(joint_targets[4]) * 0.18
        pitch = -float(joint_targets[5]) * 0.12

        camera_id_normalized = camera_id.lower()
        if "context" in camera_id_normalized:
            forward -= 0.35
            lift += 0.16
            yaw += 0.18
        elif "wrist" in camera_id_normalized:
            forward += 0.25 + float(joint_targets[3]) * 0.75
            lateral += float(joint_targets[4]) * 0.30
            lift -= 0.18 + float(joint_targets[5]) * 0.34
            pitch -= 0.20
        elif "cell" in camera_id_normalized:
            lift += 0.22
            pitch -= 0.12

        delta = _transform_matrix(
            [lateral, -lift, forward],
            yaw_radians=yaw,
            pitch_radians=pitch,
        )
        return self.anchor_pose.world_from_camera @ delta

    def _select_source_sample(self, target_world_from_camera: np.ndarray) -> PoseSample:
        target_center = target_world_from_camera[:3, 3]
        target_forward = target_world_from_camera[:3, 2]

        def _score(sample: PoseSample) -> float:
            translation_distance = float(np.linalg.norm(sample.camera_center_world - target_center))
            alignment = float(np.clip(np.dot(sample.forward_world, target_forward), -1.0, 1.0))
            rotation_penalty = math.acos(alignment)
            return translation_distance + (rotation_penalty * 0.35)

        return min(self.pose_samples, key=_score)

    def _camera_intrinsics_for(self, width: int, height: int, camera_id: str) -> tuple[float, float, float, float]:
        scale_x = float(width) / float(max(self.intrinsics_width, 1))
        scale_y = float(height) / float(max(self.intrinsics_height, 1))
        fx = self.fx * scale_x
        fy = self.fy * scale_y
        cx = self.cx * scale_x
        cy = self.cy * scale_y
        camera_id_normalized = camera_id.lower()
        if "wrist" in camera_id_normalized:
            fx *= 1.18
            fy *= 1.18
        elif "context" in camera_id_normalized:
            fx *= 0.88
            fy *= 0.88
        elif "cell" in camera_id_normalized:
            fx *= 1.05
            fy *= 1.05
        return fx, fy, cx, cy

    def _affine_fill(self, source_rgb: np.ndarray, relative_transform: np.ndarray, camera_id: str) -> np.ndarray:
        cv2 = self._cv2()
        height, width = source_rgb.shape[:2]
        fx, fy, _, _ = self._camera_intrinsics_for(width, height, camera_id)
        tx, ty, tz = [float(value) for value in relative_transform[:3, 3]]
        yaw = math.atan2(float(relative_transform[0, 2]), float(relative_transform[2, 2]))
        shift_x = int(np.clip(((-tx * fx) + (yaw * width * 0.75)), -width * 0.25, width * 0.25))
        shift_y = int(np.clip((-ty * fy), -height * 0.20, height * 0.20))
        scale = float(np.clip(1.0 + (-tz * 0.22), 0.84, 1.22))
        matrix = np.float32([[scale, 0.0, shift_x], [0.0, scale, shift_y]])
        return cv2.warpAffine(source_rgb, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)

    def render(self, world_state: Mapping[str, Any], camera_id: str) -> tuple[np.ndarray, dict[str, Any]]:
        cv2 = self._cv2()
        target_world_from_camera = self._target_pose(world_state, camera_id)
        source_sample = self._select_source_sample(target_world_from_camera)
        source_rgb = self.rgb_frame(source_sample.video_frame_index)
        depth_raw, confidence_raw = self._load_depth(source_sample)

        depth_height, depth_width = depth_raw.shape[:2]
        source_depth_rgb = cv2.resize(source_rgb, (depth_width, depth_height), interpolation=cv2.INTER_AREA)
        fx, fy, cx, cy = self._camera_intrinsics_for(depth_width, depth_height, camera_id)
        depth_m = depth_raw / 10000.0
        valid = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 8.0)
        if confidence_raw is not None:
            valid &= confidence_raw > 0.0
        if not bool(np.any(valid)):
            raise RuntimeError(f"preview depth map has no valid samples for {source_sample.frame_id}")

        grid_x, grid_y = np.meshgrid(
            np.arange(depth_width, dtype=np.float32),
            np.arange(depth_height, dtype=np.float32),
        )
        x = (grid_x - cx) / fx * depth_m
        y = (grid_y - cy) / fy * depth_m
        source_points = np.stack([x, y, depth_m, np.ones_like(depth_m)], axis=-1).reshape(-1, 4).T

        target_from_source = np.linalg.inv(target_world_from_camera) @ source_sample.world_from_camera
        target_points = target_from_source @ source_points
        target_z = target_points[2]
        projected_x = (target_points[0] / np.maximum(target_z, 1e-6)) * fx + cx
        projected_y = (target_points[1] / np.maximum(target_z, 1e-6)) * fy + cy

        flat_valid = valid.reshape(-1) & np.isfinite(target_z) & (target_z > 0.05)
        projected_u = np.rint(projected_x[flat_valid]).astype(np.int32)
        projected_v = np.rint(projected_y[flat_valid]).astype(np.int32)
        target_depth = target_z[flat_valid].astype(np.float32)
        colors = source_depth_rgb.reshape(-1, 3)[flat_valid]

        inside = (
            (projected_u >= 0)
            & (projected_u < depth_width)
            & (projected_v >= 0)
            & (projected_v < depth_height)
        )
        if not bool(np.any(inside)):
            raise RuntimeError(f"preview reprojection fell outside the viewport for {camera_id}")

        projected_u = projected_u[inside]
        projected_v = projected_v[inside]
        target_depth = target_depth[inside]
        colors = colors[inside]

        order = np.argsort(target_depth)[::-1]
        projected_u = projected_u[order]
        projected_v = projected_v[order]
        colors = colors[order]

        target_rgb = np.zeros((depth_height, depth_width, 3), dtype=np.uint8)
        coverage = np.zeros((depth_height, depth_width), dtype=np.uint8)
        target_rgb[projected_v, projected_u] = colors
        coverage[projected_v, projected_u] = 255

        affine_fill = self._affine_fill(source_depth_rgb, target_from_source, camera_id)
        holes = coverage == 0
        target_rgb[holes] = affine_fill[holes]

        if float(coverage.mean()) < 120.0:
            inpaint_input = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR)
            repaired = cv2.inpaint(inpaint_input, (coverage == 0).astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
            target_rgb = cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)

        preview_rgb = cv2.resize(
            target_rgb,
            (self.preview_width, self.preview_height),
            interpolation=cv2.INTER_CUBIC,
        )
        return preview_rgb, {
            "preview_mode": "pose_driven_rgbd",
            "source_frame_id": source_sample.frame_id,
            "source_video_frame_index": source_sample.video_frame_index,
            "coverage_ratio": round(float(np.mean(coverage > 0)), 4),
        }


class PoseDrivenPreviewRenderer:
    def __init__(self) -> None:
        self._scene_lock = threading.Lock()
        self._scenes: dict[str, SpatialPreviewScene] = {}

    def invalidate(self, site_world_id: str) -> None:
        with self._scene_lock:
            self._scenes.pop(site_world_id, None)

    def _build_scene(self, site_world_id: str, spec: Mapping[str, Any]) -> SpatialPreviewScene:
        conditioning = dict(spec.get("conditioning") or {}) if isinstance(spec.get("conditioning"), Mapping) else {}
        conditioning_paths = dict(conditioning.get("local_paths") or {}) if isinstance(conditioning.get("local_paths"), Mapping) else {}
        geometry = dict(spec.get("geometry") or {}) if isinstance(spec.get("geometry"), Mapping) else {}
        scene_memory_bundle = _read_json(
            _resolve_local_path(
                conditioning_paths.get("conditioning_bundle_path"),
                conditioning.get("conditioning_bundle_path"),
            )
            or Path()
        ) if _resolve_local_path(
            conditioning_paths.get("conditioning_bundle_path"),
            conditioning.get("conditioning_bundle_path"),
        ) else {}

        raw_video_path = _resolve_local_path(
            conditioning_paths.get("raw_video_path"),
            conditioning.get("raw_video_path"),
            scene_memory_bundle.get("raw_video_path"),
        )
        poses_path = _resolve_local_path(
            conditioning_paths.get("arkit_poses_path"),
            conditioning.get("arkit_poses_path"),
            scene_memory_bundle.get("arkit", {}).get("poses_path") if isinstance(scene_memory_bundle.get("arkit"), Mapping) else None,
        )
        intrinsics_path = _resolve_local_path(
            conditioning_paths.get("arkit_intrinsics_path"),
            conditioning.get("arkit_intrinsics_path"),
            scene_memory_bundle.get("arkit", {}).get("intrinsics_path") if isinstance(scene_memory_bundle.get("arkit"), Mapping) else None,
        )
        depth_dir = _resolve_local_path(
            conditioning_paths.get("arkit_depth_path"),
            conditioning.get("arkit_depth_path"),
            (scene_memory_bundle.get("arkit") or {}).get("depth_prefix_path") if isinstance(scene_memory_bundle.get("arkit"), Mapping) else None,
        )
        confidence_dir = _resolve_local_path(
            (scene_memory_bundle.get("arkit") or {}).get("confidence_prefix_path") if isinstance(scene_memory_bundle.get("arkit"), Mapping) else None,
            str(Path(str(conditioning_paths.get("arkit_depth_path") or "")).resolve().parent / "confidence") if conditioning_paths.get("arkit_depth_path") else "",
        )
        if raw_video_path is None:
            raw_video_path = _resolve_local_path(geometry.get("scene_memory_bundle_path"))
        if raw_video_path is None or poses_path is None or intrinsics_path is None or depth_dir is None:
            raise RuntimeError(f"site world {site_world_id} is missing spatial preview inputs")

        intrinsics = _read_json(intrinsics_path)
        width = int(intrinsics.get("width") or 0)
        height = int(intrinsics.get("height") or 0)
        fx = float(intrinsics.get("fx") or 0.0)
        fy = float(intrinsics.get("fy") or 0.0)
        cx = float(intrinsics.get("cx") or 0.0)
        cy = float(intrinsics.get("cy") or 0.0)
        if min(width, height) <= 0 or min(fx, fy) <= 0.0:
            raise RuntimeError(f"site world {site_world_id} has invalid intrinsics for spatial preview")

        cv2 = require_optional_dependency("cv2", extra="vision", purpose="pose-driven preview rendering")
        capture = cv2.VideoCapture(str(raw_video_path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"failed to open preview video: {raw_video_path}")
            video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            capture.release()
        if video_fps <= 0.0:
            video_fps = 24.0

        pose_rows = _load_jsonl(poses_path)
        pose_samples: list[PoseSample] = []
        for row in pose_rows:
            frame_id = str(row.get("frame_id") or "").strip()
            transform_rows = row.get("T_world_camera") or row.get("transform")
            if not frame_id or not isinstance(transform_rows, Sequence):
                continue
            try:
                world_from_camera = np.asarray(transform_rows, dtype=np.float32).reshape(4, 4)
            except ValueError:
                continue
            depth_candidates = [
                depth_dir / f"{frame_id}.png",
                depth_dir / f"smoothed-{frame_id}.png",
            ]
            depth_path = next((path for path in depth_candidates if path.is_file()), None)
            if depth_path is None:
                continue
            confidence_path = None
            if confidence_dir is not None:
                candidate = confidence_dir / f"{frame_id}.png"
                if candidate.is_file():
                    confidence_path = candidate
            t_device_sec = float(row.get("t_device_sec") or 0.0)
            video_frame_index = int(round(t_device_sec * video_fps))
            if frame_count > 0:
                video_frame_index = max(0, min(frame_count - 1, video_frame_index))
            camera_center_world = world_from_camera[:3, 3].astype(np.float32)
            forward_world = world_from_camera[:3, 2].astype(np.float32)
            norm = float(np.linalg.norm(forward_world))
            if norm > 1e-6:
                forward_world = forward_world / norm
            pose_samples.append(
                PoseSample(
                    frame_id=frame_id,
                    t_device_sec=t_device_sec,
                    world_from_camera=world_from_camera,
                    camera_from_world=np.linalg.inv(world_from_camera).astype(np.float32),
                    camera_center_world=camera_center_world,
                    forward_world=forward_world,
                    video_frame_index=video_frame_index,
                    depth_path=depth_path,
                    confidence_path=confidence_path,
                )
            )
        if not pose_samples:
            raise RuntimeError(f"site world {site_world_id} has no ARKit samples with depth for spatial preview")

        anchor_pose = min(pose_samples, key=lambda sample: sample.t_device_sec)
        return SpatialPreviewScene(
            site_world_id=site_world_id,
            video_path=raw_video_path,
            intrinsics_width=width,
            intrinsics_height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            video_fps=video_fps,
            video_frame_count=frame_count,
            pose_samples=pose_samples,
            anchor_pose=anchor_pose,
        )

    def scene(self, site_world_id: str, spec: Mapping[str, Any]) -> SpatialPreviewScene:
        with self._scene_lock:
            cached = self._scenes.get(site_world_id)
            if cached is not None:
                return cached
        built = self._build_scene(site_world_id, spec)
        with self._scene_lock:
            self._scenes[site_world_id] = built
        return built

    def render_camera(
        self,
        *,
        site_world_id: str,
        spec: Mapping[str, Any],
        world_state: Mapping[str, Any],
        camera_id: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        scene = self.scene(site_world_id, spec)
        return scene.render(world_state, camera_id)
