"""Shared camera/clip quality utilities for Stage 1 and Stage 2 gates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from ..common import read_json


def resolve_center_band_bounds(
    x_values: object,
    y_values: object,
    *,
    default_x: tuple[float, float] = (0.2, 0.8),
    default_y: tuple[float, float] = (0.2, 0.8),
) -> tuple[float, float, float, float]:
    """Resolve normalized frame center-band bounds."""

    def _pair(values: object, default: tuple[float, float]) -> tuple[float, float]:
        if isinstance(values, (list, tuple)) and len(values) == 2:
            lo, hi = float(values[0]), float(values[1])
        else:
            lo, hi = default
        lo = max(0.0, min(1.0, lo))
        hi = max(0.0, min(1.0, hi))
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi

    x_lo, x_hi = _pair(x_values, default_x)
    y_lo, y_hi = _pair(y_values, default_y)
    return x_lo, x_hi, y_lo, y_hi


def project_target_to_camera_path(
    clip_entry: dict,
    target_xyz: object,
) -> tuple[int, List[Dict[str, float]]]:
    """Project a 3D target point into all frames from a serialized camera path JSON."""
    camera_path = Path(str(clip_entry.get("camera_path", "")))
    if not camera_path.exists():
        return 0, []
    payload = read_json(camera_path)
    frames = payload.get("camera_path", [])
    if not isinstance(frames, list) or not frames:
        return 0, []

    resolution = clip_entry.get("resolution", [480, 640])
    if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
        height = int(resolution[0])
        width = int(resolution[1])
    else:
        height, width = 480, 640
    if height <= 0 or width <= 0:
        height, width = 480, 640

    target = np.asarray(target_xyz, dtype=np.float64)
    target_h = np.array([target[0], target[1], target[2], 1.0], dtype=np.float64)
    total_frames = 0
    visible_samples: List[Dict[str, float]] = []

    for frame_index, frame in enumerate(frames):
        c2w_raw = frame.get("camera_to_world")
        if not isinstance(c2w_raw, list) or len(c2w_raw) != 16:
            continue
        c2w = np.asarray(c2w_raw, dtype=np.float64).reshape(4, 4)
        total_frames += 1
        sample = _project_target_from_camera(
            target=target,
            target_h=target_h,
            c2w=c2w,
            width=width,
            height=height,
            fov_deg=float(frame.get("fov", 60.0)),
            frame_index=frame_index,
        )
        if sample is not None:
            visible_samples.append(sample)
    return total_frames, visible_samples


def project_target_to_poses(
    poses: Iterable[object],
    target_xyz: object,
) -> tuple[int, List[Dict[str, float]]]:
    """Project a 3D target point into an in-memory pose list."""
    target = np.asarray(target_xyz, dtype=np.float64)
    target_h = np.array([target[0], target[1], target[2], 1.0], dtype=np.float64)
    total_frames = 0
    visible_samples: List[Dict[str, float]] = []

    for frame_index, pose in enumerate(list(poses)):
        try:
            c2w = np.asarray(getattr(pose, "c2w"), dtype=np.float64)
            width = int(getattr(pose, "width"))
            height = int(getattr(pose, "height"))
            fx = float(getattr(pose, "fx"))
        except Exception:
            continue
        if c2w.shape != (4, 4) or width <= 0 or height <= 0:
            continue
        total_frames += 1
        fov_deg = 2.0 * math.degrees(math.atan2(width / 2.0, max(fx, 1e-6)))
        sample = _project_target_from_camera(
            target=target,
            target_h=target_h,
            c2w=c2w,
            width=width,
            height=height,
            fov_deg=fov_deg,
            frame_index=frame_index,
        )
        if sample is not None:
            visible_samples.append(sample)
    return total_frames, visible_samples


def analyze_target_visibility(
    *,
    total_frames: int,
    visible_samples: List[Dict[str, float]],
    angle_bin_deg: float,
    center_band_x: object,
    center_band_y: object,
) -> tuple[int, int, int, set[int]]:
    """Aggregate visibility and framing metrics from projected target samples."""
    if total_frames <= 0:
        return 0, 0, 0, set()

    visible_frames = len(visible_samples)
    center_band_frames = 0
    angle_bins: set[int] = set()
    total_bins = max(1, int(round(360.0 / max(float(angle_bin_deg), 1e-3))))
    bin_size_deg = 360.0 / total_bins
    x_lo, x_hi, y_lo, y_hi = resolve_center_band_bounds(center_band_x, center_band_y)

    for sample in visible_samples:
        yaw_norm = float(sample["yaw_deg_norm"])
        bin_idx = min(total_bins - 1, int(yaw_norm / bin_size_deg))
        angle_bins.add(bin_idx)

        u_norm = float(sample["u_norm"])
        v_norm = float(sample["v_norm"])
        if x_lo <= u_norm <= x_hi and y_lo <= v_norm <= y_hi:
            center_band_frames += 1

    return visible_frames, total_frames, center_band_frames, angle_bins


def estimate_clip_blur_score(
    video_path: Path,
    *,
    sample_every_n_frames: int,
    max_samples: int,
) -> float | None:
    """Estimate clip sharpness via Laplacian variance over sampled frames."""
    try:
        import cv2
    except Exception:
        return None

    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frame_index = 0
    sampled = 0
    scores: List[float] = []
    try:
        while sampled < max_samples:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_index % max(1, int(sample_every_n_frames)) != 0:
                frame_index += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            sampled += 1
            frame_index += 1
    finally:
        cap.release()
    if not scores:
        return None
    return float(np.median(np.asarray(scores, dtype=np.float64)))


def evaluate_clip_quality(
    *,
    clip_entry: dict,
    target_xyz: object | None,
    blur_laplacian_min: float,
    min_visible_frame_ratio: float,
    min_center_band_ratio: float,
    min_approach_angle_bins: int,
    angle_bin_deg: float,
    center_band_x: object,
    center_band_y: object,
    blur_sample_every_n_frames: int,
    blur_max_samples: int,
    min_clip_score: float,
    require_target: bool = False,
) -> Dict[str, object]:
    """Evaluate one clip and return deterministic quality metrics + pass/fail reasons."""
    reasons: List[str] = []
    blur_score = estimate_clip_blur_score(
        Path(str(clip_entry.get("video_path", ""))),
        sample_every_n_frames=max(1, int(blur_sample_every_n_frames)),
        max_samples=max(1, int(blur_max_samples)),
    )
    if blur_score is None or float(blur_score) < float(blur_laplacian_min):
        reasons.append("blur_laplacian_low")

    total_frames = 0
    visible_frames = 0
    center_band_frames = 0
    angle_bin_count = 0
    visible_ratio = 0.0
    center_ratio = 0.0
    has_target = target_xyz is not None

    if has_target:
        total_frames, visible_samples = project_target_to_camera_path(clip_entry, target_xyz)
        visible_frames, total_frames, center_band_frames, angle_bins = analyze_target_visibility(
            total_frames=total_frames,
            visible_samples=visible_samples,
            angle_bin_deg=float(angle_bin_deg),
            center_band_x=center_band_x,
            center_band_y=center_band_y,
        )
        angle_bin_count = len(angle_bins)
        if total_frames <= 0:
            reasons.append("target_projection_unavailable")
        else:
            visible_ratio = float(visible_frames) / float(total_frames)
            center_ratio = float(center_band_frames) / float(total_frames)
            if visible_ratio < float(min_visible_frame_ratio):
                reasons.append("target_visibility_low")
            if center_ratio < float(min_center_band_ratio):
                reasons.append("target_center_band_low")
            if angle_bin_count < int(min_approach_angle_bins):
                reasons.append("approach_angle_bins_low")
    elif require_target:
        reasons.append("missing_target_annotation")

    blur_component = (
        0.0
        if blur_score is None or float(blur_laplacian_min) <= 0.0
        else min(1.0, max(0.0, float(blur_score) / max(float(blur_laplacian_min), 1e-6)))
    )
    if has_target:
        visible_component = min(
            1.0,
            max(0.0, visible_ratio / max(float(min_visible_frame_ratio), 1e-6)),
        )
        center_component = min(
            1.0,
            max(0.0, center_ratio / max(float(min_center_band_ratio), 1e-6)),
        )
        angle_component = min(
            1.0,
            max(0.0, float(angle_bin_count) / max(float(min_approach_angle_bins), 1.0)),
        )
        clip_quality_score = (
            0.35 * blur_component
            + 0.25 * visible_component
            + 0.25 * center_component
            + 0.15 * angle_component
        )
    else:
        clip_quality_score = blur_component

    if clip_quality_score < float(min_clip_score):
        reasons.append("clip_quality_score_low")

    passed = len(reasons) == 0
    return {
        "target_visibility_ratio": round(float(visible_ratio), 6),
        "target_center_band_ratio": round(float(center_ratio), 6),
        "target_approach_angle_bins": int(angle_bin_count),
        "target_visible_frames": int(visible_frames),
        "target_total_frames": int(total_frames),
        "target_center_band_frames": int(center_band_frames),
        "blur_laplacian_score": (None if blur_score is None else round(float(blur_score), 6)),
        "clip_quality_score": round(float(clip_quality_score), 6),
        "quality_gate_passed": bool(passed),
        "quality_reject_reasons": list(reasons),
    }


def _project_target_from_camera(
    *,
    target: np.ndarray,
    target_h: np.ndarray,
    c2w: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    frame_index: int,
) -> Dict[str, float] | None:
    try:
        w2c = np.linalg.inv(c2w)
    except np.linalg.LinAlgError:
        return None

    cam = w2c @ target_h
    z = float(cam[2])
    if z >= -1e-6:
        return None

    tan_half_fov = math.tan(math.radians(max(1e-3, float(fov_deg)) / 2.0))
    if tan_half_fov <= 0:
        return None
    fx = width / (2.0 * tan_half_fov)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    u = fx * (float(cam[0]) / -z) + cx
    v = fy * (float(cam[1]) / -z) + cy
    if not (0.0 <= u < width and 0.0 <= v < height):
        return None

    cam_pos = c2w[:3, 3]
    delta_xy = cam_pos[:2] - target[:2]
    yaw = math.degrees(math.atan2(float(delta_xy[1]), float(delta_xy[0])))
    yaw_norm = (yaw + 360.0) % 360.0
    return {
        "frame_index": float(frame_index),
        "u_norm": float(u / max(width, 1)),
        "v_norm": float(v / max(height, 1)),
        "yaw_deg_norm": float(yaw_norm),
    }
