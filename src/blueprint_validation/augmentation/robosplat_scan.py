"""RoboSplat-inspired scan-only clip augmentation utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import RoboSplatScanConfig


@dataclass
class AugmentedClip:
    """Augmented video artifact and provenance."""

    clip_name: str
    output_video_path: Path
    output_depth_video_path: Optional[Path]
    source_clip_name: str
    source_video_path: Path
    source_depth_video_path: Optional[Path]
    augment_ops: Dict[str, float | bool]


def augment_scan_only_clip(
    video_path: Path,
    depth_video_path: Optional[Path],
    output_dir: Path,
    source_clip_name: str,
    augment_index: int,
    config: RoboSplatScanConfig,
) -> AugmentedClip:
    """Create one scan-only augmented clip from a rendered source clip."""
    rgb_frames, fps = _read_video(video_path, color=True)
    if not rgb_frames:
        raise RuntimeError(f"No RGB frames found in {video_path}")

    depth_frames: List[np.ndarray] = []
    depth_fps = fps
    if depth_video_path and depth_video_path.exists():
        depth_frames, depth_fps = _read_video(depth_video_path, color=False)

    ops = _sample_ops(source_clip_name, augment_index, config)
    rgb_frames = _resample_temporal(rgb_frames, float(ops["temporal_speed_factor"]))
    rgb_frames = [
        _augment_rgb_frame(
            frame=frame,
            yaw_deg=float(ops["yaw_jitter_deg"]),
            pitch_deg=float(ops["pitch_jitter_deg"]),
            height_shift_frac=float(ops["height_shift_frac"]),
            relight_gain=float(ops["relight_gain"]),
            color_temp_shift=bool(ops["color_temp_shift"]),
            color_temp_scale=float(ops["color_temp_scale"]),
        )
        for frame in rgb_frames
    ]

    out_name = f"{source_clip_name}_rs{augment_index:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_rgb = output_dir / f"{out_name}.mp4"
    _write_video(out_rgb, rgb_frames, fps=fps, color=True)

    out_depth: Optional[Path] = None
    if depth_frames:
        depth_frames = _resample_temporal(depth_frames, float(ops["temporal_speed_factor"]))
        depth_frames = [
            _augment_depth_frame(
                frame=frame,
                yaw_deg=float(ops["yaw_jitter_deg"]),
                pitch_deg=float(ops["pitch_jitter_deg"]),
                height_shift_frac=float(ops["height_shift_frac"]),
            )
            for frame in depth_frames
        ]
        out_depth = output_dir / f"{out_name}_depth.mp4"
        _write_video(out_depth, depth_frames, fps=depth_fps, color=False)

    return AugmentedClip(
        clip_name=out_name,
        output_video_path=out_rgb,
        output_depth_video_path=out_depth,
        source_clip_name=source_clip_name,
        source_video_path=video_path,
        source_depth_video_path=depth_video_path,
        augment_ops=ops,
    )


def _stable_seed(source_clip_name: str, augment_index: int) -> int:
    digest = hashlib.sha256(f"{source_clip_name}:{augment_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _sample_ops(
    source_clip_name: str,
    augment_index: int,
    config: RoboSplatScanConfig,
) -> Dict[str, float | bool]:
    rng = np.random.default_rng(_stable_seed(source_clip_name, augment_index))
    relight_low = min(config.relight_gain_min, config.relight_gain_max)
    relight_high = max(config.relight_gain_min, config.relight_gain_max)

    speed = config.temporal_speed_factors[
        augment_index % max(1, len(config.temporal_speed_factors))
    ]
    speed = float(speed if speed > 0 else 1.0)

    return {
        "yaw_jitter_deg": float(rng.uniform(-config.yaw_jitter_deg, config.yaw_jitter_deg)),
        "pitch_jitter_deg": float(rng.uniform(-config.pitch_jitter_deg, config.pitch_jitter_deg)),
        "height_shift_frac": float(
            rng.uniform(-config.camera_height_jitter_m, config.camera_height_jitter_m) * 0.20
        ),
        "relight_gain": float(rng.uniform(relight_low, relight_high)),
        "color_temp_shift": bool(config.color_temp_shift),
        "color_temp_scale": float(rng.uniform(0.92, 1.08)),
        "temporal_speed_factor": speed,
    }


def _read_video(video_path: Path, color: bool) -> tuple[List[np.ndarray], int]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 10
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if color:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            frames.append(frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return frames, fps


def _write_video(path: Path, frames: List[np.ndarray], fps: int, color: bool) -> None:
    import cv2

    if not frames:
        raise RuntimeError(f"Cannot write empty video: {path}")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, max(1, fps), (w, h), isColor=color)
    for frame in frames:
        if color:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            if frame.ndim != 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            writer.write(frame)
    writer.release()


def _resample_temporal(frames: List[np.ndarray], speed_factor: float) -> List[np.ndarray]:
    if not frames:
        return frames
    n = len(frames)
    src = np.linspace(0, n - 1, num=n, dtype=np.float32) * speed_factor
    src = np.clip(src, 0, n - 1)
    indices = np.rint(src).astype(int)
    return [frames[i] for i in indices]


def _augment_rgb_frame(
    frame: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    height_shift_frac: float,
    relight_gain: float,
    color_temp_shift: bool,
    color_temp_scale: float,
) -> np.ndarray:
    warped = _apply_view_warp(frame, yaw_deg, pitch_deg, height_shift_frac)
    out = np.clip(warped.astype(np.float32) * relight_gain, 0, 255)
    if color_temp_shift:
        # Warm/cool tone shift: balance red and blue channels while preserving mean luminance.
        out[..., 0] *= (2.0 - color_temp_scale)  # blue
        out[..., 2] *= color_temp_scale  # red
        out = np.clip(out, 0, 255)
    return out.astype(np.uint8)


def _augment_depth_frame(
    frame: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    height_shift_frac: float,
) -> np.ndarray:
    return _apply_view_warp(frame, yaw_deg, pitch_deg, height_shift_frac)


def _apply_view_warp(
    frame: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    height_shift_frac: float,
) -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    tx = (yaw_deg / 10.0) * (w * 0.04)
    ty = height_shift_frac * h
    rot = pitch_deg * 0.20

    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, rot, 1.0)
    mat[0, 2] += tx
    mat[1, 2] += ty

    return cv2.warpAffine(
        frame,
        mat,
        dsize=(w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
