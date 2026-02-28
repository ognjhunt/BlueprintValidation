"""Quality checks for RoboSplat-generated augmentation clips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class QualityGateResult:
    accepted: bool
    reason: str
    num_frames: int = 0
    resolution: Optional[Tuple[int, int]] = None


def validate_augmented_clip(
    video_path: Path,
    depth_video_path: Optional[Path] = None,
    expected_resolution: Optional[Tuple[int, int]] = None,
    min_frames: int = 3,
) -> QualityGateResult:
    """Validate augmented RGB/depth clips are readable and minimally consistent."""
    import cv2

    if not video_path.exists():
        return QualityGateResult(accepted=False, reason=f"missing_rgb:{video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return QualityGateResult(accepted=False, reason="rgb_open_failed")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if frame_count < min_frames:
        return QualityGateResult(
            accepted=False,
            reason=f"too_few_frames:{frame_count}<{min_frames}",
            num_frames=frame_count,
            resolution=(height, width),
        )

    if expected_resolution is not None and (height, width) != expected_resolution:
        return QualityGateResult(
            accepted=False,
            reason=f"resolution_mismatch:{height}x{width}",
            num_frames=frame_count,
            resolution=(height, width),
        )

    if depth_video_path and depth_video_path.exists():
        depth_cap = cv2.VideoCapture(str(depth_video_path))
        if not depth_cap.isOpened():
            return QualityGateResult(
                accepted=False,
                reason="depth_open_failed",
                num_frames=frame_count,
                resolution=(height, width),
            )
        depth_frames = int(depth_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        depth_cap.release()
        if depth_frames <= 0:
            return QualityGateResult(
                accepted=False,
                reason="depth_empty",
                num_frames=frame_count,
                resolution=(height, width),
            )

    return QualityGateResult(
        accepted=True,
        reason="ok",
        num_frames=frame_count,
        resolution=(height, width),
    )

