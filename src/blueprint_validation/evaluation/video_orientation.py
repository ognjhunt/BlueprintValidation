"""Shared video-orientation helpers used by Stage 2 and Stage 4 frame loading."""

from __future__ import annotations

from pathlib import Path

from ..common import get_logger

logger = get_logger("evaluation.video_orientation")

_ALLOWED_ORIENTATION_FIXES = {"none", "rotate180", "hflip", "vflip", "hvflip"}


def normalize_video_orientation_fix(raw: str | None) -> str:
    value = str(raw or "none").strip().lower()
    if value in _ALLOWED_ORIENTATION_FIXES:
        return value
    return "none"


def apply_video_orientation_fix(
    *,
    input_path: Path,
    cache_dir: Path,
    clip_name: str,
    stream_tag: str,
    orientation_fix: str,
    force_grayscale: bool,
) -> Path:
    """Materialize an orientation-corrected cache file and return its path."""
    mode = normalize_video_orientation_fix(orientation_fix)
    if mode == "none":
        return input_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    fixed_path = cache_dir / f"{clip_name}_{stream_tag}_{mode}.mp4"
    if fixed_path.exists() and fixed_path.stat().st_mtime >= input_path.stat().st_mtime:
        return fixed_path

    transform_video_orientation(
        input_path=input_path,
        output_path=fixed_path,
        orientation_fix=mode,
        force_grayscale=force_grayscale,
    )
    return fixed_path


def transform_video_orientation(
    *,
    input_path: Path,
    output_path: Path,
    orientation_fix: str,
    force_grayscale: bool,
) -> None:
    """Transform video orientation while preserving fps and frame count."""
    import cv2

    mode = normalize_video_orientation_fix(orientation_fix)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for orientation transform: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 10.0

    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        raise RuntimeError(f"No frames available in video: {input_path}")

    if force_grayscale and first.ndim == 3:
        first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    first = transform_video_frame(first, mode)
    if first.ndim == 2:
        height, width = first.shape
    else:
        height, width = first.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
        isColor=not force_grayscale,
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video writer for {output_path}")

    frame_count = 0
    try:
        frame = first
        while True:
            if force_grayscale and frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            writer.write(frame)
            frame_count += 1
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if force_grayscale and frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = transform_video_frame(frame, mode)
    finally:
        cap.release()
        writer.release()

    if frame_count <= 0:
        raise RuntimeError(f"No frames written during orientation transform: {input_path}")


def transform_video_frame(frame, orientation_fix: str):
    import cv2

    mode = normalize_video_orientation_fix(orientation_fix)
    if mode == "rotate180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    if mode == "hflip":
        return cv2.flip(frame, 1)
    if mode == "vflip":
        return cv2.flip(frame, 0)
    if mode == "hvflip":
        return cv2.flip(frame, -1)
    return frame
