"""Shared OpenCV video writing helpers with codec compatibility fallbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .common import get_logger

logger = get_logger("video_io")

_DEFAULT_MP4_CODECS: tuple[str, ...] = ("avc1", "H264", "mp4v")


def open_mp4_writer(
    *,
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
    is_color: bool = True,
    codec_candidates: Sequence[str] = _DEFAULT_MP4_CODECS,
):
    """Open an MP4 writer with a codec fallback chain for broad player compatibility."""
    import cv2

    width = int(frame_size[0])
    height = int(frame_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid frame size for video writer: {(width, height)}")

    resolved_fps = float(fps) if fps and fps > 0 else 10.0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tried: list[str] = []
    for codec in codec_candidates:
        code = str(codec or "").strip()
        if len(code) != 4:
            continue
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*code),
            resolved_fps,
            (width, height),
            isColor=bool(is_color),
        )
        if writer.isOpened():
            if code != _DEFAULT_MP4_CODECS[0]:
                logger.warning(
                    "Using fallback MP4 codec '%s' for %s (attempted: %s)",
                    code,
                    output_path,
                    ",".join(tried + [code]),
                )
            return writer
        writer.release()
        tried.append(code)

    raise RuntimeError(
        f"Could not open MP4 video writer for {output_path}; tried codecs={','.join(tried)}"
    )
