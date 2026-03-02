"""Tests for strict H.264/video-frame enforcement helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from blueprint_validation.video_io import ensure_h264_video

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="requires ffmpeg/ffprobe",
)


def _write_mp4(path: Path, *, num_frames: int, codec: str = "mp4v") -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), 10.0, (64, 48))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for test clip: {path}")
    for i in range(max(1, int(num_frames))):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 1] = np.uint8((i * 25) % 255)
        writer.write(frame)
    writer.release()


def test_ensure_h264_video_transcodes_and_validates_frames(tmp_path: Path):
    src = tmp_path / "src_mp4v.mp4"
    dst = tmp_path / "fixed_h264.mp4"
    _write_mp4(src, num_frames=5, codec="mp4v")

    result = ensure_h264_video(
        input_path=src,
        output_path=dst,
        min_decoded_frames=5,
    )

    assert result.path == dst
    assert result.codec_name == "h264"
    assert result.decoded_frames == 5
    assert result.transcoded is True
    assert dst.exists()


def test_ensure_h264_video_fails_on_short_clip(tmp_path: Path):
    src = tmp_path / "short.mp4"
    _write_mp4(src, num_frames=3, codec="mp4v")

    with pytest.raises(RuntimeError, match="required_min_frames=13"):
        ensure_h264_video(
            input_path=src,
            output_path=tmp_path / "short_fixed.mp4",
            min_decoded_frames=13,
        )
