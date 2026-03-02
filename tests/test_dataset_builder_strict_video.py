"""Tests for strict dataset video validation in DreamDojo dataset builder."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from blueprint_validation.training.dataset_builder import build_dreamdojo_dataset

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
        frame[:, :, 0] = np.uint8((i * 30) % 255)
        writer.write(frame)
    writer.release()


def test_build_dreamdojo_dataset_rejects_short_video(tmp_path: Path):
    src = tmp_path / "short.mp4"
    _write_mp4(src, num_frames=3, codec="mp4v")
    manifest_path = tmp_path / "enriched_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "output_video_path": str(src),
                        "prompt": "test prompt",
                        "variant_name": "daylight",
                        "clip_name": "clip_000",
                    }
                ]
            }
        )
    )

    with pytest.raises(RuntimeError, match="required_min_frames=13"):
        build_dreamdojo_dataset(
            enriched_manifest_path=manifest_path,
            output_dir=tmp_path / "dataset_out",
            facility_name="facility_a",
            min_decoded_frames=13,
        )
