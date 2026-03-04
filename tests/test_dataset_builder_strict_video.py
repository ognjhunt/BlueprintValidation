"""Tests for strict dataset video validation in DreamDojo dataset builder."""

from __future__ import annotations

import csv
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


def test_build_dreamdojo_dataset_disambiguates_colliding_basenames(tmp_path: Path):
    src_a = tmp_path / "a" / "clip.mp4"
    src_b = tmp_path / "b" / "clip.mp4"
    src_a.parent.mkdir(parents=True, exist_ok=True)
    src_b.parent.mkdir(parents=True, exist_ok=True)
    _write_mp4(src_a, num_frames=16, codec="mp4v")
    _write_mp4(src_b, num_frames=16, codec="mp4v")

    manifest_path = tmp_path / "enriched_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "output_video_path": str(src_a),
                        "prompt": "prompt A",
                        "variant_name": "v1",
                        "clip_name": "clip_a",
                    },
                    {
                        "output_video_path": str(src_b),
                        "prompt": "prompt B",
                        "variant_name": "v1",
                        "clip_name": "clip_b",
                    },
                ]
            }
        )
    )
    dataset_dir = build_dreamdojo_dataset(
        enriched_manifest_path=manifest_path,
        output_dir=tmp_path / "dataset_out",
        facility_name="facility_a",
        min_decoded_frames=13,
    )
    csv_path = dataset_dir / "metadata.csv"
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    video_paths = [row["video_path"] for row in rows]
    meta_paths = [row["meta_path"] for row in rows]
    assert len(rows) == 2
    assert len(set(video_paths)) == 2
    assert len(set(meta_paths)) == 2
    for rel_video, rel_meta in zip(video_paths, meta_paths):
        assert (dataset_dir / rel_video).exists()
        assert (dataset_dir / rel_meta).exists()


def test_build_dreamdojo_dataset_rejects_monochrome_content(tmp_path: Path):
    from types import SimpleNamespace

    src = tmp_path / "green.mp4"
    _write_mp4(src, num_frames=16, codec="mp4v")
    manifest_path = tmp_path / "enriched_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "output_video_path": str(src),
                        "prompt": "solid green",
                        "variant_name": "v1",
                        "clip_name": "clip_green",
                    }
                ]
            }
        )
    )

    def _fake_ensure_h264_video(*, input_path, **_kwargs):
        return SimpleNamespace(
            path=input_path,
            content_monochrome_warning=True,
            content_max_std_dev=0.0,
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "blueprint_validation.training.dataset_builder.ensure_h264_video",
        _fake_ensure_h264_video,
    )
    with pytest.raises(RuntimeError, match="monochrome-content"):
        build_dreamdojo_dataset(
            enriched_manifest_path=manifest_path,
            output_dir=tmp_path / "dataset_out",
            facility_name="facility_a",
            min_decoded_frames=13,
        )
    monkeypatch.undo()
