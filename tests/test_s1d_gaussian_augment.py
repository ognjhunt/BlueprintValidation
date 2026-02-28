"""Tests for Stage 1d scan-only Gaussian augmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_dummy_video(path: Path, frames: int = 4, color: bool = True) -> None:
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = 48, 64
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (w, h),
        isColor=color,
    )
    for i in range(frames):
        if color:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[..., 1] = 40 + i * 10
            frame[..., 2] = 90 + i * 5
            writer.write(frame)
        else:
            frame = np.full((h, w), 50 + i * 10, dtype=np.uint8)
            writer.write(frame)
    writer.release()


def test_stage_skips_when_disabled(sample_config, tmp_path):
    from blueprint_validation.stages.s1d_gaussian_augment import GaussianAugmentStage

    sample_config.robosplat_scan.enabled = False
    stage = GaussianAugmentStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, {})
    assert result.status == "skipped"


def test_stage_generates_augmented_manifest(sample_config, tmp_path):
    from blueprint_validation.common import read_json, write_json
    from blueprint_validation.stages.s1d_gaussian_augment import GaussianAugmentStage

    sample_config.robosplat_scan.enabled = True
    sample_config.robosplat_scan.num_augmented_clips_per_input = 2

    render_dir = tmp_path / "renders"
    rgb_path = render_dir / "clip_000.mp4"
    depth_path = render_dir / "clip_000_depth.mp4"
    _write_dummy_video(rgb_path, color=True)
    _write_dummy_video(depth_path, color=False)

    write_json(
        {
            "facility": "Test Facility",
            "clips": [
                {
                    "clip_name": "clip_000",
                    "path_type": "orbit",
                    "clip_index": 0,
                    "num_frames": 4,
                    "resolution": [48, 64],
                    "fps": 5,
                    "video_path": str(rgb_path),
                    "depth_video_path": str(depth_path),
                }
            ],
        },
        render_dir / "render_manifest.json",
    )

    stage = GaussianAugmentStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, {})
    assert result.status == "success"
    assert result.metrics["num_augmented_clips"] == 2
    assert result.metrics["num_total_clips"] == 3

    manifest_path = tmp_path / "gaussian_augment" / "augmented_manifest.json"
    assert manifest_path.exists()
    data = read_json(manifest_path)
    assert data["num_augmented_clips"] == 2
    assert len(data["clips"]) == 3
    augmented = [c for c in data["clips"] if c["clip_name"].startswith("clip_000_rs")]
    assert len(augmented) == 2


def test_s2_manifest_resolution_prefers_s1d(tmp_path):
    from blueprint_validation.stages.s2_enrich import _resolve_render_manifest

    gauss = tmp_path / "gaussian_augment" / "augmented_manifest.json"
    render = tmp_path / "renders" / "render_manifest.json"
    gauss.parent.mkdir(parents=True, exist_ok=True)
    render.parent.mkdir(parents=True, exist_ok=True)
    gauss.write_text("{}")
    render.write_text("{}")

    assert _resolve_render_manifest(tmp_path) == gauss
