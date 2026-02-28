"""Tests for Stage 1d RoboSplat augmentation orchestration."""

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

    sample_config.robosplat.enabled = False
    stage = GaussianAugmentStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, {})
    assert result.status == "skipped"


def test_stage_generates_augmented_manifest(sample_config, tmp_path):
    from blueprint_validation.common import read_json, write_json
    from blueprint_validation.stages.s1d_gaussian_augment import GaussianAugmentStage

    sample_config.robosplat.enabled = True
    sample_config.robosplat.backend = "legacy_scan"
    sample_config.robosplat.variants_per_input = 2
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
    assert result.metrics["backend_used"] == "legacy_scan"
    assert result.metrics["num_augmented_clips"] == 2
    assert result.metrics["num_total_clips"] == 3

    manifest_path = tmp_path / "gaussian_augment" / "augmented_manifest.json"
    assert manifest_path.exists()
    data = read_json(manifest_path)
    assert data["backend_used"] == "legacy_scan"
    assert data["num_augmented_clips"] == 2
    assert len(data["clips"]) == 3
    augmented = [c for c in data["clips"] if c["clip_name"].startswith("clip_000_rs")]
    assert len(augmented) == 2
    assert all(c["quality_gate_passed"] is True for c in augmented)


def test_stage_auto_backend_falls_back_to_legacy(sample_config, tmp_path):
    from blueprint_validation.common import write_json
    from blueprint_validation.stages.s1d_gaussian_augment import GaussianAugmentStage

    sample_config.robosplat.enabled = True
    sample_config.robosplat.backend = "auto"
    sample_config.robosplat.parity_mode = "hybrid"
    sample_config.robosplat.fallback_to_legacy_scan = True
    sample_config.robosplat.variants_per_input = 1
    sample_config.robosplat.vendor_repo_path = tmp_path / "missing_vendor"

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
                    "camera_path": str(render_dir / "clip_000_camera_path.json"),
                }
            ],
        },
        render_dir / "render_manifest.json",
    )

    fac = list(sample_config.facilities.values())[0]
    fac.ply_path = tmp_path / "invalid.ply"
    fac.ply_path.write_text("not_a_valid_ply")

    stage = GaussianAugmentStage()
    result = stage.run(sample_config, fac, tmp_path, {})
    assert result.status == "success"
    assert result.metrics["backend_used"] == "legacy_scan"
    assert result.metrics["fallback_backend"] == "legacy_scan"


def test_stage_strict_mode_fails_when_full_backend_unavailable(sample_config, tmp_path):
    from blueprint_validation.common import write_json
    from blueprint_validation.stages.s1d_gaussian_augment import GaussianAugmentStage

    sample_config.robosplat.enabled = True
    sample_config.robosplat.backend = "vendor"
    sample_config.robosplat.parity_mode = "strict"
    sample_config.robosplat.fallback_to_legacy_scan = False
    sample_config.robosplat.vendor_repo_path = tmp_path / "missing_vendor"

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
    assert result.status == "failed"


def test_stage_bootstrap_demo_manifest_created(sample_config, tmp_path):
    from blueprint_validation.common import write_json
    from blueprint_validation.stages.s1d_gaussian_augment import GaussianAugmentStage

    sample_config.robosplat.enabled = True
    sample_config.robosplat.backend = "legacy_scan"
    sample_config.robosplat.parity_mode = "hybrid"
    sample_config.robosplat.demo_source = "synthetic"
    sample_config.robosplat.bootstrap_if_missing_demo = True
    sample_config.robosplat.variants_per_input = 1

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
    demo_manifest = tmp_path / "gaussian_augment" / "bootstrap_demo" / "demo_manifest.json"
    assert demo_manifest.exists()


def test_s2_manifest_resolution_prefers_s1d(tmp_path):
    from blueprint_validation.stages.s2_enrich import _resolve_render_manifest

    gauss = tmp_path / "gaussian_augment" / "augmented_manifest.json"
    render = tmp_path / "renders" / "render_manifest.json"
    gauss.parent.mkdir(parents=True, exist_ok=True)
    render.parent.mkdir(parents=True, exist_ok=True)
    gauss.write_text("{}")
    render.write_text("{}")

    assert _resolve_render_manifest(tmp_path) == gauss
