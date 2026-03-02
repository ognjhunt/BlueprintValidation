"""Tests for shared camera-quality utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_checker_video(path: Path, *, frames: int = 8, h: int = 64, w: int = 80) -> None:
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8,
        (w, h),
    )
    for i in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = ((i * 13) % 255, (i * 31) % 255, (i * 7) % 255)
        frame[::2, ::2] = 255
        frame[1::2, 1::2] = 40
        writer.write(frame)
    writer.release()


def test_evaluate_clip_quality_passes_for_sharp_centered_target(tmp_path):
    from blueprint_validation.evaluation.camera_quality import evaluate_clip_quality
    from blueprint_validation.rendering.camera_paths import generate_manipulation_arc, save_path_to_json

    approach = np.array([0.0, 0.0, 0.5], dtype=np.float64)
    poses = generate_manipulation_arc(
        approach_point=approach,
        arc_radius=0.55,
        height=0.85,
        num_frames=8,
        resolution=(64, 80),
    )
    path_json = tmp_path / "camera_path.json"
    save_path_to_json(poses, path_json)
    video_path = tmp_path / "clip.mp4"
    _write_checker_video(video_path, frames=8, h=64, w=80)

    clip_entry = {
        "camera_path": str(path_json),
        "video_path": str(video_path),
        "resolution": [64, 80],
        "path_type": "manipulation",
    }
    metrics = evaluate_clip_quality(
        clip_entry=clip_entry,
        target_xyz=[0.0, 0.0, 0.5],
        blur_laplacian_min=5.0,
        min_visible_frame_ratio=0.2,
        min_center_band_ratio=0.2,
        min_approach_angle_bins=2,
        angle_bin_deg=45.0,
        center_band_x=[0.2, 0.8],
        center_band_y=[0.2, 0.8],
        blur_sample_every_n_frames=1,
        blur_max_samples=8,
        min_clip_score=0.5,
        require_target=True,
    )
    assert metrics["quality_gate_passed"] is True
    assert float(metrics["target_visibility_ratio"]) > 0.5
    assert float(metrics["target_center_band_ratio"]) > 0.5
    assert int(metrics["target_approach_angle_bins"]) >= 2
    assert float(metrics["clip_quality_score"]) >= 0.5


def test_evaluate_clip_quality_requires_target_for_manipulation(tmp_path):
    from blueprint_validation.evaluation.camera_quality import evaluate_clip_quality

    clip_entry = {
        "camera_path": str(tmp_path / "missing.json"),
        "video_path": str(tmp_path / "missing.mp4"),
        "resolution": [64, 80],
        "path_type": "manipulation",
    }
    metrics = evaluate_clip_quality(
        clip_entry=clip_entry,
        target_xyz=None,
        blur_laplacian_min=5.0,
        min_visible_frame_ratio=0.2,
        min_center_band_ratio=0.2,
        min_approach_angle_bins=2,
        angle_bin_deg=45.0,
        center_band_x=[0.2, 0.8],
        center_band_y=[0.2, 0.8],
        blur_sample_every_n_frames=1,
        blur_max_samples=8,
        min_clip_score=0.5,
        require_target=True,
    )
    assert metrics["quality_gate_passed"] is False
    assert "missing_target_annotation" in metrics["quality_reject_reasons"]


def test_resolve_center_band_bounds_swaps_out_of_order_pairs():
    from blueprint_validation.evaluation.camera_quality import resolve_center_band_bounds

    x_lo, x_hi, y_lo, y_hi = resolve_center_band_bounds([0.8, 0.2], [0.9, 0.1])
    assert x_lo == pytest.approx(0.2)
    assert x_hi == pytest.approx(0.8)
    assert y_lo == pytest.approx(0.1)
    assert y_hi == pytest.approx(0.9)
