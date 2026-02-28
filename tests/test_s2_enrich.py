"""Tests for S2 enrich compatibility with RoboSplat manifests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _write_dummy_video(path: Path, frames: int = 4) -> None:
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = 48, 64
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (w, h),
    )
    for i in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[..., 1] = 40 + i * 10
        frame[..., 2] = 90 + i * 5
        writer.write(frame)
    writer.release()


def test_s2_enrich_reads_robosplat_manifest(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.common import read_json, write_json
    from blueprint_validation.config import VariantSpec
    from blueprint_validation.stages.s2_enrich import EnrichStage

    work_dir = tmp_path
    gauss_dir = work_dir / "gaussian_augment"
    gauss_dir.mkdir(parents=True, exist_ok=True)
    source_video = gauss_dir / "clip_000_rb00.mp4"
    source_depth = gauss_dir / "clip_000_rb00_depth.mp4"
    _write_dummy_video(source_video)
    _write_dummy_video(source_depth)

    write_json(
        {
            "facility": "Test Facility",
            "backend_used": "native",
            "num_source_clips": 1,
            "num_augmented_clips": 1,
            "clips": [
                {
                    "clip_name": "clip_000_rb00",
                    "video_path": str(source_video),
                    "depth_video_path": str(source_depth),
                    "variant_id": "rb-00",
                    "variant_ops": {"yaw_deg": 3.0},
                    "object_source": "cluster",
                    "demo_source": "synthetic",
                    "augmentation_type": "robosplat_full",
                    "backend_used": "native",
                }
            ],
        },
        gauss_dir / "augmented_manifest.json",
    )

    monkeypatch.setattr(
        "blueprint_validation.stages.s2_enrich.get_variants",
        lambda **kwargs: [VariantSpec(name="v1", prompt="test variant")],
    )

    def _fake_enrich_clip(**kwargs):
        out = kwargs["output_dir"] / f"{kwargs['clip_name']}_v1.mp4"
        _write_dummy_video(out)
        return [
            SimpleNamespace(
                variant_name="v1",
                prompt="test variant",
                output_video_path=out,
                input_video_path=kwargs["video_path"],
            )
        ]

    monkeypatch.setattr("blueprint_validation.stages.s2_enrich.enrich_clip", _fake_enrich_clip)

    stage = EnrichStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, work_dir, {})
    assert result.status == "success"
    manifest = read_json(work_dir / "enriched" / "enriched_manifest.json")
    assert manifest["num_clips"] == 1
    assert manifest["clips"][0]["clip_name"] == "clip_000_rb00"

