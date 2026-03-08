from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from blueprint_validation.common import read_json, write_json
from blueprint_validation.stages.s5_visual_fidelity import VisualFidelityStage


def _write_dummy_video_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"00")


def test_s5_visual_fidelity_uses_stage2_lineage_manifest_when_renders_missing(
    sample_config,
    tmp_path,
    monkeypatch,
):
    work_dir = tmp_path
    source_video = work_dir / "gaussian_augment" / "clip_000.mp4"
    enriched_video = work_dir / "enriched" / "clip_000_v1.mp4"
    _write_dummy_video_file(source_video)
    _write_dummy_video_file(enriched_video)

    write_json(
        {
            "clips": [
                {
                    "clip_name": "clip_000",
                    "video_path": str(source_video),
                }
            ]
        },
        work_dir / "gaussian_augment" / "augmented_manifest.json",
    )
    write_json(
        {
            "clips": [
                {
                    "clip_name": "clip_000",
                    "variant_name": "v1",
                    "output_video_path": str(enriched_video),
                }
            ]
        },
        work_dir / "enriched" / "enriched_manifest.json",
    )

    monkeypatch.setattr(
        "blueprint_validation.stages.s5_visual_fidelity._extract_frames",
        lambda *_args, **_kwargs: [np.zeros((4, 4, 3), dtype=np.uint8)],
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s5_visual_fidelity.compute_video_metrics",
        lambda *_args, **_kwargs: SimpleNamespace(
            to_dict=lambda: {"mean_psnr": 30.0, "mean_ssim": 0.9}
        ),
    )

    stage = VisualFidelityStage()
    facility = sample_config.facilities["test_facility"]
    result = stage.run(sample_config, facility, work_dir, {})
    assert result.status == "success"
    assert result.metrics["source_stage"] == "s1d_gaussian_augment"
    assert result.metrics["num_reference_from_source_manifest"] == 1

    report = read_json(work_dir / "visual_fidelity" / "visual_fidelity.json")
    assert report["per_clip"][0]["reference_resolution_mode"] == "source_manifest"
    assert report["per_clip"][0]["reference_video_path"] == str(source_video)


def test_s5_visual_fidelity_prefers_input_video_path_over_manifest(
    sample_config,
    tmp_path,
    monkeypatch,
):
    work_dir = tmp_path
    manifest_source_video = work_dir / "renders" / "clip_000_render.mp4"
    enriched_input_video = work_dir / "enriched" / "clip_000_input.mp4"
    enriched_output_video = work_dir / "enriched" / "clip_000_out.mp4"
    for path in [manifest_source_video, enriched_input_video, enriched_output_video]:
        _write_dummy_video_file(path)

    write_json(
        {
            "clips": [
                {
                    "clip_name": "clip_000",
                    "video_path": str(manifest_source_video),
                }
            ]
        },
        work_dir / "renders" / "render_manifest.json",
    )
    write_json(
        {
            "clips": [
                {
                    "clip_name": "clip_000",
                    "variant_name": "v1",
                    "input_video_path": str(enriched_input_video),
                    "output_video_path": str(enriched_output_video),
                }
            ]
        },
        work_dir / "enriched" / "enriched_manifest.json",
    )

    monkeypatch.setattr(
        "blueprint_validation.stages.s5_visual_fidelity._extract_frames",
        lambda *_args, **_kwargs: [np.zeros((4, 4, 3), dtype=np.uint8)],
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s5_visual_fidelity.compute_video_metrics",
        lambda *_args, **_kwargs: SimpleNamespace(
            to_dict=lambda: {"mean_psnr": 31.0, "mean_ssim": 0.91}
        ),
    )

    stage = VisualFidelityStage()
    facility = sample_config.facilities["test_facility"]
    result = stage.run(sample_config, facility, work_dir, {})
    assert result.status == "success"
    assert result.metrics["num_reference_from_input_video_path"] == 1

    report = read_json(work_dir / "visual_fidelity" / "visual_fidelity.json")
    assert report["per_clip"][0]["reference_resolution_mode"] == "input_video_path"
    assert report["per_clip"][0]["reference_video_path"] == str(enriched_input_video)


def test_s5_visual_fidelity_is_diagnostic_only_when_inputs_missing(sample_config, tmp_path):
    stage = VisualFidelityStage()
    facility = sample_config.facilities["test_facility"]
    result = stage.run(sample_config, facility, tmp_path, {})
    assert result.status == "success"
    assert result.metrics["diagnostic_only"] is True
    assert result.metrics["diagnostic_status"] == "missing_inputs"
