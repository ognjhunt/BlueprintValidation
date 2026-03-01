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


def test_s2_manifest_resolution_prefers_s1e(tmp_path):
    from blueprint_validation.stages.s2_enrich import _resolve_render_manifest

    splat = tmp_path / "splatsim" / "interaction_manifest.json"
    gauss = tmp_path / "gaussian_augment" / "augmented_manifest.json"
    render = tmp_path / "renders" / "render_manifest.json"
    splat.parent.mkdir(parents=True, exist_ok=True)
    gauss.parent.mkdir(parents=True, exist_ok=True)
    render.parent.mkdir(parents=True, exist_ok=True)
    splat.write_text("{}")
    gauss.write_text("{}")
    render.write_text("{}")

    assert _resolve_render_manifest(tmp_path) == splat


def test_s2_enrich_prefers_previous_results_manifest(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.common import StageResult, write_json
    from blueprint_validation.config import VariantSpec
    from blueprint_validation.stages.s2_enrich import EnrichStage

    render_dir = tmp_path / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    video_path = render_dir / "clip_render.mp4"
    depth_path = render_dir / "clip_render_depth.mp4"
    _write_dummy_video(video_path)
    _write_dummy_video(depth_path)

    render_manifest = render_dir / "render_manifest.json"
    write_json(
        {
            "facility": "Test Facility",
            "clips": [
                {
                    "clip_name": "clip_render",
                    "video_path": str(video_path),
                    "depth_video_path": str(depth_path),
                }
            ],
        },
        render_manifest,
    )

    # Stale higher-priority manifest on disk should be ignored when previous_results are present.
    stale_splatsim = tmp_path / "splatsim" / "interaction_manifest.json"
    stale_splatsim.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        {
            "facility": "Test Facility",
            "clips": [
                {
                    "clip_name": "clip_stale",
                    "video_path": str(video_path),
                    "depth_video_path": str(depth_path),
                }
            ],
        },
        stale_splatsim,
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

    previous_results = {
        "s1_render": StageResult(
            stage_name="s1_render",
            status="success",
            elapsed_seconds=0,
            outputs={"manifest_path": str(render_manifest)},
        )
    }
    stage = EnrichStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, previous_results)

    assert result.status == "success"
    assert result.metrics["source_stage"] == "s1_render"
    assert result.metrics["source_mode"] == "previous_results"


def test_scene_specific_depth_control_rotates_180(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import blueprint_validation.stages.s2_enrich as s2

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(s2, "_FORCE_ROTATE_180_DEPTH_FACILITIES", {"kitchen_0787"})
    _maybe_prepare_depth_control = s2._maybe_prepare_depth_control

    depth_path = tmp_path / "in_depth.mp4"
    h, w = 24, 32
    writer = cv2.VideoWriter(
        str(depth_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (w, h),
        isColor=False,
    )
    frame = np.zeros((h, w), dtype=np.uint8)
    frame[0:4, 0:4] = 255  # bright corner marker at top-left
    writer.write(frame)
    writer.release()

    fixed = _maybe_prepare_depth_control(
        facility_name="kitchen_0787",
        clip_name="clip_000_orbit",
        depth_path=depth_path,
        enrich_dir=tmp_path / "enriched",
    )
    assert fixed is not None
    assert fixed.exists()
    assert fixed != depth_path

    cap = cv2.VideoCapture(str(fixed))
    ok, out_frame = cap.read()
    cap.release()
    assert ok
    if out_frame.ndim == 3:
        out_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2GRAY)
    # Marker should be at bottom-right after 180 rotation.
    assert out_frame[h - 2, w - 2] > 200
    monkeypatch.undo()


def test_scene_specific_depth_control_passthrough_for_other_facilities(tmp_path):
    from blueprint_validation.stages.s2_enrich import _maybe_prepare_depth_control

    depth_path = tmp_path / "in_depth.mp4"
    depth_path.write_bytes(b"dummy")
    got = _maybe_prepare_depth_control(
        facility_name="other_scene",
        clip_name="clip_000_orbit",
        depth_path=depth_path,
        enrich_dir=tmp_path / "enriched",
    )
    assert got == depth_path
