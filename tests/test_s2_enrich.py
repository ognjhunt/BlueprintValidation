"""Tests for S2 enrich compatibility with RoboSplat manifests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict

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


def _write_solid_video(path: Path, value: int, frames: int = 4) -> None:
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = 48, 64
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (w, h),
    )
    frame = np.full((h, w, 3), value, dtype=np.uint8)
    for _ in range(frames):
        writer.write(frame)
    writer.release()


def _write_textured_video(path: Path, frames: int = 8) -> None:
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = 48, 64
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (w, h),
    )
    rng = np.random.default_rng(7)
    for i in range(frames):
        noise = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
        frame = noise.copy()
        frame[:, :, 1] = (frame[:, :, 1].astype(np.uint16) + i * 5) % 255
        writer.write(frame.astype(np.uint8))
    writer.release()


def _write_target_camera_path(
    path: Path,
    *,
    target_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    camera_center_xyz: tuple[float, float, float] | None = None,
    num_frames: int = 8,
    resolution: tuple[int, int] = (48, 64),
) -> None:
    from blueprint_validation.rendering.camera_paths import (
        generate_manipulation_arc,
        save_path_to_json,
    )

    center = camera_center_xyz if camera_center_xyz is not None else target_xyz
    poses = generate_manipulation_arc(
        approach_point=np.asarray(center, dtype=np.float64),
        arc_radius=0.6,
        height=0.7,
        num_frames=num_frames,
        look_down_deg=45.0,
        target_z_bias_m=0.0,
        resolution=resolution,
    )
    save_path_to_json(poses, path)


def _count_video_frames(path: Path) -> int:
    cv2 = pytest.importorskip("cv2")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(0, frames)


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


def test_s2_enrich_anchor_gate_rejects_low_similarity(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.common import read_json, write_json
    from blueprint_validation.config import VariantSpec
    from blueprint_validation.stages.s2_enrich import EnrichStage

    work_dir = tmp_path
    render_dir = work_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    source_video = render_dir / "clip_000_orbit.mp4"
    source_depth = render_dir / "clip_000_orbit_depth.mp4"
    _write_solid_video(source_video, value=25)
    _write_solid_video(source_depth, value=25)

    write_json(
        {
            "facility": "Test Facility",
            "clips": [
                {
                    "clip_name": "clip_000_orbit",
                    "video_path": str(source_video),
                    "depth_video_path": str(source_depth),
                }
            ],
        },
        render_dir / "render_manifest.json",
    )

    sample_config.enrich.dynamic_variants = False
    sample_config.enrich.variants = [VariantSpec(name="v1", prompt="test variant")]
    sample_config.enrich.context_frame_index = 0
    sample_config.enrich.min_frame0_ssim = 0.95
    sample_config.enrich.delete_rejected_outputs = False

    def _fake_enrich_clip(**kwargs):
        out = kwargs["output_dir"] / f"{kwargs['clip_name']}_v1.mp4"
        _write_solid_video(out, value=220)
        return [
            SimpleNamespace(
                variant_name="v1",
                prompt="test variant",
                output_video_path=out,
                input_video_path=kwargs["video_path"],
                context_frame_index=kwargs.get("context_frame_index"),
            )
        ]

    monkeypatch.setattr("blueprint_validation.stages.s2_enrich.enrich_clip", _fake_enrich_clip)

    stage = EnrichStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, work_dir, {})
    assert result.status == "failed"
    assert result.metrics["num_rejected_anchor_similarity"] == 1
    assert result.metrics["num_enriched_clips"] == 0

    manifest = read_json(work_dir / "enriched" / "enriched_manifest.json")
    assert manifest["num_clips"] == 0


def test_s2_enrich_pretrim_limits_cosmos_input_length(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.common import read_json, write_json
    from blueprint_validation.config import VariantSpec
    from blueprint_validation.stages.s2_enrich import EnrichStage

    work_dir = tmp_path
    render_dir = work_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    source_video = render_dir / "clip_000_orbit.mp4"
    source_depth = render_dir / "clip_000_orbit_depth.mp4"
    _write_dummy_video(source_video, frames=10)
    _write_dummy_video(source_depth, frames=10)

    write_json(
        {
            "facility": "Test Facility",
            "clips": [
                {
                    "clip_name": "clip_000_orbit",
                    "video_path": str(source_video),
                    "depth_video_path": str(source_depth),
                }
            ],
        },
        render_dir / "render_manifest.json",
    )

    sample_config.enrich.dynamic_variants = False
    sample_config.enrich.variants = [VariantSpec(name="v1", prompt="test variant")]
    sample_config.enrich.context_frame_index = 8
    sample_config.enrich.max_input_frames = 4
    sample_config.enrich.min_frame0_ssim = 0.0

    captured: dict = {}

    def _fake_enrich_clip(**kwargs):
        captured["video_path"] = kwargs["video_path"]
        captured["depth_path"] = kwargs["depth_path"]
        captured["context_frame_index"] = kwargs.get("context_frame_index")
        out = kwargs["output_dir"] / f"{kwargs['clip_name']}_v1.mp4"
        _write_dummy_video(out, frames=4)
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
    assert result.metrics["num_trimmed_inputs"] == 1

    trimmed_video = captured["video_path"]
    assert trimmed_video != source_video
    assert _count_video_frames(trimmed_video) == 4
    assert captured["context_frame_index"] == 2
    assert captured["depth_path"] is not None
    assert _count_video_frames(captured["depth_path"]) == 4

    manifest = read_json(work_dir / "enriched" / "enriched_manifest.json")
    assert manifest["num_clips"] == 1
    entry = manifest["clips"][0]
    assert entry["input_trimmed"] is True
    assert entry["input_trim_num_frames"] == 4
    assert entry["context_frame_index"] == 2


def test_s2_stage1_coverage_gate_passes_with_visible_sharp_multiview_target(
    sample_config,
    tmp_path,
    monkeypatch,
):
    from blueprint_validation.common import StageResult, write_json
    from blueprint_validation.config import VariantSpec
    from blueprint_validation.stages.s2_enrich import EnrichStage

    work_dir = tmp_path
    render_dir = work_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    video_path = render_dir / "clip_000_manipulation.mp4"
    depth_path = render_dir / "clip_000_manipulation_depth.mp4"
    camera_path = render_dir / "clip_000_manipulation_camera_path.json"
    _write_textured_video(video_path, frames=8)
    _write_textured_video(depth_path, frames=8)
    _write_target_camera_path(camera_path, target_xyz=(0.0, 0.0, 0.0), num_frames=8)

    write_json(
        {
            "facility": "Test Facility",
            "clips": [
                {
                    "clip_name": "clip_000_manipulation",
                    "path_type": "manipulation",
                    "video_path": str(video_path),
                    "depth_video_path": str(depth_path),
                    "camera_path": str(camera_path),
                    "resolution": [48, 64],
                    "path_context": {"approach_point": [0.0, 0.0, 0.0]},
                }
            ],
        },
        render_dir / "render_manifest.json",
    )

    sample_config.render.stage1_coverage_gate_enabled = True
    sample_config.render.stage1_coverage_min_visible_frame_ratio = 0.2
    sample_config.render.stage1_coverage_min_approach_angle_bins = 2
    sample_config.render.stage1_coverage_angle_bin_deg = 45.0
    sample_config.render.stage1_coverage_blur_laplacian_min = 5.0
    sample_config.render.stage1_coverage_blur_sample_every_n_frames = 1
    sample_config.render.stage1_coverage_blur_max_samples_per_clip = 8
    sample_config.enrich.dynamic_variants = False
    sample_config.enrich.variants = [VariantSpec(name="v1", prompt="test variant")]

    def _fake_enrich_clip(**kwargs):
        out = kwargs["output_dir"] / f"{kwargs['clip_name']}_v1.mp4"
        _write_textured_video(out, frames=4)
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
            outputs={"manifest_path": str(render_dir / "render_manifest.json")},
        )
    }

    stage = EnrichStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, work_dir, previous_results)
    assert result.status == "success"
    assert result.metrics["coverage_gate_passed"] is True
    assert result.metrics["coverage_target_count"] == 1
    assert result.metrics["coverage_targets_passing"] == 1


def test_s2_stage1_coverage_gate_fails_blurry_clips(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.common import StageResult, write_json
    from blueprint_validation.config import VariantSpec
    from blueprint_validation.stages.s2_enrich import EnrichStage

    work_dir = tmp_path
    render_dir = work_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    video_path = render_dir / "clip_000_manipulation.mp4"
    depth_path = render_dir / "clip_000_manipulation_depth.mp4"
    camera_path = render_dir / "clip_000_manipulation_camera_path.json"
    _write_solid_video(video_path, value=40, frames=8)
    _write_solid_video(depth_path, value=40, frames=8)
    _write_target_camera_path(camera_path, target_xyz=(0.0, 0.0, 0.0), num_frames=8)

    write_json(
        {
            "facility": "Test Facility",
            "clips": [
                {
                    "clip_name": "clip_000_manipulation",
                    "path_type": "manipulation",
                    "video_path": str(video_path),
                    "depth_video_path": str(depth_path),
                    "camera_path": str(camera_path),
                    "resolution": [48, 64],
                    "path_context": {"approach_point": [0.0, 0.0, 0.0]},
                }
            ],
        },
        render_dir / "render_manifest.json",
    )

    sample_config.render.stage1_coverage_gate_enabled = True
    sample_config.render.stage1_coverage_min_visible_frame_ratio = 0.2
    sample_config.render.stage1_coverage_min_approach_angle_bins = 2
    sample_config.render.stage1_coverage_angle_bin_deg = 45.0
    sample_config.render.stage1_coverage_blur_laplacian_min = 250.0
    sample_config.render.stage1_coverage_blur_sample_every_n_frames = 1
    sample_config.render.stage1_coverage_blur_max_samples_per_clip = 8
    sample_config.enrich.dynamic_variants = False
    sample_config.enrich.variants = [VariantSpec(name="v1", prompt="test variant")]

    def _unexpected_enrich_call(**kwargs):
        raise AssertionError("enrich_clip should not run when Stage-1 coverage gate fails")

    monkeypatch.setattr(
        "blueprint_validation.stages.s2_enrich.enrich_clip",
        _unexpected_enrich_call,
    )

    previous_results = {
        "s1_render": StageResult(
            stage_name="s1_render",
            status="success",
            elapsed_seconds=0,
            outputs={"manifest_path": str(render_dir / "render_manifest.json")},
        )
    }

    stage = EnrichStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, work_dir, previous_results)
    assert result.status == "failed"
    assert result.metrics["coverage_gate_passed"] is False
    assert result.metrics["coverage_blurry_clip_count"] == 1
    assert "coverage gate failed" in (result.detail or "").lower()


def test_s2_stage1_coverage_gate_fails_when_target_has_single_angle(
    sample_config,
    tmp_path,
    monkeypatch,
):
    from blueprint_validation.common import StageResult, write_json
    from blueprint_validation.config import VariantSpec
    from blueprint_validation.stages.s2_enrich import EnrichStage

    work_dir = tmp_path
    render_dir = work_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    video_path = render_dir / "clip_000_manipulation.mp4"
    depth_path = render_dir / "clip_000_manipulation_depth.mp4"
    camera_path = render_dir / "clip_000_manipulation_camera_path.json"
    _write_textured_video(video_path, frames=1)
    _write_textured_video(depth_path, frames=1)
    _write_target_camera_path(camera_path, target_xyz=(0.0, 0.0, 0.0), num_frames=1)

    write_json(
        {
            "facility": "Test Facility",
            "clips": [
                {
                    "clip_name": "clip_000_manipulation",
                    "path_type": "manipulation",
                    "video_path": str(video_path),
                    "depth_video_path": str(depth_path),
                    "camera_path": str(camera_path),
                    "resolution": [48, 64],
                    "path_context": {"approach_point": [0.0, 0.0, 0.0]},
                }
            ],
        },
        render_dir / "render_manifest.json",
    )

    sample_config.render.stage1_coverage_gate_enabled = True
    sample_config.render.stage1_coverage_min_visible_frame_ratio = 0.2
    sample_config.render.stage1_coverage_min_approach_angle_bins = 2
    sample_config.render.stage1_coverage_angle_bin_deg = 45.0
    sample_config.render.stage1_coverage_blur_laplacian_min = 5.0
    sample_config.render.stage1_coverage_blur_sample_every_n_frames = 1
    sample_config.render.stage1_coverage_blur_max_samples_per_clip = 1
    sample_config.enrich.dynamic_variants = False
    sample_config.enrich.variants = [VariantSpec(name="v1", prompt="test variant")]

    def _unexpected_enrich_call(**kwargs):
        raise AssertionError("enrich_clip should not run when Stage-1 coverage gate fails")

    monkeypatch.setattr(
        "blueprint_validation.stages.s2_enrich.enrich_clip",
        _unexpected_enrich_call,
    )

    previous_results = {
        "s1_render": StageResult(
            stage_name="s1_render",
            status="success",
            elapsed_seconds=0,
            outputs={"manifest_path": str(render_dir / "render_manifest.json")},
        )
    }

    stage = EnrichStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, work_dir, previous_results)
    assert result.status == "failed"
    assert result.metrics["coverage_gate_passed"] is False
    assert result.metrics["coverage_targets_failing"] == 1
    assert "coverage gate failed" in (result.detail or "").lower()


def test_s2_stage1_coverage_gate_fails_center_band(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.common import StageResult, write_json
    from blueprint_validation.config import VariantSpec
    from blueprint_validation.stages.s2_enrich import EnrichStage

    work_dir = tmp_path
    render_dir = work_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    video_path = render_dir / "clip_000_manipulation.mp4"
    depth_path = render_dir / "clip_000_manipulation_depth.mp4"
    camera_path = render_dir / "clip_000_manipulation_camera_path.json"
    _write_textured_video(video_path, frames=8)
    _write_textured_video(depth_path, frames=8)
    _write_target_camera_path(
        camera_path,
        target_xyz=(0.0, 0.0, 0.0),
        camera_center_xyz=(0.8, 0.0, 0.0),
        num_frames=8,
    )

    write_json(
        {
            "facility": "Test Facility",
            "clips": [
                {
                    "clip_name": "clip_000_manipulation",
                    "path_type": "manipulation",
                    "video_path": str(video_path),
                    "depth_video_path": str(depth_path),
                    "camera_path": str(camera_path),
                    "resolution": [48, 64],
                    "path_context": {"approach_point": [0.0, 0.0, 0.0]},
                }
            ],
        },
        render_dir / "render_manifest.json",
    )

    sample_config.render.stage1_coverage_gate_enabled = True
    sample_config.render.stage1_coverage_min_visible_frame_ratio = 0.1
    sample_config.render.stage1_coverage_min_approach_angle_bins = 2
    sample_config.render.stage1_coverage_angle_bin_deg = 45.0
    sample_config.render.stage1_coverage_blur_laplacian_min = 5.0
    sample_config.render.stage1_coverage_blur_sample_every_n_frames = 1
    sample_config.render.stage1_coverage_blur_max_samples_per_clip = 8
    sample_config.render.stage1_coverage_min_center_band_ratio = 0.8
    sample_config.render.stage1_coverage_center_band_x = [0.35, 0.65]
    sample_config.render.stage1_coverage_center_band_y = [0.35, 0.65]
    sample_config.enrich.dynamic_variants = False
    sample_config.enrich.variants = [VariantSpec(name="v1", prompt="test variant")]

    def _unexpected_enrich_call(**kwargs):
        raise AssertionError("enrich_clip should not run when Stage-1 coverage gate fails")

    monkeypatch.setattr(
        "blueprint_validation.stages.s2_enrich.enrich_clip",
        _unexpected_enrich_call,
    )

    previous_results = {
        "s1_render": StageResult(
            stage_name="s1_render",
            status="success",
            elapsed_seconds=0,
            outputs={"manifest_path": str(render_dir / "render_manifest.json")},
        )
    }

    stage = EnrichStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, work_dir, previous_results)
    assert result.status == "failed"
    assert result.metrics["coverage_gate_passed"] is False
    assert result.metrics["coverage_targets_center_band_failing"] == 1


def test_s2_context_selection_prefers_target_centered_frame(
    sample_config,
    tmp_path,
    monkeypatch,
):
    from blueprint_validation.common import write_json
    from blueprint_validation.config import VariantSpec
    from blueprint_validation.stages.s2_enrich import EnrichStage, _resolve_clip_context_selection

    work_dir = tmp_path
    render_dir = work_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    video_path = render_dir / "clip_000_manipulation.mp4"
    depth_path = render_dir / "clip_000_manipulation_depth.mp4"
    camera_path = render_dir / "clip_000_manipulation_camera_path.json"
    _write_textured_video(video_path, frames=8)
    _write_textured_video(depth_path, frames=8)
    _write_target_camera_path(
        camera_path,
        target_xyz=(0.0, 0.0, 0.0),
        camera_center_xyz=(0.7, 0.0, 0.0),
        num_frames=8,
    )

    clip_entry = {
        "clip_name": "clip_000_manipulation",
        "path_type": "manipulation",
        "video_path": str(video_path),
        "depth_video_path": str(depth_path),
        "camera_path": str(camera_path),
        "resolution": [48, 64],
        "path_context": {"approach_point": [0.0, 0.0, 0.0]},
    }
    write_json({"facility": "Test Facility", "clips": [clip_entry]}, render_dir / "render_manifest.json")

    sample_config.render.stage1_coverage_gate_enabled = False
    sample_config.enrich.dynamic_variants = False
    sample_config.enrich.variants = [VariantSpec(name="v1", prompt="test variant")]
    sample_config.enrich.context_frame_mode = "target_centered"
    sample_config.enrich.context_frame_index = 0

    expected_idx, expected_mode, _ = _resolve_clip_context_selection(clip_entry, sample_config)
    captured: Dict[str, object] = {}

    def _fake_enrich_clip(**kwargs):
        captured["context_frame_index"] = kwargs.get("context_frame_index")
        out = kwargs["output_dir"] / f"{kwargs['clip_name']}_v1.mp4"
        _write_textured_video(out, frames=4)
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
    assert captured["context_frame_index"] == expected_idx
    assert result.metrics["context_selection_target_centered_count"] == (
        1 if expected_mode == "target_centered" else 0
    )


def test_s2_context_selection_falls_back_to_fixed_when_target_missing(
    sample_config,
    tmp_path,
    monkeypatch,
):
    from blueprint_validation.common import write_json
    from blueprint_validation.config import VariantSpec
    from blueprint_validation.stages.s2_enrich import EnrichStage

    work_dir = tmp_path
    render_dir = work_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    video_path = render_dir / "clip_000_manipulation.mp4"
    depth_path = render_dir / "clip_000_manipulation_depth.mp4"
    _write_textured_video(video_path, frames=8)
    _write_textured_video(depth_path, frames=8)

    write_json(
        {
            "facility": "Test Facility",
            "clips": [
                {
                    "clip_name": "clip_000_manipulation",
                    "path_type": "manipulation",
                    "video_path": str(video_path),
                    "depth_video_path": str(depth_path),
                    "resolution": [48, 64],
                    "path_context": {},
                }
            ],
        },
        render_dir / "render_manifest.json",
    )

    sample_config.render.stage1_coverage_gate_enabled = False
    sample_config.enrich.dynamic_variants = False
    sample_config.enrich.variants = [VariantSpec(name="v1", prompt="test variant")]
    sample_config.enrich.context_frame_mode = "target_centered"
    sample_config.enrich.context_frame_index = 3

    captured: Dict[str, object] = {}

    def _fake_enrich_clip(**kwargs):
        captured["context_frame_index"] = kwargs.get("context_frame_index")
        out = kwargs["output_dir"] / f"{kwargs['clip_name']}_v1.mp4"
        _write_textured_video(out, frames=4)
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
    assert captured["context_frame_index"] == 3
    assert result.metrics["context_selection_fixed_count"] == 1


def test_s2_task_targeted_clip_selection_prefers_manipulation_clip(
    sample_config,
    tmp_path,
    monkeypatch,
):
    from blueprint_validation.common import write_json
    from blueprint_validation.config import VariantSpec
    from blueprint_validation.stages.s2_enrich import EnrichStage

    work_dir = tmp_path
    render_dir = work_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    orbit_video = render_dir / "clip_000_orbit.mp4"
    orbit_depth = render_dir / "clip_000_orbit_depth.mp4"
    manip_video = render_dir / "clip_001_manipulation.mp4"
    manip_depth = render_dir / "clip_001_manipulation_depth.mp4"
    _write_dummy_video(orbit_video, frames=6)
    _write_dummy_video(orbit_depth, frames=6)
    _write_dummy_video(manip_video, frames=6)
    _write_dummy_video(manip_depth, frames=6)

    write_json(
        {
            "facility": "Test Facility",
            "clips": [
                {
                    "clip_name": "clip_000_orbit",
                    "clip_index": 0,
                    "path_type": "orbit",
                    "video_path": str(orbit_video),
                    "depth_video_path": str(orbit_depth),
                },
                {
                    "clip_name": "clip_001_manipulation",
                    "clip_index": 1,
                    "path_type": "manipulation",
                    "video_path": str(manip_video),
                    "depth_video_path": str(manip_depth),
                },
            ],
        },
        render_dir / "render_manifest.json",
    )

    hints_path = work_dir / "task_targets.synthetic.json"
    write_json(
        {
            "manipulation_candidates": [
                {
                    "instance_id": "157",
                    "label": "trash_can",
                    "boundingBox": {"center": [0.0, 0.0, 0.0], "extents": [0.3, 0.3, 0.5]},
                }
            ]
        },
        hints_path,
    )
    facility = list(sample_config.facilities.values())[0]
    facility.task_hints_path = hints_path

    sample_config.enrich.dynamic_variants = False
    sample_config.enrich.variants = [VariantSpec(name="v1", prompt="test variant")]
    sample_config.enrich.max_source_clips = 1
    sample_config.enrich.source_clip_selection_mode = "task_targeted"
    sample_config.enrich.source_clip_task = "Pick up trash_can_157 and place it in the target zone"

    captured = {"clip_names": []}

    def _fake_enrich_clip(**kwargs):
        captured["clip_names"].append(kwargs["clip_name"])
        out = kwargs["output_dir"] / f"{kwargs['clip_name']}_v1.mp4"
        _write_dummy_video(out, frames=4)
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
    result = stage.run(sample_config, facility, work_dir, {})
    assert result.status == "success"
    assert captured["clip_names"] == ["clip_001_manipulation"]
    assert result.metrics["num_selected_source_clips"] == 1


def test_s2_orientation_fix_rotate180_applies_to_rgb_and_depth(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from blueprint_validation.config import FacilityConfig
    from blueprint_validation.stages.s2_enrich import _resolve_oriented_inputs_for_clip

    source_rgb = tmp_path / "clip_rgb.mp4"
    source_depth = tmp_path / "clip_depth.mp4"
    h, w = 24, 32

    writer_rgb = cv2.VideoWriter(
        str(source_rgb),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (w, h),
    )
    rgb_frame = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_frame[0:4, 0:4] = [0, 0, 255]
    writer_rgb.write(rgb_frame)
    writer_rgb.release()

    writer_depth = cv2.VideoWriter(
        str(source_depth),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (w, h),
        isColor=False,
    )
    depth_frame = np.zeros((h, w), dtype=np.uint8)
    depth_frame[0:4, 0:4] = 255
    writer_depth.write(depth_frame)
    writer_depth.release()

    facility = FacilityConfig(
        name="Kitchen",
        ply_path=tmp_path / "dummy.ply",
        video_orientation_fix="rotate180",
    )
    facility.ply_path.write_bytes(b"x")
    out_rgb, out_depth, mode = _resolve_oriented_inputs_for_clip(
        facility=facility,
        clip_name="clip_000_orbit",
        enrich_dir=tmp_path / "enriched",
        video_path=source_rgb,
        depth_path=source_depth,
    )

    assert mode == "rotate180"
    assert out_rgb != source_rgb
    assert out_depth is not None
    assert out_depth != source_depth

    cap_rgb = cv2.VideoCapture(str(out_rgb))
    ok_rgb, rotated_rgb = cap_rgb.read()
    cap_rgb.release()
    assert ok_rgb
    assert rotated_rgb[h - 2, w - 2, 2] > 200

    cap_depth = cv2.VideoCapture(str(out_depth))
    ok_depth, rotated_depth = cap_depth.read()
    cap_depth.release()
    assert ok_depth
    if rotated_depth.ndim == 3:
        rotated_depth = cv2.cvtColor(rotated_depth, cv2.COLOR_BGR2GRAY)
    assert rotated_depth[h - 2, w - 2] > 200


def test_s2_multi_view_context_uses_image_context_path_and_omits_context_index(
    sample_config,
    tmp_path,
    monkeypatch,
):
    from blueprint_validation.common import write_json
    from blueprint_validation.config import VariantSpec
    from blueprint_validation.stages.s2_enrich import EnrichStage

    work_dir = tmp_path
    render_dir = work_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    video_path = render_dir / "clip_000_orbit.mp4"
    depth_path = render_dir / "clip_000_orbit_depth.mp4"
    _write_textured_video(video_path, frames=12)
    _write_textured_video(depth_path, frames=12)

    write_json(
        {
            "facility": "Test Facility",
            "clips": [
                {
                    "clip_name": "clip_000_orbit",
                    "path_type": "orbit",
                    "video_path": str(video_path),
                    "depth_video_path": str(depth_path),
                }
            ],
        },
        render_dir / "render_manifest.json",
    )

    sample_config.enrich.dynamic_variants = False
    sample_config.enrich.variants = [VariantSpec(name="v1", prompt="test variant")]
    sample_config.enrich.multi_view_context_enabled = True
    sample_config.enrich.multi_view_context_offsets = [-3, 0, 3]
    sample_config.enrich.scene_index_enabled = False

    captured: Dict[str, object] = {}

    def _fake_enrich_clip(**kwargs):
        captured["context_frame_index"] = kwargs.get("context_frame_index")
        captured["image_context_path"] = kwargs.get("image_context_path")
        out = kwargs["output_dir"] / f"{kwargs['clip_name']}_v1.mp4"
        _write_dummy_video(out, frames=4)
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
    facility = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, facility, work_dir, {})
    assert result.status == "success"
    assert captured["context_frame_index"] is None
    assert captured["image_context_path"] is not None
    assert Path(str(captured["image_context_path"])).exists()


def test_s2_max_input_frames_zero_does_not_pretrim(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.common import write_json
    from blueprint_validation.config import VariantSpec
    from blueprint_validation.stages.s2_enrich import EnrichStage

    work_dir = tmp_path
    render_dir = work_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    source_video = render_dir / "clip_000_orbit.mp4"
    source_depth = render_dir / "clip_000_orbit_depth.mp4"
    _write_dummy_video(source_video, frames=9)
    _write_dummy_video(source_depth, frames=9)

    write_json(
        {
            "facility": "Test Facility",
            "clips": [
                {
                    "clip_name": "clip_000_orbit",
                    "video_path": str(source_video),
                    "depth_video_path": str(source_depth),
                }
            ],
        },
        render_dir / "render_manifest.json",
    )

    sample_config.enrich.dynamic_variants = False
    sample_config.enrich.variants = [VariantSpec(name="v1", prompt="test variant")]
    sample_config.enrich.max_input_frames = 0

    captured: Dict[str, object] = {}

    def _fake_enrich_clip(**kwargs):
        captured["video_path"] = kwargs["video_path"]
        out = kwargs["output_dir"] / f"{kwargs['clip_name']}_v1.mp4"
        _write_dummy_video(out, frames=4)
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
    facility = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, facility, work_dir, {})
    assert result.status == "success"
    assert captured["video_path"] == source_video
    assert result.metrics["num_trimmed_inputs"] == 0


def test_select_source_clips_task_targeted_fallback_orders_manipulation_first(sample_config):
    from blueprint_validation.stages.s2_enrich import _select_source_clips

    sample_config.enrich.max_source_clips = 1
    sample_config.enrich.source_clip_selection_mode = "task_targeted"
    sample_config.enrich.source_clip_task = "Pick up trash_can_157 and place it in the target zone"
    facility = list(sample_config.facilities.values())[0]
    facility.task_hints_path = None

    render_manifest = {
        "clips": [
            {"clip_name": "clip_000_orbit", "path_type": "orbit"},
            {"clip_name": "clip_001_manipulation", "path_type": "manipulation"},
            {"clip_name": "clip_002_sweep", "path_type": "sweep"},
        ]
    }
    selected, meta = _select_source_clips(
        render_manifest=render_manifest,
        config=sample_config,
        facility=facility,
    )

    assert [c["clip_name"] for c in selected] == ["clip_001_manipulation"]
    assert meta["fallback"] == "missing_task_hints"


def test_resolve_multi_view_context_indices_clamps_and_dedupes():
    from blueprint_validation.stages.s2_enrich import _resolve_multi_view_context_indices

    indices = _resolve_multi_view_context_indices(
        anchor_index=4,
        total_frames=6,
        offsets=[-9, -2, 0, 2, 2, 20],
    )
    assert indices == [0, 2, 4, 5]
