"""Regression tests for CPU-only dataset quality gates."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_validation.common import read_json
from blueprint_validation.config import DatasetDistributionConfig, DatasetQualityConfig
from blueprint_validation.training import dataset_builder as builder_mod
from blueprint_validation.training.dataset_builder import build_dreamdojo_dataset


def _make_clip(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return path


def _write_manifest(path: Path, clips: list[dict]) -> Path:
    path.write_text(json.dumps({"clips": clips}), encoding="utf-8")
    return path


def _patch_video_quality(monkeypatch: pytest.MonkeyPatch, *, fingerprints: list[str]) -> None:
    index = {"value": 0}

    def _fingerprint(_path: Path) -> str:
        i = index["value"]
        index["value"] = i + 1
        return fingerprints[min(i, len(fingerprints) - 1)]

    def _ensure_h264_video(*, input_path: Path, **_kwargs):
        return SimpleNamespace(
            path=input_path,
            content_monochrome_warning=False,
            content_max_std_dev=12.0,
            decoded_frames=24,
        )

    monkeypatch.setattr(builder_mod, "ensure_h264_video", _ensure_h264_video)
    monkeypatch.setattr(
        builder_mod,
        "analyze_temporal_quality",
        lambda *_args, **_kwargs: {
            "decoded_frames": 24.0,
            "mean_interframe_delta": 3.0,
            "freeze_ratio": 0.1,
            "abrupt_cut_ratio": 0.1,
            "blockiness_score": 0.1,
        },
    )
    monkeypatch.setattr(builder_mod, "temporal_reject_reasons", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(builder_mod, "fingerprint_video_content", _fingerprint)


def test_dataset_builder_quarantines_duplicate_clip_prompt_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_video_quality(monkeypatch, fingerprints=["fp_same", "fp_same"])
    clip_a = _make_clip(tmp_path / "input" / "a.mp4")
    clip_b = _make_clip(tmp_path / "input" / "b.mp4")
    manifest_path = _write_manifest(
        tmp_path / "enriched_manifest.json",
        [
            {
                "clip_name": "clip_a",
                "variant_name": "daylight",
                "prompt": "robot opens cabinet door",
                "output_video_path": str(clip_a),
            },
            {
                "clip_name": "clip_b",
                "variant_name": "daylight",
                "prompt": "robot opens cabinet door",
                "output_video_path": str(clip_b),
            },
        ],
    )
    quality = DatasetQualityConfig(
        fail_on_rejections=False,
        max_reject_fraction=1.0,
        distribution=DatasetDistributionConfig(enabled=False),
    )

    dataset_dir = build_dreamdojo_dataset(
        enriched_manifest_path=manifest_path,
        output_dir=tmp_path / "dataset_out",
        facility_name="facility_a",
        quality_config=quality,
        dataset_tag="facility_a:s3",
    )

    with (dataset_dir / "metadata.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1

    quarantine_manifest = read_json(dataset_dir / "quarantine" / "quarantine_manifest.json")
    assert quarantine_manifest["num_rejected_clips"] == 1
    rejection = quarantine_manifest["rejections"][0]
    assert "duplicate_clip_prompt_pair" in rejection["detail"]
    assert Path(rejection["quarantine_video_path"]).exists()


def test_dataset_builder_blocks_cross_dataset_leakage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_a = _make_clip(tmp_path / "run_a" / "clip.mp4")
    clip_b = _make_clip(tmp_path / "run_b" / "clip.mp4")
    quality = DatasetQualityConfig(
        fail_on_rejections=False,
        max_reject_fraction=1.0,
        distribution=DatasetDistributionConfig(enabled=False),
    )
    leakage_index_path = tmp_path / "quality" / "dataset_fingerprint_index.json"

    def _ensure_h264_video(*, input_path: Path, **_kwargs):
        return SimpleNamespace(
            path=input_path,
            content_monochrome_warning=False,
            content_max_std_dev=12.0,
            decoded_frames=24,
        )

    monkeypatch.setattr(builder_mod, "ensure_h264_video", _ensure_h264_video)
    monkeypatch.setattr(
        builder_mod,
        "analyze_temporal_quality",
        lambda *_args, **_kwargs: {"decoded_frames": 24.0},
    )
    monkeypatch.setattr(builder_mod, "temporal_reject_reasons", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(builder_mod, "fingerprint_video_content", lambda _path: "fp_leak")

    manifest_a = _write_manifest(
        tmp_path / "manifest_a.json",
        [
            {
                "clip_name": "clip_a",
                "variant_name": "v1",
                "prompt": "robot opens cabinet door",
                "output_video_path": str(clip_a),
            }
        ],
    )
    build_dreamdojo_dataset(
        enriched_manifest_path=manifest_a,
        output_dir=tmp_path / "out_a",
        facility_name="facility_a",
        quality_config=quality,
        leakage_index_path=leakage_index_path,
        dataset_tag="facility_a:s3",
    )
    assert leakage_index_path.exists()

    manifest_b = _write_manifest(
        tmp_path / "manifest_b.json",
        [
            {
                "clip_name": "clip_b",
                "variant_name": "v2",
                "prompt": "robot opens cabinet door",
                "output_video_path": str(clip_b),
            }
        ],
    )
    with pytest.raises(RuntimeError, match="cross_dataset_leakage"):
        build_dreamdojo_dataset(
            enriched_manifest_path=manifest_b,
            output_dir=tmp_path / "out_b",
            facility_name="facility_b",
            quality_config=quality,
            leakage_index_path=leakage_index_path,
            dataset_tag="facility_b:s3",
        )


def test_dataset_builder_fails_on_distribution_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_video_quality(monkeypatch, fingerprints=["fp0", "fp1", "fp2"])
    clips = []
    for idx in range(3):
        clip_path = _make_clip(tmp_path / "distribution" / f"clip_{idx}.mp4")
        clips.append(
            {
                "clip_name": f"clip_{idx}",
                "variant_name": "same_variant",
                "prompt": "robot opens cabinet door",
                "output_video_path": str(clip_path),
            }
        )
    manifest_path = _write_manifest(tmp_path / "distribution_manifest.json", clips)
    quality = DatasetQualityConfig(
        fail_on_rejections=False,
        max_reject_fraction=1.0,
        distribution=DatasetDistributionConfig(
            enabled=True,
            min_total_clips_for_caps=1,
            min_unique_variants=1,
            min_unique_source_clips=1,
            max_single_variant_fraction=0.60,
            max_single_source_clip_fraction=1.0,
            max_prompt_dominance_fraction=0.60,
        ),
    )

    with pytest.raises(RuntimeError, match="distribution_checks_failed"):
        build_dreamdojo_dataset(
            enriched_manifest_path=manifest_path,
            output_dir=tmp_path / "distribution_out",
            facility_name="facility_a",
            quality_config=quality,
            dataset_tag="facility_a:s3",
        )


def test_dataset_builder_prompt_lint_rejects_empty_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_video_quality(monkeypatch, fingerprints=["fp0", "fp1"])
    clip_a = _make_clip(tmp_path / "prompt" / "a.mp4")
    clip_b = _make_clip(tmp_path / "prompt" / "b.mp4")
    manifest_path = _write_manifest(
        tmp_path / "prompt_manifest.json",
        [
            {
                "clip_name": "clip_bad",
                "variant_name": "v1",
                "prompt": "",
                "output_video_path": str(clip_a),
            },
            {
                "clip_name": "clip_good",
                "variant_name": "v2",
                "prompt": "robot closes drawer carefully",
                "output_video_path": str(clip_b),
            },
        ],
    )
    quality = DatasetQualityConfig(
        fail_on_rejections=False,
        max_reject_fraction=1.0,
        distribution=DatasetDistributionConfig(enabled=False),
    )

    dataset_dir = build_dreamdojo_dataset(
        enriched_manifest_path=manifest_path,
        output_dir=tmp_path / "prompt_out",
        facility_name="facility_a",
        quality_config=quality,
        dataset_tag="facility_a:s3",
    )

    report = read_json(dataset_dir / "dataset_quality_report.json")
    assert report["num_accepted_clips"] == 1
    assert report["num_rejected_clips"] == 1
    rejection_detail = read_json(dataset_dir / "quarantine" / "quarantine_manifest.json")[
        "rejections"
    ][0]["detail"]
    assert "prompt_empty" in rejection_detail
