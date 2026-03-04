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
from blueprint_validation.training.data_quality import fingerprint_video_content
from blueprint_validation.training.dataset_builder import build_dreamdojo_dataset


def _make_clip(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return path


def _write_manifest(path: Path, clips: list[dict]) -> Path:
    path.write_text(json.dumps({"clips": clips}), encoding="utf-8")
    return path


def _write_video(path: Path, *, seed: int = 0, frames: int = 8) -> Path:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (24, 24))
    if not writer.isOpened():
        raise RuntimeError(f"failed opening writer: {path}")
    for idx in range(frames):
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        frame[:, :, 0] = np.uint8((idx * 17) % 255)
        frame[:, :, 1] = np.uint8((20 + idx * 11) % 255)
        frame[:, :, 2] = rng.integers(0, 40, size=(24, 24), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _patch_video_gate_checks_without_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_fingerprint_video_content_is_path_independent(tmp_path: Path) -> None:
    original = _write_video(tmp_path / "path_a" / "clip.mp4", seed=11)
    copied = tmp_path / "path_b" / "clip.mp4"
    copied.parent.mkdir(parents=True, exist_ok=True)
    copied.write_bytes(original.read_bytes())
    assert fingerprint_video_content(original) == fingerprint_video_content(copied)


def test_dataset_builder_detects_duplicate_content_across_paths_without_mocked_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_video_gate_checks_without_fingerprint(monkeypatch)
    clip_a = _write_video(tmp_path / "dup_a" / "clip.mp4", seed=19)
    clip_b = tmp_path / "dup_b" / "clip.mp4"
    clip_b.parent.mkdir(parents=True, exist_ok=True)
    clip_b.write_bytes(clip_a.read_bytes())
    manifest_path = _write_manifest(
        tmp_path / "dup_manifest.json",
        [
            {
                "clip_name": "clip_a",
                "variant_name": "v1",
                "prompt": "robot opens cabinet door",
                "output_video_path": str(clip_a),
            },
            {
                "clip_name": "clip_b",
                "variant_name": "v2",
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
        output_dir=tmp_path / "dup_out",
        facility_name="facility_a",
        quality_config=quality,
        dataset_tag="facility_a:s3",
    )
    with (dataset_dir / "metadata.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    quarantine_manifest = read_json(dataset_dir / "quarantine" / "quarantine_manifest.json")
    assert "duplicate_clip_prompt_pair" in quarantine_manifest["rejections"][0]["detail"]


def test_dataset_builder_does_not_persist_leakage_index_when_hard_gate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_video_quality(monkeypatch, fingerprints=["fp_accept", "fp_reject"])
    clip_a = _make_clip(tmp_path / "hard_fail" / "a.mp4")
    clip_b = _make_clip(tmp_path / "hard_fail" / "b.mp4")
    manifest_path = _write_manifest(
        tmp_path / "hard_fail_manifest.json",
        [
            {
                "clip_name": "clip_good",
                "variant_name": "v1",
                "prompt": "robot closes drawer carefully",
                "output_video_path": str(clip_a),
            },
            {
                "clip_name": "clip_bad",
                "variant_name": "v2",
                "prompt": "",
                "output_video_path": str(clip_b),
            },
        ],
    )
    leakage_index_path = tmp_path / "quality" / "dataset_fingerprint_index.json"
    quality = DatasetQualityConfig(
        fail_on_rejections=True,
        max_reject_fraction=1.0,
        distribution=DatasetDistributionConfig(enabled=False),
    )
    with pytest.raises(RuntimeError, match="quality_rejections_present"):
        build_dreamdojo_dataset(
            enriched_manifest_path=manifest_path,
            output_dir=tmp_path / "hard_fail_out",
            facility_name="facility_a",
            quality_config=quality,
            leakage_index_path=leakage_index_path,
            dataset_tag="facility_a:s3",
        )
    assert not leakage_index_path.exists()


def test_dataset_builder_migrates_legacy_leakage_index_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_video_gate_checks_without_fingerprint(monkeypatch)
    legacy_video = _write_video(tmp_path / "legacy" / "clip.mp4", seed=31)
    new_video = _write_video(tmp_path / "new" / "clip.mp4", seed=41)
    leakage_index_path = tmp_path / "quality" / "dataset_fingerprint_index.json"
    leakage_index_path.parent.mkdir(parents=True, exist_ok=True)
    leakage_index_path.write_text(
        json.dumps(
            {
                "num_fingerprints": 1,
                "fingerprints": {
                    "legacy_fp_001": {
                        "dataset_tag": "legacy:s3",
                        "facility": "legacy",
                        "clip_name": "legacy_clip",
                        "variant_name": "v0",
                        "prompt_norm": "legacy prompt",
                        "video_path": str(legacy_video),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = _write_manifest(
        tmp_path / "migrate_manifest.json",
        [
            {
                "clip_name": "new_clip",
                "variant_name": "v1",
                "prompt": "robot opens cabinet door",
                "output_video_path": str(new_video),
            }
        ],
    )
    quality = DatasetQualityConfig(
        fail_on_rejections=False,
        max_reject_fraction=1.0,
        enable_leakage_detection=False,
        distribution=DatasetDistributionConfig(enabled=False),
    )
    build_dreamdojo_dataset(
        enriched_manifest_path=manifest_path,
        output_dir=tmp_path / "migrate_out",
        facility_name="facility_a",
        quality_config=quality,
        leakage_index_path=leakage_index_path,
        dataset_tag="facility_a:s3",
    )
    migrated = read_json(leakage_index_path)
    assert migrated["schema_version"] == 2
    assert migrated["fingerprint_version"] == 2
    assert migrated["num_unmigrated_legacy"] == 0
    assert migrated["unmigrated_legacy"] == []
    legacy_v2_fp = fingerprint_video_content(legacy_video)
    assert legacy_v2_fp in migrated["fingerprints"]
    assert migrated["fingerprints"][legacy_v2_fp]["fingerprint_version"] == 2
    assert migrated["fingerprints"][legacy_v2_fp]["legacy_fingerprint"] == "legacy_fp_001"


def test_dataset_builder_keeps_missing_legacy_entries_in_unmigrated_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_video_gate_checks_without_fingerprint(monkeypatch)
    new_video = _write_video(tmp_path / "new" / "clip.mp4", seed=77)
    leakage_index_path = tmp_path / "quality" / "dataset_fingerprint_index.json"
    leakage_index_path.parent.mkdir(parents=True, exist_ok=True)
    leakage_index_path.write_text(
        json.dumps(
            {
                "num_fingerprints": 1,
                "fingerprints": {
                    "legacy_fp_missing": {
                        "dataset_tag": "legacy:s3",
                        "video_path": str(tmp_path / "missing" / "gone.mp4"),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = _write_manifest(
        tmp_path / "unmigrated_manifest.json",
        [
            {
                "clip_name": "new_clip",
                "variant_name": "v1",
                "prompt": "robot opens cabinet door",
                "output_video_path": str(new_video),
            }
        ],
    )
    quality = DatasetQualityConfig(
        fail_on_rejections=False,
        max_reject_fraction=1.0,
        enable_leakage_detection=False,
        distribution=DatasetDistributionConfig(enabled=False),
    )
    build_dreamdojo_dataset(
        enriched_manifest_path=manifest_path,
        output_dir=tmp_path / "unmigrated_out",
        facility_name="facility_a",
        quality_config=quality,
        leakage_index_path=leakage_index_path,
        dataset_tag="facility_a:s3",
    )
    migrated = read_json(leakage_index_path)
    assert migrated["schema_version"] == 2
    assert migrated["num_unmigrated_legacy"] == 1
    assert migrated["unmigrated_legacy"][0]["legacy_fingerprint"] == "legacy_fp_missing"
    assert migrated["unmigrated_legacy"][0]["reason"] == "video_path_missing"
