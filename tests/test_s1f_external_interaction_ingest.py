"""Tests for Stage 1f external interaction ingest."""

from __future__ import annotations

import json
from pathlib import Path


def _write_stage1_source_manifest(path: Path, video_path: Path) -> None:
    from blueprint_validation.common import write_json

    write_json(
        {
            "num_clips": 1,
            "clips": [
                {
                    "clip_name": "ext_clip_000",
                    "video_path": str(video_path),
                }
            ],
        },
        path,
    )


def test_s1f_external_ingest_success(sample_config, tmp_path):
    from blueprint_validation.stages.s1f_external_interaction_ingest import (
        ExternalInteractionIngestStage,
    )

    fac = sample_config.facilities["test_facility"]
    video = tmp_path / "external.mp4"
    video.write_bytes(b"video")
    src_manifest = tmp_path / "source_manifest.json"
    _write_stage1_source_manifest(src_manifest, video)

    sample_config.external_interaction.enabled = True
    sample_config.external_interaction.manifest_path = src_manifest
    sample_config.external_interaction.source_name = "polaris"

    result = ExternalInteractionIngestStage().run(sample_config, fac, tmp_path, {})
    assert result.status == "success"
    out_path = Path(result.outputs["manifest_path"])
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["source_name"] == "polaris"
    assert payload["num_source_clips"] == 1


def test_s1f_external_ingest_missing_manifest_fails(sample_config, tmp_path):
    from blueprint_validation.stages.s1f_external_interaction_ingest import (
        ExternalInteractionIngestStage,
    )

    fac = sample_config.facilities["test_facility"]
    sample_config.external_interaction.enabled = True
    sample_config.external_interaction.manifest_path = tmp_path / "missing.json"

    result = ExternalInteractionIngestStage().run(sample_config, fac, tmp_path, {})
    assert result.status == "failed"
    assert "not found" in result.detail.lower()


def test_s1f_external_ingest_invalid_schema_fails(sample_config, tmp_path):
    from blueprint_validation.common import write_json
    from blueprint_validation.stages.s1f_external_interaction_ingest import (
        ExternalInteractionIngestStage,
    )

    fac = sample_config.facilities["test_facility"]
    invalid_manifest = tmp_path / "invalid.json"
    write_json({"clips": [{"video_path": ""}]}, invalid_manifest)

    sample_config.external_interaction.enabled = True
    sample_config.external_interaction.manifest_path = invalid_manifest

    result = ExternalInteractionIngestStage().run(sample_config, fac, tmp_path, {})
    assert result.status == "failed"
    assert "invalid external interaction manifest" in result.detail.lower()


def test_s1f_external_ingest_enforces_video_path_exists(sample_config, tmp_path):
    from blueprint_validation.common import write_json
    from blueprint_validation.stages.s1f_external_interaction_ingest import (
        ExternalInteractionIngestStage,
    )

    fac = sample_config.facilities["test_facility"]
    missing_video = tmp_path / "does_not_exist.mp4"
    manifest = tmp_path / "source_manifest.json"
    write_json(
        {
            "clips": [
                {
                    "clip_name": "ext_clip_000",
                    "video_path": str(missing_video),
                }
            ]
        },
        manifest,
    )

    sample_config.external_interaction.enabled = True
    sample_config.external_interaction.manifest_path = manifest

    result = ExternalInteractionIngestStage().run(sample_config, fac, tmp_path, {})
    assert result.status == "failed"
    assert "does not exist" in result.detail.lower()


def test_s1f_external_ingest_merges_splatsim_manifest_with_deduped_names(sample_config, tmp_path):
    from blueprint_validation.common import StageResult, write_json
    from blueprint_validation.stages.s1f_external_interaction_ingest import (
        ExternalInteractionIngestStage,
    )

    fac = sample_config.facilities["test_facility"]
    ext_video = tmp_path / "ext.mp4"
    splat_video = tmp_path / "splat.mp4"
    ext_video.write_bytes(b"video")
    splat_video.write_bytes(b"video")

    ext_manifest = tmp_path / "external_manifest.json"
    write_json(
        {"clips": [{"clip_name": "shared_clip", "video_path": str(ext_video)}]},
        ext_manifest,
    )
    splat_manifest = tmp_path / "splatsim_manifest.json"
    write_json(
        {"clips": [{"clip_name": "shared_clip", "video_path": str(splat_video)}]},
        splat_manifest,
    )

    sample_config.external_interaction.enabled = True
    sample_config.external_interaction.manifest_path = ext_manifest
    sample_config.external_interaction.source_name = "polaris"

    previous_results = {
        "s1e_splatsim_interaction": StageResult(
            stage_name="s1e_splatsim_interaction",
            status="success",
            elapsed_seconds=0,
            outputs={"manifest_path": str(splat_manifest)},
        )
    }
    result = ExternalInteractionIngestStage().run(sample_config, fac, tmp_path, previous_results)
    assert result.status == "success"
    assert result.outputs["merged_with_splatsim"] is True
    assert result.metrics["num_external_source_clips"] == 1
    assert result.metrics["num_splatsim_source_clips"] == 1
    assert result.metrics["num_clip_names_renamed"] == 1
    out_path = Path(result.outputs["manifest_path"])
    payload = json.loads(out_path.read_text())
    assert payload["num_source_clips"] == 2
    clip_names = [c["clip_name"] for c in payload["clips"]]
    assert clip_names == ["shared_clip", "shared_clip__splatsim_01"]
    source_stages = [c["source_stage"] for c in payload["clips"]]
    assert source_stages == ["s1f_external_interaction_ingest", "s1e_splatsim_interaction"]


def test_s1f_external_ingest_fails_when_splatsim_manifest_missing(sample_config, tmp_path):
    from blueprint_validation.common import StageResult, write_json
    from blueprint_validation.stages.s1f_external_interaction_ingest import (
        ExternalInteractionIngestStage,
    )

    fac = sample_config.facilities["test_facility"]
    ext_video = tmp_path / "ext.mp4"
    ext_video.write_bytes(b"video")
    ext_manifest = tmp_path / "external_manifest.json"
    write_json(
        {"clips": [{"clip_name": "ext_clip_000", "video_path": str(ext_video)}]},
        ext_manifest,
    )

    sample_config.external_interaction.enabled = True
    sample_config.external_interaction.manifest_path = ext_manifest

    previous_results = {
        "s1e_splatsim_interaction": StageResult(
            stage_name="s1e_splatsim_interaction",
            status="success",
            elapsed_seconds=0,
            outputs={"manifest_path": str(tmp_path / "missing_splatsim_manifest.json")},
        )
    }
    result = ExternalInteractionIngestStage().run(sample_config, fac, tmp_path, previous_results)
    assert result.status == "failed"
    assert "missing" in result.detail.lower()
