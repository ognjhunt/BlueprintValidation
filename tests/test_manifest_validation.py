"""Tests for strict manifest schema validation utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_validation.validation import ManifestValidationError, load_and_validate_manifest
from blueprint_validation.validation.manifest_validation import validate_manifest_schema


def test_validate_enriched_manifest_requires_required_fields(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"x")

    ok = {
        "clips": [
            {
                "clip_name": "clip_000",
                "output_video_path": str(clip_path),
                "input_video_path": str(clip_path),
            }
        ]
    }
    validated = validate_manifest_schema(
        ok,
        manifest_type="enriched",
        require_existing_paths=True,
    )
    assert len(validated["clips"]) == 1

    bad = {"clips": [{"clip_name": "clip_001", "output_video_path": ""}]}
    with pytest.raises(ManifestValidationError, match="missing non-empty 'output_video_path'"):
        validate_manifest_schema(
            bad,
            manifest_type="enriched",
            require_existing_paths=False,
        )


def test_validate_stage1_source_manifest_checks_video_path_when_enabled(tmp_path: Path) -> None:
    payload = {
        "clips": [
            {
                "clip_name": "clip_000",
                "video_path": str(tmp_path / "missing.mp4"),
            }
        ]
    }

    with pytest.raises(ManifestValidationError, match="path 'video_path' does not exist"):
        validate_manifest_schema(
            payload,
            manifest_type="stage1_source",
            require_existing_paths=True,
        )

    validated = validate_manifest_schema(
        payload,
        manifest_type="stage1_source",
        require_existing_paths=False,
    )
    assert validated["clips"][0]["clip_name"] == "clip_000"


def test_load_and_validate_policy_scores_manifest(tmp_path: Path) -> None:
    score_path = tmp_path / "score_input.mp4"
    score_path.write_bytes(b"x")
    manifest_path = tmp_path / "scores.json"
    manifest_path.write_text(
        json.dumps(
            {
                "scores": [
                    {
                        "video_path": str(score_path),
                        "task_score": 7.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    validated = load_and_validate_manifest(
        manifest_path,
        manifest_type="policy_scores",
        require_existing_paths=True,
    )
    assert len(validated["scores"]) == 1
