"""Tests for finetune.dataset_quality config parsing."""

from __future__ import annotations

import textwrap

from blueprint_validation.config import load_config


def test_load_config_parses_dataset_quality_block(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            schema_version: v1
            project_name: test
            facilities:
              a:
                name: A
                ply_path: {tmp_path / "a.ply"}
            finetune:
              dataset_quality:
                strict_manifest_validation: false
                quarantine_rejections: false
                fail_on_rejections: false
                max_reject_fraction: 0.25
                enable_duplicate_detection: false
                enable_leakage_detection: false
                prompt_lint:
                  enabled: true
                  min_chars: 12
                  min_tokens: 3
                  min_unique_token_ratio: 0.55
                  allow_generic_substrings: true
                temporal_gates:
                  enabled: true
                  min_frames_for_check: 10
                  max_frames_to_sample: 44
                  min_mean_interframe_delta: 2.5
                  max_freeze_ratio: 0.5
                  max_abrupt_cut_ratio: 0.2
                  max_blockiness_score: 0.3
                distribution:
                  enabled: true
                  min_total_clips_for_caps: 8
                  min_unique_variants: 3
                  min_unique_source_clips: 5
                  max_single_variant_fraction: 0.7
                  max_single_source_clip_fraction: 0.4
                  max_prompt_dominance_fraction: 0.45
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    dq = config.finetune.dataset_quality
    assert dq.strict_manifest_validation is False
    assert dq.quarantine_rejections is False
    assert dq.fail_on_rejections is False
    assert dq.max_reject_fraction == 0.25
    assert dq.enable_duplicate_detection is False
    assert dq.enable_leakage_detection is False

    assert dq.prompt_lint.min_chars == 12
    assert dq.prompt_lint.min_tokens == 3
    assert dq.prompt_lint.min_unique_token_ratio == 0.55
    assert dq.prompt_lint.allow_generic_substrings is True

    assert dq.temporal_gates.min_frames_for_check == 10
    assert dq.temporal_gates.max_frames_to_sample == 44
    assert dq.temporal_gates.min_mean_interframe_delta == 2.5
    assert dq.temporal_gates.max_freeze_ratio == 0.5
    assert dq.temporal_gates.max_abrupt_cut_ratio == 0.2
    assert dq.temporal_gates.max_blockiness_score == 0.3

    assert dq.distribution.min_total_clips_for_caps == 8
    assert dq.distribution.min_unique_variants == 3
    assert dq.distribution.min_unique_source_clips == 5
    assert dq.distribution.max_single_variant_fraction == 0.7
    assert dq.distribution.max_single_source_clip_fraction == 0.4
    assert dq.distribution.max_prompt_dominance_fraction == 0.45


def test_load_config_rejects_invalid_dataset_quality_ranges(tmp_path) -> None:
    import pytest

    config_path = tmp_path / "config_invalid_dataset_quality.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            schema_version: v1
            project_name: test
            facilities:
              a:
                name: A
                ply_path: {tmp_path / "a.ply"}
            finetune:
              dataset_quality:
                max_reject_fraction: 1.2
                prompt_lint:
                  min_unique_token_ratio: 1.1
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finetune.dataset_quality.max_reject_fraction"):
        load_config(config_path)
