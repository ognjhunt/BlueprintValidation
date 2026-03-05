"""Pure-logic Stage-2 tests that do not depend on OpenCV."""

from __future__ import annotations


def test_s2_enrich_missing_source_manifest_sets_error_code(sample_config, tmp_path):
    from blueprint_validation.stages.s2_enrich import EnrichStage

    facility = list(sample_config.facilities.values())[0]
    result = EnrichStage().run(sample_config, facility, tmp_path, previous_results={})
    assert result.status == "failed"
    assert result.outputs.get("error_code") == "s2_source_manifest_missing"
    assert result.metrics.get("error_code") == "s2_source_manifest_missing"


def test_s2_resolve_multi_view_context_indices_anchor_fallback():
    from blueprint_validation.stages.s2_enrich import _resolve_multi_view_context_indices

    indices = _resolve_multi_view_context_indices(
        anchor_index=7,
        total_frames=20,
        offsets=[],
    )
    assert indices == [7]


def test_s2_select_source_clips_explicit_fail_closed_metadata(sample_config):
    from blueprint_validation.stages.s2_enrich import _select_source_clips

    sample_config.enrich.source_clip_selection_mode = "explicit"
    sample_config.enrich.source_clip_name = "missing_clip"
    facility = list(sample_config.facilities.values())[0]

    selected, meta = _select_source_clips(
        render_manifest={"clips": [{"clip_name": "clip_000_orbit"}]},
        config=sample_config,
        facility=facility,
    )
    assert selected == []
    assert meta["fail_closed"] is True
    assert meta["fallback"] == "explicit_clip_not_found"
