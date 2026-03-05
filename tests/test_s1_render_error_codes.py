"""Focused tests for Stage-1 machine-readable error codes."""

from __future__ import annotations


def test_s1_render_sets_error_code_when_strict_task_hints_missing(sample_config, tmp_path):
    from blueprint_validation.stages.s1_render import RenderStage

    sample_config.render.stage1_strict_require_task_hints = True
    facility = list(sample_config.facilities.values())[0]
    facility.task_hints_path = None

    result = RenderStage().run(sample_config, facility, tmp_path, previous_results={})
    assert result.status == "failed"
    assert result.outputs.get("error_code") == "s1_task_hints_required_missing"
    assert result.metrics.get("error_code") == "s1_task_hints_required_missing"
