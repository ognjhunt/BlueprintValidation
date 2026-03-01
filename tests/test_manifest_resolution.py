"""Tests for stage manifest lineage resolution helpers."""

from __future__ import annotations

from pathlib import Path


def test_resolver_prefers_previous_results_over_filesystem_stale(tmp_path):
    from blueprint_validation.common import StageResult
    from blueprint_validation.stages.manifest_resolution import (
        ManifestCandidate,
        resolve_manifest_source,
    )

    stale_composite = tmp_path / "robot_composite" / "composited_manifest.json"
    render_manifest = tmp_path / "renders" / "render_manifest.json"
    stale_composite.parent.mkdir(parents=True, exist_ok=True)
    render_manifest.parent.mkdir(parents=True, exist_ok=True)
    stale_composite.write_text("{}")
    render_manifest.write_text("{}")

    previous_results = {
        "s1_render": StageResult(
            stage_name="s1_render",
            status="success",
            elapsed_seconds=0,
            outputs={"manifest_path": str(render_manifest)},
        )
    }
    resolved = resolve_manifest_source(
        work_dir=tmp_path,
        previous_results=previous_results,
        candidates=[
            ManifestCandidate("s1b_robot_composite", Path("robot_composite/composited_manifest.json")),
            ManifestCandidate("s1_render", Path("renders/render_manifest.json")),
        ],
    )
    assert resolved is not None
    assert resolved.source_mode == "previous_results"
    assert resolved.source_stage == "s1_render"
    assert resolved.source_manifest_path == render_manifest


def test_resolver_does_not_fallback_when_previous_results_present(tmp_path):
    from blueprint_validation.common import StageResult
    from blueprint_validation.stages.manifest_resolution import (
        ManifestCandidate,
        resolve_manifest_source,
    )

    render_manifest = tmp_path / "renders" / "render_manifest.json"
    render_manifest.parent.mkdir(parents=True, exist_ok=True)
    render_manifest.write_text("{}")

    previous_results = {
        "s1_render": StageResult(
            stage_name="s1_render",
            status="skipped",
            elapsed_seconds=0,
            outputs={"manifest_path": str(render_manifest)},
        )
    }
    resolved = resolve_manifest_source(
        work_dir=tmp_path,
        previous_results=previous_results,
        candidates=[ManifestCandidate("s1_render", Path("renders/render_manifest.json"))],
    )
    assert resolved is None


def test_resolver_filesystem_fallback_when_standalone(tmp_path):
    from blueprint_validation.stages.manifest_resolution import (
        ManifestCandidate,
        resolve_manifest_source,
    )

    render_manifest = tmp_path / "renders" / "render_manifest.json"
    render_manifest.parent.mkdir(parents=True, exist_ok=True)
    render_manifest.write_text("{}")

    resolved = resolve_manifest_source(
        work_dir=tmp_path,
        previous_results={},
        candidates=[ManifestCandidate("s1_render", Path("renders/render_manifest.json"))],
    )
    assert resolved is not None
    assert resolved.source_mode == "filesystem_fallback"
    assert resolved.source_stage == "s1_render"
    assert resolved.source_manifest_path == render_manifest

