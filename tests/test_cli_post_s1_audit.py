"""Tests for CLI post-s1-audit command wiring."""

from __future__ import annotations

import pytest

pytest.importorskip("click")
from click.testing import CliRunner


def test_post_s1_audit_command_invokes_stage(monkeypatch, tmp_path):
    import blueprint_validation.cli as cli_module
    import blueprint_validation.stages.s1_render as s1_render_module
    from blueprint_validation.config import CameraPathSpec, FacilityConfig, RenderConfig, ValidationConfig

    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: v1\n")
    ply_path = tmp_path / "scene.ply"
    ply_path.write_bytes(b"ply\n")
    cfg = ValidationConfig(
        project_name="Test",
        facilities={
            "kitchen_0787": FacilityConfig(
                name="Kitchen Scene 0787 (InteriorGS)",
                ply_path=ply_path,
                description="test",
            )
        },
        render=RenderConfig(
            resolution=(120, 160),
            camera_paths=[CameraPathSpec(type="orbit", radius_m=2.0)],
        ),
    )
    monkeypatch.setattr(cli_module, "load_config", lambda _path: cfg)

    captured: dict[str, object] = {}

    class FakeRenderStage:
        def run_post_s1_audit(self, **kwargs):
            captured.update(kwargs)
            return {
                "num_clips_in_manifest": 12,
                "num_videos_missing": 0,
                "num_videos_invalid": 1,
                "num_monochrome_warnings": 2,
                "num_quality_gate_failed": 3,
                "vlm_rows_scored": 6,
                "vlm_rows_total": 6,
                "summary_path": str(tmp_path / "summary.json"),
                "clip_rows_path": str(tmp_path / "clip_rows.jsonl"),
                "vlm_rows_path": str(tmp_path / "vlm_rows.jsonl"),
            }

    monkeypatch.setattr(s1_render_module, "RenderStage", FakeRenderStage)

    result = CliRunner().invoke(
        cli_module.cli,
        [
            "--config",
            str(config_path),
            "--work-dir",
            str(tmp_path / "outputs"),
            "post-s1-audit",
            "--facility",
            "kitchen_0787",
            "--geometry-max-clips",
            "9",
            "--geometry-probe-frames",
            "10",
            "--vlm-rescore-first",
            "4",
        ],
    )

    assert result.exit_code == 0
    assert int(captured["geometry_max_specs"]) == 9
    assert int(captured["geometry_probe_frames_override"]) == 10
    assert int(captured["vlm_rescore_first"]) == 4
    assert "Post-S1 audit complete" in result.output
