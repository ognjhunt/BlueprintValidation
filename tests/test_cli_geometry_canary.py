"""Tests for CLI geometry-canary command wiring."""

from __future__ import annotations

import pytest

pytest.importorskip("click")
from click.testing import CliRunner


def test_geometry_canary_command_invokes_stage(monkeypatch, tmp_path):
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
        def run_geometry_canary(self, **kwargs):
            captured.update(kwargs)
            return {
                "num_rows": 6,
                "num_target_grounded_rows": 6,
                "first6_target_grounded_rows": 6,
                "first6_target_missing_count": 2,
                "rows_path": str(tmp_path / "rows.jsonl"),
                "summary_path": str(tmp_path / "summary.json"),
            }

    monkeypatch.setattr(s1_render_module, "RenderStage", FakeRenderStage)

    result = CliRunner().invoke(
        cli_module.cli,
        [
            "--config",
            str(config_path),
            "--work-dir",
            str(tmp_path / "outputs"),
            "geometry-canary",
            "--facility",
            "kitchen_0787",
            "--max-clips",
            "7",
            "--probe-frames",
            "10",
            "--all-clips",
        ],
    )

    assert result.exit_code == 0
    assert int(captured["max_specs"]) == 7
    assert int(captured["probe_frames_override"]) == 10
    assert bool(captured["targeted_only"]) is False
    assert "Geometry canary complete" in result.output
