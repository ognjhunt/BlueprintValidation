"""Tests for CLI preflight profile handling."""

from __future__ import annotations

import pytest

pytest.importorskip("click")
from click.testing import CliRunner


def test_preflight_audit_mode_maps_to_audit_profile(monkeypatch, tmp_path):
    import blueprint_validation.cli as cli_module
    import blueprint_validation.preflight as preflight_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: v1\n")
    seen = {}

    monkeypatch.setattr(cli_module, "load_config", lambda _path: object())

    def _run_preflight(_cfg, **kwargs):
        seen["profile"] = kwargs.get("profile")
        return []

    monkeypatch.setattr(
        preflight_module,
        "run_preflight",
        _run_preflight,
    )

    result = CliRunner().invoke(
        cli_module.cli,
        ["--config", str(config_path), "preflight", "--audit-mode"],
    )
    assert result.exit_code == 0
    assert seen["profile"] == "audit"
    assert "deprecated; using --profile audit" in result.output


def test_preflight_profile_keeps_failures_fatal(monkeypatch, tmp_path):
    import blueprint_validation.cli as cli_module
    import blueprint_validation.preflight as preflight_module
    from blueprint_validation.common import PreflightCheck

    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: v1\n")

    monkeypatch.setattr(cli_module, "load_config", lambda _path: object())
    monkeypatch.setattr(
        preflight_module,
        "run_preflight",
        lambda _cfg, **_kwargs: [
            PreflightCheck(name="dep:pytorch", passed=False, detail="Cannot import torch"),
        ],
    )

    result = CliRunner().invoke(
        cli_module.cli,
        ["--config", str(config_path), "preflight", "--profile", "audit"],
    )
    assert result.exit_code == 1
    assert "FAIL: dep:pytorch" in result.output


def test_run_all_invokes_preflight_by_default(monkeypatch, tmp_path):
    import blueprint_validation.cli as cli_module
    import blueprint_validation.preflight as preflight_module
    import blueprint_validation.pipeline as pipeline_module
    from blueprint_validation.common import PreflightCheck, StageResult

    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: v1\n")
    calls = {"preflight": 0, "run_all": 0}

    monkeypatch.setattr(cli_module, "load_config", lambda _path: object())
    monkeypatch.setattr(
        preflight_module,
        "run_preflight",
        lambda _cfg, **_kwargs: (
            calls.__setitem__("preflight", calls["preflight"] + 1)
            or [PreflightCheck(name="gpu", passed=True, detail="ok")]
        ),
    )

    class _DummyPipeline:
        def __init__(self, config, work_dir):
            del config, work_dir

        def run_all(self, **_kwargs):
            calls["run_all"] += 1
            return {"facility/s1_render": StageResult("s1_render", "success", 0.0)}

    monkeypatch.setattr(pipeline_module, "ValidationPipeline", _DummyPipeline)

    result = CliRunner().invoke(
        cli_module.cli,
        ["--config", str(config_path), "run-all"],
    )
    assert result.exit_code == 0
    assert calls["preflight"] == 1
    assert calls["run_all"] == 1


def test_run_all_skip_preflight_bypasses_gate(monkeypatch, tmp_path):
    import blueprint_validation.cli as cli_module
    import blueprint_validation.preflight as preflight_module
    import blueprint_validation.pipeline as pipeline_module
    from blueprint_validation.common import StageResult

    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: v1\n")
    calls = {"preflight": 0, "run_all": 0}

    monkeypatch.setattr(cli_module, "load_config", lambda _path: object())
    monkeypatch.setattr(
        preflight_module,
        "run_preflight",
        lambda _cfg, **_kwargs: calls.__setitem__("preflight", calls["preflight"] + 1) or [],
    )

    class _DummyPipeline:
        def __init__(self, config, work_dir):
            del config, work_dir

        def run_all(self, **_kwargs):
            calls["run_all"] += 1
            return {"facility/s1_render": StageResult("s1_render", "success", 0.0)}

    monkeypatch.setattr(pipeline_module, "ValidationPipeline", _DummyPipeline)

    result = CliRunner().invoke(
        cli_module.cli,
        ["--config", str(config_path), "run-all", "--skip-preflight"],
    )
    assert result.exit_code == 0
    assert calls["preflight"] == 0
    assert calls["run_all"] == 1
