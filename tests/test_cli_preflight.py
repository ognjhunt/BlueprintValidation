"""Tests for CLI preflight audit-mode behavior."""

from __future__ import annotations

import pytest

pytest.importorskip("click")
from click.testing import CliRunner


def test_preflight_audit_mode_ignores_gpu_failure(monkeypatch, tmp_path):
    import blueprint_validation.cli as cli_module
    import blueprint_validation.preflight as preflight_module
    from blueprint_validation.common import PreflightCheck

    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: v1\n")

    monkeypatch.setattr(cli_module, "load_config", lambda _path: object())
    monkeypatch.setattr(
        preflight_module,
        "run_preflight",
        lambda _cfg: [
            PreflightCheck(name="gpu", passed=False, detail="No CUDA GPU detected"),
        ],
    )

    result = CliRunner().invoke(
        cli_module.cli,
        ["--config", str(config_path), "preflight", "--audit-mode"],
    )
    assert result.exit_code == 0
    assert "Audit mode: ignoring GPU preflight failure." in result.output


def test_preflight_audit_mode_keeps_non_gpu_failures_fatal(monkeypatch, tmp_path):
    import blueprint_validation.cli as cli_module
    import blueprint_validation.preflight as preflight_module
    from blueprint_validation.common import PreflightCheck

    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: v1\n")

    monkeypatch.setattr(cli_module, "load_config", lambda _path: object())
    monkeypatch.setattr(
        preflight_module,
        "run_preflight",
        lambda _cfg: [
            PreflightCheck(name="gpu", passed=False, detail="No CUDA GPU detected"),
            PreflightCheck(name="dep:pytorch", passed=False, detail="Cannot import torch"),
        ],
    )

    result = CliRunner().invoke(
        cli_module.cli,
        ["--config", str(config_path), "preflight", "--audit-mode"],
    )
    assert result.exit_code == 1
    assert "FAIL: dep:pytorch" in result.output
