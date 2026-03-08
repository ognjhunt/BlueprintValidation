"""Tests for CLI reproducibility guardrails."""

from __future__ import annotations

import pytest


def test_cli_requires_pinned_commit_when_guardrail_enabled(monkeypatch, tmp_path):
    pytest.importorskip("click")
    import blueprint_validation.cli as cli_module
    from click.testing import CliRunner

    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: v1\n")
    monkeypatch.setattr(cli_module, "load_config", lambda _path: object())

    result = CliRunner().invoke(
        cli_module.cli,
        ["--config", str(config_path), "preflight"],
        env={"BLUEPRINT_REQUIRE_PINNED_CHECKOUT": "true"},
    )
    assert result.exit_code != 0
    assert "BLUEPRINT_PINNED_GIT_COMMIT is unset" in result.output


def test_cli_fails_when_pinned_commit_mismatches(monkeypatch, tmp_path):
    pytest.importorskip("click")
    import blueprint_validation.cli as cli_module
    from click.testing import CliRunner

    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: v1\n")
    monkeypatch.setattr(cli_module, "load_config", lambda _path: object())
    monkeypatch.setattr(cli_module, "_resolve_git_commit", lambda _cwd: "abc123456789")

    result = CliRunner().invoke(
        cli_module.cli,
        ["--config", str(config_path), "preflight"],
        env={
            "BLUEPRINT_REQUIRE_PINNED_CHECKOUT": "true",
            "BLUEPRINT_PINNED_GIT_COMMIT": "deadbeef",
        },
    )
    assert result.exit_code != 0
    assert "Pinned commit mismatch" in result.output


def test_load_local_env_defaults_uses_repo_root(monkeypatch):
    import blueprint_validation.cli as cli_module

    loaded = []
    monkeypatch.setattr(cli_module, "_load_local_env_file", lambda path: loaded.append(path))

    cli_module._load_local_env_defaults()

    repo_root = cli_module.Path(cli_module.__file__).resolve().parents[2]
    assert loaded == [
        repo_root / "scripts" / "runtime_env.local",
        repo_root / ".env.local",
        repo_root / ".env",
    ]
