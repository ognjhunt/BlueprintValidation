from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from blueprint_validation.cli import _resolve_config_path, cli


def test_cli_only_exposes_kept_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for name in ("runtime", "session", "preflight", "report"):
        assert name in result.output
    for removed in ("run-all", "build_scene_package", "eval-policy", "warmup", "teleop"):
        assert removed not in result.output


def test_resolve_config_path_falls_back_to_example_for_default_name(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_config_path("validation.yaml")

    assert resolved.name == "example_validation.yaml"
    assert resolved.exists()


def test_resolve_config_path_keeps_missing_explicit_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_config_path("custom-validation.yaml")

    assert resolved == (tmp_path / "custom-validation.yaml").resolve()
