from __future__ import annotations

import pytest

pytest.importorskip("click")
from click.testing import CliRunner


def test_root_help_does_not_load_config_when_default_path_is_missing(monkeypatch, tmp_path) -> None:
    import blueprint_validation.cli as cli_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "load_config",
        lambda _path: (_ for _ in ()).throw(AssertionError("load_config should not run for help")),
    )

    result = CliRunner().invoke(cli_module.cli, ["--help"])

    assert result.exit_code == 0
    assert "Consume built site-world packages" in result.output
    assert result.output.index("session") < result.output.index("run-all")
    assert "Legacy: build a direct scene package" in result.output
    assert "Legacy: run the compatibility pipeline" in result.output


def test_session_help_does_not_load_config_when_default_path_is_missing(monkeypatch, tmp_path) -> None:
    import blueprint_validation.cli as cli_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "load_config",
        lambda _path: (_ for _ in ()).throw(AssertionError("load_config should not run for help")),
    )

    result = CliRunner().invoke(cli_module.cli, ["session", "--help"])

    assert result.exit_code == 0
    assert "built site-world registrations" in result.output


def test_session_create_help_does_not_load_config_when_default_path_is_missing(monkeypatch, tmp_path) -> None:
    import blueprint_validation.cli as cli_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "load_config",
        lambda _path: (_ for _ in ()).throw(AssertionError("load_config should not run for help")),
    )

    result = CliRunner().invoke(cli_module.cli, ["session", "create", "--help"])

    assert result.exit_code == 0
    assert "--site-world-registration" in result.output
