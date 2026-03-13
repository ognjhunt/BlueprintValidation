from __future__ import annotations

from click.testing import CliRunner

from blueprint_validation.cli import cli


def test_cli_only_exposes_kept_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for name in ("session", "preflight", "report"):
        assert name in result.output
    for removed in ("run-all", "build_scene_package", "eval-policy", "warmup", "teleop"):
        assert removed not in result.output
