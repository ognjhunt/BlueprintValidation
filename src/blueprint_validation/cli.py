"""CLI entry point for the NeoVerse-only validation workflow."""

from __future__ import annotations

import json
import os
import shlex
from dataclasses import replace
from pathlib import Path
from typing import Optional

import click

from .common import setup_logging
from .config import load_config
from .runtime_backend import KNOWN_RUNTIME_BACKEND_KINDS, normalize_runtime_kind


_PREFERRED_COMMAND_ORDER = {
    "session": 0,
    "preflight": 1,
    "report": 2,
}

_SESSION_COMMAND_ORDER = {
    "create": 0,
    "reset": 1,
    "step": 2,
    "run-batch": 3,
    "export": 4,
    "stop": 5,
}


class BlueprintCliGroup(click.Group):
    def list_commands(self, ctx: click.Context) -> list[str]:
        commands = list(super().list_commands(ctx))
        return sorted(commands, key=lambda name: (_PREFERRED_COMMAND_ORDER.get(name, 100), name))


class SessionCliGroup(click.Group):
    def list_commands(self, ctx: click.Context) -> list[str]:
        commands = list(super().list_commands(ctx))
        return sorted(commands, key=lambda name: (_SESSION_COMMAND_ORDER.get(name, 100), name))


def _load_local_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        try:
            parsed = shlex.split(value.strip())
            os.environ[key] = parsed[0] if parsed else ""
        except ValueError:
            os.environ[key] = value.strip().strip("'").strip('"')


def _load_local_env_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (
        repo_root / "scripts" / "runtime_env.local",
        repo_root / ".env.local",
        repo_root / ".env",
    ):
        _load_local_env_file(candidate)


def _load_cli_config(ctx: click.Context):
    if "config" not in ctx.obj:
        config_path = Path(ctx.obj["config_path"]).expanduser().resolve()
        if not config_path.exists():
            raise click.ClickException(f"Config path does not exist: {config_path}")
        config = load_config(config_path)
        override_kind = ctx.obj.get("required_runtime_kind")
        if override_kind:
            config = replace(
                config,
                scene_memory_runtime=replace(
                    config.scene_memory_runtime,
                    required_runtime_kind=normalize_runtime_kind(override_kind),
                ),
            )
        ctx.obj["config"] = config
    return ctx.obj["config"]


@click.group(cls=BlueprintCliGroup)
@click.option("--config", "config_path", type=click.Path(), default="validation.yaml", help="Path to validation YAML config.")
@click.option("--work-dir", type=click.Path(), default="./data/outputs", help="Working directory for reports.")
@click.option(
    "--required-runtime-kind",
    type=click.Choice(sorted(KNOWN_RUNTIME_BACKEND_KINDS)),
    default=None,
    help="Override the runtime kind this command requires from the remote runtime service.",
)
@click.option("--verbose/--quiet", default=True, help="Logging verbosity.")
@click.pass_context
def cli(
    ctx: click.Context,
    config_path: str,
    work_dir: str,
    required_runtime_kind: Optional[str],
    verbose: bool,
) -> None:
    """Consume built site-world packages through NeoVerse runtime sessions and exports."""
    setup_logging(verbose)
    _load_local_env_defaults()
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    ctx.obj["work_dir"] = Path(work_dir)
    ctx.obj["required_runtime_kind"] = required_runtime_kind


@cli.command()
@click.option(
    "--site-world-registration",
    type=click.Path(exists=True),
    default=None,
    help="Optional site_world_registration.json to validate alongside runtime readiness.",
)
@click.pass_context
def preflight(ctx: click.Context, site_world_registration: Optional[str]) -> None:
    """Run NeoVerse runtime readiness checks for built site-world packages."""
    from .preflight import run_preflight

    checks = run_preflight(_load_cli_config(ctx), site_world_registration=site_world_registration)
    failed = [check for check in checks if not check.passed]
    for check in checks:
        label = "PASS" if check.passed else "FAIL"
        click.echo(f"{label}: {check.name} - {check.detail}")
    if failed:
        raise click.ClickException(f"{len(failed)} preflight check(s) failed.")


@cli.command()
@click.option("--format", "fmt", type=click.Choice(["markdown", "json"]), default="markdown")
@click.option("--output", "output_path", type=click.Path(), default="validation_report.md")
@click.pass_context
def report(ctx: click.Context, fmt: str, output_path: str) -> None:
    """Generate a minimal report from NeoVerse session/export artifacts."""
    from .reporting.report_builder import build_report

    report_path = build_report(
        _load_cli_config(ctx),
        Path(ctx.obj["work_dir"]),
        fmt=fmt,
        output_path=Path(output_path),
    )
    click.echo(f"Report written to {report_path}")


@cli.group("session", cls=SessionCliGroup)
def session_group() -> None:
    """Run hosted NeoVerse session workflows from built site-world registrations."""


@session_group.command("create")
@click.option("--session-id", required=True)
@click.option("--session-work-dir", type=click.Path(file_okay=False), required=True)
@click.option("--site-world-registration", type=click.Path(exists=True), required=True)
@click.option("--robot-profile-id", required=True)
@click.option("--task-id", required=True)
@click.option("--scenario-id", required=True)
@click.option("--start-state-id", required=True)
@click.option("--canonical-package-uri", default=None)
@click.option("--canonical-package-version", default=None)
@click.option("--prompt", default=None)
@click.option("--trajectory-json", default=None)
@click.option("--presentation-model", default=None)
@click.option("--debug-mode/--no-debug-mode", default=False, show_default=True)
@click.option("--unsafe-allow-blocked-site-world/--no-unsafe-allow-blocked-site-world", default=False, show_default=True)
@click.option("--notes", default="", show_default=True)
@click.pass_context
def session_create(
    ctx: click.Context,
    session_id: str,
    session_work_dir: str,
    site_world_registration: str,
    robot_profile_id: str,
    task_id: str,
    scenario_id: str,
    start_state_id: str,
    canonical_package_uri: Optional[str],
    canonical_package_version: Optional[str],
    prompt: Optional[str],
    trajectory_json: Optional[str],
    presentation_model: Optional[str],
    debug_mode: bool,
    unsafe_allow_blocked_site_world: bool,
    notes: str,
) -> None:
    from .hosted_session import HostedSessionError, create_session

    try:
        payload = create_session(
            config=_load_cli_config(ctx),
            session_id=session_id,
            session_work_dir=Path(session_work_dir),
            registration_path=Path(site_world_registration),
            robot_profile_id=robot_profile_id,
            task_id=task_id,
            scenario_id=scenario_id,
            start_state_id=start_state_id,
            notes=notes,
            canonical_package_uri=canonical_package_uri,
            canonical_package_version=canonical_package_version,
            prompt=prompt,
            trajectory=json.loads(trajectory_json) if trajectory_json else None,
            presentation_model=presentation_model,
            debug_mode=debug_mode,
            unsafe_allow_blocked_site_world=unsafe_allow_blocked_site_world,
        )
    except (HostedSessionError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(payload))


@session_group.command("reset")
@click.option("--session-id", required=True)
@click.option("--session-work-dir", type=click.Path(file_okay=False), required=True)
@click.option("--task-id", default=None)
@click.option("--scenario-id", default=None)
@click.option("--start-state-id", default=None)
@click.pass_context
def session_reset(
    ctx: click.Context,
    session_id: str,
    session_work_dir: str,
    task_id: Optional[str],
    scenario_id: Optional[str],
    start_state_id: Optional[str],
) -> None:
    from .hosted_session import HostedSessionError, reset_session

    try:
        payload = reset_session(
            config=_load_cli_config(ctx),
            session_id=session_id,
            session_work_dir=Path(session_work_dir),
            task_id=task_id,
            scenario_id=scenario_id,
            start_state_id=start_state_id,
        )
    except HostedSessionError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(payload))


@session_group.command("step")
@click.option("--session-work-dir", type=click.Path(file_okay=False), required=True)
@click.option("--episode-id", required=True)
@click.option("--action-json", required=True)
@click.pass_context
def session_step(ctx: click.Context, session_work_dir: str, episode_id: str, action_json: str) -> None:
    from .hosted_session import HostedSessionError, step_session

    try:
        payload = step_session(
            config=_load_cli_config(ctx),
            session_work_dir=Path(session_work_dir),
            episode_id=episode_id,
            action=json.loads(action_json),
        )
    except (HostedSessionError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(payload))


@session_group.command("run-batch")
@click.option("--session-work-dir", type=click.Path(file_okay=False), required=True)
@click.option("--num-episodes", type=int, required=True)
@click.option("--task-id", default=None)
@click.option("--scenario-id", default=None)
@click.option("--start-state-id", default=None)
@click.option("--max-steps", type=int, default=None)
@click.pass_context
def session_run_batch(
    ctx: click.Context,
    session_work_dir: str,
    num_episodes: int,
    task_id: Optional[str],
    scenario_id: Optional[str],
    start_state_id: Optional[str],
    max_steps: Optional[int],
) -> None:
    from .hosted_session import HostedSessionError, run_batch

    try:
        payload = run_batch(
            config=_load_cli_config(ctx),
            session_work_dir=Path(session_work_dir),
            num_episodes=num_episodes,
            task_id=task_id,
            scenario_id=scenario_id,
            start_state_id=start_state_id,
            max_steps=max_steps,
        )
    except HostedSessionError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(payload))


@session_group.command("stop")
@click.option("--session-id", required=True)
@click.option("--session-work-dir", type=click.Path(file_okay=False), required=True)
def session_stop(session_id: str, session_work_dir: str) -> None:
    from .hosted_session import stop_session

    payload = stop_session(session_work_dir=Path(session_work_dir))
    payload["sessionId"] = session_id
    click.echo(json.dumps(payload))


@session_group.command("export")
@click.option("--session-id", required=True)
@click.option("--session-work-dir", type=click.Path(file_okay=False), required=True)
def session_export(session_id: str, session_work_dir: str) -> None:
    from .hosted_session import export_session

    payload = export_session(session_work_dir=Path(session_work_dir))
    payload["sessionId"] = session_id
    click.echo(json.dumps(payload))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
