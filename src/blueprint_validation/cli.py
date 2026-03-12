"""CLI entry point for blueprint-validate."""

from __future__ import annotations

import os
import shlex
import hashlib
import subprocess
import sys
import json
from pathlib import Path
from typing import Optional

import click

from .common import get_logger, setup_logging
from .config import load_config

logger = get_logger("cli")


def _load_local_env_file(path: Path) -> None:
    """Load simple KEY=VALUE or export KEY=VALUE entries if env is currently unset."""
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw in lines:
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
    """Load env defaults from this repository root only (never from arbitrary CWD)."""
    repo_root = Path(__file__).resolve().parents[2]
    candidates = (
        repo_root / "scripts" / "runtime_env.local",
        repo_root / ".env.local",
        repo_root / ".env",
    )
    for candidate in candidates:
        _load_local_env_file(candidate)


def _env_flag(name: str) -> bool:
    return (os.environ.get(name, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_git_commit(cwd: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return ""


def _stable_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _stage1_code_hash(cwd: Path) -> str:
    files = [
        cwd / "src/blueprint_validation/stages/s1_isaac_render.py",
        cwd / "src/blueprint_validation/stages/s1_render.py",
        cwd / "src/blueprint_validation/rendering/stage1_active_perception.py",
        cwd / "src/blueprint_validation/rendering/camera_paths.py",
    ]
    h = hashlib.sha256()
    for file_path in files:
        h.update(file_path.read_bytes())
    return h.hexdigest()


def _enforce_repro_guardrails(config_path: Path) -> None:
    if not _env_flag("BLUEPRINT_REQUIRE_PINNED_CHECKOUT"):
        return

    cwd = Path.cwd()
    expected_commit = (os.environ.get("BLUEPRINT_PINNED_GIT_COMMIT", "") or "").strip()
    if not expected_commit:
        raise click.ClickException(
            "Pinned checkout required but BLUEPRINT_PINNED_GIT_COMMIT is unset."
        )
    actual_commit = _resolve_git_commit(cwd)
    if not actual_commit:
        raise click.ClickException(
            "Pinned checkout required but git commit could not be resolved "
            "(run from a real git checkout, not an unversioned copy)."
        )
    if not actual_commit.startswith(expected_commit):
        raise click.ClickException(
            f"Pinned commit mismatch: expected={expected_commit} actual={actual_commit}."
        )

    expected_config_hash = (os.environ.get("BLUEPRINT_EXPECT_CONFIG_HASH", "") or "").strip()
    if expected_config_hash:
        actual_config_hash = _stable_sha256(config_path.resolve())
        if actual_config_hash != expected_config_hash:
            raise click.ClickException(
                "Config hash mismatch: "
                f"expected={expected_config_hash} actual={actual_config_hash}."
            )

    expected_stage1_hash = (os.environ.get("BLUEPRINT_EXPECT_STAGE1_CODE_HASH", "") or "").strip()
    if expected_stage1_hash:
        actual_stage1_hash = _stage1_code_hash(cwd)
        if actual_stage1_hash != expected_stage1_hash:
            raise click.ClickException(
                "Stage-1 code hash mismatch: "
                f"expected={expected_stage1_hash} actual={actual_stage1_hash}."
            )


@click.group()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default="validation.yaml",
    help="Path to validation YAML config.",
)
@click.option(
    "--work-dir",
    type=click.Path(),
    default="./data/outputs",
    help="Working directory for pipeline outputs.",
)
@click.option("--verbose/--quiet", default=True, help="Logging verbosity.")
@click.option("--dry-run", is_flag=True, default=False, help="Print actions without executing.")
@click.pass_context
def cli(ctx: click.Context, config_path: str, work_dir: str, verbose: bool, dry_run: bool) -> None:
    """BlueprintValidation: post-qualification evaluation and adaptation pipeline."""
    setup_logging(verbose)
    _load_local_env_defaults()
    _enforce_repro_guardrails(Path(config_path))
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(Path(config_path))
    ctx.obj["work_dir"] = Path(work_dir)
    ctx.obj["verbose"] = verbose
    ctx.obj["dry_run"] = dry_run
    os.environ["BLUEPRINT_DRY_RUN"] = "1" if dry_run else "0"


@cli.command()
@click.option(
    "--profile",
    type=click.Choice(["audit", "runtime-local", "runtime-cloud"]),
    default="runtime-local",
    show_default=True,
    help="Preflight profile to run.",
)
@click.option(
    "--audit-mode",
    is_flag=True,
    default=False,
    help="Deprecated alias for --profile audit.",
)
@click.pass_context
def preflight(ctx: click.Context, profile: str, audit_mode: bool) -> None:
    """Run preflight checks for the selected execution profile."""
    from .preflight import normalize_preflight_profile, run_preflight

    config = ctx.obj["config"]
    if audit_mode:
        click.echo("Warning: --audit-mode is deprecated; using --profile audit.")
        profile = "audit"
    checks = run_preflight(
        config,
        work_dir=ctx.obj["work_dir"],
        profile=normalize_preflight_profile(profile),
    )
    failed = [c for c in checks if not c.passed]
    if failed:
        click.echo(f"\n{len(failed)} preflight check(s) failed:", err=True)
        for c in failed:
            click.echo(f"  FAIL: {c.name} — {c.detail}", err=True)
        sys.exit(1)
    click.echo(f"All {len(checks)} preflight checks passed.")


def _get_facility(ctx: click.Context, facility_id: str):
    config = ctx.obj["config"]
    if facility_id not in config.facilities:
        available = ", ".join(config.facilities.keys()) or "(none)"
        click.echo(
            f"Unknown evaluation target '{facility_id}'. Available: {available}",
            err=True,
        )
        sys.exit(1)
    return config.facilities[facility_id]


def _get_stage_work_dir(ctx: click.Context, facility_id: str) -> Path:
    work_dir = ctx.obj["work_dir"] / facility_id
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def _bootstrap_scene_memory_runtime_stage(
    config,
    facility,
    work_dir: Path,
    previous_results: dict,
) -> None:
    from .stages.s0b_scene_memory_runtime import SceneMemoryRuntimeStage

    runtime_stage = SceneMemoryRuntimeStage()
    runtime_result = runtime_stage.execute(config, facility, work_dir, previous_results)
    runtime_result.save(work_dir / "s0b_scene_memory_runtime_result.json")
    previous_results["s0b_scene_memory_runtime"] = runtime_result


def _bootstrap_scene_package_stage(
    config,
    facility,
    work_dir: Path,
    previous_results: dict,
) -> None:
    from .stages.s0a_scene_package import ScenePackageStage

    scene_stage = ScenePackageStage()
    scene_result = scene_stage.execute(config, facility, work_dir, previous_results)
    scene_result.save(work_dir / "s0a_scene_package_result.json")
    previous_results["s0a_scene_package"] = scene_result


def _parse_key_value_pairs(values: tuple[str, ...]) -> dict[str, str]:
    payload: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise click.ClickException(f"Expected KEY=VALUE format, got '{item}'.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise click.ClickException(f"Expected KEY=VALUE format, got '{item}'.")
        payload[key] = value
    return payload


def _target_option(*, required: bool = True, help_text: str | None = None):
    default_help = (
        "Qualified opportunity / evaluation target ID."
        if required
        else "Qualified opportunity / evaluation target ID (omit to target all configured entries)."
    )
    return click.option(
        "--opportunity",
        "--facility",
        "facility",
        required=required,
        default=None,
        help=(help_text or default_help) + " Legacy alias retained: --facility.",
    )


@cli.command()
@click.pass_context
def build_scene_package(ctx: click.Context) -> None:
    """Build a direct scene package from a raw PLY and local USD assets."""
    from .scene_builder import SceneAssetManifestError, build_scene_package as _build_scene_package

    config = ctx.obj["config"]
    try:
        result = _build_scene_package(config)
    except (SceneAssetManifestError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "scene_root": str(result.scene_root),
                "scene_manifest_path": str(result.scene_manifest_path),
                "usd_scene_path": str(result.usd_scene_path),
                "visual_usd_scene_path": str(result.visual_usd_scene_path),
                "physics_usd_scene_path": str(result.physics_usd_scene_path),
                "replacement_manifest_path": str(result.replacement_manifest_path),
                "support_surfaces_path": str(result.support_surfaces_path),
                "physics_qc_path": str(result.physics_qc_path),
                "task_config_path": str(result.task_config_path),
                "isaac_lab_package_root": str(result.isaac_lab_package_root),
            },
            indent=2,
        )
    )


@cli.command()
@click.option("--scene-root", type=click.Path(exists=True, file_okay=False), required=True)
def validate_scene_package(scene_root: str) -> None:
    """Validate a local scene handoff directory for teleop use."""
    from .teleop import TeleopManifestError, load_and_validate_scene_package

    try:
        payload = load_and_validate_scene_package(Path(scene_root))
    except TeleopManifestError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(payload, indent=2))


@cli.command("build-teleop-manifests")
@click.option("--scene-id", required=True, help="Scene ID.")
@click.option("--task-id", required=True, help="Task identifier.")
@click.option("--task-text", required=True, help="Task description.")
@click.option("--demo-index", type=int, default=0, show_default=True)
@click.option("--robot-type", default="franka", show_default=True)
@click.option("--robot-asset-ref", required=True, help="Robot asset reference.")
@click.option("--teleop-device", required=True, help="Teleop device name.")
@click.option("--sim-backend", default="isaac_sim", show_default=True)
@click.option("--action-space", default="ee_delta_pose_gripper", show_default=True)
@click.option("--action-dim", type=int, default=7, show_default=True)
@click.option("--lerobot-root", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--episode-ref", required=True, help="Episode reference within the teleop dataset.")
@click.option("--action-sequence-path", type=click.Path(exists=True), required=True)
@click.option("--state-sequence-path", type=click.Path(exists=True), required=True)
@click.option("--joint-name", multiple=True, help="Repeat for each joint name.")
@click.option("--state-key", multiple=True, help="Repeat for each state key.")
@click.option("--video", "videos", multiple=True, help="Camera video mapping CAMERA_ID=/abs/path.mp4")
@click.option(
    "--calibration",
    "calibrations",
    multiple=True,
    help="Camera calibration mapping CAMERA_ID=/abs/path.json",
)
@click.option("--output-dir", type=click.Path(file_okay=False), required=True)
def build_teleop_manifests(
    scene_id: str,
    task_id: str,
    task_text: str,
    demo_index: int,
    robot_type: str,
    robot_asset_ref: str,
    teleop_device: str,
    sim_backend: str,
    action_space: str,
    action_dim: int,
    lerobot_root: str,
    episode_ref: str,
    action_sequence_path: str,
    state_sequence_path: str,
    joint_name: tuple[str, ...],
    state_key: tuple[str, ...],
    videos: tuple[str, ...],
    calibrations: tuple[str, ...],
    output_dir: str,
) -> None:
    """Build teleop manifests from recorded video/action/state files."""
    from .teleop import TeleopManifestError, write_teleop_manifests

    video_map = _parse_key_value_pairs(videos)
    calib_map = _parse_key_value_pairs(calibrations)
    missing = sorted(set(video_map) - set(calib_map))
    if missing:
        raise click.ClickException(
            f"Missing calibration mapping for camera ids: {', '.join(missing)}."
        )

    start_state_hash = hashlib.sha256(
        json.dumps(
            {
                "scene_id": scene_id,
                "task_id": task_id,
                "demo_index": demo_index,
                "robot_type": robot_type,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    session = {
        "session_id": f"{scene_id}::{task_id}::{demo_index:03d}",
        "scene_id": scene_id,
        "task_id": task_id,
        "task_text": task_text,
        "demo_index": int(demo_index),
        "success": True,
        "sim_backend": sim_backend,
        "teleop_device": teleop_device,
        "robot_type": robot_type,
        "robot_asset_ref": robot_asset_ref,
        "action_space": action_space,
        "action_dim": int(action_dim),
        "joint_names": list(joint_name),
        "state_keys": list(state_key),
        "camera_ids": list(video_map.keys()),
        "video_paths": video_map,
        "calibration_refs": calib_map,
        "lerobot_root": str(Path(lerobot_root).resolve()),
        "episode_ref": episode_ref,
        "start_state_hash": start_state_hash,
        "action_sequence_path": str(Path(action_sequence_path).resolve()),
        "state_sequence_path": str(Path(state_sequence_path).resolve()),
    }
    try:
        outputs = write_teleop_manifests(
            output_dir=Path(output_dir),
            source_name="teleop",
            sessions=[session],
        )
    except TeleopManifestError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2))


@cli.command()
@click.option("--listen-host", default="0.0.0.0", show_default=True)
@click.option("--listen-port", type=int, default=49111, show_default=True)
@click.option("--target-host", default="127.0.0.1", show_default=True)
@click.option("--target-port", type=int, default=49110, show_default=True)
@click.option("--action-dim", type=int, default=7, show_default=True)
@click.option("--connect-timeout-s", type=float, default=120.0, show_default=True)
@click.option("--idle-timeout-s", type=float, default=10.0, show_default=True)
@click.option("--target-retry-seconds", type=float, default=0.5, show_default=True)
@click.option("--translation-scale", type=float, default=1.0, show_default=True)
@click.option("--rotation-scale", type=float, default=1.0, show_default=True)
@click.option("--gripper-open-value", type=float, default=1.0, show_default=True)
@click.option("--gripper-close-value", type=float, default=-1.0, show_default=True)
@click.option("--packet-log-path", type=click.Path(), default=None)
def run_vision_pro_relay(
    listen_host: str,
    listen_port: int,
    target_host: str,
    target_port: int,
    action_dim: int,
    connect_timeout_s: float,
    idle_timeout_s: float,
    target_retry_seconds: float,
    translation_scale: float,
    rotation_scale: float,
    gripper_open_value: float,
    gripper_close_value: float,
    packet_log_path: Optional[str],
) -> None:
    """Run the Vision Pro JSON bridge relay on the remote GPU box."""
    from .teleop import (
        VisionProRelayConfig,
        VisionProRelayError,
        run_vision_pro_relay as _run_vision_pro_relay,
    )

    cfg = VisionProRelayConfig(
        listen_host=listen_host,
        listen_port=int(listen_port),
        target_host=target_host,
        target_port=int(target_port),
        action_dim=int(action_dim),
        connect_timeout_s=float(connect_timeout_s),
        idle_timeout_s=float(idle_timeout_s),
        target_retry_seconds=float(target_retry_seconds),
        packet_log_path=Path(packet_log_path) if packet_log_path else None,
        translation_scale=float(translation_scale),
        rotation_scale=float(rotation_scale),
        gripper_open_value=float(gripper_open_value),
        gripper_close_value=float(gripper_close_value),
    )
    try:
        _run_vision_pro_relay(cfg)
    except VisionProRelayError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.option("--scene-root", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--task-id", required=True, help="Task identifier.")
@click.option("--task-text", required=True, help="Task description.")
@click.option("--output-dir", type=click.Path(file_okay=False), required=True)
@click.option("--demo-index", type=int, default=0, show_default=True)
@click.option("--robot-type", default="franka", show_default=True)
@click.option("--robot-asset-ref", default="robot/franka/franka.usd", show_default=True)
@click.option("--teleop-device", default="keyboard", show_default=True)
@click.option("--sim-backend", default="isaac_sim", show_default=True)
@click.option("--action-space", default="ee_delta_pose_gripper", show_default=True)
@click.option("--action-dim", type=int, default=7, show_default=True)
@click.option("--max-steps", type=int, default=200, show_default=True)
@click.option("--headless/--windowed", default=False, show_default=True)
@click.option(
    "--success-flag",
    type=click.Choice(["auto", "success", "failure"]),
    default="auto",
    show_default=True,
    help="Use auto to prompt after each attempt; success/failure forces the outcome.",
)
@click.option("--task-package", default=None, help="Isaac Lab task package import path.")
@click.option("--env-cfg-class", default="TeleopEnvCfg", show_default=True)
@click.option("--camera-key", multiple=True, help="Observation key(s) to treat as RGB cameras.")
@click.option("--state-key", multiple=True, help="Observation key(s) to persist as state.")
@click.option("--scripted-command", multiple=True, help="Optional scripted keyboard commands for non-interactive runs.")
@click.option("--translation-step-m", type=float, default=0.02, show_default=True)
@click.option("--rotation-step-rad", type=float, default=0.12, show_default=True)
@click.option("--gripper-step", type=float, default=1.0, show_default=True)
@click.option("--spacemouse-deadzone", type=float, default=0.08, show_default=True)
@click.option("--spacemouse-translation-scale", type=float, default=0.03, show_default=True)
@click.option("--spacemouse-rotation-scale", type=float, default=0.18, show_default=True)
@click.option("--bridge-host", default="0.0.0.0", show_default=True)
@click.option("--bridge-port", type=int, default=49110, show_default=True)
@click.option("--bridge-connect-timeout-s", type=float, default=120.0, show_default=True)
@click.option("--bridge-idle-timeout-s", type=float, default=10.0, show_default=True)
@click.option("--bridge-packet-log-enabled/--no-bridge-packet-log-enabled", default=True, show_default=True)
@click.option("--confirm-success/--no-confirm-success", default=True, show_default=True)
@click.option("--max-attempts", type=int, default=1, show_default=True)
@click.option("--attempt-pause-seconds", type=float, default=0.5, show_default=True)
def record_teleop(
    scene_root: str,
    task_id: str,
    task_text: str,
    output_dir: str,
    demo_index: int,
    robot_type: str,
    robot_asset_ref: str,
    teleop_device: str,
    sim_backend: str,
    action_space: str,
    action_dim: int,
    max_steps: int,
    headless: bool,
    success_flag: str,
    task_package: Optional[str],
    env_cfg_class: str,
    camera_key: tuple[str, ...],
    state_key: tuple[str, ...],
    scripted_command: tuple[str, ...],
    translation_step_m: float,
    rotation_step_rad: float,
    gripper_step: float,
    spacemouse_deadzone: float,
    spacemouse_translation_scale: float,
    spacemouse_rotation_scale: float,
    bridge_host: str,
    bridge_port: int,
    bridge_connect_timeout_s: float,
    bridge_idle_timeout_s: float,
    bridge_packet_log_enabled: bool,
    confirm_success: bool,
    max_attempts: int,
    attempt_pause_seconds: float,
) -> None:
    """Record one local Isaac teleop demo and emit teleop manifests."""
    from .teleop import IsaacTeleopRuntimeError, TeleopRecorderConfig, record_teleop_session

    cfg = TeleopRecorderConfig(
        scene_root=Path(scene_root),
        output_dir=Path(output_dir),
        task_id=task_id,
        task_text=task_text,
        demo_index=int(demo_index),
        robot_type=robot_type,
        robot_asset_ref=robot_asset_ref,
        teleop_device=teleop_device,
        sim_backend=sim_backend,
        action_space=action_space,
        action_dim=int(action_dim),
        max_steps=int(max_steps),
        headless=bool(headless),
        success=(
            None
            if success_flag == "auto"
            else True if success_flag == "success" else False
        ),
        task_package=task_package,
        env_cfg_class=env_cfg_class,
        camera_keys=list(camera_key),
        state_keys=list(state_key),
        scripted_commands=list(scripted_command),
        translation_step_m=float(translation_step_m),
        rotation_step_rad=float(rotation_step_rad),
        gripper_step=float(gripper_step),
        spacemouse_deadzone=float(spacemouse_deadzone),
        spacemouse_translation_scale=float(spacemouse_translation_scale),
        spacemouse_rotation_scale=float(spacemouse_rotation_scale),
        bridge_host=str(bridge_host),
        bridge_port=int(bridge_port),
        bridge_connect_timeout_s=float(bridge_connect_timeout_s),
        bridge_idle_timeout_s=float(bridge_idle_timeout_s),
        bridge_packet_log_enabled=bool(bridge_packet_log_enabled),
        confirm_success=bool(confirm_success),
        max_attempts=int(max_attempts),
        attempt_pause_seconds=float(attempt_pause_seconds),
    )
    try:
        outputs = record_teleop_session(cfg)
    except IsaacTeleopRuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2))


@cli.command()
@_target_option(help_text="Qualified opportunity / evaluation target ID to render.")
@click.option(
    "--ply-path",
    type=click.Path(exists=True),
    default=None,
    help="Legacy direct-geometry override for the resolved evaluation target PLY.",
)
@click.pass_context
def render(ctx: click.Context, facility: str, ply_path: Optional[str]) -> None:
    """Stage 1: Render clips using the resolved gsplat or Isaac backend."""
    from .stages.render_backend import active_render_backend
    from .stages.s1_isaac_render import IsaacRenderStage
    from .stages.s1_render import RenderStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    if ply_path:
        fac.ply_path = Path(ply_path)
    work_dir = _get_stage_work_dir(ctx, facility)

    previous_results = {}
    _bootstrap_scene_memory_runtime_stage(config, fac, work_dir, previous_results)
    _bootstrap_scene_package_stage(config, fac, work_dir, previous_results)

    backend = active_render_backend(config, fac, previous_results)
    stage = IsaacRenderStage() if backend == "isaac_scene" else RenderStage()
    result = stage.execute(config, fac, work_dir, previous_results)
    result.save(work_dir / f"{stage.name}_result.json")
    click.echo(
        f"Render complete via {backend}: {result.status} ({result.elapsed_seconds:.1f}s)"
    )


@cli.command("geometry-canary")
@_target_option(help_text="Qualified opportunity / evaluation target ID to evaluate.")
@click.option(
    "--max-clips",
    type=int,
    default=12,
    show_default=True,
    help="Maximum camera specs to evaluate in this canary run.",
)
@click.option(
    "--probe-frames",
    type=int,
    default=0,
    show_default=True,
    help="Override probe frame count (0 uses config/budget).",
)
@click.option(
    "--targeted-only/--all-clips",
    default=True,
    show_default=True,
    help="Restrict checks to target-grounded clips only.",
)
@click.pass_context
def geometry_canary(
    ctx: click.Context,
    facility: str,
    max_clips: int,
    probe_frames: int,
    targeted_only: bool,
) -> None:
    """Stage 1: Run a no-render geometry canary for target-presence checks."""
    from .stages.s1_render import RenderStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = RenderStage()
    summary = stage.run_geometry_canary(
        config=config,
        facility=fac,
        work_dir=work_dir,
        max_specs=max(1, int(max_clips)),
        probe_frames_override=max(0, int(probe_frames)),
        targeted_only=bool(targeted_only),
    )

    click.echo(
        "Geometry canary complete: "
        f"rows={int(summary.get('num_rows', 0))} "
        f"target_rows={int(summary.get('num_target_grounded_rows', 0))} "
        f"first6={int(summary.get('first6_target_grounded_rows', 0))} "
        f"first6_target_missing={int(summary.get('first6_target_missing_count', 0))}"
    )
    click.echo(f"Rows JSONL: {summary.get('rows_path')}")
    click.echo(f"Summary JSON: {summary.get('summary_path')}")


@cli.command("post-s1-audit")
@_target_option(help_text="Qualified opportunity / evaluation target ID to audit.")
@click.option(
    "--geometry-max-clips",
    type=int,
    default=12,
    show_default=True,
    help="Max target-grounded clips to evaluate in geometry canary.",
)
@click.option(
    "--geometry-probe-frames",
    type=int,
    default=0,
    show_default=True,
    help="Override geometry canary probe frame count (0 uses config/budget).",
)
@click.option(
    "--vlm-rescore-first",
    type=int,
    default=6,
    show_default=True,
    help="Rescore first N validated clips with VLM (0 disables VLM rescoring).",
)
@click.pass_context
def post_s1_audit(
    ctx: click.Context,
    facility: str,
    geometry_max_clips: int,
    geometry_probe_frames: int,
    vlm_rescore_first: int,
) -> None:
    """Run consolidated CPU-only Stage-1 post-run reliability audit."""
    from .stages.s1_render import RenderStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = RenderStage()
    summary = stage.run_post_s1_audit(
        config=config,
        facility=fac,
        work_dir=work_dir,
        geometry_max_specs=max(1, int(geometry_max_clips)),
        geometry_probe_frames_override=max(0, int(geometry_probe_frames)),
        vlm_rescore_first=max(0, int(vlm_rescore_first)),
    )

    click.echo(
        "Post-S1 audit complete: "
        f"clips={int(summary.get('num_clips_in_manifest', 0))} "
        f"missing={int(summary.get('num_videos_missing', 0))} "
        f"invalid={int(summary.get('num_videos_invalid', 0))} "
        f"monochrome={int(summary.get('num_monochrome_warnings', 0))} "
        f"quality_failed={int(summary.get('num_quality_gate_failed', 0))} "
        f"vlm_scored={int(summary.get('vlm_rows_scored', 0))}/"
        f"{int(summary.get('vlm_rows_total', 0))}"
    )
    click.echo(f"Summary JSON: {summary.get('summary_path')}")
    click.echo(f"Clip rows JSONL: {summary.get('clip_rows_path')}")
    click.echo(f"VLM rows JSONL: {summary.get('vlm_rows_path')}")


@cli.command("compose-robot")
@_target_option()
@click.pass_context
def compose_robot(ctx: click.Context, facility: str) -> None:
    """Stage 1b: Composite URDF robot arm into rendered clips."""
    from .stages.s1b_robot_composite import RobotCompositeStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = RobotCompositeStage()
    result = stage.execute(config, fac, work_dir, {})
    result.save(work_dir / "s1b_robot_composite_result.json")
    click.echo(f"Robot composite complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("polish-gemini")
@_target_option()
@click.pass_context
def polish_gemini(ctx: click.Context, facility: str) -> None:
    """Stage 1c: Optional Gemini image polish for composited clips."""
    from .stages.s1c_gemini_polish import GeminiPolishStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = GeminiPolishStage()
    result = stage.execute(config, fac, work_dir, {})
    result.save(work_dir / "s1c_gemini_polish_result.json")
    click.echo(f"Gemini polish complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("augment-gaussian")
@_target_option()
@click.pass_context
def augment_gaussian(ctx: click.Context, facility: str) -> None:
    """Stage 1d: Full RoboSplat-default augmentation."""
    from .stages.s1d_gaussian_augment import GaussianAugmentStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = GaussianAugmentStage()
    result = stage.execute(config, fac, work_dir, {})
    result.save(work_dir / "s1d_gaussian_augment_result.json")
    click.echo(f"Gaussian augment complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("augment-robosplat")
@_target_option()
@click.pass_context
def augment_robosplat(ctx: click.Context, facility: str) -> None:
    """Alias for Stage 1d full RoboSplat augmentation."""
    from .stages.s1d_gaussian_augment import GaussianAugmentStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = GaussianAugmentStage()
    result = stage.execute(config, fac, work_dir, {})
    result.save(work_dir / "s1d_gaussian_augment_result.json")
    click.echo(f"RoboSplat augment complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("ingest-external-interaction")
@_target_option()
@click.pass_context
def ingest_external_interaction(ctx: click.Context, facility: str) -> None:
    """Stage 1f: Ingest external interaction manifest into stage-1 source format."""
    from .stages.s1f_external_interaction_ingest import ExternalInteractionIngestStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = ExternalInteractionIngestStage()
    result = stage.execute(config, fac, work_dir, {})
    result.save(work_dir / "s1f_external_interaction_ingest_result.json")
    click.echo(
        f"External interaction ingest complete: {result.status} ({result.elapsed_seconds:.1f}s)"
    )


@cli.command("ingest-external-rollouts")
@_target_option()
@click.pass_context
def ingest_external_rollouts(ctx: click.Context, facility: str) -> None:
    """Stage 1g: Ingest external teleop rollouts into action-labeled rows."""
    from .stages.s1g_external_rollout_ingest import ExternalRolloutIngestStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = ExternalRolloutIngestStage()
    result = stage.execute(config, fac, work_dir, {})
    result.save(work_dir / "s1g_external_rollout_ingest_result.json")
    click.echo(
        f"External rollout ingest complete: {result.status} ({result.elapsed_seconds:.1f}s)"
    )


@cli.command()
@_target_option()
@click.pass_context
def enrich(ctx: click.Context, facility: str) -> None:
    """Stage 2: Cosmos Transfer 2.5 enrichment."""
    from .stages.s2_enrich import EnrichStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    previous_results = {}
    _bootstrap_scene_memory_runtime_stage(config, fac, work_dir, previous_results)
    stage = EnrichStage()
    result = stage.execute(config, fac, work_dir, previous_results)
    result.save(work_dir / "s2_enrich_result.json")
    click.echo(f"Enrich complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command()
@_target_option()
@click.pass_context
def finetune(ctx: click.Context, facility: str) -> None:
    """Stage 3: DreamDojo fine-tuning."""
    from .stages.s3_finetune import FinetuneStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = FinetuneStage()
    result = stage.execute(config, fac, work_dir, {})
    result.save(work_dir / "s3_finetune_result.json")
    click.echo(f"Finetune complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("finetune-policy")
@_target_option()
@click.pass_context
def finetune_policy(ctx: click.Context, facility: str) -> None:
    """Optional Stage 3b: OpenVLA-OFT policy fine-tuning."""
    from .stages.s3b_policy_finetune import PolicyFinetuneStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = PolicyFinetuneStage()
    result = stage.execute(config, fac, work_dir, {})
    result.save(work_dir / "s3b_policy_finetune_result.json")
    click.echo(f"Policy finetune complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("rl-loop-policy")
@_target_option()
@click.pass_context
def rl_loop_policy(ctx: click.Context, facility: str) -> None:
    """Stage 3c: World-VLA-Loop-style policy RL loop."""
    from .stages.s3c_policy_rl_loop import PolicyRLLoopStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = PolicyRLLoopStage()
    result = stage.execute(config, fac, work_dir, {})
    result.save(work_dir / "s3c_policy_rl_loop_result.json")
    click.echo(f"Policy RL loop complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("eval-policy")
@_target_option()
@click.pass_context
def eval_policy(ctx: click.Context, facility: str) -> None:
    """Stage 4: OpenVLA-OFT policy evaluation with VLM judge scoring."""
    from .stages.s4_policy_eval import PolicyEvalStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    previous_results = {}
    _bootstrap_scene_memory_runtime_stage(config, fac, work_dir, previous_results)
    stage = PolicyEvalStage()
    result = stage.execute(config, fac, work_dir, previous_results)
    result.save(work_dir / "s4_policy_eval_result.json")
    click.echo(f"Policy eval complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("export-rlds")
@_target_option()
@click.pass_context
def export_rlds(ctx: click.Context, facility: str) -> None:
    """Stage 4a: Export adapted rollouts to RLDS TFRecord dataset."""
    from .stages.s4a_rlds_export import RLDSExportStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    previous_results = {}
    _bootstrap_scene_memory_runtime_stage(config, fac, work_dir, previous_results)
    stage = RLDSExportStage()
    result = stage.execute(config, fac, work_dir, previous_results)
    result.save(work_dir / "s4a_rlds_export_result.json")
    click.echo(f"RLDS export complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("export-rollouts")
@_target_option()
@click.pass_context
def export_rollouts(ctx: click.Context, facility: str) -> None:
    """Stage 4b: Export rollouts to RLDS-style datasets."""
    from .stages.s4b_rollout_dataset import RolloutDatasetStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    previous_results = {}
    _bootstrap_scene_memory_runtime_stage(config, fac, work_dir, previous_results)
    stage = RolloutDatasetStage()
    result = stage.execute(config, fac, work_dir, previous_results)
    result.save(work_dir / "s4b_rollout_dataset_result.json")
    click.echo(f"Rollout export complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("train-policy-pair")
@_target_option()
@click.pass_context
def train_policy_pair(ctx: click.Context, facility: str) -> None:
    """Stage 4c: Train policy_base and policy_site from paired datasets."""
    from .stages.s4c_policy_pair_train import PolicyPairTrainStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = PolicyPairTrainStage()
    result = stage.execute(config, fac, work_dir, {})
    result.save(work_dir / "s4c_policy_pair_train_result.json")
    click.echo(f"Policy pair train complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("eval-policy-pair")
@_target_option()
@click.pass_context
def eval_policy_pair(ctx: click.Context, facility: str) -> None:
    """Stage 4d: Evaluate policy_base vs policy_site on heldout rollouts."""
    from .stages.s4d_policy_pair_eval import PolicyPairEvalStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    previous_results = {}
    _bootstrap_scene_memory_runtime_stage(config, fac, work_dir, previous_results)
    stage = PolicyPairEvalStage()
    result = stage.execute(config, fac, work_dir, previous_results)
    result.save(work_dir / "s4d_policy_pair_eval_result.json")
    click.echo(f"Policy pair eval complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("eval-trained-policy")
@_target_option()
@click.pass_context
def eval_trained_policy(ctx: click.Context, facility: str) -> None:
    """Stage 4e: Evaluate Stage 3b fine-tuned policy in adapted world model."""
    from .stages.s4e_trained_eval import TrainedPolicyEvalStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    previous_results = {}
    _bootstrap_scene_memory_runtime_stage(config, fac, work_dir, previous_results)
    stage = TrainedPolicyEvalStage()
    result = stage.execute(config, fac, work_dir, previous_results)
    result.save(work_dir / "s4e_trained_eval_result.json")
    click.echo(f"Trained policy eval complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("eval-visual")
@_target_option()
@click.pass_context
def eval_visual(ctx: click.Context, facility: str) -> None:
    """Stage 5: Visual fidelity metrics (PSNR/SSIM/LPIPS)."""
    from .stages.s5_visual_fidelity import VisualFidelityStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = VisualFidelityStage()
    result = stage.execute(config, fac, work_dir, {})
    result.save(work_dir / "s5_visual_fidelity_result.json")
    click.echo(f"Visual fidelity complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("eval-spatial")
@_target_option()
@click.pass_context
def eval_spatial(ctx: click.Context, facility: str) -> None:
    """Stage 6: Spatial accuracy verification."""
    from .stages.s6_spatial_accuracy import SpatialAccuracyStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = SpatialAccuracyStage()
    result = stage.execute(config, fac, work_dir, {})
    result.save(work_dir / "s6_spatial_accuracy_result.json")
    click.echo(f"Spatial accuracy complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("eval-crosssite")
@click.pass_context
def eval_crosssite(ctx: click.Context) -> None:
    """Stage 7: Cross-site discrimination test (requires 2 facilities)."""
    from .stages.s7_cross_site import CrossSiteStage

    config = ctx.obj["config"]
    if len(config.facilities) < 2:
        click.echo("Cross-site test requires at least 2 facilities in config.", err=True)
        sys.exit(1)

    work_dir = ctx.obj["work_dir"]
    work_dir.mkdir(parents=True, exist_ok=True)

    stage = CrossSiteStage()
    # Cross-site uses all facilities, pass first as primary
    fac_ids = list(config.facilities.keys())
    fac = config.facilities[fac_ids[0]]
    result = stage.execute(config, fac, work_dir, {})
    result.save(work_dir / "s7_cross_site_result.json")
    click.echo(f"Cross-site complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command()
@_target_option(required=False)
@click.pass_context
def warmup(ctx: click.Context, facility: str | None) -> None:
    """Pre-compute CPU-only artifacts (occupancy grids, camera paths, variant prompts).

    Run this before a GPU session to save 5-15 min per facility at render time.
    """
    from .warmup import warmup_facility

    config = ctx.obj["config"]
    work_dir = ctx.obj["work_dir"]

    if facility:
        fac = _get_facility(ctx, facility)
        fac_work_dir = work_dir / facility
        fac_work_dir.mkdir(parents=True, exist_ok=True)
        summary = warmup_facility(config, fac, fac_work_dir)
        click.echo(
            f"Warmup {facility}: {summary.get('num_clips', 0)} clips, "
            f"{summary.get('elapsed_seconds', 0):.1f}s"
        )
    else:
        for fid, fac in config.facilities.items():
            fac_work_dir = work_dir / fid
            fac_work_dir.mkdir(parents=True, exist_ok=True)
            summary = warmup_facility(config, fac, fac_work_dir)
            click.echo(
                f"Warmup {fid}: {summary.get('num_clips', 0)} clips, "
                f"{summary.get('elapsed_seconds', 0):.1f}s"
            )
        click.echo(f"All {len(config.facilities)} facilities warmed up.")


@cli.command("bootstrap-task-hints")
@_target_option(
    required=False,
    help_text="Qualified opportunity / evaluation target ID (omit to bootstrap all configured entries).",
)
@click.pass_context
def bootstrap_task_hints(ctx: click.Context, facility: str | None) -> None:
    """Stage 0: Bootstrap synthetic task_targets.json when source hints are missing."""
    from .stages.s0_task_hints_bootstrap import TaskHintsBootstrapStage

    config = ctx.obj["config"]
    work_dir = ctx.obj["work_dir"]

    stage = TaskHintsBootstrapStage()
    if facility:
        fac = _get_facility(ctx, facility)
        fac_work_dir = work_dir / facility
        fac_work_dir.mkdir(parents=True, exist_ok=True)
        result = stage.execute(config, fac, fac_work_dir, {})
        result.save(fac_work_dir / "s0_task_hints_bootstrap_result.json")
        click.echo(
            f"Bootstrap {facility}: {result.status} "
            f"(hints={result.outputs.get('task_hints_path', 'none')})"
        )
        if result.status == "failed":
            sys.exit(1)
        return

    any_failed = False
    for fid, fac in config.facilities.items():
        fac_work_dir = work_dir / fid
        fac_work_dir.mkdir(parents=True, exist_ok=True)
        result = stage.execute(config, fac, fac_work_dir, {})
        result.save(fac_work_dir / "s0_task_hints_bootstrap_result.json")
        click.echo(
            f"Bootstrap {fid}: {result.status} "
            f"(hints={result.outputs.get('task_hints_path', 'none')})"
        )
        any_failed = any_failed or result.status == "failed"
    if any_failed:
        sys.exit(1)


@cli.command("run-all")
@click.option(
    "--continue-on-failure",
    is_flag=True,
    default=False,
    help="Continue after stage failures (default is fail-fast).",
)
@click.option(
    "--resume",
    "resume_from_results",
    is_flag=True,
    default=False,
    help="Reuse successful/skipped *_result.json files and continue from incomplete stages.",
)
@click.option(
    "--skip-preflight",
    is_flag=True,
    default=False,
    help="Skip the mandatory preflight gate before pipeline execution.",
)
@click.pass_context
def run_all(
    ctx: click.Context,
    continue_on_failure: bool,
    resume_from_results: bool,
    skip_preflight: bool,
) -> None:
    """Run the supported site-world preparation pipeline."""
    from .pipeline import ValidationPipeline
    from .preflight import run_preflight

    config = ctx.obj["config"]
    work_dir = ctx.obj["work_dir"]

    if not skip_preflight:
        checks = run_preflight(config, work_dir=work_dir, profile="runtime_local")
        failed = [c for c in checks if not c.passed]
        if failed:
            click.echo(
                "\nrun-all preflight failed. Resolve these issues or rerun with --skip-preflight:",
                err=True,
            )
            for check in failed:
                click.echo(f"  FAIL: {check.name} — {check.detail}", err=True)
            sys.exit(1)

    pipeline = ValidationPipeline(config, work_dir)
    summary = pipeline.run_all(
        fail_fast=not continue_on_failure,
        resume_from_results=resume_from_results,
    )

    click.echo("\n=== Pipeline Summary ===")
    for stage_name, result in summary.items():
        click.echo(f"  {stage_name}: {result.status} ({result.elapsed_seconds:.1f}s)")

    failed = [name for name, result in summary.items() if result.status == "failed"]
    if failed:
        click.echo("\nPipeline failed stages:", err=True)
        for name in failed:
            click.echo(f"  - {name}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--format", "fmt", type=click.Choice(["markdown", "json"]), default="markdown")
@click.option("--output", "output_path", type=click.Path(), default="validation_report.md")
@click.pass_context
def report(ctx: click.Context, fmt: str, output_path: str) -> None:
    """Generate final validation report from existing pipeline outputs."""
    from .reporting.report_builder import build_report

    config = ctx.obj["config"]
    work_dir = ctx.obj["work_dir"]

    report_path = build_report(config, work_dir, fmt=fmt, output_path=Path(output_path))
    click.echo(f"Report written to {report_path}")


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current site-world preparation status for all facilities."""
    from .common import read_json

    config = ctx.obj["config"]
    work_dir = ctx.obj["work_dir"]

    stages = [
        "s0_task_hints_bootstrap",
        "s0a_scene_package",
        "s0b_scene_memory_runtime",
        "s1_isaac_render",
        "s1_render",
        "s1b_robot_composite",
        "s1c_gemini_polish",
        "s1d_gaussian_augment",
        "s1f_external_interaction_ingest",
        "s1g_external_rollout_ingest",
    ]

    for fid in config.facilities:
        click.echo(f"\nFacility: {fid}")
        fdir = work_dir / fid
        for sname in stages:
            result_file = fdir / f"{sname}_result.json"
            if result_file.exists():
                data = read_json(result_file)
                click.echo(f"  {sname}: {data['status']} ({data['elapsed_seconds']:.1f}s)")
            else:
                click.echo(f"  {sname}: not started")

@cli.group("session")
def session_group() -> None:
    """Hosted-session runtime commands for WebApp orchestration."""


@session_group.command("create")
@click.option("--session-id", required=True)
@click.option("--session-work-dir", type=click.Path(file_okay=False), required=True)
@click.option(
    "--site-world-registration",
    "--runtime-manifest",
    "registration_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to site_world_registration.json. Legacy alias retained: --runtime-manifest.",
)
@click.option("--robot-profile-id", required=True)
@click.option("--task-id", required=True)
@click.option("--scenario-id", required=True)
@click.option("--start-state-id", required=True)
@click.option("--policy-json", required=False, default=None, help="Optional JSON payload describing adapter/model/checkpoint.")
@click.option("--canonical-package-uri", default=None)
@click.option("--canonical-package-version", default=None)
@click.option("--prompt", default=None)
@click.option("--trajectory-json", default=None, help="Optional JSON payload describing a presentation trajectory.")
@click.option("--presentation-model", default=None)
@click.option("--debug-mode/--no-debug-mode", default=False, show_default=True)
@click.option(
    "--unsafe-allow-blocked-site-world/--no-unsafe-allow-blocked-site-world",
    default=False,
    show_default=True,
    help="Unsafe local-only bypass for blocked site worlds during smoke/debug sessions.",
)
@click.option("--export-mode", "export_modes", multiple=True)
@click.option("--robot-profile-override-json", default=None)
@click.option("--notes", default="", show_default=True)
@click.pass_context
def session_create(
    ctx: click.Context,
    session_id: str,
    session_work_dir: str,
    registration_path: str,
    robot_profile_id: str,
    task_id: str,
    scenario_id: str,
    start_state_id: str,
    policy_json: Optional[str],
    canonical_package_uri: Optional[str],
    canonical_package_version: Optional[str],
    prompt: Optional[str],
    trajectory_json: Optional[str],
    presentation_model: Optional[str],
    debug_mode: bool,
    unsafe_allow_blocked_site_world: bool,
    export_modes: tuple[str, ...],
    robot_profile_override_json: Optional[str],
    notes: str,
) -> None:
    from .hosted_session import HostedSessionError, create_session

    try:
        policy_payload = json.loads(policy_json) if policy_json else {}
        if canonical_package_uri is not None:
            policy_payload["canonical_package_uri"] = canonical_package_uri
        if canonical_package_version is not None:
            policy_payload["canonical_package_version"] = canonical_package_version
        if prompt is not None:
            policy_payload["prompt"] = prompt
        if trajectory_json is not None:
            policy_payload["trajectory"] = json.loads(trajectory_json)
        if presentation_model is not None:
            policy_payload["presentation_model"] = presentation_model
        if debug_mode:
            policy_payload["debug_mode"] = True
        payload = create_session(
            config=ctx.obj["config"],
            session_id=session_id,
            session_work_dir=Path(session_work_dir),
            registration_path=Path(registration_path),
            robot_profile_id=robot_profile_id,
            task_id=task_id,
            scenario_id=scenario_id,
            start_state_id=start_state_id,
            policy_payload=(policy_payload or None),
            export_modes=export_modes,
            robot_profile_override=(
                json.loads(robot_profile_override_json) if robot_profile_override_json else None
            ),
            notes=notes,
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
@click.option("--seed", type=int, default=None)
@click.pass_context
def session_reset(
    ctx: click.Context,
    session_id: str,
    session_work_dir: str,
    task_id: Optional[str],
    scenario_id: Optional[str],
    start_state_id: Optional[str],
    seed: Optional[int],
) -> None:
    from .hosted_session import HostedSessionError, reset_session

    try:
        payload = reset_session(
            config=ctx.obj["config"],
            session_id=session_id,
            session_work_dir=Path(session_work_dir),
            task_id=task_id,
            scenario_id=scenario_id,
            start_state_id=start_state_id,
            seed=seed,
        )
    except HostedSessionError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(payload))


@session_group.command("step")
@click.option("--session-id", required=True)
@click.option("--session-work-dir", type=click.Path(file_okay=False), required=True)
@click.option("--episode-id", required=True)
@click.option("--action-json", default=None)
@click.option("--auto-policy/--no-auto-policy", default=True, show_default=True)
@click.pass_context
def session_step(
    ctx: click.Context,
    session_id: str,
    session_work_dir: str,
    episode_id: str,
    action_json: Optional[str],
    auto_policy: bool,
) -> None:
    from .hosted_session import HostedSessionError, step_session

    try:
        payload = step_session(
            config=ctx.obj["config"],
            session_work_dir=Path(session_work_dir),
            episode_id=episode_id,
            action=json.loads(action_json) if action_json else None,
            auto_policy=auto_policy,
        )
    except (HostedSessionError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(payload))


@session_group.command("run-batch")
@click.option("--session-id", required=True)
@click.option("--session-work-dir", type=click.Path(file_okay=False), required=True)
@click.option("--num-episodes", type=int, required=True)
@click.option("--task-id", default=None)
@click.option("--scenario-id", default=None)
@click.option("--start-state-id", default=None)
@click.option("--seed", type=int, default=None)
@click.option("--max-steps", type=int, default=None)
@click.pass_context
def session_run_batch(
    ctx: click.Context,
    session_id: str,
    session_work_dir: str,
    num_episodes: int,
    task_id: Optional[str],
    scenario_id: Optional[str],
    start_state_id: Optional[str],
    seed: Optional[int],
    max_steps: Optional[int],
) -> None:
    from .hosted_session import HostedSessionError, run_batch

    try:
        payload = run_batch(
            config=ctx.obj["config"],
            session_work_dir=Path(session_work_dir),
            num_episodes=num_episodes,
            task_id=task_id,
            scenario_id=scenario_id,
            start_state_id=start_state_id,
            seed=seed,
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
