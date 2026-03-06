"""CLI entry point for blueprint-validate."""

from __future__ import annotations

import os
import shlex
import hashlib
import subprocess
import sys
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
    """Load repo-local env defaults for non-interactive runs (without overriding process env)."""
    cwd = Path.cwd()
    candidates = (
        cwd / "scripts" / "runtime_env.local",
        cwd / ".env.local",
        cwd / ".env",
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
    """BlueprintValidation: Gaussian splat to robot world model validation pipeline."""
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
        click.echo(f"Unknown facility '{facility_id}'. Available: {available}", err=True)
        sys.exit(1)
    return config.facilities[facility_id]


def _get_stage_work_dir(ctx: click.Context, facility_id: str) -> Path:
    work_dir = ctx.obj["work_dir"] / facility_id
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


@cli.command()
@click.option("--facility", required=True, help="Facility ID to render.")
@click.option("--ply-path", type=click.Path(exists=True), default=None, help="Override PLY path.")
@click.pass_context
def render(ctx: click.Context, facility: str, ply_path: Optional[str]) -> None:
    """Stage 1: Render PLY to video clips via gsplat."""
    from .stages.s1_render import RenderStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    if ply_path:
        fac.ply_path = Path(ply_path)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = RenderStage()
    result = stage.execute(config, fac, work_dir, {})
    result.save(work_dir / "s1_render_result.json")
    click.echo(f"Render complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("geometry-canary")
@click.option("--facility", required=True, help="Facility ID to evaluate.")
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
@click.option("--facility", required=True, help="Facility ID to audit.")
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
@click.option("--facility", required=True, help="Facility ID.")
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
@click.option("--facility", required=True, help="Facility ID.")
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
@click.option("--facility", required=True, help="Facility ID.")
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
@click.option("--facility", required=True, help="Facility ID.")
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


@cli.command("simulate-interaction")
@click.option("--facility", required=True, help="Facility ID.")
@click.pass_context
def simulate_interaction(ctx: click.Context, facility: str) -> None:
    """Stage 1e: Optional PyBullet interaction-validated clip generation."""
    from .stages.s1e_splatsim_interaction import SplatSimInteractionStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = SplatSimInteractionStage()
    result = stage.execute(config, fac, work_dir, {})
    result.save(work_dir / "s1e_splatsim_interaction_result.json")
    click.echo(f"SplatSim interaction complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("ingest-external-interaction")
@click.option("--facility", required=True, help="Facility ID.")
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


@cli.command()
@click.option("--facility", required=True, help="Facility ID.")
@click.pass_context
def enrich(ctx: click.Context, facility: str) -> None:
    """Stage 2: Cosmos Transfer 2.5 enrichment."""
    from .stages.s2_enrich import EnrichStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = EnrichStage()
    result = stage.execute(config, fac, work_dir, {})
    result.save(work_dir / "s2_enrich_result.json")
    click.echo(f"Enrich complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command()
@click.option("--facility", required=True, help="Facility ID.")
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
@click.option("--facility", required=True, help="Facility ID.")
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
@click.option("--facility", required=True, help="Facility ID.")
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
@click.option("--facility", required=True, help="Facility ID.")
@click.pass_context
def eval_policy(ctx: click.Context, facility: str) -> None:
    """Stage 4: OpenVLA-OFT policy evaluation with VLM judge scoring."""
    from .stages.s4_policy_eval import PolicyEvalStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = PolicyEvalStage()
    result = stage.execute(config, fac, work_dir, {})
    result.save(work_dir / "s4_policy_eval_result.json")
    click.echo(f"Policy eval complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("export-rlds")
@click.option("--facility", required=True, help="Facility ID.")
@click.pass_context
def export_rlds(ctx: click.Context, facility: str) -> None:
    """Stage 4a: Export adapted rollouts to RLDS TFRecord dataset."""
    from .stages.s4a_rlds_export import RLDSExportStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = RLDSExportStage()
    result = stage.execute(config, fac, work_dir, {})
    result.save(work_dir / "s4a_rlds_export_result.json")
    click.echo(f"RLDS export complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("export-rollouts")
@click.option("--facility", required=True, help="Facility ID.")
@click.pass_context
def export_rollouts(ctx: click.Context, facility: str) -> None:
    """Stage 4b: Export rollouts to RLDS-style datasets."""
    from .stages.s4b_rollout_dataset import RolloutDatasetStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = RolloutDatasetStage()
    result = stage.execute(config, fac, work_dir, {})
    result.save(work_dir / "s4b_rollout_dataset_result.json")
    click.echo(f"Rollout export complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("train-policy-pair")
@click.option("--facility", required=True, help="Facility ID.")
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
@click.option("--facility", required=True, help="Facility ID.")
@click.pass_context
def eval_policy_pair(ctx: click.Context, facility: str) -> None:
    """Stage 4d: Evaluate policy_base vs policy_site on heldout rollouts."""
    from .stages.s4d_policy_pair_eval import PolicyPairEvalStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = PolicyPairEvalStage()
    result = stage.execute(config, fac, work_dir, {})
    result.save(work_dir / "s4d_policy_pair_eval_result.json")
    click.echo(f"Policy pair eval complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("eval-trained-policy")
@click.option("--facility", required=True, help="Facility ID.")
@click.pass_context
def eval_trained_policy(ctx: click.Context, facility: str) -> None:
    """Stage 4e: Evaluate Stage 3b fine-tuned policy in adapted world model."""
    from .stages.s4e_trained_eval import TrainedPolicyEvalStage

    config = ctx.obj["config"]
    fac = _get_facility(ctx, facility)
    work_dir = _get_stage_work_dir(ctx, facility)

    stage = TrainedPolicyEvalStage()
    result = stage.execute(config, fac, work_dir, {})
    result.save(work_dir / "s4e_trained_eval_result.json")
    click.echo(f"Trained policy eval complete: {result.status} ({result.elapsed_seconds:.1f}s)")


@cli.command("eval-visual")
@click.option("--facility", required=True, help="Facility ID.")
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
@click.option("--facility", required=True, help="Facility ID.")
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
@click.option("--facility", default=None, help="Facility ID (omit to warmup all facilities).")
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
@click.option("--facility", default=None, help="Facility ID (omit to bootstrap all facilities).")
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
    """Run the full validation pipeline, all stages sequentially."""
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
    """Show current pipeline status for all facilities."""
    from .common import read_json

    config = ctx.obj["config"]
    work_dir = ctx.obj["work_dir"]

    stages = [
        "s0_task_hints_bootstrap",
        "s1_render",
        "s1b_robot_composite",
        "s1c_gemini_polish",
        "s1d_gaussian_augment",
        "s1e_splatsim_interaction",
        "s1f_external_interaction_ingest",
        "s2_enrich",
        "s3_finetune",
        "s3b_policy_finetune",
        "s3c_policy_rl_loop",
        "s4_policy_eval",
        "s4a_rlds_export",
        "s4e_trained_eval",
        "s4b_rollout_dataset",
        "s4c_policy_pair_train",
        "s4d_policy_pair_eval",
        "s5_visual_fidelity",
        "s6_spatial_accuracy",
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

    # Cross-site
    cs_file = work_dir / "s7_cross_site_result.json"
    if cs_file.exists():
        data = read_json(cs_file)
        click.echo(f"\nCross-site: {data['status']} ({data['elapsed_seconds']:.1f}s)")
    else:
        click.echo("\nCross-site: not started")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
