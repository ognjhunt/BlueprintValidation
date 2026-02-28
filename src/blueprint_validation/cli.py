"""CLI entry point for blueprint-validate."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from .common import get_logger, setup_logging
from .config import load_config

logger = get_logger("cli")


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
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(Path(config_path))
    ctx.obj["work_dir"] = Path(work_dir)
    ctx.obj["verbose"] = verbose
    ctx.obj["dry_run"] = dry_run


@cli.command()
@click.pass_context
def preflight(ctx: click.Context) -> None:
    """Run preflight checks for GPU, dependencies, and model weights."""
    from .preflight import run_preflight

    config = ctx.obj["config"]
    checks = run_preflight(config)
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
        return

    for fid, fac in config.facilities.items():
        fac_work_dir = work_dir / fid
        fac_work_dir.mkdir(parents=True, exist_ok=True)
        result = stage.execute(config, fac, fac_work_dir, {})
        result.save(fac_work_dir / "s0_task_hints_bootstrap_result.json")
        click.echo(
            f"Bootstrap {fid}: {result.status} "
            f"(hints={result.outputs.get('task_hints_path', 'none')})"
        )


@cli.command("run-all")
@click.pass_context
def run_all(ctx: click.Context) -> None:
    """Run the full validation pipeline, all stages sequentially."""
    from .pipeline import ValidationPipeline

    config = ctx.obj["config"]
    work_dir = ctx.obj["work_dir"]

    pipeline = ValidationPipeline(config, work_dir)
    summary = pipeline.run_all()

    click.echo("\n=== Pipeline Summary ===")
    for stage_name, result in summary.items():
        click.echo(f"  {stage_name}: {result.status} ({result.elapsed_seconds:.1f}s)")


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
        "s1_render", "s1b_robot_composite", "s1c_gemini_polish", "s1d_gaussian_augment",
        "s1e_splatsim_interaction",
        "s2_enrich", "s3_finetune", "s3b_policy_finetune", "s3c_policy_rl_loop",
        "s4_policy_eval", "s4a_rlds_export",
        "s4e_trained_eval", "s4b_rollout_dataset", "s4c_policy_pair_train",
        "s4d_policy_pair_eval",
        "s5_visual_fidelity", "s6_spatial_accuracy",
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
