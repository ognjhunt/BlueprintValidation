"""Build JSON and Markdown validation reports from pipeline outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from ..common import get_logger, read_json, write_json, write_text_atomic
from ..config import ValidationConfig
from .stage_catalog import FACILITY_STAGE_RESULT_NAMES

logger = get_logger("reporting.report_builder")


def build_report(
    config: ValidationConfig,
    work_dir: Path,
    fmt: str = "markdown",
    output_path: Path = Path("validation_report.md"),
) -> Path:
    """Build a validation report from pipeline outputs.

    Args:
        config: Pipeline configuration
        work_dir: Working directory containing pipeline outputs
        fmt: Output format ("markdown" or "json")
        output_path: Where to write the report
    """
    # Collect all results
    report_data = _collect_results(config, work_dir)

    if fmt == "json":
        output_path = output_path.with_suffix(".json")
        write_json(report_data, output_path)
    else:
        output_path = output_path.with_suffix(".md")
        md = _render_markdown(report_data, config)
        write_text_atomic(output_path, md)

    logger.info("Report written to %s", output_path)
    return output_path


def _collect_results(config: ValidationConfig, work_dir: Path) -> Dict[str, Any]:
    """Collect all stage results into a single dict."""
    results: Dict[str, Any] = {
        "project_name": config.project_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "facilities": {},
        "cross_site": None,
        "policy_eval_matrix": None,
        "claim_portfolio": None,
    }

    for fid in config.facilities:
        fac_dir = work_dir / fid
        fac_results = {}
        for stage in FACILITY_STAGE_RESULT_NAMES:
            result_file = fac_dir / f"{stage}_result.json"
            if result_file.exists():
                fac_results[stage] = read_json(result_file)
        results["facilities"][fid] = fac_results

    # Cross-site
    cs_file = work_dir / "s7_cross_site_result.json"
    if cs_file.exists():
        results["cross_site"] = read_json(cs_file)

    # Pipeline summary
    summary_file = work_dir / "pipeline_summary.json"
    if summary_file.exists():
        results["pipeline_summary"] = read_json(summary_file)

    matrix_file = work_dir / "policy_eval" / "matrix_report.json"
    if matrix_file.exists():
        results["policy_eval_matrix"] = read_json(matrix_file)
    claim_portfolio_file = work_dir / "claim_portfolio_report.json"
    if claim_portfolio_file.exists():
        results["claim_portfolio"] = read_json(claim_portfolio_file)

    return results


def _is_fixed_world_claim_protocol(config: ValidationConfig | None) -> bool:
    if config is None or len(config.facilities) != 1:
        return False
    return (
        str(getattr(config.eval_policy, "claim_protocol", "none") or "none").strip().lower()
        == "fixed_same_facility_uplift"
    )


def _append_s4_policy_eval_section(
    lines: list[str], fac_data: Dict[str, Any], *, supporting: bool
) -> None:
    if "s4_policy_eval" not in fac_data:
        return
    pe = fac_data["s4_policy_eval"]
    metrics = pe.get("metrics", {})
    lines.append(
        "### Supporting Evidence: Frozen Policy Baseline vs Adapted World Model (S4)\n"
        if supporting
        else "### Policy Performance (Primary Test)\n"
    )
    if supporting:
        lines.append(
            "*Supporting world-model evidence only. This section does not determine the top-line "
            "single-facility claim result.*\n"
        )
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Baseline mean task score | {metrics.get('baseline_mean_task_score', 'N/A')} |")
    lines.append(f"| Adapted mean task score | {metrics.get('adapted_mean_task_score', 'N/A')} |")
    lines.append(f"| Absolute difference | {metrics.get('absolute_difference', 'N/A')} |")
    lines.append(f"| Improvement | {metrics.get('improvement_pct', 'N/A')}% |")
    lines.append(f"| Win rate | {metrics.get('win_rate', 'N/A')} |")
    lines.append(f"| p-value | {metrics.get('p_value', 'N/A')} |")
    lines.append("")

    pairwise = metrics.get("pairwise", {})
    if pairwise:
        lines.append("#### Pairwise Condition Comparisons\n")
        lines.append(
            "| Comparison | Score A | Score B | Abs Diff | Improvement | Win Rate | p-value |"
        )
        lines.append(
            "|------------|---------|---------|----------|-------------|----------|---------|"
        )
        for pair_key, pair_data in pairwise.items():
            parts = pair_key.split("_vs_")
            if len(parts) == 2:
                c1, c2 = parts
                lines.append(
                    f"| {c1} vs {c2} "
                    f"| {pair_data.get(f'{c1}_mean', 'N/A')} "
                    f"| {pair_data.get(f'{c2}_mean', 'N/A')} "
                    f"| {pair_data.get('absolute_difference', 'N/A')} "
                    f"| {pair_data.get('improvement_pct', 'N/A')}% "
                    f"| {pair_data.get('win_rate', 'N/A')} "
                    f"| {pair_data.get('p_value', 'N/A')} |"
                )
        lines.append("")

    per_condition = metrics.get("per_condition", {})
    has_manip = any(v.get("manipulation_success_rate", 0) > 0 for v in per_condition.values())
    if has_manip:
        lines.append("#### Manipulation Performance\n")
        lines.append("| Condition | Success Rate | Mean Task Score |")
        lines.append("|-----------|-------------|-----------------|")
        for cond, cdata in per_condition.items():
            lines.append(
                f"| {cond} "
                f"| {cdata.get('manipulation_success_rate', 'N/A')} "
                f"| {cdata.get('mean_task_score', 'N/A')} |"
            )
        lines.append("")


def _append_scene_runtime_section(lines: list[str], fac_data: Dict[str, Any]) -> None:
    runtime_stage = fac_data.get("s0b_scene_memory_runtime", {}) or {}
    runtime_outputs = runtime_stage.get("outputs", {}) or {}
    runtime_metrics = runtime_stage.get("metrics", {}) or {}

    intake_lineage = None
    for stage_name in ("s1_isaac_render", "s1_render", "s2_enrich", "s4_policy_eval"):
        payload = fac_data.get(stage_name, {}) or {}
        outputs = payload.get("outputs", {}) or {}
        metrics = payload.get("metrics", {}) or {}
        if isinstance(outputs.get("intake_lineage"), dict):
            intake_lineage = outputs.get("intake_lineage")
            break
        if isinstance(metrics.get("intake_lineage"), dict):
            intake_lineage = metrics.get("intake_lineage")
            break

    if not runtime_stage and not intake_lineage:
        return

    lines.append("### Intake And Runtime\n")
    if isinstance(intake_lineage, dict):
        lines.append(
            f"- Preferred intake: {intake_lineage.get('preferred_intake_kind', 'N/A')}"
        )
        lines.append(f"- Intake mode: {intake_lineage.get('intake_mode', 'N/A')}")
    if runtime_stage:
        lines.append(f"- Runtime stage status: {runtime_stage.get('status', 'N/A')}")
        lines.append(
            f"- Selected scene-memory runtime: {runtime_outputs.get('selected_backend', 'N/A')}"
        )
        lines.append(
            f"- Secondary runtime: {runtime_outputs.get('secondary_backend', 'N/A')}"
        )
        lines.append(
            f"- Fallback runtime: {runtime_outputs.get('fallback_backend', 'N/A')}"
        )
    lines.append("")


def _append_s4e_trained_eval_section(
    lines: list[str], fac_data: Dict[str, Any], *, primary: bool
) -> None:
    if "s4e_trained_eval" not in fac_data:
        return
    te = fac_data["s4e_trained_eval"]
    te_metrics = te.get("metrics", {})
    lines.append(
        "### Primary Headline: Same-Facility WM Policy Uplift (S4e)\n"
        if primary
        else "### Exploratory Trained Policy Eval (S4e)\n"
    )
    if primary:
        lines.append(
            "*This is the sole primary gate for `headline_scope=wm_uplift`. It is same-facility "
            "world-model evidence only and does not answer whether the uplift carries over IRL in "
            "that exact same facility.*\n"
        )
    else:
        lines.append(
            "*Exploratory only. This section never determines the canonical single-facility "
            "headline; use the fixed-world S4d claim protocol for that answer.*\n"
        )
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(
        f"| Trained mean task score | {te_metrics.get('trained_mean_task_score', 'N/A')} |"
    )
    lines.append(
        f"| Trained manipulation success | "
        f"{te_metrics.get('trained_manipulation_success_rate', 'N/A')} |"
    )
    lines.append(f"| Num rollouts | {te_metrics.get('num_rollouts_trained', 'N/A')} |")
    lines.append(f"| Claim comparison | {te_metrics.get('claim_comparison_key', 'N/A')} |")
    lines.append(
        "| Claim comparison world fixed | "
        f"{te_metrics.get('claim_comparison_world_fixed', 'N/A')} |"
    )
    lines.append("")

    te_pairwise = te_metrics.get("pairwise", {})
    if te_pairwise:
        lines.append("#### Trained vs Frozen Comparisons\n")
        lines.append(
            "| Comparison | Score A | Score B | Abs Diff | Improvement | Win Rate | p-value |"
        )
        lines.append(
            "|------------|---------|---------|----------|-------------|----------|---------|"
        )
        for pair_key, pair_data in te_pairwise.items():
            parts = pair_key.split("_vs_")
            if len(parts) == 2:
                c1, c2 = parts
                lines.append(
                    f"| {c1} vs {c2} "
                    f"| {pair_data.get(f'{c1}_mean', 'N/A')} "
                    f"| {pair_data.get(f'{c2}_mean', 'N/A')} "
                    f"| {pair_data.get('absolute_difference', 'N/A')} "
                    f"| {pair_data.get('improvement_pct', 'N/A')}% "
                    f"| {pair_data.get('win_rate', 'N/A')} "
                    f"| {pair_data.get('p_value', 'N/A')} |"
                )
        lines.append("")


def _append_s4f_polaris_eval_section(
    lines: list[str], fac_data: Dict[str, Any], *, primary: bool
) -> None:
    if "s4f_polaris_eval" not in fac_data:
        return
    pe = fac_data["s4f_polaris_eval"]
    metrics = pe.get("metrics", {})
    lines.append(
        "### Primary Headline: PolaRiS Deployment Gate (S4f)\n"
        if primary
        else "### PolaRiS Deployment Gate (S4f)\n"
    )
    if primary:
        lines.append(
            "*This is the default outer-loop decision gate when PolaRiS is enabled and valid. "
            "World-model stages remain supporting evidence for training/adaptation.*\n"
        )
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Winner | {metrics.get('winner', 'N/A')} |")
    lines.append(f"| Frozen success rate | {metrics.get('frozen_success_rate', 'N/A')} |")
    lines.append(f"| Adapted success rate | {metrics.get('adapted_success_rate', 'N/A')} |")
    lines.append(f"| Frozen mean progress | {metrics.get('frozen_mean_progress', 'N/A')} |")
    lines.append(f"| Adapted mean progress | {metrics.get('adapted_mean_progress', 'N/A')} |")
    lines.append(f"| Delta vs frozen | {metrics.get('delta_vs_frozen', 'N/A')} |")
    lines.append(f"| Scene mode | {metrics.get('scene_mode', 'N/A')} |")
    lines.append("")


def _append_s4d_policy_pair_eval_section(
    lines: list[str], fac_data: Dict[str, Any], *, supporting: bool
) -> None:
    if "s4d_policy_pair_eval" not in fac_data:
        return
    pe2 = fac_data["s4d_policy_pair_eval"]
    metrics = pe2.get("metrics", {})
    lines.append(
        "### Supporting Evidence: Policy Training Attribution Control (S4d)\n"
        if supporting
        else "### Policy Training A/B (Heldout)\n"
    )
    if supporting:
        lines.append(
            "*Supporting attribution/control evidence only. This section does not determine the "
            "top-line single-facility claim result.*\n"
        )
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(
        f"| Policy base mean task score | {metrics.get('policy_base_mean_task_score', 'N/A')} |"
    )
    lines.append(
        f"| Policy site mean task score | {metrics.get('policy_site_mean_task_score', 'N/A')} |"
    )
    lines.append(
        f"| Absolute difference | {metrics.get('task_score_absolute_difference', 'N/A')} |"
    )
    lines.append(f"| Improvement | {metrics.get('task_score_improvement_pct', 'N/A')}% |")
    lines.append(f"| Policy base success rate | {metrics.get('policy_base_success_rate', 'N/A')} |")
    lines.append(f"| Policy site success rate | {metrics.get('policy_site_success_rate', 'N/A')} |")
    lines.append(f"| Win rate (site over base) | {metrics.get('win_rate_site_over_base', 'N/A')} |")
    lines.append(f"| p-value (task score) | {metrics.get('p_value_task_score', 'N/A')} |")
    lines.append("")


def _append_claim_eval_section(lines: list[str], fac_data: Dict[str, Any]) -> None:
    if "s4d_policy_pair_eval" not in fac_data:
        return
    metrics = fac_data["s4d_policy_pair_eval"].get("metrics", {})
    if str(metrics.get("claim_protocol", "")).strip().lower() != "fixed_same_facility_uplift":
        return
    bootstrap = metrics.get("bootstrap_site_vs_frozen", {})
    generic_bootstrap = metrics.get("bootstrap_generic_vs_frozen", {})
    attribution = metrics.get("site_vs_generic_attribution", {})
    claim_outcome = _claim_outcome(metrics)
    lines.append("### Primary Headline: Fixed-World Same-Facility Claim (S4d)\n")
    lines.append(
        "*This is the sole primary gate for the fixed-world claim protocol. It is same-facility "
        "simulation evidence only and does not answer any IRL transfer question.*\n"
    )
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Primary endpoint | {metrics.get('primary_endpoint', 'N/A')} |")
    lines.append(f"| Num eval cells | {metrics.get('num_eval_cells', 'N/A')} |")
    lines.append(
        f"| Frozen baseline task-success rate | "
        f"{(metrics.get('arm_summary', {}).get('frozen_baseline', {}) or {}).get('success_rate', 'N/A')} |"
    )
    lines.append(
        f"| Site-trained mean task-success rate | "
        f"{_mean_seed_success_rate((metrics.get('arm_summary', {}).get('site_trained', {}) or {}).get('per_seed_success_rate', {}))} |"
    )
    lines.append(
        f"| generic_control mean task-success rate | "
        f"{_mean_seed_success_rate((metrics.get('arm_summary', {}).get('generic_control', {}) or {}).get('per_seed_success_rate', {}))} |"
    )
    lines.append(f"| Mean uplift (pp) | {bootstrap.get('mean_lift_pp', 'N/A')} |")
    lines.append(f"| 95% CI low (pp) | {bootstrap.get('ci_low_pp', 'N/A')} |")
    lines.append(f"| 95% CI high (pp) | {bootstrap.get('ci_high_pp', 'N/A')} |")
    lines.append(f"| Two-sided p-value | {bootstrap.get('p_value_two_sided', 'N/A')} |")
    lines.append(
        f"| Positive seeds | {bootstrap.get('positive_seed_count', 'N/A')} / "
        f"{len((metrics.get('arm_summary', {}).get('site_trained', {}) or {}).get('per_seed_success_rate', {}))} |"
    )
    lines.append(f"| Claim outcome | {claim_outcome} |")
    lines.append(
        f"| generic_control mean uplift (pp) | {generic_bootstrap.get('mean_lift_pp', 'N/A')} |"
    )
    lines.append(
        f"| site minus generic uplift (pp) | {attribution.get('mean_lift_delta_pp', 'N/A')} |"
    )
    lines.append("")


def _append_claim_portfolio_section(lines: list[str], data: Dict[str, Any]) -> None:
    portfolio = data.get("claim_portfolio")
    if not isinstance(portfolio, dict):
        return
    gate = portfolio.get("go_to_robot_gate", {}) or {}
    pooled_frozen = portfolio.get("pooled_site_vs_frozen", {}) or {}
    pooled_generic = portfolio.get("pooled_site_vs_generic", {}) or {}
    lines.append("## Claim Portfolio\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Eligible facilities | {portfolio.get('eligible_facility_count', 'N/A')} |")
    lines.append(f"| Go-to-robot gate | {'PASS' if bool(gate.get('passed', False)) else 'FAIL'} |")
    lines.append(
        f"| Pooled site vs frozen mean uplift (pp) | {pooled_frozen.get('mean_lift_pp', 'N/A')} |"
    )
    lines.append(
        f"| Pooled site vs frozen 95% CI low (pp) | {pooled_frozen.get('ci_low_pp', 'N/A')} |"
    )
    lines.append(
        f"| Pooled site vs generic mean uplift (pp) | {pooled_generic.get('mean_lift_pp', 'N/A')} |"
    )
    lines.append(
        f"| Pooled site vs generic 95% CI low (pp) | {pooled_generic.get('ci_low_pp', 'N/A')} |"
    )
    lines.append("")
    failures = list(gate.get("failures", []) or [])
    if failures:
        lines.append("Gate failures:")
        for failure in failures:
            lines.append(f"- {failure}")
        lines.append("")
    facility_claims = list(portfolio.get("facility_claims", []) or [])
    if facility_claims:
        lines.append(
            "| Facility | Eligible | Site-Frozen (pp) | Site-Generic (pp) | Generic Control |"
        )
        lines.append(
            "|----------|----------|------------------|-------------------|-----------------|"
        )
        for claim in facility_claims:
            lines.append(
                f"| {claim.get('facility_id', 'N/A')} "
                f"| {claim.get('eligible', False)} "
                f"| {claim.get('site_vs_frozen_lift_pp', 'N/A')} "
                f"| {claim.get('site_vs_generic_lift_pp', 'N/A')} "
                f"| {claim.get('generic_control_mode', 'N/A')} |"
            )
        lines.append("")


def _claim_outcome(metrics: Dict[str, Any]) -> str:
    outcome = str(metrics.get("claim_outcome", "") or "").strip().upper()
    if outcome in {"PASS", "FAIL", "INCONCLUSIVE", "INELIGIBLE"}:
        return outcome
    return "PASS" if bool(metrics.get("claim_passed", False)) else "FAIL"


def _mean_seed_success_rate(per_seed: Dict[str, Any] | Dict[int, Any]) -> str:
    if not isinstance(per_seed, dict) or not per_seed:
        return "N/A"
    values = []
    for value in per_seed.values():
        try:
            values.append(float(value))
        except Exception:
            continue
    if not values:
        return "N/A"
    return f"{sum(values) / len(values):.3f}"


def _stage_succeeded(fac_data: Dict[str, Any], stage_name: str) -> bool:
    payload = fac_data.get(stage_name, {})
    return str(payload.get("status", "")).strip().lower() == "success"


def _policy_eval_matrix_axis_rows(matrix_mode: str) -> list[tuple[str, str]]:
    if matrix_mode == "single_facility_same_site_policy_uplift":
        return [
            (
                "Seen tasks, same facility: frozen vs trained",
                "seen_task_same_facility_frozen_vs_trained",
            ),
            (
                "Heldout tasks, same facility: frozen vs trained",
                "heldout_task_same_facility_frozen_vs_trained",
            ),
            (
                "Heldout tasks, same facility: policy base vs policy site control",
                "heldout_task_same_facility_policy_base_vs_policy_site",
            ),
        ]
    return [
        ("Seen task, seen environment", "seen_task_seen_env"),
        ("Unseen object, seen environment", "unseen_object_seen_env"),
        ("Seen task, novel environment", "seen_task_novel_env"),
    ]


def _render_markdown(data: Dict[str, Any], config: ValidationConfig) -> str:
    """Render the report as Markdown."""
    lines = []
    lines.append(f"# Validation Report: {data.get('project_name', 'BlueprintValidation')}")
    lines.append(f"*Generated: {data.get('generated_at', '')}*\n")
    fixed_world_claim = _is_fixed_world_claim_protocol(config)
    if fixed_world_claim:
        lines.append(
            "*Scope: fixed-world same-facility claim protocol in simulation only. "
            "No matched real-robot evidence is included, so this report does not answer whether "
            "the observed uplift carries over IRL in the exact same facility.*\n"
        )
    elif len(config.facilities) == 1:
        lines.append(
            "*Scope: exploratory same-facility world-model evidence only. The canonical "
            "fixed-world claim protocol is not enabled, so this report does not provide the "
            "official single-facility yes/no answer. No matched real-robot evidence is included.*\n"
        )

    # Executive Summary
    lines.append("## Executive Summary\n")
    _add_executive_summary(lines, data, config)

    _append_claim_portfolio_section(lines, data)

    # Per-Facility Results
    for fid, fac_data in data.get("facilities", {}).items():
        fac_config = config.facilities.get(fid)
        fac_name = fac_config.name if fac_config else fid
        lines.append(f"\n## Facility: {fac_name}\n")
        _append_scene_runtime_section(lines, fac_data)

        polaris_primary = bool(config.eval_polaris.enabled) and bool(
            config.eval_polaris.default_as_primary_gate
        )
        polaris_available = _stage_succeeded(fac_data, "s4f_polaris_eval")
        if polaris_primary and polaris_available:
            _append_s4f_polaris_eval_section(lines, fac_data, primary=True)
            _append_s4_policy_eval_section(lines, fac_data, supporting=True)
            _append_s4e_trained_eval_section(lines, fac_data, primary=False)
            _append_s4d_policy_pair_eval_section(lines, fac_data, supporting=True)
        elif fixed_world_claim:
            _append_claim_eval_section(lines, fac_data)
            _append_s4_policy_eval_section(lines, fac_data, supporting=True)
            _append_s4e_trained_eval_section(lines, fac_data, primary=False)
        elif len(config.facilities) == 1:
            _append_s4_policy_eval_section(lines, fac_data, supporting=True)
            _append_s4e_trained_eval_section(lines, fac_data, primary=False)
            _append_s4d_policy_pair_eval_section(lines, fac_data, supporting=True)
        else:
            _append_s4_policy_eval_section(lines, fac_data, supporting=False)
            _append_s4e_trained_eval_section(lines, fac_data, primary=False)
            _append_s4d_policy_pair_eval_section(lines, fac_data, supporting=False)

        # Visual Fidelity
        if "s5_visual_fidelity" in fac_data:
            vf = fac_data["s5_visual_fidelity"]
            metrics = vf.get("metrics", {})
            lines.append("### Visual Fidelity\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for key in ["overall_mean_psnr", "overall_mean_ssim", "overall_mean_lpips"]:
                if key in metrics:
                    lines.append(f"| {key} | {metrics[key]} |")
            lines.append("")

        # Spatial Accuracy
        if "s6_spatial_accuracy" in fac_data:
            sa = fac_data["s6_spatial_accuracy"]
            metrics = sa.get("metrics", {})
            lines.append("### Spatial Accuracy\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Mean spatial score | {metrics.get('mean_spatial_score', 'N/A')} |")
            lines.append(f"| Mean visual score | {metrics.get('mean_visual_score', 'N/A')} |")
            lines.append(f"| Mean landmark score | {metrics.get('mean_landmark_score', 'N/A')} |")
            lines.append("")

        # Render & Enrich stats
        render_stage_name = "s1_isaac_render" if "s1_isaac_render" in fac_data else "s1_render"
        if render_stage_name in fac_data:
            r = fac_data[render_stage_name].get("metrics", {})
            lines.append("### Render Stats\n")
            lines.append(f"- Stage: {render_stage_name}")
            lines.append(f"- Clips rendered: {r.get('num_clips', 'N/A')}")
            lines.append(f"- Total frames: {r.get('total_frames', 'N/A')}")
            lines.append("")

        if "s1d_gaussian_augment" in fac_data:
            ga = fac_data["s1d_gaussian_augment"]
            gm = ga.get("metrics", {})
            lines.append("### RoboSplat Augmentation (S1d)\n")
            lines.append(f"- Status: {ga.get('status', 'N/A')}")
            lines.append(f"- Backend used: {gm.get('backend_used', 'N/A')}")
            lines.append(f"- Fallback backend: {gm.get('fallback_backend', 'N/A')}")
            lines.append(f"- Source clips: {gm.get('num_source_clips', 'N/A')}")
            lines.append(f"- Augmented clips accepted: {gm.get('num_augmented_clips', 'N/A')}")
            lines.append(f"- Rejected by quality gate: {gm.get('num_rejected_quality', 'N/A')}")
            lines.append("")

        if "s1f_external_interaction_ingest" in fac_data:
            ei = fac_data["s1f_external_interaction_ingest"]
            em = ei.get("metrics", {})
            eo = ei.get("outputs", {})
            lines.append("### External Interaction Ingest (S1f)\n")
            lines.append(f"- Status: {ei.get('status', 'N/A')}")
            lines.append(f"- Source: {eo.get('source_name', em.get('source_name', 'N/A'))}")
            lines.append(f"- Clips ingested: {em.get('num_clips', 'N/A')}")
            lines.append("")

        if "s1g_external_rollout_ingest" in fac_data:
            er = fac_data["s1g_external_rollout_ingest"]
            em = er.get("metrics", {})
            lines.append("### External Rollout Ingest (S1g)\n")
            lines.append(f"- Status: {er.get('status', 'N/A')}")
            lines.append(f"- Sessions ingested: {em.get('num_sessions', 'N/A')}")
            lines.append(f"- Output mode: {em.get('mode', 'N/A')}")
            if em.get("mode") in {"wm_only", "wm_and_policy"}:
                lines.append(
                    "- Note: current pipeline wiring uses these sessions for policy training only; "
                    "DreamDojo/world-model ingestion still comes from `external_interaction`."
                )
            lines.append("")

        if "s3_finetune" in fac_data:
            ft = fac_data["s3_finetune"].get("metrics", {})
            lines.append("### Fine-tuning\n")
            lines.append(f"- Epochs: {ft.get('num_epochs', 'N/A')}")
            lines.append(f"- Final loss: {ft.get('final_loss', 'N/A')}")
            lines.append(f"- Training time: {ft.get('training_seconds', 'N/A')}s")
            lines.append("")

        if "s3b_policy_finetune" in fac_data:
            pf = fac_data["s3b_policy_finetune"]
            metrics = pf.get("metrics", {})
            outputs = pf.get("outputs", {})
            adapted_checkpoint = (
                outputs.get("adapted_policy_checkpoint")
                or outputs.get("adapted_openvla_checkpoint")
                or "N/A"
            )
            lines.append("### Policy Fine-tuning (Selected Policy Adapter)\n")
            lines.append(f"- Status: {pf.get('status', 'N/A')}")
            lines.append(f"- Dataset: {metrics.get('dataset_name', 'N/A')}")
            lines.append(f"- Return code: {metrics.get('returncode', 'N/A')}")
            lines.append(f"- Adapted checkpoint: {adapted_checkpoint}")
            lines.append("")

        if "s3c_policy_rl_loop" in fac_data:
            rl = fac_data["s3c_policy_rl_loop"]
            metrics = rl.get("metrics", {})
            outputs = rl.get("outputs", {})
            rl_checkpoint = (
                outputs.get("adapted_policy_checkpoint_rl")
                or outputs.get("adapted_openvla_checkpoint_rl")
                or "N/A"
            )
            lines.append("### Policy RL Loop (World-VLA-Loop Style)\n")
            lines.append(f"- Status: {rl.get('status', 'N/A')}")
            lines.append(f"- Iterations completed: {metrics.get('iterations_completed', 'N/A')}")
            lines.append(f"- Reward mode: {metrics.get('reward_mode', 'N/A')}")
            lines.append(f"- RL checkpoint: {rl_checkpoint}")
            lines.append("")

        if "s3d_wm_refresh_loop" in fac_data:
            wm = fac_data["s3d_wm_refresh_loop"]
            metrics = wm.get("metrics", {})
            outputs = wm.get("outputs", {})
            lines.append("### World Model Refresh Loop (S3d)\n")
            lines.append(f"- Status: {wm.get('status', 'N/A')}")
            lines.append(f"- Iterations completed: {metrics.get('iterations_completed', 'N/A')}")
            lines.append(f"- Source condition: {metrics.get('source_condition', 'N/A')}")
            lines.append(
                f"- Final adapted checkpoint: "
                f"{outputs.get('final_adapted_checkpoint_path', 'N/A')}"
            )
            lines.append("")

    # Cross-Site Discrimination
    if data.get("cross_site"):
        cs = data["cross_site"]
        metrics = cs.get("metrics", {})
        lines.append("\n## Cross-Site Discrimination\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Overall accuracy | {metrics.get('overall_accuracy', 'N/A')} |")
        lines.append(f"| Inter-facility LPIPS | {metrics.get('inter_facility_lpips', 'N/A')} |")
        lines.append(f"| Intra-facility LPIPS | {metrics.get('intra_facility_lpips', 'N/A')} |")
        lines.append("")

        # Per-model accuracy
        if "model_accuracies" in metrics:
            lines.append("### Per-Model Accuracy\n")
            lines.append("| Facility | Accuracy |")
            lines.append("|----------|----------|")
            for fid, acc in metrics["model_accuracies"].items():
                lines.append(f"| {fid} | {acc} |")
            lines.append("")

        # Confusion matrix
        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            fids = list(cm.keys())
            lines.append("### Confusion Matrix\n")
            header = "| Source \\ Predicted | " + " | ".join(fids) + " |"
            sep = "|" + "---|" * (len(fids) + 1)
            lines.append(header)
            lines.append(sep)
            for src in fids:
                row = f"| {src} | " + " | ".join(str(cm[src].get(p, 0)) for p in fids) + " |"
                lines.append(row)
            lines.append("")

    if data.get("policy_eval_matrix"):
        matrix = data["policy_eval_matrix"]
        axes = matrix.get("axes", {})
        lines.append("\n## Policy Eval Matrix\n")
        lines.append("| Axis | Available | Abs Diff | p-value |")
        lines.append("|------|-----------|----------|---------|")
        matrix_mode = str(matrix.get("mode", ""))
        for axis_label, axis in _policy_eval_matrix_axis_rows(matrix_mode):
            row = axes.get(axis, {})
            lines.append(
                f"| {axis_label} | {row.get('available', False)} | "
                f"{row.get('task_score_absolute_difference', 'N/A')} | "
                f"{row.get('p_value_task_score', 'N/A')} |"
            )
        if matrix_mode != "single_facility_same_site_policy_uplift":
            lines.append(
                f"- Forgetting ratio: {matrix.get('forgetting_ratio', 'N/A')} "
                f"(gate <= {matrix.get('forgetting_ratio_gate', 'N/A')})"
            )
        lines.append("")

    # Configuration
    lines.append("\n## Configuration\n")
    lines.append(f"- Project: {config.project_name}")
    lines.append(f"- Facilities: {', '.join(config.facilities.keys())}")
    lines.append(f"- Render resolution: {config.render.resolution}")
    lines.append(f"- Model size: {config.finetune.model_size}")
    lines.append(
        f"- LoRA: {'rank=' + str(config.finetune.lora_rank) if config.finetune.use_lora else 'disabled (full fine-tuning)'}"
    )
    lines.append(f"- Policy finetune enabled: {config.policy_finetune.enabled}")
    lines.append(f"- PolaRiS enabled: {config.eval_polaris.enabled}")
    lines.append(f"- PolaRiS primary gate: {config.eval_polaris.default_as_primary_gate}")
    lines.append(f"- Policy RL loop enabled: {config.policy_rl_loop.enabled}")
    lines.append(f"- Policy dataset: {config.policy_finetune.dataset_name}")
    lines.append(f"- Policy adapter: {config.policy_adapter.name}")
    lines.append(f"- VLM judge: {config.eval_policy.vlm_judge.model}")
    lines.append(f"- Agentic Vision: {config.eval_policy.vlm_judge.enable_agentic_vision}")

    return "\n".join(lines) + "\n"


def _add_executive_summary(lines: list, data: dict, config: ValidationConfig = None) -> None:
    """Add executive summary to the report."""
    min_abs_diff = 0.5
    if config is not None:
        min_abs_diff = config.eval_policy.min_absolute_difference
    fixed_world_claim = _is_fixed_world_claim_protocol(config)
    single_facility = bool(config is not None and len(config.facilities) == 1)
    cross_site_applicable = bool(
        (config is not None and len(config.facilities) > 1) or data.get("cross_site")
    )
    claim_portfolio = data.get("claim_portfolio") if isinstance(data, dict) else None
    polaris_primary = bool(
        config is not None and config.eval_polaris.enabled and config.eval_polaris.default_as_primary_gate
    )

    if claim_portfolio:
        gate = claim_portfolio.get("go_to_robot_gate", {}) or {}
        pooled_frozen = claim_portfolio.get("pooled_site_vs_frozen", {}) or {}
        pooled_generic = claim_portfolio.get("pooled_site_vs_generic", {}) or {}
        lines.append("| Test | Result |")
        lines.append("|------|--------|")
        lines.append(
            "| Investor-Grade Multi-Facility Claim | "
            f"{'PASS' if bool(gate.get('passed', False)) else 'FAIL'} |"
        )
        lines.append("")
        if bool(gate.get("passed", False)):
            lines.append(
                "**Portfolio gate passed.** "
                f"{claim_portfolio.get('eligible_facility_count', 0)} facilities cleared the "
                "same fixed-world claim protocol, with pooled site-vs-frozen lift "
                f"{pooled_frozen.get('mean_lift_pp', 'N/A')}pp and pooled site-vs-generic lift "
                f"{pooled_generic.get('mean_lift_pp', 'N/A')}pp.\n"
            )
        else:
            lines.append(
                "**Portfolio gate not yet passed.** "
                "The investor-grade multi-facility claim either lacks enough eligible facilities "
                "or misses one or more pooled lift / integrity thresholds.\n"
            )
        return

    polaris_metrics = None
    if polaris_primary:
        for _, fac_data in data.get("facilities", {}).items():
            s4f = fac_data.get("s4f_polaris_eval", {})
            if s4f and str(s4f.get("status", "")).strip().lower() == "success":
                polaris_metrics = s4f.get("metrics", {})
                break
    if polaris_primary and polaris_metrics is not None:
        winner = str((polaris_metrics or {}).get("winner", "PENDING/FAIL"))
        adapted_rate = (polaris_metrics or {}).get("adapted_success_rate")
        frozen_rate = (polaris_metrics or {}).get("frozen_success_rate")
        lines.append("| Test | Result |")
        lines.append("|------|--------|")
        lines.append(
            f"| Primary Headline: PolaRiS Deployment Gate | {winner if winner else 'PENDING/FAIL'} |"
        )
        lines.append(
            "| Supporting Evidence: World-Model Inner Loop | "
            f"{'PASS' if polaris_metrics is not None else 'PENDING/FAIL'} |"
        )
        lines.append("")
        if polaris_metrics is not None:
            lines.append(
                "**Default deployment recommendation comes from PolaRiS.** "
                f"Frozen success rate={frozen_rate}, adapted success rate={adapted_rate}, "
                f"winner={winner}.\n"
            )
        return

    if fixed_world_claim:
        claim_metrics = None
        for _, fac_data in data.get("facilities", {}).items():
            s4d = fac_data.get("s4d_policy_pair_eval", {})
            metrics = s4d.get("metrics", {})
            if (
                str(metrics.get("claim_protocol", "")).strip().lower()
                == "fixed_same_facility_uplift"
            ):
                claim_metrics = metrics
                break
        claim_outcome = _claim_outcome(claim_metrics or {})
        bootstrap = (claim_metrics or {}).get("bootstrap_site_vs_frozen", {})
        lines.append("| Test | Result |")
        lines.append("|------|--------|")
        lines.append(f"| Primary Headline: Fixed-World Same-Facility Claim | {claim_outcome} |")
        lines.append("")
        if claim_outcome == "PASS":
            lines.append(
                "**Canonical executable answer: yes in fixed-world simulation.** "
                "Site-trained policies beat the frozen baseline on disjoint same-facility eval "
                "cells inside one frozen adapted world snapshot, using task success as the primary "
                f"endpoint (mean uplift {bootstrap.get('mean_lift_pp', 'N/A')}pp, 95% CI lower "
                f"bound {bootstrap.get('ci_low_pp', 'N/A')}pp).\n"
            )
        elif claim_outcome == "INCONCLUSIVE":
            lines.append(
                "**Canonical executable answer: inconclusive in fixed-world simulation.** "
                "The fixed-world claim protocol ran, but the evidence bar for a site-specific "
                "same-facility uplift was not met or could not be resolved cleanly.\n"
            )
        elif claim_outcome == "INELIGIBLE":
            lines.append(
                "**Canonical executable answer: ineligible.** "
                "The fixed-world claim protocol is configured, but the run did not satisfy the "
                "minimum headline-eligibility requirements needed to answer the question.\n"
            )
        else:
            lines.append(
                "**Canonical executable answer: not yet established.** "
                "The fixed-world claim protocol does not yet show a passing same-facility "
                "task-success uplift over the frozen baseline on disjoint eval cells.\n"
            )
        return

    # Check if primary test passed
    primary_passed = False
    for _, fac_data in data.get("facilities", {}).items():
        if "s4_policy_eval" not in fac_data:
            continue
        metrics = fac_data["s4_policy_eval"].get("metrics", {})
        abs_diff = metrics.get("absolute_difference", 0)
        p_value = metrics.get("p_value")
        if abs_diff >= min_abs_diff and (p_value is None or p_value < 0.05):
            primary_passed = True

    # Trained policy headline is driven only by S4e adapted_vs_trained.
    trained_passed = False
    for _, fac_data in data.get("facilities", {}).items():
        if "s4e_trained_eval" not in fac_data:
            continue
        te_metrics = fac_data["s4e_trained_eval"].get("metrics", {})
        if not bool(te_metrics.get("claim_comparison_world_fixed", False)):
            continue
        if te_metrics.get("claim_comparison_key") not in {
            "adapted_vs_trained",
            "trained_vs_adapted",
        }:
            continue
        abs_d = te_metrics.get("claim_comparison_absolute_difference", 0)
        pv = te_metrics.get("claim_comparison_p_value")
        if abs_d >= min_abs_diff and (pv is None or pv < 0.05):
            trained_passed = True
            break

    cross_site_passed = False
    if data.get("cross_site"):
        cs_metrics = data["cross_site"].get("metrics", {})
        if cs_metrics.get("overall_accuracy", 0) > 0.7:
            cross_site_passed = True

    lines.append("| Test | Result |")
    lines.append("|------|--------|")
    if single_facility:
        lines.append("| Canonical Headline: Fixed-World Same-Facility Claim | INELIGIBLE |")
        lines.append(
            "| Supporting Evidence: Frozen Baseline vs Adapted World Model | "
            f"{'PASS' if primary_passed else 'PENDING/FAIL'} |"
        )
        lines.append(
            "| Exploratory Evidence: Trained Policy Eval | "
            f"{'PASS' if trained_passed else 'PENDING/FAIL'} |"
        )
    else:
        lines.append(
            f"| Frozen Policy Performance | {'PASS' if primary_passed else 'PENDING/FAIL'} |"
        )
        lines.append(
            f"| Trained Policy Improvement | {'PASS' if trained_passed else 'PENDING/FAIL'} |"
        )
    if cross_site_applicable:
        lines.append(
            f"| Cross-Site Discrimination | {'PASS' if cross_site_passed else 'PENDING/FAIL'} |"
        )
    lines.append("")

    if single_facility:
        lines.append(
            "**Canonical single-facility answer is unavailable in this configuration.** "
            "This run did not enable the fixed-world S4d claim protocol, so S4 and S4e are "
            "supporting or exploratory world-model evidence only.\n"
        )
        if primary_passed or trained_passed:
            lines.append(
                f"Observed evidence can still be useful for debugging and prioritization "
                f"(for example, absolute differences above {min_abs_diff} with p < 0.05), but it "
                "must not be treated as the canonical answer to the same-facility uplift question.\n"
            )
    elif primary_passed:
        lines.append(
            f"**The site-adapted world model produced higher policy task scores than the "
            f"baseline (absolute difference >= {min_abs_diff}, p < 0.05).** This validates "
            "that the site-adapted world model is a stronger evaluation environment for robot "
            "policies than the generic baseline.\n"
        )
    if trained_passed and not single_facility:
        lines.append(
            "**Policies fine-tuned on site-adapted rollout data outperform frozen baselines in the "
            "evaluated world model.** This remains simulator/world-model evidence unless matched "
            "real-robot runs are added.\n"
        )
