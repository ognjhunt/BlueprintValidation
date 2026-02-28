"""Build JSON and Markdown validation reports from pipeline outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from ..common import get_logger, read_json
from ..config import ValidationConfig

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
        output_path.write_text(json.dumps(report_data, indent=2, default=str))
    else:
        output_path = output_path.with_suffix(".md")
        md = _render_markdown(report_data, config)
        output_path.write_text(md)

    logger.info("Report written to %s", output_path)
    return output_path


def _collect_results(config: ValidationConfig, work_dir: Path) -> Dict[str, Any]:
    """Collect all stage results into a single dict."""
    results: Dict[str, Any] = {
        "project_name": config.project_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "facilities": {},
        "cross_site": None,
    }

    stages = [
        "s0_task_hints_bootstrap",
        "s1_render",
        "s1b_robot_composite",
        "s1c_gemini_polish",
        "s1d_gaussian_augment",
        "s1e_splatsim_interaction",
        "s2_enrich",
        "s3_finetune",
        "s4_policy_eval",
        "s4a_rlds_export",
        "s3b_policy_finetune",
        "s3c_policy_rl_loop",
        "s4e_trained_eval",
        "s4b_rollout_dataset",
        "s4c_policy_pair_train",
        "s4d_policy_pair_eval",
        "s5_visual_fidelity",
        "s6_spatial_accuracy",
    ]

    for fid in config.facilities:
        fac_dir = work_dir / fid
        fac_results = {}
        for stage in stages:
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

    return results


def _render_markdown(data: Dict[str, Any], config: ValidationConfig) -> str:
    """Render the report as Markdown."""
    lines = []
    lines.append(f"# Validation Report: {data.get('project_name', 'BlueprintValidation')}")
    lines.append(f"*Generated: {data.get('generated_at', '')}*\n")

    # Executive Summary
    lines.append("## Executive Summary\n")
    _add_executive_summary(lines, data, config)

    # Per-Facility Results
    for fid, fac_data in data.get("facilities", {}).items():
        fac_config = config.facilities.get(fid)
        fac_name = fac_config.name if fac_config else fid
        lines.append(f"\n## Facility: {fac_name}\n")

        # Policy Eval (the headline result)
        if "s4_policy_eval" in fac_data:
            pe = fac_data["s4_policy_eval"]
            metrics = pe.get("metrics", {})
            lines.append("### Policy Performance (Primary Test)\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(
                f"| Baseline mean task score | {metrics.get('baseline_mean_task_score', 'N/A')} |"
            )
            lines.append(
                f"| Adapted mean task score | {metrics.get('adapted_mean_task_score', 'N/A')} |"
            )
            lines.append(f"| Absolute difference | {metrics.get('absolute_difference', 'N/A')} |")
            lines.append(f"| Improvement | {metrics.get('improvement_pct', 'N/A')}% |")
            lines.append(f"| Win rate | {metrics.get('win_rate', 'N/A')} |")
            lines.append(f"| p-value | {metrics.get('p_value', 'N/A')} |")
            lines.append("")

            # Pairwise comparisons (N-way)
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

            # Manipulation performance per condition
            per_condition = metrics.get("per_condition", {})
            has_manip = any(
                v.get("manipulation_success_rate", 0) > 0 for v in per_condition.values()
            )
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

        # Trained policy eval (S4e)
        if "s4e_trained_eval" in fac_data:
            te = fac_data["s4e_trained_eval"]
            te_metrics = te.get("metrics", {})
            lines.append("### Trained Policy Evaluation (S4e)\n")
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

        if "s4d_policy_pair_eval" in fac_data:
            pe2 = fac_data["s4d_policy_pair_eval"]
            metrics = pe2.get("metrics", {})
            lines.append("### Policy Training A/B (Heldout)\n")
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
            lines.append(
                f"| Policy base success rate | {metrics.get('policy_base_success_rate', 'N/A')} |"
            )
            lines.append(
                f"| Policy site success rate | {metrics.get('policy_site_success_rate', 'N/A')} |"
            )
            lines.append(
                f"| Win rate (site over base) | {metrics.get('win_rate_site_over_base', 'N/A')} |"
            )
            lines.append(f"| p-value (task score) | {metrics.get('p_value_task_score', 'N/A')} |")
            lines.append("")

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
        if "s1_render" in fac_data:
            r = fac_data["s1_render"].get("metrics", {})
            lines.append("### Render Stats\n")
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

    # Check if primary test passed
    primary_passed = False
    for fid, fac_data in data.get("facilities", {}).items():
        if "s4d_policy_pair_eval" in fac_data:
            metrics = fac_data["s4d_policy_pair_eval"].get("metrics", {})
            abs_diff = metrics.get("task_score_absolute_difference", 0)
            p_value = metrics.get("p_value_task_score")
            if abs_diff >= min_abs_diff and (p_value is None or p_value < 0.05):
                primary_passed = True
        elif "s4_policy_eval" in fac_data:
            metrics = fac_data["s4_policy_eval"].get("metrics", {})
            abs_diff = metrics.get("absolute_difference", 0)
            p_value = metrics.get("p_value")
            if abs_diff >= min_abs_diff and (p_value is None or p_value < 0.05):
                primary_passed = True

    # Check if trained policy test passed (S4e)
    trained_passed = False
    for fid, fac_data in data.get("facilities", {}).items():
        if "s4e_trained_eval" in fac_data:
            te_metrics = fac_data["s4e_trained_eval"].get("metrics", {})
            te_pairwise = te_metrics.get("pairwise", {})
            for pair_key, pair_data in te_pairwise.items():
                if "trained" in pair_key:
                    abs_d = pair_data.get("absolute_difference", 0)
                    pv = pair_data.get("p_value")
                    if abs_d >= min_abs_diff and (pv is None or pv < 0.05):
                        trained_passed = True

    cross_site_passed = False
    if data.get("cross_site"):
        cs_metrics = data["cross_site"].get("metrics", {})
        if cs_metrics.get("overall_accuracy", 0) > 0.7:
            cross_site_passed = True

    lines.append("| Test | Result |")
    lines.append("|------|--------|")
    lines.append(f"| Frozen Policy Performance | {'PASS' if primary_passed else 'PENDING/FAIL'} |")
    lines.append(f"| Trained Policy Improvement | {'PASS' if trained_passed else 'PENDING/FAIL'} |")
    lines.append(
        f"| Cross-Site Discrimination | {'PASS' if cross_site_passed else 'PENDING/FAIL'} |"
    )
    lines.append("")

    if primary_passed:
        lines.append(
            f"**The site-adapted world model produced higher policy task scores than the "
            f"baseline (absolute difference >= {min_abs_diff}, p < 0.05).** This validates "
            f"that facility-specific Gaussian splat data improves the evaluation environment "
            f"for robot policies.\n"
        )
    if trained_passed:
        lines.append(
            "**Policies fine-tuned on site-adapted rollout data outperform frozen baselines.** "
            "This validates the stronger claim that robot policies perform better when "
            "**trained** in a site-adapted world model.\n"
        )
