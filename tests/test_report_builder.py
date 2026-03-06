"""Tests for report builder."""

import json


def test_report_builder_markdown(tmp_path, sample_config):
    from blueprint_validation.common import write_json
    from blueprint_validation.reporting.report_builder import build_report

    work_dir = tmp_path / "outputs"
    fac_dir = work_dir / "test_facility"
    fac_dir.mkdir(parents=True)

    # Create mock stage results
    write_json(
        {
            "stage_name": "s1_render",
            "status": "success",
            "elapsed_seconds": 10.5,
            "metrics": {"num_clips": 6, "total_frames": 294},
        },
        fac_dir / "s1_render_result.json",
    )
    write_json(
        {
            "stage_name": "s4_policy_eval",
            "status": "success",
            "elapsed_seconds": 300.0,
            "metrics": {
                "baseline_mean_task_score": 4.2,
                "adapted_mean_task_score": 6.8,
                "improvement_pct": 61.9,
                "win_rate": 0.78,
                "p_value": 0.003,
            },
        },
        fac_dir / "s4_policy_eval_result.json",
    )
    write_json(
        {
            "stage_name": "s1d_gaussian_augment",
            "status": "success",
            "elapsed_seconds": 40.0,
            "metrics": {
                "backend_used": "native",
                "fallback_backend": "none",
                "num_source_clips": 6,
                "num_augmented_clips": 24,
                "num_rejected_quality": 2,
            },
        },
        fac_dir / "s1d_gaussian_augment_result.json",
    )
    write_json(
        {
            "stage_name": "s3c_policy_rl_loop",
            "status": "success",
            "elapsed_seconds": 200.0,
            "outputs": {"adapted_policy_checkpoint_rl": "/tmp/ckpt"},
            "metrics": {"iterations_completed": 2, "reward_mode": "hybrid"},
        },
        fac_dir / "s3c_policy_rl_loop_result.json",
    )

    output_path = tmp_path / "report.md"
    result = build_report(sample_config, work_dir, fmt="markdown", output_path=output_path)
    assert result.exists()
    content = result.read_text()
    assert "Validation Report" in content
    assert "Supporting Evidence: Frozen Policy Baseline vs Adapted World Model" in content
    assert "RoboSplat Augmentation" in content
    assert "Policy RL Loop" in content
    assert "61.9%" in content


def test_report_builder_json(tmp_path, sample_config):
    from blueprint_validation.reporting.report_builder import build_report

    work_dir = tmp_path / "outputs"
    (work_dir / "test_facility").mkdir(parents=True)

    output_path = tmp_path / "report.json"
    result = build_report(sample_config, work_dir, fmt="json", output_path=output_path)
    assert result.exists()
    data = json.loads(result.read_text())
    assert data["project_name"] == "Test Project"


def test_executive_summary_uses_s4e_only_when_world_fixed(sample_config):
    from blueprint_validation.reporting.report_builder import _add_executive_summary

    lines: list[str] = []
    data = {
        "facilities": {
            "test_facility": {
                "s4e_trained_eval": {
                    "metrics": {
                        "claim_comparison_key": "baseline_vs_trained",
                        "claim_comparison_world_fixed": False,
                        "claim_comparison_absolute_difference": 2.0,
                        "claim_comparison_p_value": 0.001,
                    }
                }
            }
        }
    }
    _add_executive_summary(lines, data, sample_config)
    rendered = "\n".join(lines)
    assert "| Canonical Headline: Fixed-World Same-Facility Claim | INELIGIBLE |" in rendered
    assert "| Exploratory Evidence: Trained Policy Eval | PENDING/FAIL |" in rendered


def test_executive_summary_accepts_world_fixed_s4e_fallback(sample_config):
    from blueprint_validation.reporting.report_builder import _add_executive_summary

    sample_config.eval_policy.headline_scope = "wm_uplift"
    lines: list[str] = []
    data = {
        "facilities": {
            "test_facility": {
                "s4_policy_eval": {
                    "metrics": {
                        "absolute_difference": 1.4,
                        "p_value": 0.01,
                    }
                },
                "s4e_trained_eval": {
                    "metrics": {
                        "claim_comparison_key": "adapted_vs_trained",
                        "claim_comparison_world_fixed": True,
                        "claim_comparison_absolute_difference": 2.0,
                        "claim_comparison_p_value": 0.001,
                    }
                }
            }
        }
    }
    _add_executive_summary(lines, data, sample_config)
    rendered = "\n".join(lines)
    assert "| Canonical Headline: Fixed-World Same-Facility Claim | INELIGIBLE |" in rendered
    assert "| Supporting Evidence: Frozen Baseline vs Adapted World Model | PASS |" in rendered
    assert "| Exploratory Evidence: Trained Policy Eval | PASS |" in rendered
    assert "Canonical single-facility answer is unavailable in this configuration" in rendered
    assert "Cross-Site Discrimination" not in rendered


def test_executive_summary_does_not_let_s4d_satisfy_trained_headline(sample_config):
    from blueprint_validation.reporting.report_builder import _add_executive_summary

    lines: list[str] = []
    data = {
        "facilities": {
            "test_facility": {
                "s4d_policy_pair_eval": {
                    "metrics": {
                        "task_score_absolute_difference": 1.5,
                        "p_value_task_score": 0.01,
                    }
                },
                "s4e_trained_eval": {
                    "metrics": {
                        "claim_comparison_key": "baseline_vs_trained",
                        "claim_comparison_world_fixed": False,
                        "claim_comparison_absolute_difference": 2.0,
                        "claim_comparison_p_value": 0.001,
                    }
                },
            }
        }
    }
    _add_executive_summary(lines, data, sample_config)
    rendered = "\n".join(lines)
    assert "| Canonical Headline: Fixed-World Same-Facility Claim | INELIGIBLE |" in rendered
    assert "| Exploratory Evidence: Trained Policy Eval | PENDING/FAIL |" in rendered


def test_executive_summary_wm_uplift_does_not_let_s4_only_satisfy_primary_headline(sample_config):
    from blueprint_validation.reporting.report_builder import _add_executive_summary

    sample_config.eval_policy.headline_scope = "wm_uplift"
    lines: list[str] = []
    data = {
        "facilities": {
            "test_facility": {
                "s4_policy_eval": {
                    "metrics": {
                        "absolute_difference": 1.5,
                        "p_value": 0.01,
                    }
                }
            }
        }
    }
    _add_executive_summary(lines, data, sample_config)
    rendered = "\n".join(lines)
    assert "| Canonical Headline: Fixed-World Same-Facility Claim | INELIGIBLE |" in rendered
    assert "| Supporting Evidence: Frozen Baseline vs Adapted World Model | PASS |" in rendered
    assert "must not be treated as the canonical answer" in rendered


def test_executive_summary_does_not_let_s4d_satisfy_frozen_policy_headline(sample_config):
    from blueprint_validation.reporting.report_builder import _add_executive_summary

    lines: list[str] = []
    data = {
        "facilities": {
            "test_facility": {
                "s4d_policy_pair_eval": {
                    "metrics": {
                        "task_score_absolute_difference": 1.5,
                        "p_value_task_score": 0.01,
                    }
                }
            }
        }
    }
    _add_executive_summary(lines, data, sample_config)
    rendered = "\n".join(lines)
    assert "| Supporting Evidence: Frozen Baseline vs Adapted World Model | PENDING/FAIL |" in rendered


def test_report_builder_renders_single_facility_policy_matrix_without_forgetting_ratio(
    tmp_path, sample_config
):
    from blueprint_validation.common import write_json
    from blueprint_validation.reporting.report_builder import build_report

    work_dir = tmp_path / "outputs"
    fac_dir = work_dir / "test_facility"
    fac_dir.mkdir(parents=True)
    write_json(
        {
            "mode": "single_facility_same_site_policy_uplift",
            "axes": {
                "seen_task_same_facility_frozen_vs_trained": {
                    "available": True,
                    "task_score_absolute_difference": 1.2,
                    "p_value_task_score": 0.02,
                },
                "heldout_task_same_facility_frozen_vs_trained": {
                    "available": True,
                    "task_score_absolute_difference": 1.0,
                    "p_value_task_score": 0.03,
                },
                "heldout_task_same_facility_policy_base_vs_policy_site": {
                    "available": False,
                    "task_score_absolute_difference": None,
                    "p_value_task_score": None,
                },
            },
        },
        work_dir / "policy_eval" / "matrix_report.json",
    )

    output_path = tmp_path / "report.md"
    sample_config.eval_policy.headline_scope = "wm_uplift"
    result = build_report(sample_config, work_dir, fmt="markdown", output_path=output_path)
    content = result.read_text()
    assert "Seen tasks, same facility: frozen vs trained" in content
    assert "Heldout tasks, same facility: frozen vs trained" in content
    assert "Heldout tasks, same facility: policy base vs policy site control" in content
    assert "Forgetting ratio" not in content


def test_report_builder_wm_uplift_reorders_sections_and_marks_supporting_evidence(
    tmp_path, sample_config
):
    from blueprint_validation.common import write_json
    from blueprint_validation.reporting.report_builder import build_report

    sample_config.eval_policy.headline_scope = "wm_uplift"
    work_dir = tmp_path / "outputs"
    fac_dir = work_dir / "test_facility"
    fac_dir.mkdir(parents=True)

    write_json(
        {
            "stage_name": "s4_policy_eval",
            "status": "success",
            "metrics": {
                "baseline_mean_task_score": 4.0,
                "adapted_mean_task_score": 5.5,
                "absolute_difference": 1.5,
                "improvement_pct": 37.5,
                "win_rate": 0.75,
                "p_value": 0.01,
            },
        },
        fac_dir / "s4_policy_eval_result.json",
    )
    write_json(
        {
            "stage_name": "s4e_trained_eval",
            "status": "success",
            "metrics": {
                "trained_mean_task_score": 6.5,
                "trained_manipulation_success_rate": 0.5,
                "num_rollouts_trained": 8,
                "claim_comparison_key": "adapted_vs_trained",
                "claim_comparison_world_fixed": True,
                "claim_comparison_absolute_difference": 1.0,
                "claim_comparison_p_value": 0.01,
            },
        },
        fac_dir / "s4e_trained_eval_result.json",
    )
    write_json(
        {
            "stage_name": "s4d_policy_pair_eval",
            "status": "success",
            "metrics": {
                "policy_base_mean_task_score": 5.0,
                "policy_site_mean_task_score": 5.8,
                "task_score_absolute_difference": 0.8,
                "task_score_improvement_pct": 16.0,
                "policy_base_success_rate": 0.25,
                "policy_site_success_rate": 0.5,
                "win_rate_site_over_base": 0.625,
                "p_value_task_score": 0.03,
            },
        },
        fac_dir / "s4d_policy_pair_eval_result.json",
    )

    output_path = tmp_path / "report.md"
    result = build_report(sample_config, work_dir, fmt="markdown", output_path=output_path)
    content = result.read_text()
    assert "exploratory same-facility world-model evidence only" in content
    assert "canonical fixed-world claim protocol is not enabled" in content
    assert "### Exploratory Trained Policy Eval (S4e)" in content
    assert "### Supporting Evidence: Frozen Policy Baseline vs Adapted World Model (S4)" in content
    assert "### Supporting Evidence: Policy Training Attribution Control (S4d)" in content
    assert content.index("### Supporting Evidence: Frozen Policy Baseline vs Adapted World Model (S4)") < content.index(
        "### Exploratory Trained Policy Eval (S4e)"
    )
    assert content.index("### Exploratory Trained Policy Eval (S4e)") < content.index(
        "### Supporting Evidence: Policy Training Attribution Control (S4d)"
    )


def test_report_builder_fixed_world_claim_uses_s4d_as_primary_headline(tmp_path, sample_config):
    from blueprint_validation.common import write_json
    from blueprint_validation.reporting.report_builder import build_report

    sample_config.eval_policy.claim_protocol = "fixed_same_facility_uplift"
    sample_config.eval_policy.primary_endpoint = "task_success"
    sample_config.eval_policy.freeze_world_snapshot = True

    work_dir = tmp_path / "outputs"
    fac_dir = work_dir / "test_facility"
    fac_dir.mkdir(parents=True)

    write_json(
        {
            "stage_name": "s4d_policy_pair_eval",
            "status": "success",
            "metrics": {
                "claim_protocol": "fixed_same_facility_uplift",
                "primary_endpoint": "task_success",
                "num_eval_cells": 12,
                "claim_outcome": "PASS",
                "claim_passed": True,
                "bootstrap_site_vs_frozen": {
                    "mean_lift_pp": 12.0,
                    "ci_low_pp": 4.0,
                    "ci_high_pp": 18.0,
                    "p_value_two_sided": 0.01,
                    "positive_seed_count": 4,
                },
                "arm_summary": {
                    "site_trained": {
                        "per_seed_success_rate": {"0": 0.7, "1": 0.8, "2": 0.75, "3": 0.72}
                    }
                },
            },
        },
        fac_dir / "s4d_policy_pair_eval_result.json",
    )

    output_path = tmp_path / "report.md"
    result = build_report(sample_config, work_dir, fmt="markdown", output_path=output_path)
    content = result.read_text()
    assert "Primary Headline: Fixed-World Same-Facility Claim" in content
    assert "fixed-world same-facility claim protocol in simulation only" in content
    assert "Primary Headline: Fixed-World Same-Facility Claim | PASS" in content
