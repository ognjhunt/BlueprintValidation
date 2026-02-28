"""Tests for report builder."""

import json

import pytest


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
    assert "Policy Performance" in content
    assert "RoboSplat Augmentation" in content
    assert "Policy RL Loop" in content
    assert "61.9%" in content


def test_report_builder_json(tmp_path, sample_config):
    from blueprint_validation.common import write_json
    from blueprint_validation.reporting.report_builder import build_report

    work_dir = tmp_path / "outputs"
    (work_dir / "test_facility").mkdir(parents=True)

    output_path = tmp_path / "report.json"
    result = build_report(sample_config, work_dir, fmt="json", output_path=output_path)
    assert result.exists()
    data = json.loads(result.read_text())
    assert data["project_name"] == "Test Project"
