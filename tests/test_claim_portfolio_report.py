from __future__ import annotations


def test_report_builder_renders_claim_portfolio_section(sample_config, tmp_path):
    from blueprint_validation.common import write_json
    from blueprint_validation.config import FacilityConfig
    from blueprint_validation.reporting.report_builder import build_report

    sample_config.facilities["facility_b"] = FacilityConfig(
        name="facility_b",
        ply_path=tmp_path / "facility_b.ply",
        claim_benchmark_path=tmp_path / "facility_b_benchmark.json",
    )
    sample_config.facilities["facility_b"].ply_path.write_text("")
    sample_config.facilities["facility_b"].claim_benchmark_path.write_text('{"version": 1, "task_specs": [], "assignments": []}')

    work_dir = tmp_path / "outputs"
    write_json(
        {
            "eligible_facility_count": 3,
            "facility_claims": [
                {
                    "facility_id": "test_facility",
                    "eligible": True,
                    "site_vs_frozen_lift_pp": 10.0,
                    "site_vs_generic_lift_pp": 3.0,
                    "generic_control_mode": "leave_one_facility_out",
                }
            ],
            "pooled_site_vs_frozen": {"mean_lift_pp": 10.0, "ci_low_pp": 2.0},
            "pooled_site_vs_generic": {"mean_lift_pp": 3.0, "ci_low_pp": 1.0},
            "go_to_robot_gate": {"passed": True, "failures": []},
        },
        work_dir / "claim_portfolio_report.json",
    )

    result = build_report(sample_config, work_dir, fmt="markdown", output_path=tmp_path / "report.md")
    content = result.read_text()
    assert "Investor-Grade Multi-Facility Claim | PASS" in content
    assert "## Claim Portfolio" in content
