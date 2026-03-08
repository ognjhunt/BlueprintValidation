"""Preflight and report coverage for PolaRiS primary-gate behavior."""

from __future__ import annotations

import json


def test_preflight_fails_when_polaris_primary_gate_is_unrunnable(sample_config, monkeypatch):
    import blueprint_validation.preflight as preflight
    from blueprint_validation.common import PreflightCheck

    sample_config.eval_polaris.enabled = True
    sample_config.eval_polaris.default_as_primary_gate = True

    def _ok(name: str) -> PreflightCheck:
        return PreflightCheck(name=name, passed=True, detail="ok")

    monkeypatch.setattr(preflight, "check_gpu", lambda: _ok("gpu"))
    monkeypatch.setattr(preflight, "check_dependency", lambda *a, **k: _ok("dep"))
    monkeypatch.setattr(preflight, "check_external_tool", lambda *a, **k: _ok("tool"))
    monkeypatch.setattr(preflight, "check_facility_ply", lambda *a, **k: _ok("ply"))
    monkeypatch.setattr(preflight, "check_model_weights", lambda *a, **k: _ok("weights"))
    monkeypatch.setattr(preflight, "check_hf_auth", lambda: _ok("hf_auth"))
    monkeypatch.setattr(preflight, "check_cosmos_predict_tokenizer_access", lambda *a, **k: _ok("cosmos"))
    monkeypatch.setattr(preflight, "check_task_hints_bootstrap_readiness", lambda *a, **k: _ok("task_hints"))
    monkeypatch.setattr(preflight, "check_openvla_local_checkpoint_requirement", lambda *a, **k: _ok("local_ckpt"))
    monkeypatch.setattr(preflight, "check_claim_benchmark_readiness", lambda *a, **k: _ok("claim"))
    monkeypatch.setattr(preflight, "check_python_import_from_path", lambda *a, **k: _ok("import"))
    monkeypatch.setattr(preflight, "check_api_key", lambda *a, **k: _ok("api"))
    monkeypatch.setattr(preflight, "check_api_key_for_scope", lambda *a, **k: _ok("api_scope"))
    monkeypatch.setattr(preflight, "check_external_interaction_manifest", lambda *a, **k: _ok("ext"))
    monkeypatch.setattr(preflight, "check_cloud_budget_enforcement", lambda *a, **k: _ok("budget"))
    monkeypatch.setattr(preflight, "check_cloud_shutdown_enforcement", lambda *a, **k: _ok("shutdown"))

    checks = preflight.run_preflight(sample_config)
    by_name = {check.name: check for check in checks}
    assert by_name["polaris:runtime"].passed is False
    assert by_name["polaris:primary_gate:test_facility"].passed is False


def test_report_builder_uses_polaris_as_primary_headline(tmp_path, sample_config):
    from blueprint_validation.common import write_json
    from blueprint_validation.reporting.report_builder import build_report

    sample_config.eval_polaris.enabled = True
    sample_config.eval_polaris.default_as_primary_gate = True
    work_dir = tmp_path / "outputs"
    fac_dir = work_dir / "test_facility"
    fac_dir.mkdir(parents=True)

    write_json(
        {
            "stage_name": "s4f_polaris_eval",
            "status": "success",
            "metrics": {
                "winner": "adapted_openvla",
                "frozen_success_rate": 0.41,
                "adapted_success_rate": 0.67,
                "frozen_mean_progress": 0.50,
                "adapted_mean_progress": 0.74,
                "delta_vs_frozen": 0.26,
                "scene_mode": "scene_package_bridge",
            },
        },
        fac_dir / "s4f_polaris_eval_result.json",
    )
    write_json(
        {
            "stage_name": "s4_policy_eval",
            "status": "success",
            "metrics": {"absolute_difference": 1.2, "p_value": 0.01},
        },
        fac_dir / "s4_policy_eval_result.json",
    )

    result = build_report(sample_config, work_dir, fmt="markdown", output_path=tmp_path / "report.md")
    content = result.read_text()
    assert "Primary Headline: PolaRiS Deployment Gate" in content
    assert "Default deployment recommendation comes from PolaRiS" in content
    assert "Supporting Evidence: Frozen Policy Baseline vs Adapted World Model" in content
