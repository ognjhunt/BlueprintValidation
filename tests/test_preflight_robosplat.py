"""RoboSplat-specific preflight checks."""

from __future__ import annotations


def test_preflight_includes_robosplat_checks(sample_config, monkeypatch):
    from blueprint_validation.common import PreflightCheck
    import blueprint_validation.preflight as preflight

    # Short-circuit unrelated checks for deterministic unit test behavior.
    monkeypatch.setattr(preflight, "check_gpu", lambda: PreflightCheck("gpu", True))
    monkeypatch.setattr(preflight, "check_dependency", lambda *a, **k: PreflightCheck("dep", True))
    monkeypatch.setattr(preflight, "check_external_tool", lambda *a, **k: PreflightCheck("tool", True))
    monkeypatch.setattr(preflight, "check_facility_ply", lambda *a, **k: PreflightCheck("ply", True))
    monkeypatch.setattr(preflight, "check_model_weights", lambda *a, **k: PreflightCheck("weights", True))
    monkeypatch.setattr(preflight, "check_hf_auth", lambda: PreflightCheck("hf", True))
    monkeypatch.setattr(preflight, "check_path_exists", lambda *a, **k: PreflightCheck("path", True))
    monkeypatch.setattr(preflight, "check_path_exists_under", lambda *a, **k: PreflightCheck("path_under", True))
    monkeypatch.setattr(preflight, "check_python_import_from_path", lambda *a, **k: PreflightCheck("imp", True))
    monkeypatch.setattr(preflight, "check_api_key", lambda *a, **k: PreflightCheck("api", True))
    monkeypatch.setattr(preflight, "check_api_key_for_scope", lambda *a, **k: PreflightCheck("api_scope", True))

    sample_config.robosplat.enabled = True
    sample_config.robosplat.backend = "auto"
    checks = preflight.run_preflight(sample_config)
    names = {c.name for c in checks}
    assert "robosplat:backend" in names
    assert "robosplat:parity_mode" in names
    assert "robosplat:variants_per_input" in names
    assert "robosplat:vendor_repo" in names


def test_preflight_flags_required_real_demo(sample_config, monkeypatch):
    from blueprint_validation.common import PreflightCheck
    import blueprint_validation.preflight as preflight

    monkeypatch.setattr(preflight, "check_gpu", lambda: PreflightCheck("gpu", True))
    monkeypatch.setattr(preflight, "check_dependency", lambda *a, **k: PreflightCheck("dep", True))
    monkeypatch.setattr(preflight, "check_external_tool", lambda *a, **k: PreflightCheck("tool", True))
    monkeypatch.setattr(preflight, "check_facility_ply", lambda *a, **k: PreflightCheck("ply", True))
    monkeypatch.setattr(preflight, "check_model_weights", lambda *a, **k: PreflightCheck("weights", True))
    monkeypatch.setattr(preflight, "check_hf_auth", lambda: PreflightCheck("hf", True))
    monkeypatch.setattr(preflight, "check_path_exists", lambda *a, **k: PreflightCheck("path", True))
    monkeypatch.setattr(preflight, "check_path_exists_under", lambda *a, **k: PreflightCheck("path_under", True))
    monkeypatch.setattr(preflight, "check_python_import_from_path", lambda *a, **k: PreflightCheck("imp", True))
    monkeypatch.setattr(preflight, "check_api_key", lambda *a, **k: PreflightCheck("api", True))
    monkeypatch.setattr(preflight, "check_api_key_for_scope", lambda *a, **k: PreflightCheck("api_scope", True))

    sample_config.robosplat.enabled = True
    sample_config.robosplat.demo_source = "required_real"
    checks = preflight.run_preflight(sample_config)
    real_demo_checks = [c for c in checks if c.name == "robosplat:required_real_demo"]
    assert len(real_demo_checks) == 1
    assert real_demo_checks[0].passed is False

