"""Tests for preflight checks."""

from __future__ import annotations

import sys
import types


def test_check_gpu_uses_total_memory(monkeypatch):
    from blueprint_validation.preflight import check_gpu

    props = types.SimpleNamespace(total_memory=80 * 1024**3)
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda idx: "Fake H100",
        get_device_properties=lambda idx: props,
    )
    fake_torch = types.SimpleNamespace(cuda=cuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = check_gpu()
    assert result.passed is True
    assert "80GB" in result.detail


def test_check_gpu_fallback_total_mem(monkeypatch):
    from blueprint_validation.preflight import check_gpu

    props = types.SimpleNamespace(total_mem=24 * 1024**3)
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda idx: "Legacy GPU",
        get_device_properties=lambda idx: props,
    )
    fake_torch = types.SimpleNamespace(cuda=cuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = check_gpu()
    assert result.passed is True
    assert "24GB" in result.detail


def test_check_gpu_handles_runtime_errors(monkeypatch):
    from blueprint_validation.preflight import check_gpu

    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda idx: (_ for _ in ()).throw(RuntimeError("driver error")),
        get_device_properties=lambda idx: None,
    )
    fake_torch = types.SimpleNamespace(cuda=cuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = check_gpu()
    assert result.passed is False
    assert "GPU check failed" in result.detail


def test_preflight_pi05_invalid_profile_runtime_backend(sample_config, monkeypatch, tmp_path):
    import blueprint_validation.preflight as preflight
    from blueprint_validation.common import PreflightCheck

    sample_config.policy_adapter.name = "pi05"
    sample_config.policy_adapter.pi05.profile = "invalid_profile"
    sample_config.policy_adapter.pi05.runtime_mode = "remote"
    sample_config.policy_adapter.pi05.train_backend = "jax"
    sample_config.policy_adapter.pi05.openpi_repo = tmp_path / "missing_openpi"
    sample_config.policy_finetune.enabled = True

    def _ok(name: str) -> PreflightCheck:
        return PreflightCheck(name=name, passed=True, detail="ok")

    monkeypatch.setattr(preflight, "check_gpu", lambda: _ok("gpu"))
    monkeypatch.setattr(preflight, "check_dependency", lambda *a, **k: _ok("dep"))
    monkeypatch.setattr(preflight, "check_external_tool", lambda *a, **k: _ok("tool"))
    monkeypatch.setattr(preflight, "check_facility_ply", lambda *a, **k: _ok("ply"))
    monkeypatch.setattr(preflight, "check_model_weights", lambda *a, **k: _ok("weights"))
    monkeypatch.setattr(preflight, "check_hf_auth", lambda: _ok("hf_auth"))
    monkeypatch.setattr(preflight, "check_path_exists", lambda *a, **k: _ok(k.get("name", "path")))
    monkeypatch.setattr(
        preflight, "check_path_exists_under", lambda *a, **k: _ok(k.get("name", "path_under"))
    )
    monkeypatch.setattr(preflight, "check_cosmos_wrapper_contract", lambda *a, **k: _ok("cosmos"))
    monkeypatch.setattr(preflight, "check_dreamdojo_contract", lambda *a, **k: _ok("dreamdojo"))
    monkeypatch.setattr(preflight, "check_python_import_from_path", lambda *a, **k: _ok("import"))
    monkeypatch.setattr(preflight, "check_api_key", lambda *a, **k: _ok("api_key"))
    monkeypatch.setattr(preflight, "check_api_key_for_scope", lambda *a, **k: _ok("api_scope"))
    monkeypatch.setattr(preflight, "check_cloud_budget_enforcement", lambda *a, **k: _ok("budget"))
    monkeypatch.setattr(
        preflight, "check_cloud_shutdown_enforcement", lambda *a, **k: _ok("shutdown")
    )

    # Ensure OpenVLA-only checks are skipped for pi05.
    monkeypatch.setattr(
        preflight,
        "check_openvla_finetune_contract",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("OpenVLA contract should not run")),
    )
    monkeypatch.setattr(
        preflight,
        "check_openvla_dataset_registry",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("OpenVLA registry should not run")),
    )

    checks = preflight.run_preflight(sample_config)
    by_name = {c.name: c for c in checks}
    assert by_name["policy_adapter:pi05:profile"].passed is False
    assert by_name["policy_adapter:pi05:runtime_mode"].passed is False
    assert by_name["policy_adapter:pi05:train_backend"].passed is False
