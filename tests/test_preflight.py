"""Tests for preflight checks."""

from __future__ import annotations

import sys
import types
from pathlib import Path


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


def _make_openpi_scripts(repo_root: Path) -> None:
    scripts = repo_root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (repo_root / "src" / "openpi").mkdir(parents=True, exist_ok=True)
    (repo_root / "src" / "openpi" / "__init__.py").write_text("")
    (scripts / "train_pytorch.py").write_text(
        """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name")
parser.add_argument("--run_root_dir")
parser.add_argument("--dataset_root")
parser.add_argument("--dataset_name")
parser.add_argument("--base_model")
parser.add_argument("--batch_size")
parser.add_argument("--learning_rate")
parser.add_argument("--max_steps")
_ = parser.parse_args()
""".strip()
    )
    (scripts / "compute_norm_stats.py").write_text(
        """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root")
parser.add_argument("--dataset_name")
parser.add_argument("--profile")
_ = parser.parse_args()
""".strip()
    )


def _patch_preflight_fast(monkeypatch, preflight):
    from blueprint_validation.common import PreflightCheck

    def _ok(name: str) -> PreflightCheck:
        return PreflightCheck(name=name, passed=True, detail="ok")

    monkeypatch.setattr(preflight, "check_gpu", lambda: _ok("gpu"))
    monkeypatch.setattr(preflight, "check_dependency", lambda *a, **k: _ok("dep"))
    monkeypatch.setattr(preflight, "check_external_tool", lambda *a, **k: _ok("tool"))
    monkeypatch.setattr(preflight, "check_facility_ply", lambda *a, **k: _ok("ply"))
    monkeypatch.setattr(preflight, "check_model_weights", lambda *a, **k: _ok("weights"))
    monkeypatch.setattr(preflight, "check_hf_auth", lambda: _ok("hf_auth"))
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


def test_preflight_pi05_base_reference_fails_when_openvla_like(
    sample_config, monkeypatch, tmp_path
):
    import blueprint_validation.preflight as preflight

    _patch_preflight_fast(monkeypatch, preflight)
    _make_openpi_scripts(tmp_path / "openpi")

    sample_config.policy_adapter.name = "pi05"
    sample_config.policy_adapter.pi05.openpi_repo = tmp_path / "openpi"
    sample_config.policy_finetune.enabled = True
    sample_config.eval_policy.model_name = "openvla/openvla-7b"
    sample_config.eval_policy.checkpoint_path = tmp_path / "openvla-7b"

    checks = preflight.run_preflight(sample_config)
    by_name = {c.name: c for c in checks}
    assert by_name["policy:base_reference"].passed is False


def test_preflight_pi05_base_reference_passes_with_explicit_model(
    sample_config, monkeypatch, tmp_path
):
    import blueprint_validation.preflight as preflight

    _patch_preflight_fast(monkeypatch, preflight)
    _make_openpi_scripts(tmp_path / "openpi")

    sample_config.policy_adapter.name = "pi05"
    sample_config.policy_adapter.pi05.openpi_repo = tmp_path / "openpi"
    sample_config.policy_finetune.enabled = True
    sample_config.eval_policy.model_name = "openpi/pi05"
    sample_config.eval_policy.checkpoint_path = tmp_path / "openvla-7b"

    checks = preflight.run_preflight(sample_config)
    by_name = {c.name: c for c in checks}
    assert by_name["policy:base_reference"].passed is True


def test_preflight_pi05_dataset_dir_required_when_rollout_dataset_disabled(
    sample_config,
    monkeypatch,
    tmp_path,
):
    import blueprint_validation.preflight as preflight

    _patch_preflight_fast(monkeypatch, preflight)
    _make_openpi_scripts(tmp_path / "openpi")

    sample_config.policy_adapter.name = "pi05"
    sample_config.policy_adapter.pi05.openpi_repo = tmp_path / "openpi"
    sample_config.policy_finetune.enabled = True
    sample_config.rollout_dataset.enabled = False
    sample_config.policy_finetune.data_root_dir = tmp_path / "datasets"
    sample_config.policy_finetune.data_root_dir.mkdir(parents=True, exist_ok=True)
    sample_config.policy_finetune.dataset_name = "bridge_orig"
    sample_config.eval_policy.model_name = "openpi/pi05"

    checks = preflight.run_preflight(sample_config)
    by_name = {c.name: c for c in checks}
    assert by_name["policy_finetune:dataset_dir"].passed is False


def test_preflight_pi05_dataset_dir_passes_when_present(sample_config, monkeypatch, tmp_path):
    import blueprint_validation.preflight as preflight

    _patch_preflight_fast(monkeypatch, preflight)
    _make_openpi_scripts(tmp_path / "openpi")

    sample_config.policy_adapter.name = "pi05"
    sample_config.policy_adapter.pi05.openpi_repo = tmp_path / "openpi"
    sample_config.policy_finetune.enabled = True
    sample_config.rollout_dataset.enabled = False
    sample_config.policy_finetune.data_root_dir = tmp_path / "datasets"
    sample_config.policy_finetune.dataset_name = "bridge_orig"
    (
        sample_config.policy_finetune.data_root_dir / sample_config.policy_finetune.dataset_name
    ).mkdir(parents=True, exist_ok=True)
    sample_config.eval_policy.model_name = "openpi/pi05"

    checks = preflight.run_preflight(sample_config)
    by_name = {c.name: c for c in checks}
    assert by_name["policy_finetune:dataset_dir"].passed is True


def test_check_pi05_script_contracts_pass(tmp_path):
    from blueprint_validation.preflight import (
        check_pi05_norm_stats_contract,
        check_pi05_train_contract,
    )

    repo = tmp_path / "openpi"
    _make_openpi_scripts(repo)

    train = check_pi05_train_contract(repo, "scripts/train_pytorch.py")
    norm = check_pi05_norm_stats_contract(repo, "scripts/compute_norm_stats.py")
    assert train.passed is True
    assert norm.passed is True


def test_check_pi05_script_contracts_fail_when_flags_missing(tmp_path):
    from blueprint_validation.preflight import (
        check_pi05_norm_stats_contract,
        check_pi05_train_contract,
    )

    repo = tmp_path / "openpi"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "train_pytorch.py").write_text("print('no flags')")
    (scripts / "compute_norm_stats.py").write_text("print('no flags')")

    train = check_pi05_train_contract(repo, "scripts/train_pytorch.py")
    norm = check_pi05_norm_stats_contract(repo, "scripts/compute_norm_stats.py")
    assert train.passed is False
    assert norm.passed is False
    assert "--dataset_root" in train.detail or "CLI options" in train.detail
    assert "--dataset_root" in norm.detail or "CLI options" in norm.detail
