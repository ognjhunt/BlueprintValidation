"""Tests for preflight checks."""

from __future__ import annotations

import importlib
import json
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


def test_check_python_import_from_path_checks_layout_without_importing(tmp_path):
    from blueprint_validation.preflight import check_python_import_from_path

    repo = tmp_path / "repo"
    module_dir = repo / "pkg"
    module_dir.mkdir(parents=True)
    marker = tmp_path / "executed.txt"
    (module_dir / "mod.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )

    result = check_python_import_from_path("pkg.mod", repo, "import:pkg_mod")

    assert result.passed is True
    assert "Module layout" in result.detail
    assert marker.exists() is False


def test_check_python_import_from_path_reports_missing_module_layout(tmp_path):
    from blueprint_validation.preflight import check_python_import_from_path

    repo = tmp_path / "repo"
    repo.mkdir(parents=True)

    result = check_python_import_from_path(
        ("missing.mod", "still_missing"),
        repo,
        "import:missing",
    )

    assert result.passed is False
    assert "Cannot find module layout" in result.detail


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
    monkeypatch.setattr(
        preflight,
        "check_cosmos_predict_tokenizer_access",
        lambda *a, **k: _ok("cosmos_predict"),
    )
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
    monkeypatch.setattr(
        preflight,
        "check_task_hints_bootstrap_readiness",
        lambda *a, **k: _ok("task_hints_bootstrap"),
    )
    monkeypatch.setattr(
        preflight,
        "check_openvla_local_checkpoint_requirement",
        lambda *a, **k: _ok("local_checkpoint"),
    )
    monkeypatch.setattr(
        preflight,
        "check_claim_benchmark_readiness",
        lambda *a, **k: _ok("claim_benchmark"),
    )
    monkeypatch.setattr(
        preflight,
        "check_external_interaction_manifest",
        lambda *a, **k: _ok("external_interaction"),
    )
    monkeypatch.setattr(
        preflight,
        "check_dreamzero_runtime_contract",
        lambda *a, **k: _ok("dreamzero_runtime"),
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
        preflight,
        "check_cosmos_predict_tokenizer_access",
        lambda *a, **k: _ok("cosmos_predict"),
    )
    monkeypatch.setattr(
        preflight, "check_path_exists_under", lambda *a, **k: _ok(k.get("name", "path_under"))
    )
    monkeypatch.setattr(preflight, "check_cosmos_wrapper_contract", lambda *a, **k: _ok("cosmos"))
    monkeypatch.setattr(preflight, "check_dreamdojo_contract", lambda *a, **k: _ok("dreamdojo"))
    monkeypatch.setattr(preflight, "check_python_import_from_path", lambda *a, **k: _ok("import"))
    monkeypatch.setattr(
        preflight,
        "check_external_interaction_manifest",
        lambda *a, **k: _ok("external_interaction"),
    )
    monkeypatch.setattr(
        preflight, "check_dreamzero_runtime_contract", lambda *a, **k: _ok("dreamzero_runtime")
    )
    monkeypatch.setattr(
        preflight, "check_dreamzero_train_contract", lambda *a, **k: _ok("dreamzero_train_contract")
    )
    monkeypatch.setattr(preflight, "check_api_key", lambda *a, **k: _ok("api_key"))
    monkeypatch.setattr(preflight, "check_api_key_for_scope", lambda *a, **k: _ok("api_scope"))
    monkeypatch.setattr(preflight, "check_cloud_budget_enforcement", lambda *a, **k: _ok("budget"))
    monkeypatch.setattr(
        preflight, "check_cloud_shutdown_enforcement", lambda *a, **k: _ok("shutdown")
    )
    monkeypatch.setattr(
        preflight,
        "check_task_hints_bootstrap_readiness",
        lambda *a, **k: _ok("task_hints_bootstrap"),
    )
    monkeypatch.setattr(
        preflight,
        "check_openvla_local_checkpoint_requirement",
        lambda *a, **k: _ok("local_checkpoint"),
    )
    monkeypatch.setattr(
        preflight,
        "check_claim_benchmark_readiness",
        lambda *a, **k: _ok("claim_benchmark"),
    )


def test_preflight_requires_stage2_runtime_dependencies(sample_config, monkeypatch):
    import blueprint_validation.preflight as preflight
    from blueprint_validation.common import PreflightCheck

    seen_dependencies: list[tuple[str, str]] = []

    def _ok(name: str) -> PreflightCheck:
        return PreflightCheck(name=name, passed=True, detail="ok")

    def _dependency(module_name: str, package_name: str = "") -> PreflightCheck:
        seen_dependencies.append((module_name, package_name))
        pkg = package_name or module_name
        return PreflightCheck(name=f"dep:{pkg}", passed=True, detail="ok")

    monkeypatch.setattr(preflight, "check_gpu", lambda: _ok("gpu"))
    monkeypatch.setattr(preflight, "check_dependency", _dependency)
    monkeypatch.setattr(preflight, "check_external_tool", lambda *a, **k: _ok("tool"))
    monkeypatch.setattr(preflight, "check_facility_ply", lambda *a, **k: _ok("ply"))
    monkeypatch.setattr(preflight, "check_model_weights", lambda *a, **k: _ok("weights"))
    monkeypatch.setattr(preflight, "check_hf_auth", lambda: _ok("hf_auth"))
    monkeypatch.setattr(
        preflight,
        "check_cosmos_predict_tokenizer_access",
        lambda *a, **k: _ok("cosmos_predict"),
    )
    monkeypatch.setattr(preflight, "check_path_exists", lambda *a, **k: _ok("path"))
    monkeypatch.setattr(preflight, "check_path_exists_under", lambda *a, **k: _ok("path_under"))
    monkeypatch.setattr(preflight, "check_cosmos_wrapper_contract", lambda *a, **k: _ok("cosmos"))
    monkeypatch.setattr(preflight, "check_dreamdojo_contract", lambda *a, **k: _ok("dreamdojo"))
    monkeypatch.setattr(
        preflight, "check_policy_base_reference_for_adapter", lambda *a, **k: _ok("policy_base")
    )
    monkeypatch.setattr(preflight, "check_python_import_from_path", lambda *a, **k: _ok("import"))
    monkeypatch.setattr(preflight, "check_api_key", lambda *a, **k: _ok("api_key"))
    monkeypatch.setattr(preflight, "check_api_key_for_scope", lambda *a, **k: _ok("api_scope"))
    monkeypatch.setattr(
        preflight, "check_openvla_finetune_contract", lambda *a, **k: _ok("openvla_contract")
    )
    monkeypatch.setattr(
        preflight, "check_openvla_dataset_registry", lambda *a, **k: _ok("dataset_registry")
    )
    monkeypatch.setattr(preflight, "check_cloud_budget_enforcement", lambda *a, **k: _ok("budget"))
    monkeypatch.setattr(
        preflight, "check_cloud_shutdown_enforcement", lambda *a, **k: _ok("shutdown")
    )

    _ = preflight.run_preflight(sample_config)

    assert ("sam2", "sam2") in seen_dependencies
    assert ("natsort", "natsort") in seen_dependencies
    assert ("lightning", "lightning") in seen_dependencies


def test_preflight_claim_uses_auto_selected_dreamdojo_experiment_for_action_dim(
    sample_config, monkeypatch, tmp_path
):
    import blueprint_validation.preflight as preflight

    _patch_preflight_fast(monkeypatch, preflight)

    dreamdojo_repo = tmp_path / "DreamDojo"
    configs_dir = dreamdojo_repo / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    (configs_dir / "2b_480_640_gr1.yaml").write_text("action_dim: 7\n")

    sample_config.eval_policy.mode = "claim"
    sample_config.eval_policy.headline_scope = "dual"
    sample_config.eval_policy.required_action_dim = 7
    sample_config.policy_adapter.openvla.policy_action_dim = 7
    sample_config.finetune.dreamdojo_repo = dreamdojo_repo
    sample_config.finetune.eval_world_experiment = None
    sample_config.finetune.experiment_config = None

    checks = preflight.run_preflight(sample_config)
    by_name = {c.name: c for c in checks}
    assert by_name["claim:world_model_action_dim"].passed is True


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


def test_preflight_dreamzero_base_reference_rejects_openvla_like_model(sample_config, monkeypatch):
    import blueprint_validation.preflight as preflight

    _patch_preflight_fast(monkeypatch, preflight)
    sample_config.policy_adapter.name = "dreamzero"
    sample_config.policy_adapter.dreamzero.base_model_name = "openvla/openvla-7b"
    sample_config.eval_policy.headline_scope = "dual"

    checks = preflight.run_preflight(sample_config)
    by_name = {c.name: c for c in checks}
    assert by_name["policy:base_reference"].passed is False
    assert "openvla-like" in by_name["policy:base_reference"].detail.lower()


def test_preflight_single_facility_policy_compare_uses_same_facility_matrix_mode(
    sample_config, monkeypatch
):
    import blueprint_validation.preflight as preflight

    _patch_preflight_fast(monkeypatch, preflight)
    sample_config.policy_compare.enabled = True

    checks = preflight.run_preflight(sample_config)
    by_name = {c.name: c for c in checks}
    assert by_name["policy_eval_matrix:mode"].passed is True
    assert "single_facility" in by_name["policy_eval_matrix:mode"].detail


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


def test_check_pi05_script_contracts_do_not_execute_scripts(tmp_path, monkeypatch):
    import blueprint_validation.preflight as preflight

    repo = tmp_path / "openpi"
    _make_openpi_scripts(repo)

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("subprocess.run should not be used for CLI contract checks")

    monkeypatch.setattr(preflight.subprocess, "run", _fail_if_called)

    train = preflight.check_pi05_train_contract(repo, "scripts/train_pytorch.py")
    norm = preflight.check_pi05_norm_stats_contract(repo, "scripts/compute_norm_stats.py")
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


def test_check_cosmos_predict_tokenizer_access_passes_with_local_file(tmp_path):
    from blueprint_validation.preflight import check_cosmos_predict_tokenizer_access

    cosmos_transfer_dir = tmp_path / "cosmos-transfer-2.5-2b"
    cosmos_transfer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = tmp_path / "cosmos-predict-2.5-2b" / "tokenizer.pth"
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path.write_bytes(b"ok")

    result = check_cosmos_predict_tokenizer_access(cosmos_transfer_dir)
    assert result.passed is True
    assert "Local tokenizer present" in result.detail


def test_check_cosmos_predict_tokenizer_access_fails_without_token_or_file(tmp_path, monkeypatch):
    from blueprint_validation.preflight import check_cosmos_predict_tokenizer_access

    monkeypatch.delenv("HF_TOKEN", raising=False)
    cosmos_transfer_dir = tmp_path / "cosmos-transfer-2.5-2b"
    cosmos_transfer_dir.mkdir(parents=True, exist_ok=True)

    result = check_cosmos_predict_tokenizer_access(cosmos_transfer_dir)
    assert result.passed is False
    assert "HF_TOKEN is unset" in result.detail


def test_check_openvla_local_checkpoint_requirement_fails_without_checkpoint(
    sample_config, monkeypatch
):
    from blueprint_validation.preflight import check_openvla_local_checkpoint_requirement

    monkeypatch.delenv("BLUEPRINT_ALLOW_REMOTE_OPENVLA_MODEL", raising=False)
    sample_config.eval_policy.mode = "claim"
    sample_config.eval_policy.headline_scope = "dual"

    result = check_openvla_local_checkpoint_requirement(sample_config, "openvla_oft")
    assert result.passed is False
    assert "local checkpoint" in result.detail.lower()


def test_check_openvla_local_checkpoint_requirement_allows_explicit_remote_override(
    sample_config, monkeypatch
):
    from blueprint_validation.preflight import check_openvla_local_checkpoint_requirement

    monkeypatch.setenv("BLUEPRINT_ALLOW_REMOTE_OPENVLA_MODEL", "1")
    sample_config.eval_policy.mode = "claim"
    sample_config.eval_policy.headline_scope = "dual"

    result = check_openvla_local_checkpoint_requirement(sample_config, "openvla_oft")
    assert result.passed is True
    assert "explicitly allowed" in result.detail.lower()


def test_preflight_wm_only_defers_policy_pipeline(sample_config, monkeypatch):
    import blueprint_validation.preflight as preflight

    _patch_preflight_fast(monkeypatch, preflight)
    sample_config.eval_policy.mode = "claim"
    sample_config.eval_policy.headline_scope = "wm_only"
    sample_config.eval_policy.rollout_driver = "scripted"
    sample_config.finetune.eval_world_experiment = "cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320"
    sample_config.policy_finetune.enabled = True

    checks = preflight.run_preflight(sample_config)
    by_name = {c.name: c for c in checks}
    assert by_name["wm_only:rollout_driver"].passed is True
    assert by_name["wm_only:world_model_action_dim_resolved"].passed is True
    assert by_name["policy_pipeline:deferred"].passed is True
    assert "policy_finetune:openvla_repo" not in by_name


def test_preflight_claim_dual_keeps_action_contract_checks(sample_config, monkeypatch):
    import blueprint_validation.preflight as preflight

    _patch_preflight_fast(monkeypatch, preflight)
    sample_config.eval_policy.mode = "claim"
    sample_config.eval_policy.headline_scope = "dual"
    sample_config.eval_policy.required_action_dim = 7
    sample_config.policy_adapter.openvla.policy_action_dim = 7
    sample_config.finetune.eval_world_experiment = "cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320"

    checks = preflight.run_preflight(sample_config)
    by_name = {c.name: c for c in checks}
    assert by_name["claim:policy_action_dim"].passed is True
    assert by_name["claim:world_model_action_dim"].passed is True


def test_preflight_dreamzero_checks_action_dim(sample_config, monkeypatch):
    import blueprint_validation.preflight as preflight

    _patch_preflight_fast(monkeypatch, preflight)
    sample_config.policy_adapter.name = "dreamzero"
    sample_config.policy_adapter.dreamzero.policy_action_dim = 0
    sample_config.eval_policy.mode = "research"
    sample_config.eval_policy.headline_scope = "dual"

    checks = preflight.run_preflight(sample_config)
    by_name = {c.name: c for c in checks}
    assert by_name["policy_adapter:dreamzero:action_dim"].passed is False


def test_preflight_dreamzero_checks_inference_import(sample_config, monkeypatch):
    import blueprint_validation.preflight as preflight
    from blueprint_validation.common import PreflightCheck

    _patch_preflight_fast(monkeypatch, preflight)
    sample_config.policy_adapter.name = "dreamzero"
    sample_config.eval_policy.mode = "research"
    sample_config.eval_policy.headline_scope = "dual"

    def _import_check(module_name, extra_path, name):
        if name == "policy_adapter:dreamzero:inference_import":
            return PreflightCheck(name=name, passed=False, detail="cannot import")
        return PreflightCheck(name=name, passed=True, detail="ok")

    monkeypatch.setattr(preflight, "check_python_import_from_path", _import_check)

    checks = preflight.run_preflight(sample_config)
    by_name = {c.name: c for c in checks}
    assert by_name["policy_adapter:dreamzero:inference_import"].passed is False


def test_preflight_dreamzero_runtime_checks_run_without_policy_finetune(sample_config, monkeypatch):
    import blueprint_validation.preflight as preflight
    from blueprint_validation.common import PreflightCheck

    _patch_preflight_fast(monkeypatch, preflight)
    sample_config.policy_adapter.name = "dreamzero"
    sample_config.policy_finetune.enabled = False
    sample_config.eval_policy.mode = "research"
    sample_config.eval_policy.headline_scope = "dual"

    monkeypatch.setattr(
        preflight,
        "check_path_exists",
        lambda path, name: PreflightCheck(name=name, passed=True, detail=str(path)),
    )

    checks = preflight.run_preflight(sample_config)
    by_name = {c.name: c for c in checks}
    assert by_name["policy_adapter:dreamzero:repo_path"].passed is True
    assert by_name["policy_adapter:dreamzero:checkpoint_path"].passed is True


def test_preflight_external_interaction_manifest_fails_on_invalid_schema(sample_config, tmp_path):
    from blueprint_validation.common import write_json
    import blueprint_validation.preflight as preflight

    invalid_manifest = tmp_path / "bad_external_manifest.json"
    write_json({"clips": [{"video_path": ""}]}, invalid_manifest)

    sample_config.external_interaction.enabled = True
    sample_config.external_interaction.manifest_path = invalid_manifest

    result = preflight.check_external_interaction_manifest(sample_config)
    assert result.passed is False
    assert "invalid stage1_source manifest" in result.detail.lower()


def test_preflight_external_interaction_manifest_autoskips_when_no_manifest(sample_config):
    import blueprint_validation.preflight as preflight

    sample_config.external_interaction.enabled = True
    sample_config.external_interaction.manifest_path = None
    result = preflight.check_external_interaction_manifest(sample_config)
    assert result.passed is True
    assert "auto-skip" in result.detail.lower()


def test_preflight_external_interaction_manifest_passes_with_valid_schema(sample_config, tmp_path):
    from blueprint_validation.common import write_json
    import blueprint_validation.preflight as preflight

    video = tmp_path / "external.mp4"
    video.write_bytes(b"video")
    manifest = tmp_path / "external_manifest.json"
    write_json({"clips": [{"clip_name": "ext_000", "video_path": str(video)}]}, manifest)

    sample_config.external_interaction.enabled = True
    sample_config.external_interaction.manifest_path = manifest

    result = preflight.check_external_interaction_manifest(sample_config)
    assert result.passed is True
    assert "validated stage1_source manifest" in result.detail.lower()


def test_preflight_external_rollout_manifest_autoskips_when_no_manifest(sample_config):
    import blueprint_validation.preflight as preflight

    sample_config.external_rollouts.enabled = True
    sample_config.external_rollouts.manifest_path = None
    result = preflight.check_external_rollout_manifest(sample_config)
    assert result.passed is True
    assert "auto-skip" in result.detail.lower()


def test_preflight_warns_when_external_rollouts_request_wm_ingest(sample_config):
    import blueprint_validation.preflight as preflight

    sample_config.external_rollouts.enabled = True
    sample_config.external_rollouts.mode = "wm_and_policy"

    result = preflight.check_external_rollouts_wm_ingest_notice(sample_config)
    assert result.passed is True
    assert "policy-training datasets only" in result.detail
    assert "external_interaction" in result.detail


def test_preflight_dreamzero_training_required_when_effective_policy_finetune_enabled(
    sample_config, monkeypatch
):
    import blueprint_validation.preflight as preflight

    _patch_preflight_fast(monkeypatch, preflight)
    sample_config.policy_adapter.name = "dreamzero"
    sample_config.policy_adapter.dreamzero.allow_training = False
    sample_config.policy_finetune.enabled = True
    sample_config.eval_policy.mode = "research"
    sample_config.eval_policy.headline_scope = "dual"

    checks = preflight.run_preflight(sample_config)
    by_name = {c.name: c for c in checks}
    assert by_name["policy_adapter:dreamzero:allow_training"].passed is False
    assert (
        "requires policy training"
        in by_name["policy_adapter:dreamzero:allow_training"].detail.lower()
    )


def test_preflight_dreamzero_training_required_when_action_boost_auto_enables_policy_finetune(
    sample_config, monkeypatch
):
    import blueprint_validation.preflight as preflight

    _patch_preflight_fast(monkeypatch, preflight)
    sample_config.policy_adapter.name = "dreamzero"
    sample_config.policy_adapter.dreamzero.allow_training = False
    sample_config.policy_finetune.enabled = False
    sample_config.action_boost.enabled = True
    sample_config.action_boost.auto_enable_policy_finetune = True
    sample_config.eval_policy.mode = "research"
    sample_config.eval_policy.headline_scope = "dual"

    checks = preflight.run_preflight(sample_config)
    by_name = {c.name: c for c in checks}
    assert by_name["action_boost:policy_finetune_override"].passed is True
    assert by_name["policy_adapter:dreamzero:allow_training"].passed is False


def test_check_dreamzero_runtime_contract_passes_for_supported_class(tmp_path):
    import blueprint_validation.preflight as preflight

    repo = tmp_path / "dz_repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "dummy_runtime.py").write_text(
        """
class Runtime:
    @classmethod
    def from_pretrained(cls, model_id, device=None):
        return cls()

    def predict_action(self, *, frames, prompt):
        return {"action": [0.0]}
""".strip()
    )

    result = preflight.check_dreamzero_runtime_contract(
        repo_path=repo,
        inference_module="dummy_runtime",
        inference_class="Runtime",
    )
    assert result.passed is True


def test_check_dreamzero_runtime_contract_fails_without_inference_entrypoint(tmp_path):
    import blueprint_validation.preflight as preflight

    repo = tmp_path / "dz_repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "bad_runtime.py").write_text(
        """
class Runtime:
    def __init__(self, model_path=None):
        self.model_path = model_path
""".strip()
    )

    result = preflight.check_dreamzero_runtime_contract(
        repo_path=repo,
        inference_module="bad_runtime",
        inference_class="Runtime",
    )
    assert result.passed is False
    assert "missing inference entrypoints" in result.detail.lower()


def test_run_preflight_audit_profile_marks_runtime_requirements_advisory(
    sample_config, monkeypatch
):
    import blueprint_validation.preflight as preflight
    from blueprint_validation.common import PreflightCheck

    _patch_preflight_fast(monkeypatch, preflight)
    observed = {}

    def _task_hints(_facility_id, _facility, *, deep_scan=True):
        observed["deep_scan"] = deep_scan
        return PreflightCheck(name="task_hints_bootstrap", passed=True, detail="ok")

    monkeypatch.setattr(preflight, "check_task_hints_bootstrap_readiness", _task_hints)
    for attr in [
        "check_gpu",
        "check_model_weights",
        "check_hf_auth",
        "check_cosmos_predict_tokenizer_access",
        "check_openvla_local_checkpoint_requirement",
        "check_api_key",
        "check_api_key_for_scope",
        "check_python_import_from_path",
    ]:
        monkeypatch.setattr(
            preflight,
            attr,
            lambda *a, _attr=attr, **k: (_ for _ in ()).throw(
                AssertionError(f"{_attr} should be skipped in audit profile")
            ),
        )

    checks = preflight.run_preflight(sample_config, profile="audit")
    by_name = {c.name: c for c in checks}
    assert observed["deep_scan"] is False
    assert by_name["gpu"].passed is True
    assert "Audit profile" in by_name["gpu"].detail
    assert by_name["weights:DreamDojo"].passed is True
    assert by_name["policy:local_checkpoint_requirement"].passed is True
    assert by_name["hf_auth"].passed is True
    assert by_name["import:cosmos_predict2"].passed is True
    assert by_name["api_key:eval_spatial"].passed is True


def test_run_preflight_audit_profile_skips_runtime_repo_and_policy_contracts(
    sample_config, monkeypatch
):
    import blueprint_validation.preflight as preflight

    _patch_preflight_fast(monkeypatch, preflight)

    for attr in [
        "check_path_exists",
        "check_path_exists_under",
        "check_cosmos_wrapper_contract",
        "check_dreamdojo_contract",
        "check_openvla_finetune_contract",
        "check_openvla_dataset_registry",
    ]:
        monkeypatch.setattr(
            preflight,
            attr,
            lambda *a, _attr=attr, **k: (_ for _ in ()).throw(
                AssertionError(f"{_attr} should be skipped in audit profile")
            ),
        )

    checks = preflight.run_preflight(sample_config, profile="audit")
    by_name = {c.name: c for c in checks}
    assert by_name["repo:cosmos_transfer"].passed is True
    assert by_name["repo:dreamdojo"].passed is True
    assert by_name["finetune:dreamdojo_contract"].passed is True
    assert by_name["policy_finetune:openvla_repo"].passed is True
    assert by_name["policy_finetune:dataset_registry"].passed is True


def test_run_preflight_runtime_local_marks_cloud_guards_advisory(sample_config, monkeypatch):
    import blueprint_validation.preflight as preflight

    _patch_preflight_fast(monkeypatch, preflight)
    monkeypatch.setattr(
        preflight,
        "check_cloud_budget_enforcement",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("cloud budget enforcement should be advisory in runtime-local")
        ),
    )
    monkeypatch.setattr(
        preflight,
        "check_cloud_shutdown_enforcement",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("cloud shutdown enforcement should be advisory in runtime-local")
        ),
    )

    checks = preflight.run_preflight(sample_config, profile="runtime-local")
    by_name = {c.name: c for c in checks}
    assert by_name["cloud:budget_enforcement"].passed is True
    assert "runtime-local profile" in by_name["cloud:budget_enforcement"].detail
    assert by_name["cloud:auto_shutdown_enforcement"].passed is True


def test_preflight_module_import_is_lazy_for_heavy_helpers(monkeypatch):
    for name in [
        "blueprint_validation.preflight",
        "blueprint_validation.warmup",
        "blueprint_validation.enrichment.cosmos_runner",
        "blueprint_validation.training.dreamdojo_finetune",
    ]:
        sys.modules.pop(name, None)

    def _blocking_module(module_name: str) -> types.ModuleType:
        module = types.ModuleType(module_name)

        def __getattr__(attr):
            raise AssertionError(f"{module_name} accessed during preflight import via {attr}")

        module.__getattr__ = __getattr__  # type: ignore[attr-defined]
        return module

    monkeypatch.setitem(
        sys.modules,
        "blueprint_validation.warmup",
        _blocking_module("blueprint_validation.warmup"),
    )
    monkeypatch.setitem(
        sys.modules,
        "blueprint_validation.enrichment.cosmos_runner",
        _blocking_module("blueprint_validation.enrichment.cosmos_runner"),
    )
    monkeypatch.setitem(
        sys.modules,
        "blueprint_validation.training.dreamdojo_finetune",
        _blocking_module("blueprint_validation.training.dreamdojo_finetune"),
    )

    module = importlib.import_module("blueprint_validation.preflight")
    assert hasattr(module, "run_preflight")


def test_check_qualified_opportunity_handoff_gate_passes_for_ready_target(sample_config):
    import blueprint_validation.preflight as preflight

    facility = sample_config.facilities["test_facility"]
    facility.opportunity_handoff_path = Path("/tmp/opportunity_handoff.json")
    facility.qualification_state = "ready"
    facility.downstream_evaluation_eligibility = True

    result = preflight.check_qualified_opportunity_handoff_gate("test_facility", facility)
    assert result.passed is True
    assert "accepted" in result.detail.lower()


def test_check_qualified_opportunity_handoff_gate_fails_for_non_ready_target(sample_config):
    import blueprint_validation.preflight as preflight

    facility = sample_config.facilities["test_facility"]
    facility.opportunity_handoff_path = Path("/tmp/opportunity_handoff.json")
    facility.qualification_state = "risky"
    facility.downstream_evaluation_eligibility = False

    result = preflight.check_qualified_opportunity_handoff_gate("test_facility", facility)
    assert result.passed is False
    assert "not ready" in result.detail.lower()


def test_check_legacy_intake_notice_is_advisory_for_legacy_targets(sample_config):
    import blueprint_validation.preflight as preflight

    facility = sample_config.facilities["test_facility"]
    facility.opportunity_handoff_path = None

    result = preflight.check_legacy_intake_notice("test_facility", facility)
    assert result.passed is True
    assert "legacy direct geometry intake" in result.detail.lower()


def test_run_preflight_includes_qualified_opportunity_gate(sample_config, monkeypatch):
    import blueprint_validation.preflight as preflight

    _patch_preflight_fast(monkeypatch, preflight)
    facility = sample_config.facilities["test_facility"]
    facility.opportunity_handoff_path = Path("/tmp/opportunity_handoff.json")
    facility.qualification_state = "ready"
    facility.downstream_evaluation_eligibility = True

    checks = preflight.run_preflight(sample_config, profile="audit")
    by_name = {c.name: c for c in checks}
    assert by_name["qualified_opportunity:eligibility:test_facility"].passed is True
    assert by_name["intake:legacy_direct_notice:test_facility"].passed is True


def test_task_hints_bootstrap_readiness_passes_with_labels_only_bundle(sample_config):
    import blueprint_validation.preflight as preflight

    facility = sample_config.facilities["test_facility"]
    bundle_root = facility.ply_path.parent / "advanced_geometry"
    bundle_root.mkdir()
    (bundle_root / "labels.json").write_text("{}")
    (bundle_root / "structure.json").write_text("{}")
    facility.geometry_bundle_path = bundle_root
    facility.labels_path = bundle_root / "labels.json"
    facility.structure_path = bundle_root / "structure.json"
    facility.task_hints_path = None

    result = preflight.check_task_hints_bootstrap_readiness("test_facility", facility, deep_scan=False)
    assert result.passed is True
    assert "metadata present" in result.detail.lower()


def test_run_preflight_accepts_capture_pipeline_handoff_with_inferred_bundle(tmp_path, monkeypatch):
    import yaml

    import blueprint_validation.preflight as preflight
    from blueprint_validation.config import load_config

    real_check_facility_ply = preflight.check_facility_ply
    real_check_task_hints_bootstrap_readiness = preflight.check_task_hints_bootstrap_readiness
    _patch_preflight_fast(monkeypatch, preflight)
    monkeypatch.setattr(preflight, "check_facility_ply", real_check_facility_ply)
    monkeypatch.setattr(
        preflight,
        "check_task_hints_bootstrap_readiness",
        real_check_task_hints_bootstrap_readiness,
    )

    pipeline_dir = tmp_path / "pipeline"
    bundle_root = pipeline_dir / "advanced_geometry"
    bundle_root.mkdir(parents=True)
    (bundle_root / "3dgs_compressed.ply").write_text("ply\n")
    (bundle_root / "labels.json").write_text("{}")
    (bundle_root / "structure.json").write_text("{}")

    handoff_path = pipeline_dir / "opportunity_handoff.json"
    handoff_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene_demo",
                "capture_id": "capture_demo",
                "readiness_state": "ready",
                "match_ready": True,
            }
        )
    )

    config_path = tmp_path / "qualified.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "schema_version": "v1",
                "qualified_opportunities": {
                    "opp_capture": {"opportunity_handoff_path": str(handoff_path)}
                },
            }
        )
    )

    config = load_config(config_path)
    checks = preflight.run_preflight(config, profile="audit")
    by_name = {c.name: c for c in checks}

    assert by_name["qualified_opportunity:eligibility:opp_capture"].passed is True
    assert by_name["ply:opp_capture"].passed is True
    assert by_name["task_hints_bootstrap:opp_capture"].passed is True
    assert "metadata present" in by_name["task_hints_bootstrap:opp_capture"].detail.lower()


def test_run_preflight_fails_capture_pipeline_handoff_gate_when_not_match_ready(tmp_path, monkeypatch):
    import yaml

    import blueprint_validation.preflight as preflight
    from blueprint_validation.config import load_config

    _patch_preflight_fast(monkeypatch, preflight)

    pipeline_dir = tmp_path / "pipeline"
    bundle_root = pipeline_dir / "advanced_geometry"
    bundle_root.mkdir(parents=True)
    (bundle_root / "3dgs_compressed.ply").write_text("ply\n")

    handoff_path = pipeline_dir / "opportunity_handoff.json"
    handoff_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene_demo",
                "capture_id": "capture_demo",
                "readiness_state": "risky",
                "match_ready": False,
            }
        )
    )

    config_path = tmp_path / "qualified.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "schema_version": "v1",
                "qualified_opportunities": {
                    "opp_capture": {"opportunity_handoff_path": str(handoff_path)}
                },
            }
        )
    )

    config = load_config(config_path)
    checks = preflight.run_preflight(config, profile="audit")
    by_name = {c.name: c for c in checks}

    assert by_name["qualified_opportunity:eligibility:opp_capture"].passed is False
    assert "not ready" in by_name["qualified_opportunity:eligibility:opp_capture"].detail.lower()


def test_run_preflight_reports_missing_inferred_bundle_ply_path(tmp_path, monkeypatch):
    import yaml

    import blueprint_validation.preflight as preflight
    from blueprint_validation.config import load_config

    real_check_facility_ply = preflight.check_facility_ply
    real_check_task_hints_bootstrap_readiness = preflight.check_task_hints_bootstrap_readiness
    _patch_preflight_fast(monkeypatch, preflight)
    monkeypatch.setattr(preflight, "check_facility_ply", real_check_facility_ply)
    monkeypatch.setattr(
        preflight,
        "check_task_hints_bootstrap_readiness",
        real_check_task_hints_bootstrap_readiness,
    )

    pipeline_dir = tmp_path / "pipeline"
    bundle_root = pipeline_dir / "advanced_geometry"
    bundle_root.mkdir(parents=True)
    (bundle_root / "labels.json").write_text("{}")
    (bundle_root / "structure.json").write_text("{}")

    handoff_path = pipeline_dir / "opportunity_handoff.json"
    handoff_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene_demo",
                "capture_id": "capture_demo",
                "readiness_state": "ready",
                "match_ready": True,
            }
        )
    )

    config_path = tmp_path / "qualified.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "schema_version": "v1",
                "qualified_opportunities": {
                    "opp_capture": {"opportunity_handoff_path": str(handoff_path)}
                },
            }
        )
    )

    config = load_config(config_path)
    checks = preflight.run_preflight(config, profile="audit")
    by_name = {c.name: c for c in checks}

    assert by_name["ply:opp_capture"].passed is False
    assert str(bundle_root / "3dgs_compressed.ply") in by_name["ply:opp_capture"].detail
    assert by_name["task_hints_bootstrap:opp_capture"].passed is True
