"""Preflight checks for GPU, dependencies, and model weights."""

from __future__ import annotations

import ast
import inspect
import os
import re
import shutil
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import Callable, List, Literal, cast

from .common import PreflightCheck, get_logger
from .config import ValidationConfig
from .evaluation.claim_benchmark import (
    claim_benchmark_alignment_failures,
    claim_benchmark_strictness_failures,
    load_pinned_claim_benchmark,
)
from .teleop.contracts import TeleopManifestError, load_and_validate_teleop_manifest
from .validation import ManifestValidationError, load_and_validate_manifest

logger = get_logger("preflight")

PreflightProfile = Literal["audit", "runtime_local", "runtime_cloud"]

_PI05_REQUIRED_TRAIN_FLAGS = {
    "--exp_name",
    "--run_root_dir",
    "--dataset_root",
    "--dataset_name",
    "--base_model",
    "--batch_size",
    "--learning_rate",
    "--max_steps",
}

_PI05_REQUIRED_NORM_STATS_FLAGS = {
    "--dataset_root",
    "--dataset_name",
    "--profile",
}

_COSMOS_PREDICT_REPO = "nvidia/Cosmos-Predict2.5-2B"
_COSMOS_PREDICT_REVISION = "6787e176dce74a101d922174a95dba29fa5f0c55"
_COSMOS_PREDICT_TOKENIZER_FILENAME = "tokenizer.pth"
_PROFILE_ALIASES = {
    "audit": "audit",
    "runtime_local": "runtime_local",
    "runtime-local": "runtime_local",
    "runtime_cloud": "runtime_cloud",
    "runtime-cloud": "runtime_cloud",
}


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def normalize_preflight_profile(profile: str | None) -> PreflightProfile:
    token = str(profile or "runtime_local").strip().lower()
    normalized = _PROFILE_ALIASES.get(token)
    if normalized is None:
        allowed = ", ".join(sorted(_PROFILE_ALIASES))
        raise ValueError(f"Unsupported preflight profile '{profile}'. Expected one of: {allowed}")
    return cast(PreflightProfile, normalized)


def _advisory_check(name: str, detail: str) -> PreflightCheck:
    return PreflightCheck(name=name, passed=True, detail=detail)


def _advisory_detail(profile: PreflightProfile, detail: str) -> str:
    label = "Audit profile" if profile == "audit" else "runtime-local profile"
    return f"{label}: {detail}"


def _append_profiled_check(
    checks: list[PreflightCheck],
    *,
    enforce: bool,
    advisory_name: str,
    advisory_detail: str,
    factory: Callable[[], PreflightCheck],
) -> None:
    if enforce:
        checks.append(factory())
        return
    checks.append(_advisory_check(advisory_name, advisory_detail))


def _append_profiled_checks(
    checks: list[PreflightCheck],
    *,
    enforce: bool,
    specs: list[tuple[str, str, Callable[[], PreflightCheck]]],
) -> None:
    for advisory_name, advisory_detail, factory in specs:
        _append_profiled_check(
            checks,
            enforce=enforce,
            advisory_name=advisory_name,
            advisory_detail=advisory_detail,
            factory=factory,
        )


def _load_build_controlnet_spec():
    from .enrichment.cosmos_runner import build_controlnet_spec

    return build_controlnet_spec


def _load_resolve_dreamdojo_experiment_name():
    from .training.dreamdojo_finetune import resolve_dreamdojo_experiment_name

    return resolve_dreamdojo_experiment_name


def _load_ply_means_and_colors_numpy():
    from .warmup import load_ply_means_and_colors_numpy

    return load_ply_means_and_colors_numpy


def check_gpu() -> PreflightCheck:
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_memory = getattr(props, "total_memory", None)
            if total_memory is None:
                total_memory = getattr(props, "total_mem", None)
            if total_memory is not None:
                vram_gb = total_memory / (1024**3)
                detail = f"{gpu_name} ({vram_gb:.0f}GB VRAM)"
            else:
                detail = f"{gpu_name} (VRAM unknown)"
            return PreflightCheck(
                name="gpu",
                passed=True,
                detail=detail,
            )
        return PreflightCheck(name="gpu", passed=False, detail="No CUDA GPU detected")
    except ImportError:
        return PreflightCheck(name="gpu", passed=False, detail="PyTorch not installed")
    except Exception as e:  # pragma: no cover - defensive against driver/API mismatches
        return PreflightCheck(name="gpu", passed=False, detail=f"GPU check failed: {e}")


def check_dependency(module_name: str, package_name: str = "") -> PreflightCheck:
    pkg = package_name or module_name
    try:
        import_module(module_name)
        return PreflightCheck(name=f"dep:{pkg}", passed=True)
    except ImportError:
        return PreflightCheck(
            name=f"dep:{pkg}", passed=False, detail=f"Cannot import {module_name}"
        )


def check_model_weights(path: Path, name: str) -> PreflightCheck:
    if path.exists() and any(path.iterdir()):
        return PreflightCheck(name=f"weights:{name}", passed=True, detail=str(path))
    return PreflightCheck(
        name=f"weights:{name}",
        passed=False,
        detail=f"Not found at {path}. Run: bash scripts/download_models.sh",
    )


def check_api_key(env_var: str) -> PreflightCheck:
    val = os.environ.get(env_var, "")
    if val:
        return PreflightCheck(name=f"api_key:{env_var}", passed=True, detail="Set")
    return PreflightCheck(
        name=f"api_key:{env_var}",
        passed=False,
        detail=f"Environment variable {env_var} not set",
    )


def check_external_tool(cmd: str) -> PreflightCheck:
    if shutil.which(cmd):
        return PreflightCheck(name=f"tool:{cmd}", passed=True)
    return PreflightCheck(name=f"tool:{cmd}", passed=False, detail=f"{cmd} not found in PATH")


def check_path_exists(path: Path, name: str) -> PreflightCheck:
    if path.exists():
        return PreflightCheck(name=name, passed=True, detail=str(path))
    return PreflightCheck(name=name, passed=False, detail=f"Not found at {path}")


def check_path_exists_under(root: Path, rel_path: str, name: str) -> PreflightCheck:
    target = root / rel_path
    if target.exists():
        return PreflightCheck(name=name, passed=True, detail=str(target))
    return PreflightCheck(name=name, passed=False, detail=f"Missing required path: {target}")


def _canonical_policy_adapter_name(name: str) -> str:
    key = (name or "").strip().lower()
    if key in {"openvla_oft", "openvla-oft", "oft", "openvla", "open-vla"}:
        return "openvla_oft"
    if key in {"pi05", "pi0.5", "openpi"}:
        return "pi05"
    if key in {"dreamzero", "dream-zero", "dz"}:
        return "dreamzero"
    return key


def _effective_headline_scope(config: ValidationConfig) -> str:
    """Resolve headline scope after action-boost runtime overrides."""
    scope = (getattr(config.eval_policy, "headline_scope", "wm_only") or "wm_only").strip().lower()
    action_boost = getattr(config, "action_boost", None)
    if (
        scope == "wm_only"
        and bool(getattr(action_boost, "enabled", False))
        and bool(getattr(action_boost, "auto_switch_headline_scope_to_dual", True))
    ):
        return "wm_uplift"
    return scope


def _effective_policy_finetune_enabled(config: ValidationConfig) -> bool:
    """Resolve policy_finetune.enabled after action-boost runtime overrides."""
    if bool(config.policy_finetune.enabled):
        return True
    action_boost = getattr(config, "action_boost", None)
    return bool(getattr(action_boost, "enabled", False)) and bool(
        getattr(action_boost, "auto_enable_policy_finetune", True)
    )


def _resolve_eval_world_action_dim(config: ValidationConfig) -> int | None:
    """Resolve expected DreamDojo world-model action dim from configured experiment hints."""
    token = (
        config.finetune.eval_world_experiment or config.finetune.experiment_config or ""
    ).strip()
    if not token:
        try:
            token = _load_resolve_dreamdojo_experiment_name()(
                config.finetune.dreamdojo_repo,
                None,
            )
        except Exception:
            return None

    # Explicit experiment names (current supported set in openvla_runner mappings).
    if token.lower().startswith("cosmos_predict2"):
        mapping = {
            "cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame": 384,
            "cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320": 7,
            "cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_480_640_": 7,
        }
        return mapping.get(token)

    # Resolve local yaml config and parse action_dim.
    maybe = Path(token)
    candidate: Path
    if maybe.is_absolute() or "/" in token or token.endswith(".yaml"):
        candidate = (
            maybe if maybe.is_absolute() else (config.finetune.dreamdojo_repo / "configs" / maybe)
        )
        if candidate.suffix != ".yaml":
            yaml_candidate = candidate.with_suffix(".yaml")
            if yaml_candidate.exists():
                candidate = yaml_candidate
    else:
        stem = token.lower()
        if stem.startswith("dreamdojo_"):
            stem = stem[len("dreamdojo_") :]
        candidate = config.finetune.dreamdojo_repo / "configs" / f"{Path(stem).stem}.yaml"

    if not candidate.exists():
        return None
    try:
        text = candidate.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^\s*action_dim\s*:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    if not match:
        return None
    return int(match.group(1))


def _append_runtime_repo_contract_checks(
    checks: list[PreflightCheck],
    *,
    config: ValidationConfig,
    normalized_profile: PreflightProfile,
    enforce_runtime_requirements: bool,
) -> None:
    detail = _advisory_detail(
        normalized_profile,
        "Local runtime repos and wrapper contracts are enforced only for runtime execution.",
    )
    _append_profiled_checks(
        checks,
        enforce=enforce_runtime_requirements,
        specs=[
            (
                "repo:cosmos_transfer",
                detail,
                lambda: check_path_exists(config.enrich.cosmos_repo, "repo:cosmos_transfer"),
            ),
            (
                "repo:cosmos_transfer:inference_script",
                detail,
                lambda: check_path_exists_under(
                    config.enrich.cosmos_repo,
                    "examples/inference.py",
                    "repo:cosmos_transfer:inference_script",
                ),
            ),
            (
                "enrich:cosmos_wrapper_contract",
                detail,
                lambda: check_cosmos_wrapper_contract(config.enrich.cosmos_repo),
            ),
            (
                "repo:dreamdojo",
                detail,
                lambda: check_path_exists(config.finetune.dreamdojo_repo, "repo:dreamdojo"),
            ),
            (
                "repo:dreamdojo:train_entrypoint",
                detail,
                lambda: check_path_exists_under(
                    config.finetune.dreamdojo_repo,
                    "scripts/train.py",
                    "repo:dreamdojo:train_entrypoint",
                ),
            ),
            (
                "repo:dreamdojo:configs",
                detail,
                lambda: check_path_exists_under(
                    config.finetune.dreamdojo_repo,
                    "configs",
                    "repo:dreamdojo:configs",
                ),
            ),
            (
                "finetune:dreamdojo_contract",
                detail,
                lambda: check_dreamdojo_contract(
                    config.finetune.dreamdojo_repo,
                    config.finetune.experiment_config,
                ),
            ),
        ],
    )


def _append_openvla_policy_runtime_checks(
    checks: list[PreflightCheck],
    *,
    config: ValidationConfig,
    normalized_profile: PreflightProfile,
    enforce_runtime_requirements: bool,
) -> None:
    openvla_backend = config.policy_adapter.openvla
    script_path = Path(openvla_backend.finetune_script)
    if not script_path.is_absolute():
        script_path = openvla_backend.openvla_repo / script_path
    dataset_names = {config.policy_finetune.dataset_name}
    if config.rollout_dataset.enabled:
        dataset_names.add(config.rollout_dataset.baseline_dataset_name)
        dataset_names.add(config.rollout_dataset.adapted_dataset_name)

    detail = _advisory_detail(
        normalized_profile,
        "Policy-training runtime repos and scripts are enforced only for runtime execution.",
    )
    _append_profiled_checks(
        checks,
        enforce=enforce_runtime_requirements,
        specs=[
            ("tool:torchrun", detail, lambda: check_external_tool("torchrun")),
            (
                "policy_finetune:openvla_repo",
                detail,
                lambda: check_path_exists(
                    openvla_backend.openvla_repo,
                    "policy_finetune:openvla_repo",
                ),
            ),
            (
                "policy_finetune:finetune_script",
                detail,
                lambda: check_path_exists(script_path, "policy_finetune:finetune_script"),
            ),
            (
                "policy_finetune:wrapper_contract",
                detail,
                lambda: check_openvla_finetune_contract(
                    openvla_backend.openvla_repo,
                    openvla_backend.finetune_script,
                ),
            ),
            (
                "policy_finetune:dataset_registry",
                detail,
                lambda: check_openvla_dataset_registry(
                    openvla_backend.openvla_repo,
                    dataset_names,
                ),
            ),
        ],
    )


def _append_pi05_policy_runtime_checks(
    checks: list[PreflightCheck],
    *,
    config: ValidationConfig,
    normalized_profile: PreflightProfile,
    enforce_runtime_requirements: bool,
) -> None:
    pi05 = config.policy_adapter.pi05
    train_script = Path(pi05.train_script)
    norm_script = Path(pi05.norm_stats_script)
    if not train_script.is_absolute():
        train_script = pi05.openpi_repo / train_script
    if not norm_script.is_absolute():
        norm_script = pi05.openpi_repo / norm_script
    detail = _advisory_detail(
        normalized_profile,
        "Policy-training runtime repos and scripts are enforced only for runtime execution.",
    )
    _append_profiled_checks(
        checks,
        enforce=enforce_runtime_requirements,
        specs=[
            (
                "policy_adapter:pi05:openpi_repo",
                detail,
                lambda: check_path_exists(
                    pi05.openpi_repo,
                    "policy_adapter:pi05:openpi_repo",
                ),
            ),
            (
                "policy_adapter:pi05:train_script",
                detail,
                lambda: check_path_exists(train_script, "policy_adapter:pi05:train_script"),
            ),
            (
                "policy_adapter:pi05:norm_stats_script",
                detail,
                lambda: check_path_exists(norm_script, "policy_adapter:pi05:norm_stats_script"),
            ),
            (
                "policy_adapter:pi05:train_contract",
                detail,
                lambda: check_pi05_train_contract(pi05.openpi_repo, pi05.train_script),
            ),
            (
                "policy_adapter:pi05:norm_stats_contract",
                detail,
                lambda: check_pi05_norm_stats_contract(pi05.openpi_repo, pi05.norm_stats_script),
            ),
            (
                "import:openpi",
                detail,
                lambda: check_python_import_from_path(
                    "openpi",
                    pi05.openpi_repo / "src",
                    "import:openpi",
                ),
            ),
        ],
    )


def _append_dreamzero_policy_runtime_checks(
    checks: list[PreflightCheck],
    *,
    config: ValidationConfig,
    normalized_profile: PreflightProfile,
    enforce_runtime_requirements: bool,
) -> None:
    dz = config.policy_adapter.dreamzero
    train_script = Path(dz.train_script)
    if not train_script.is_absolute():
        train_script = dz.repo_path / train_script
    detail = _advisory_detail(
        normalized_profile,
        "Policy-training runtime repos and scripts are enforced only for runtime execution.",
    )
    _append_profiled_checks(
        checks,
        enforce=enforce_runtime_requirements,
        specs=[
            (
                "policy_adapter:dreamzero:train_script",
                detail,
                lambda: check_path_exists(train_script, "policy_adapter:dreamzero:train_script"),
            ),
            (
                "policy_adapter:dreamzero:train_contract",
                detail,
                lambda: check_dreamzero_train_contract(dz.repo_path, dz.train_script),
            ),
        ],
    )


def _append_dreamzero_adapter_runtime_checks(
    checks: list[PreflightCheck],
    *,
    config: ValidationConfig,
    normalized_profile: PreflightProfile,
    enforce_runtime_requirements: bool,
) -> None:
    dz = config.policy_adapter.dreamzero
    detail = _advisory_detail(
        normalized_profile,
        "Adapter runtime repos and imports are enforced only for runtime execution.",
    )
    _append_profiled_checks(
        checks,
        enforce=enforce_runtime_requirements,
        specs=[
            (
                "policy_adapter:dreamzero:repo_path",
                detail,
                lambda: check_path_exists(dz.repo_path, "policy_adapter:dreamzero:repo_path"),
            ),
            (
                "policy_adapter:dreamzero:checkpoint_path",
                detail,
                lambda: check_path_exists(
                    dz.checkpoint_path,
                    "policy_adapter:dreamzero:checkpoint_path",
                ),
            ),
            (
                "policy_adapter:dreamzero:inference_import",
                detail,
                lambda: check_python_import_from_path(
                    dz.inference_module,
                    dz.repo_path,
                    "policy_adapter:dreamzero:inference_import",
                ),
            ),
            (
                "policy_adapter:dreamzero:runtime_contract",
                detail,
                lambda: check_dreamzero_runtime_contract(
                    repo_path=dz.repo_path,
                    inference_module=dz.inference_module,
                    inference_class=dz.inference_class,
                ),
            ),
        ],
    )


def _claim_mode_checks(
    config: ValidationConfig,
    adapter_name: str,
    *,
    scope_override: str | None = None,
) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    mode = (config.eval_policy.mode or "").strip().lower()
    scope = (
        str(scope_override or getattr(config.eval_policy, "headline_scope", "wm_only") or "wm_only")
        .strip()
        .lower()
    )
    checks.append(
        PreflightCheck(
            name="eval_policy:mode",
            passed=mode in {"claim", "research"},
            detail=mode or "<unset>",
        )
    )
    checks.append(
        PreflightCheck(
            name="eval_policy:headline_scope",
            passed=scope in {"wm_only", "wm_uplift", "dual"},
            detail=scope or "<unset>",
        )
    )
    if mode != "claim":
        return checks

    if scope == "wm_only":
        rollout_driver = (
            (getattr(config.eval_policy, "rollout_driver", "scripted") or "").strip().lower()
        )
        checks.append(
            PreflightCheck(
                name="wm_only:rollout_driver",
                passed=rollout_driver in {"scripted", "both"},
                detail=rollout_driver or "<unset>",
            )
        )
        world_dim = _resolve_eval_world_action_dim(config)
        checks.append(
            PreflightCheck(
                name="wm_only:world_model_action_dim_resolved",
                passed=world_dim is not None,
                detail=(
                    f"world={world_dim}"
                    if world_dim is not None
                    else (
                        "Could not resolve world action_dim from finetune.eval_world_experiment/"
                        "finetune.experiment_config."
                    )
                ),
            )
        )
        return checks

    checks.append(
        PreflightCheck(
            name="claim:policy_adapter",
            passed=adapter_name == "openvla_oft",
            detail=(
                "Claim mode only supports openvla_oft."
                if adapter_name != "openvla_oft"
                else "openvla_oft"
            ),
        )
    )
    checks.append(
        PreflightCheck(
            name="claim:require_native_action_compat",
            passed=bool(config.eval_policy.require_native_action_compat),
            detail=str(config.eval_policy.require_native_action_compat),
        )
    )

    required_dim = int(config.eval_policy.required_action_dim)
    policy_dim = int(config.policy_adapter.openvla.policy_action_dim)
    checks.append(
        PreflightCheck(
            name="claim:policy_action_dim",
            passed=policy_dim == required_dim,
            detail=f"policy={policy_dim}, required={required_dim}",
        )
    )
    world_dim = _resolve_eval_world_action_dim(config)
    checks.append(
        PreflightCheck(
            name="claim:world_model_action_dim",
            passed=world_dim == required_dim,
            detail=(
                f"world={world_dim}, required={required_dim}"
                if world_dim is not None
                else (
                    "Could not resolve world action_dim from finetune.eval_world_experiment/"
                    "finetune.experiment_config."
                )
            ),
        )
    )
    checks.append(
        PreflightCheck(
            name="claim:rollout_dataset_dim_contract",
            passed=bool(config.rollout_dataset.require_consistent_action_dim),
            detail=str(config.rollout_dataset.require_consistent_action_dim),
        )
    )
    return checks


def check_policy_base_reference(model_name: str, checkpoint_path: Path) -> PreflightCheck:
    if checkpoint_path.exists() and checkpoint_path.is_dir() and any(checkpoint_path.iterdir()):
        return PreflightCheck(
            name="policy:base_checkpoint",
            passed=True,
            detail=str(checkpoint_path),
        )
    if checkpoint_path.exists() and checkpoint_path.is_file():
        return PreflightCheck(
            name="policy:base_checkpoint",
            passed=True,
            detail=str(checkpoint_path),
        )
    if model_name:
        return PreflightCheck(
            name="policy:base_model",
            passed=True,
            detail=f"Using model reference '{model_name}' (checkpoint path missing: {checkpoint_path})",
        )
    return PreflightCheck(
        name="policy:base_model",
        passed=False,
        detail="Neither eval_policy.model_name nor eval_policy.checkpoint_path is usable.",
    )


def _is_openvla_like_reference(model_name: str, checkpoint_path: Path) -> bool:
    model_text = (model_name or "").strip().lower()
    ckpt_text = str(checkpoint_path).strip().lower()
    return "openvla" in model_text or "openvla" in ckpt_text


def _resolve_policy_base_reference_for_adapter(
    config: ValidationConfig,
    adapter_name: str,
) -> tuple[str, Path]:
    if adapter_name == "openvla_oft":
        backend = config.policy_adapter.openvla
        model_name = str(backend.base_model_name or config.eval_policy.model_name or "").strip()
        checkpoint_path = backend.base_checkpoint_path or config.eval_policy.checkpoint_path
        return model_name, checkpoint_path
    if adapter_name == "dreamzero":
        backend = config.policy_adapter.dreamzero
        model_name = str(backend.base_model_name or "").strip()
        checkpoint_path = backend.checkpoint_path
        return model_name, checkpoint_path
    return config.eval_policy.model_name, config.eval_policy.checkpoint_path


def check_policy_base_reference_for_adapter(
    adapter_name: str,
    model_name: str,
    checkpoint_path: Path,
) -> PreflightCheck:
    if adapter_name not in {"pi05", "dreamzero"}:
        return check_policy_base_reference(model_name=model_name, checkpoint_path=checkpoint_path)

    ckpt_exists = checkpoint_path.exists() and (
        checkpoint_path.is_file() or (checkpoint_path.is_dir() and any(checkpoint_path.iterdir()))
    )
    if ckpt_exists:
        if _is_openvla_like_reference("", checkpoint_path):
            label = "dreamzero" if adapter_name == "dreamzero" else "pi05"
            return PreflightCheck(
                name="policy:base_reference",
                passed=False,
                detail=(
                    f"{label} adapter selected but configured checkpoint points to an OpenVLA-like "
                    f"artifact: {checkpoint_path}. Set a {label}-compatible checkpoint instead."
                ),
            )
        return PreflightCheck(
            name="policy:base_reference",
            passed=True,
            detail=f"{adapter_name} checkpoint reference: {checkpoint_path}",
        )

    if adapter_name == "dreamzero" and model_name:
        if _is_openvla_like_reference(model_name, Path("")):
            return PreflightCheck(
                name="policy:base_reference",
                passed=False,
                detail=(
                    "dreamzero adapter selected but configured model reference is OpenVLA-like: "
                    f"{model_name}. Set policy_adapter.dreamzero.base_model_name and/or "
                    "policy_adapter.dreamzero.checkpoint_path to DreamZero-compatible values."
                ),
            )
        return PreflightCheck(
            name="policy:base_reference",
            passed=True,
            detail=f"dreamzero model reference: {model_name}",
        )

    if model_name and not _is_openvla_like_reference(model_name, Path("")):
        return PreflightCheck(
            name="policy:base_reference",
            passed=True,
            detail=f"pi05 model reference: {model_name}",
        )

    if adapter_name == "dreamzero":
        return PreflightCheck(
            name="policy:base_reference",
            passed=False,
            detail=(
                "dreamzero adapter selected but no DreamZero-compatible base reference is configured. "
                "Set policy_adapter.dreamzero.base_model_name and/or "
                "policy_adapter.dreamzero.checkpoint_path."
            ),
        )

    return PreflightCheck(
        name="policy:base_reference",
        passed=False,
        detail=(
            "pi05 adapter selected but eval_policy reference is missing or OpenVLA-like. "
            "Set eval_policy.model_name and/or eval_policy.checkpoint_path to a pi05/OpenPI reference."
        ),
    )


def check_hf_auth() -> PreflightCheck:
    """Check that Hugging Face auth is available for gated model downloads."""
    if (os.environ.get("HF_TOKEN") or "").strip():
        return PreflightCheck(name="hf_auth", passed=True, detail="HF_TOKEN is set")

    hf_cli = shutil.which("huggingface-cli")
    hf_new = shutil.which("hf")
    if not hf_cli and not hf_new:
        return PreflightCheck(
            name="hf_auth",
            passed=False,
            detail="hf/huggingface-cli not found and HF_TOKEN not set",
        )

    whoami_cmd = (
        [hf_cli, "whoami"] if hf_cli else [hf_new, "auth", "whoami"]  # type: ignore[list-item]
    )
    try:
        proc = subprocess.run(
            whoami_cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return PreflightCheck(name="hf_auth", passed=False, detail=f"HF auth check failed: {exc}")

    if proc.returncode == 0:
        return PreflightCheck(name="hf_auth", passed=True, detail="huggingface-cli authenticated")
    return PreflightCheck(
        name="hf_auth",
        passed=False,
        detail="Run `huggingface-cli login` or set HF_TOKEN",
    )


def check_cosmos_predict_tokenizer_access(cosmos_checkpoint_path: Path) -> PreflightCheck:
    """Validate access to the Cosmos Predict tokenizer required by Stage 2 runtime."""
    name = "enrich:cosmos_predict_tokenizer"
    local_candidates: list[Path] = []

    # Preferred location used by scripts/download_models.sh.
    if cosmos_checkpoint_path:
        local_candidates.append(
            cosmos_checkpoint_path.parent
            / "cosmos-predict-2.5-2b"
            / _COSMOS_PREDICT_TOKENIZER_FILENAME
        )
    # Canonical cloud/default checkpoint roots.
    local_candidates.extend(
        [
            Path("/models/checkpoints")
            / "cosmos-predict-2.5-2b"
            / _COSMOS_PREDICT_TOKENIZER_FILENAME,
            Path("/app/data/checkpoints")
            / "cosmos-predict-2.5-2b"
            / _COSMOS_PREDICT_TOKENIZER_FILENAME,
        ]
    )

    seen: set[str] = set()
    for candidate in local_candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return PreflightCheck(
                name=name,
                passed=True,
                detail=f"Local tokenizer present: {candidate}",
            )

    token = (os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        return PreflightCheck(
            name=name,
            passed=False,
            detail=(
                "Missing local tokenizer and HF_TOKEN is unset. "
                "Grant access to nvidia/Cosmos-Predict2.5-2B and run scripts/download_models.sh."
            ),
        )

    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
    except Exception as exc:
        return PreflightCheck(
            name=name,
            passed=False,
            detail=f"huggingface_hub unavailable for access check: {exc}",
        )

    try:
        url = hf_hub_url(
            repo_id=_COSMOS_PREDICT_REPO,
            filename=_COSMOS_PREDICT_TOKENIZER_FILENAME,
            revision=_COSMOS_PREDICT_REVISION,
            repo_type="model",
        )
        get_hf_file_metadata(url=url, token=token)
    except Exception as exc:
        msg = str(exc).replace("\n", " ").strip()
        if len(msg) > 220:
            msg = msg[:220] + "..."
        return PreflightCheck(
            name=name,
            passed=False,
            detail=(
                "Cannot access required HF asset "
                f"{_COSMOS_PREDICT_REPO}@{_COSMOS_PREDICT_REVISION}/{_COSMOS_PREDICT_TOKENIZER_FILENAME}: "
                f"{msg}"
            ),
        )

    return PreflightCheck(
        name=name,
        passed=True,
        detail=(f"HF access verified: {_COSMOS_PREDICT_REPO}/{_COSMOS_PREDICT_TOKENIZER_FILENAME}"),
    )


def check_python_import_from_path(
    module_name: str | tuple[str, ...] | list[str], extra_path: Path, name: str
) -> PreflightCheck:
    """Check if one (or any) module name can be imported with an extra sys.path entry."""
    if not extra_path.exists():
        return PreflightCheck(name=name, passed=False, detail=f"Path not found: {extra_path}")

    if isinstance(module_name, str):
        module_candidates = [module_name]
    else:
        module_candidates = list(module_name)

    if not module_candidates:
        return PreflightCheck(name=name, passed=False, detail="No module candidates provided")

    inserted = False
    errors: list[str] = []
    try:
        path_text = str(extra_path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
            inserted = True

        for candidate in module_candidates:
            try:
                import_module(candidate)
                return PreflightCheck(
                    name=name,
                    passed=True,
                    detail=f"Importable as '{candidate}' from {extra_path}",
                )
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
            finally:
                for key in list(sys.modules.keys()):
                    if key == candidate or key.startswith(candidate + "."):
                        sys.modules.pop(key, None)

        joined = "; ".join(errors)
        return PreflightCheck(
            name=name,
            passed=False,
            detail=f"Cannot import any candidate ({', '.join(module_candidates)}): {joined}",
        )
    finally:
        if inserted:
            try:
                sys.path.remove(str(extra_path))
            except ValueError:
                pass


def check_api_key_for_scope(env_var: str, scope: str) -> PreflightCheck:
    val = os.environ.get(env_var, "")
    if val:
        return PreflightCheck(name=f"api_key:{scope}", passed=True, detail=f"{env_var} set")
    return PreflightCheck(
        name=f"api_key:{scope}",
        passed=False,
        detail=f"{env_var} not set",
    )


def check_external_interaction_manifest(config: ValidationConfig) -> PreflightCheck:
    """Preflight strict schema/path validation for Stage 1f input manifests."""
    ext_cfg = config.external_interaction
    if not bool(ext_cfg.enabled):
        return PreflightCheck(
            name="external_interaction:manifest",
            passed=True,
            detail="external_interaction.enabled=false",
        )

    manifest_path = ext_cfg.manifest_path
    if manifest_path is None:
        return PreflightCheck(
            name="external_interaction:manifest",
            passed=False,
            detail=(
                "external_interaction.enabled=true but manifest_path is not set. "
                "Set external_interaction.manifest_path to a valid stage1_source manifest."
            ),
        )
    if not manifest_path.exists():
        return PreflightCheck(
            name="external_interaction:manifest",
            passed=False,
            detail=f"External interaction manifest not found: {manifest_path}",
        )

    try:
        payload = load_and_validate_manifest(
            manifest_path,
            manifest_type="stage1_source",
            require_existing_paths=True,
        )
    except ManifestValidationError as exc:
        return PreflightCheck(
            name="external_interaction:manifest",
            passed=False,
            detail=f"Invalid stage1_source manifest: {exc}",
        )
    except Exception as exc:  # pragma: no cover - defensive
        return PreflightCheck(
            name="external_interaction:manifest",
            passed=False,
            detail=f"Failed reading manifest: {exc}",
        )

    return PreflightCheck(
        name="external_interaction:manifest",
        passed=True,
        detail=f"Validated stage1_source manifest ({len(payload.get('clips', []))} clips): {manifest_path}",
    )


def check_external_rollout_manifest(config: ValidationConfig) -> PreflightCheck:
    """Preflight strict schema/path validation for action-labeled teleop manifests."""
    ext_cfg = config.external_rollouts
    if not bool(ext_cfg.enabled):
        return PreflightCheck(
            name="external_rollouts:manifest",
            passed=True,
            detail="external_rollouts.enabled=false",
        )

    manifest_path = ext_cfg.manifest_path
    if manifest_path is None:
        return PreflightCheck(
            name="external_rollouts:manifest",
            passed=False,
            detail=(
                "external_rollouts.enabled=true but manifest_path is not set. "
                "Set external_rollouts.manifest_path to a valid teleop_session_manifest.json."
            ),
        )
    if not manifest_path.exists():
        return PreflightCheck(
            name="external_rollouts:manifest",
            passed=False,
            detail=f"External rollout manifest not found: {manifest_path}",
        )

    try:
        payload = load_and_validate_teleop_manifest(manifest_path, require_existing_paths=True)
    except TeleopManifestError as exc:
        return PreflightCheck(
            name="external_rollouts:manifest",
            passed=False,
            detail=f"Invalid teleop manifest: {exc}",
        )
    except Exception as exc:  # pragma: no cover - defensive
        return PreflightCheck(
            name="external_rollouts:manifest",
            passed=False,
            detail=f"Failed reading teleop manifest: {exc}",
        )

    return PreflightCheck(
        name="external_rollouts:manifest",
        passed=True,
        detail=(
            f"Validated teleop manifest ({len(payload.get('sessions', []))} sessions): "
            f"{manifest_path}"
        ),
    )


def check_dreamzero_runtime_contract(
    repo_path: Path,
    inference_module: str,
    inference_class: str,
) -> PreflightCheck:
    """Validate DreamZero runtime class exposes expected constructor/inference hooks."""
    name = "policy_adapter:dreamzero:runtime_contract"
    if not repo_path.exists():
        return PreflightCheck(name=name, passed=False, detail=f"Path not found: {repo_path}")

    inserted = False
    try:
        repo_str = str(repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
            inserted = True

        module = import_module(str(inference_module).strip())
        runtime_cls = getattr(module, str(inference_class).strip(), None)
        if runtime_cls is None:
            return PreflightCheck(
                name=name,
                passed=False,
                detail=f"Inference class not found: {inference_module}.{inference_class}",
            )

        from_pretrained = getattr(runtime_cls, "from_pretrained", None)
        supports_from_pretrained = callable(from_pretrained)

        supports_constructor = False
        try:
            signature = inspect.signature(runtime_cls)
            params = list(signature.parameters.values())
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
            keyword_names = {
                p.name
                for p in params
                if p.kind
                in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            supports_constructor = has_var_kw or bool(
                {"model_path", "checkpoint_path", "model_name"} & keyword_names
            )
        except Exception:
            # Some extension classes do not expose a Python signature; rely on from_pretrained/inference hooks.
            supports_constructor = False

        supports_inference = any(
            callable(getattr(runtime_cls, m, None))
            for m in ("predict_action", "infer_action", "infer", "generate")
        )
        if not supports_inference:
            # Only accept __call__ when explicitly implemented on the runtime class.
            cls_call = runtime_cls.__dict__.get("__call__")
            supports_inference = callable(cls_call)

        if not (supports_from_pretrained or supports_constructor):
            return PreflightCheck(
                name=name,
                passed=False,
                detail=(
                    "Runtime class does not expose a recognized constructor contract "
                    "(from_pretrained/model_path/checkpoint_path/model_name)."
                ),
            )
        if not supports_inference:
            return PreflightCheck(
                name=name,
                passed=False,
                detail=(
                    "Runtime class is missing inference entrypoints "
                    "(predict_action/infer_action/infer/generate/__call__)."
                ),
            )
        return PreflightCheck(
            name=name,
            passed=True,
            detail=f"Validated runtime contract: {inference_module}.{inference_class}",
        )
    except Exception as exc:
        return PreflightCheck(
            name=name,
            passed=False,
            detail=f"Runtime contract check failed: {exc}",
        )
    finally:
        if inserted:
            try:
                sys.path.remove(str(repo_path))
            except ValueError:
                pass


def check_facility_ply(facility_id: str, ply_path: Path) -> PreflightCheck:
    if ply_path.exists() and ply_path.stat().st_size > 0:
        size_mb = ply_path.stat().st_size / (1024**2)
        return PreflightCheck(
            name=f"ply:{facility_id}",
            passed=True,
            detail=f"{size_mb:.1f}MB at {ply_path}",
        )
    return PreflightCheck(
        name=f"ply:{facility_id}",
        passed=False,
        detail=f"PLY not found at {ply_path}",
    )


def check_task_hints_bootstrap_readiness(
    facility_id: str,
    facility,
    *,
    deep_scan: bool = True,
) -> PreflightCheck:
    """Validate that Stage 0 can synthesize task hints when no hints file is provided."""
    name = f"task_hints_bootstrap:{facility_id}"
    if facility.task_hints_path is not None and facility.task_hints_path.exists():
        return PreflightCheck(
            name=name,
            passed=True,
            detail=f"Existing task hints: {facility.task_hints_path}",
        )

    labels_path = facility.ply_path.parent / "labels.json"
    if labels_path.exists():
        return PreflightCheck(
            name=name,
            passed=True,
            detail=f"InteriorGS metadata present: {labels_path}",
        )

    if not facility.ply_path.exists():
        return PreflightCheck(
            name=name,
            passed=False,
            detail=f"PLY not found for Stage 0 bootstrap: {facility.ply_path}",
        )

    if not deep_scan:
        return PreflightCheck(
            name=name,
            passed=True,
            detail="Audit profile: full PLY bootstrap decode skipped; runtime preflight validates it.",
        )

    try:
        means, colors = _load_ply_means_and_colors_numpy()(facility.ply_path)
    except Exception as exc:
        return PreflightCheck(
            name=name,
            passed=False,
            detail=(
                "Stage 0 bootstrap cannot read the facility PLY without task hints or InteriorGS "
                f"metadata: {exc}"
            ),
        )

    if len(means) <= 0:
        return PreflightCheck(
            name=name,
            passed=False,
            detail="Stage 0 bootstrap loaded zero points from the facility PLY.",
        )

    return PreflightCheck(
        name=name,
        passed=True,
        detail=(
            f"PLY bootstrap path ready ({len(means)} points, colors="
            f"{'yes' if colors is not None else 'no'})"
        ),
    )


def check_openvla_local_checkpoint_requirement(
    config: ValidationConfig,
    adapter_name: str,
    *,
    scope_override: str | None = None,
) -> PreflightCheck:
    name = "policy:local_checkpoint_requirement"
    scope = (
        str(scope_override or getattr(config.eval_policy, "headline_scope", "wm_only") or "wm_only")
        .strip()
        .lower()
    )
    if adapter_name != "openvla_oft":
        return PreflightCheck(
            name=name, passed=True, detail=f"{adapter_name} does not use OpenVLA checkpoints"
        )
    if scope == "wm_only":
        return PreflightCheck(
            name=name, passed=True, detail="Deferred because eval_policy.headline_scope=wm_only"
        )
    if str(config.eval_policy.mode or "claim").strip().lower() != "claim":
        return PreflightCheck(
            name=name,
            passed=True,
            detail="Local OpenVLA checkpoint not required outside claim mode",
        )
    if _env_truthy("BLUEPRINT_ALLOW_REMOTE_OPENVLA_MODEL"):
        return PreflightCheck(
            name=name,
            passed=True,
            detail="Remote OpenVLA model resolution explicitly allowed by BLUEPRINT_ALLOW_REMOTE_OPENVLA_MODEL",
        )

    _model_name, checkpoint_path = _resolve_policy_base_reference_for_adapter(config, adapter_name)
    exists = checkpoint_path.exists() and (
        checkpoint_path.is_file() or (checkpoint_path.is_dir() and any(checkpoint_path.iterdir()))
    )
    if exists:
        return PreflightCheck(name=name, passed=True, detail=str(checkpoint_path))

    return PreflightCheck(
        name=name,
        passed=False,
        detail=(
            "Claim-mode OpenVLA runs require a local checkpoint to avoid implicit remote/model-license "
            f"resolution during execution. Missing checkpoint: {checkpoint_path}. "
            "Download the weights locally or set BLUEPRINT_ALLOW_REMOTE_OPENVLA_MODEL=1 to override."
        ),
    )


def check_claim_benchmark_readiness(
    config: ValidationConfig,
    facility_id: str,
    facility,
    *,
    work_dir: Path | None = None,
) -> PreflightCheck:
    name = f"claim_benchmark:{facility_id}"
    if (
        str(getattr(config.eval_policy, "claim_protocol", "none") or "none").strip().lower()
        != "fixed_same_facility_uplift"
    ):
        return PreflightCheck(
            name=name, passed=True, detail="Claim benchmark not required for this protocol"
        )

    benchmark_path = facility.claim_benchmark_path
    if benchmark_path is None:
        return PreflightCheck(
            name=name,
            passed=False,
            detail="facility.claim_benchmark_path is required for fixed_same_facility_uplift",
        )
    if not benchmark_path.exists():
        return PreflightCheck(
            name=name,
            passed=False,
            detail=f"Claim benchmark manifest not found: {benchmark_path}",
        )
    if work_dir is None:
        return PreflightCheck(
            name=name,
            passed=True,
            detail="Benchmark manifest present. Re-run preflight with the pipeline work dir after Stage 1 for render-alignment validation.",
        )

    render_manifest_path = work_dir / facility_id / "renders" / "render_manifest.json"
    if not render_manifest_path.exists():
        return PreflightCheck(
            name=name,
            passed=True,
            detail=f"Benchmark manifest present; render manifest not found yet at {render_manifest_path}",
        )

    try:
        render_manifest = load_and_validate_manifest(
            render_manifest_path,
            manifest_type="stage1_source",
            require_existing_paths=False,
        )
    except ManifestValidationError as exc:
        return PreflightCheck(
            name=name,
            passed=False,
            detail=f"Invalid render manifest for benchmark alignment: {exc}",
        )

    alignment_failures = claim_benchmark_alignment_failures(
        benchmark_path=benchmark_path,
        render_manifest=render_manifest,
    )
    if alignment_failures:
        return PreflightCheck(
            name=name,
            passed=False,
            detail=alignment_failures[0],
        )

    try:
        benchmark = load_pinned_claim_benchmark(
            benchmark_path=benchmark_path,
            render_manifest=render_manifest,
            video_orientation_fix=getattr(facility, "video_orientation_fix", "none"),
        )
    except ValueError as exc:
        return PreflightCheck(
            name=name,
            passed=False,
            detail=f"Invalid claim benchmark manifest: {exc}",
        )

    strictness_failures = claim_benchmark_strictness_failures(
        benchmark=benchmark,
        min_eval_task_specs=int(config.eval_policy.claim_strictness.min_eval_task_specs),
        min_eval_start_clips=int(config.eval_policy.claim_strictness.min_eval_start_clips),
        min_common_eval_cells=int(config.eval_policy.claim_strictness.min_common_eval_cells),
    )
    if strictness_failures:
        return PreflightCheck(
            name=name,
            passed=False,
            detail=strictness_failures[0],
        )

    return PreflightCheck(
        name=name,
        passed=True,
        detail=(
            f"Benchmark aligns with render manifest ({len(benchmark.task_specs)} task specs, "
            f"{len(benchmark.assignments)} assignments)"
        ),
    )


def _extract_class_fields(path: Path, class_name: str) -> set[str]:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            fields: set[str] = set()
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.add(stmt.target.id)
                elif isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            fields.add(target.id)
            return fields
    return set()


def _extract_dict_keys(path: Path, dict_name: str) -> set[str]:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == dict_name
                and isinstance(node.value, ast.Dict)
            ):
                keys: set[str] = set()
                for key in node.value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        keys.add(key.value)
                return keys
    return set()


def _resolve_script_path(repo_root: Path, script_value: str) -> Path:
    script_path = Path(script_value)
    if script_path.is_absolute():
        return script_path
    return repo_root / script_path


def _extract_argparse_option_flags(path: Path) -> set[str]:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    flags: set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        for arg in node.args:
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("--")
            ):
                flags.add(arg.value)
    return flags


def _extract_help_flags(script_path: Path, cwd: Path) -> set[str]:
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=12,
            cwd=str(cwd),
        )
    except Exception:
        return set()
    help_text = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    return set(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9_-]*", help_text))


def _check_script_cli_contract(
    *,
    check_name: str,
    script_path: Path,
    required_flags: set[str],
    cwd: Path,
    remediation: str,
) -> PreflightCheck:
    if not script_path.exists():
        return PreflightCheck(
            name=check_name,
            passed=False,
            detail=f"Script not found: {script_path}",
        )

    ast_flags = _extract_argparse_option_flags(script_path)
    combined_flags = set(ast_flags)
    source = "AST"
    if not required_flags.issubset(combined_flags):
        help_flags = _extract_help_flags(script_path, cwd=cwd)
        if help_flags:
            combined_flags |= help_flags
            source = "AST+--help" if ast_flags else "--help"

    if not combined_flags:
        return PreflightCheck(
            name=check_name,
            passed=False,
            detail=f"Could not determine CLI options for {script_path}",
        )

    missing = sorted(required_flags - combined_flags)
    if missing:
        return PreflightCheck(
            name=check_name,
            passed=False,
            detail=(
                f"Missing required CLI flags in {script_path}: {', '.join(missing)}. {remediation}"
            ),
        )
    return PreflightCheck(
        name=check_name,
        passed=True,
        detail=f"Validated CLI contract via {source}: {script_path}",
    )


def check_pi05_train_contract(openpi_repo: Path, train_script: str) -> PreflightCheck:
    script_path = _resolve_script_path(openpi_repo, train_script)
    return _check_script_cli_contract(
        check_name="policy_adapter:pi05:train_contract",
        script_path=script_path,
        required_flags=_PI05_REQUIRED_TRAIN_FLAGS,
        cwd=openpi_repo,
        remediation=(
            "Update policy_adapter.pi05.train_script or use a compatible OpenPI train entrypoint."
        ),
    )


def check_pi05_norm_stats_contract(openpi_repo: Path, norm_stats_script: str) -> PreflightCheck:
    script_path = _resolve_script_path(openpi_repo, norm_stats_script)
    return _check_script_cli_contract(
        check_name="policy_adapter:pi05:norm_stats_contract",
        script_path=script_path,
        required_flags=_PI05_REQUIRED_NORM_STATS_FLAGS,
        cwd=openpi_repo,
        remediation=(
            "Update policy_adapter.pi05.norm_stats_script or use a compatible OpenPI norm-stats entrypoint."
        ),
    )


def check_openvla_finetune_contract(openvla_repo: Path, finetune_script: str) -> PreflightCheck:
    script_path = Path(finetune_script)
    if not script_path.is_absolute():
        script_path = openvla_repo / script_path
    if not script_path.exists():
        return PreflightCheck(
            name="policy_finetune:wrapper_contract",
            passed=False,
            detail=f"Finetune script missing: {script_path}",
        )
    fields = _extract_class_fields(script_path, "FinetuneConfig")
    if not fields:
        return PreflightCheck(
            name="policy_finetune:wrapper_contract",
            passed=False,
            detail=f"Could not parse FinetuneConfig fields from {script_path}",
        )
    required = {
        "vla_path",
        "data_root_dir",
        "dataset_name",
        "run_root_dir",
        "lora_rank",
        "batch_size",
        "grad_accumulation_steps",
        "learning_rate",
        "save_freq",
        "max_steps",
        "image_aug",
        "use_l1_regression",
    }
    missing = sorted(required - fields)
    if missing:
        return PreflightCheck(
            name="policy_finetune:wrapper_contract",
            passed=False,
            detail=f"Vendored FinetuneConfig missing expected fields: {', '.join(missing)}",
        )
    return PreflightCheck(
        name="policy_finetune:wrapper_contract",
        passed=True,
        detail=f"Wrapper-compatible fields verified in {script_path}",
    )


def check_dreamzero_train_contract(repo_path: Path, train_script: str) -> PreflightCheck:
    script_path = _resolve_script_path(repo_path, train_script)
    return _check_script_cli_contract(
        check_name="policy_adapter:dreamzero:train_contract",
        script_path=script_path,
        required_flags={"--base_model", "--dataset_root", "--dataset_name", "--output_dir"},
        cwd=repo_path,
        remediation=(
            "Update policy_adapter.dreamzero.train_script or align DreamZeroPolicyAdapter.train_policy "
            "with the actual train-script CLI."
        ),
    )


def check_openvla_dataset_registry(openvla_repo: Path, dataset_names: set[str]) -> PreflightCheck:
    registry_path = openvla_repo / "prismatic" / "vla" / "datasets" / "rlds" / "oxe" / "configs.py"
    if not registry_path.exists():
        return PreflightCheck(
            name="policy_finetune:dataset_registry",
            passed=False,
            detail=f"Dataset registry missing: {registry_path}",
        )
    registered = _extract_dict_keys(registry_path, "OXE_DATASET_CONFIGS")
    if not registered:
        return PreflightCheck(
            name="policy_finetune:dataset_registry",
            passed=False,
            detail=f"Failed to parse OXE_DATASET_CONFIGS from {registry_path}",
        )
    missing = sorted(name for name in dataset_names if name not in registered)
    if missing:
        sample = ", ".join(sorted(list(registered))[:6])
        return PreflightCheck(
            name="policy_finetune:dataset_registry",
            passed=False,
            detail=(
                f"Dataset name(s) not in vendored OXE registry: {', '.join(missing)}. "
                f"Examples: {sample}"
            ),
        )
    return PreflightCheck(
        name="policy_finetune:dataset_registry",
        passed=True,
        detail=f"Validated dataset names: {', '.join(sorted(dataset_names))}",
    )


def check_cosmos_wrapper_contract(cosmos_repo: Path) -> PreflightCheck:
    config_path = cosmos_repo / "cosmos_transfer2" / "config.py"
    if not config_path.exists():
        return PreflightCheck(
            name="enrich:cosmos_wrapper_contract",
            passed=False,
            detail=f"Missing Cosmos config schema: {config_path}",
        )

    common_fields = _extract_class_fields(config_path, "CommonInferenceArguments")
    inference_fields = _extract_class_fields(config_path, "InferenceArguments")
    allowed_fields = common_fields | inference_fields
    if not allowed_fields:
        return PreflightCheck(
            name="enrich:cosmos_wrapper_contract",
            passed=False,
            detail=f"Failed to parse inference fields from {config_path}",
        )

    spec = _load_build_controlnet_spec()(
        video_path=Path("/tmp/input.mp4"),
        depth_path=Path("/tmp/depth.mp4"),
        prompt="contract-check",
        output_path=Path("/tmp/output.mp4"),
        guidance=7.0,
        controlnet_inputs=["rgb", "depth"],
    )
    extra = sorted(k for k in spec if k not in allowed_fields)
    if extra:
        return PreflightCheck(
            name="enrich:cosmos_wrapper_contract",
            passed=False,
            detail=f"Wrapper emits unsupported inference keys: {', '.join(extra)}",
        )
    return PreflightCheck(
        name="enrich:cosmos_wrapper_contract",
        passed=True,
        detail="Wrapper spec keys match vendored InferenceArguments schema",
    )


def check_dreamdojo_contract(
    dreamdojo_repo: Path, configured_experiment: str | None
) -> PreflightCheck:
    train_script = dreamdojo_repo / "scripts" / "train.py"
    if not train_script.exists():
        return PreflightCheck(
            name="finetune:dreamdojo_contract",
            passed=False,
            detail=f"DreamDojo train entrypoint missing: {train_script}",
        )
    try:
        experiment = _load_resolve_dreamdojo_experiment_name()(
            dreamdojo_repo,
            configured_experiment,
        )
    except Exception as exc:
        return PreflightCheck(
            name="finetune:dreamdojo_contract",
            passed=False,
            detail=str(exc),
        )
    return PreflightCheck(
        name="finetune:dreamdojo_contract",
        passed=True,
        detail=f"Resolved experiment={experiment}",
    )


def check_cloud_budget_enforcement(config: ValidationConfig) -> PreflightCheck:
    if config.cloud.max_cost_usd <= 0:
        return PreflightCheck(
            name="cloud:budget_enforcement",
            passed=True,
            detail="max_cost_usd<=0 (budget cap disabled)",
        )
    raw_rate = (os.environ.get("BLUEPRINT_GPU_HOURLY_RATE_USD") or "").strip()
    try:
        rate = float(raw_rate)
    except ValueError:
        rate = 0.0
    if rate > 0:
        return PreflightCheck(
            name="cloud:budget_enforcement",
            passed=True,
            detail=f"Hourly rate from env: ${rate:.4f}/hour",
        )
    return PreflightCheck(
        name="cloud:budget_enforcement",
        passed=False,
        detail=(
            "Set BLUEPRINT_GPU_HOURLY_RATE_USD to enforce cloud.max_cost_usd during pipeline execution."
        ),
    )


def check_cloud_shutdown_enforcement(config: ValidationConfig) -> PreflightCheck:
    if not config.cloud.auto_shutdown:
        return PreflightCheck(
            name="cloud:auto_shutdown_enforcement",
            passed=True,
            detail="auto_shutdown=false",
        )
    shutdown_cmd = (os.environ.get("BLUEPRINT_AUTO_SHUTDOWN_CMD") or "").strip()
    if shutdown_cmd:
        return PreflightCheck(
            name="cloud:auto_shutdown_enforcement",
            passed=True,
            detail="Shutdown command configured via BLUEPRINT_AUTO_SHUTDOWN_CMD",
        )
    return PreflightCheck(
        name="cloud:auto_shutdown_enforcement",
        passed=False,
        detail="Set BLUEPRINT_AUTO_SHUTDOWN_CMD to enforce cloud.auto_shutdown.",
    )


def run_preflight(
    config: ValidationConfig,
    work_dir: Path | None = None,
    profile: str = "runtime_local",
) -> List[PreflightCheck]:
    """Run preflight checks for the requested profile and return results."""
    normalized_profile = normalize_preflight_profile(profile)
    enforce_runtime_requirements = normalized_profile != "audit"
    enforce_cloud_requirements = normalized_profile == "runtime_cloud"

    checks: List[PreflightCheck] = []
    adapter_name = _canonical_policy_adapter_name(config.policy_adapter.name)
    configured_scope = (
        (getattr(config.eval_policy, "headline_scope", "wm_only") or "wm_only").strip().lower()
    )
    effective_scope = _effective_headline_scope(config)
    wm_only_scope = effective_scope == "wm_only"
    effective_policy_finetune_enabled = _effective_policy_finetune_enabled(config)
    checks.extend(_claim_mode_checks(config, adapter_name, scope_override=effective_scope))
    if configured_scope != effective_scope:
        checks.append(
            PreflightCheck(
                name="action_boost:headline_scope_override",
                passed=True,
                detail=f"Configured '{configured_scope}' -> effective '{effective_scope}' at runtime.",
            )
        )
    if bool(config.policy_finetune.enabled) != bool(effective_policy_finetune_enabled):
        checks.append(
            PreflightCheck(
                name="action_boost:policy_finetune_override",
                passed=True,
                detail=(
                    f"Configured policy_finetune.enabled={bool(config.policy_finetune.enabled)} -> "
                    f"effective={bool(effective_policy_finetune_enabled)} at runtime."
                ),
            )
        )

    _append_profiled_check(
        checks,
        enforce=enforce_runtime_requirements,
        advisory_name="gpu",
        advisory_detail=_advisory_detail(
            normalized_profile,
            "GPU availability is only enforced for runtime execution.",
        ),
        factory=check_gpu,
    )

    for mod, pkg in [
        ("gsplat", "gsplat"),
        ("torch", "pytorch"),
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("lpips", "lpips"),
        ("torchmetrics", "torchmetrics"),
        ("cv2", "opencv-python"),
        ("google.genai", "google-genai"),
    ]:
        checks.append(check_dependency(mod, pkg))

    checks.append(check_external_tool("ffmpeg"))
    checks.append(check_external_tool("ffprobe"))

    for fid, fconf in config.facilities.items():
        checks.append(check_facility_ply(fid, fconf.ply_path))
        if fconf.task_hints_path is not None:
            checks.append(check_path_exists(fconf.task_hints_path, f"task_hints:{fid}"))
        checks.append(
            check_task_hints_bootstrap_readiness(
                fid,
                fconf,
                deep_scan=enforce_runtime_requirements,
            )
        )

    _append_profiled_check(
        checks,
        enforce=enforce_runtime_requirements,
        advisory_name="weights:DreamDojo",
        advisory_detail=_advisory_detail(
            normalized_profile,
            "Local DreamDojo weights are not required for CPU audit runs.",
        ),
        factory=lambda: check_model_weights(config.finetune.dreamdojo_checkpoint, "DreamDojo"),
    )
    _append_profiled_check(
        checks,
        enforce=enforce_runtime_requirements,
        advisory_name="weights:Cosmos-Transfer-2.5",
        advisory_detail=_advisory_detail(
            normalized_profile,
            "Local Cosmos Transfer weights are not required for CPU audit runs.",
        ),
        factory=lambda: check_model_weights(
            config.enrich.cosmos_checkpoint,
            "Cosmos-Transfer-2.5",
        ),
    )
    if wm_only_scope:
        checks.append(
            PreflightCheck(
                name="policy:base_reference",
                passed=True,
                detail="Deferred because eval_policy.headline_scope=wm_only",
            )
        )
    else:
        model_name, checkpoint_path = _resolve_policy_base_reference_for_adapter(
            config,
            adapter_name,
        )
        checks.append(
            check_policy_base_reference_for_adapter(
                adapter_name=adapter_name,
                model_name=model_name,
                checkpoint_path=checkpoint_path,
            )
        )
        _append_profiled_check(
            checks,
            enforce=enforce_runtime_requirements,
            advisory_name="policy:local_checkpoint_requirement",
            advisory_detail=_advisory_detail(
                normalized_profile,
                "Local policy checkpoints are enforced only for runtime execution.",
            ),
            factory=lambda: check_openvla_local_checkpoint_requirement(
                config,
                adapter_name,
                scope_override=effective_scope,
            ),
        )
    _append_profiled_check(
        checks,
        enforce=enforce_runtime_requirements,
        advisory_name="hf_auth",
        advisory_detail=_advisory_detail(
            normalized_profile,
            "HF authentication is only enforced for runtime asset resolution.",
        ),
        factory=check_hf_auth,
    )
    _append_profiled_check(
        checks,
        enforce=enforce_runtime_requirements,
        advisory_name="enrich:cosmos_predict_tokenizer",
        advisory_detail=_advisory_detail(
            normalized_profile,
            "Remote/local Cosmos tokenizer access is only enforced for Stage 2 runtime.",
        ),
        factory=lambda: check_cosmos_predict_tokenizer_access(config.enrich.cosmos_checkpoint),
    )
    for fid, facility in config.facilities.items():
        checks.append(
            check_claim_benchmark_readiness(
                config,
                fid,
                facility,
                work_dir=work_dir,
            )
        )

    _append_runtime_repo_contract_checks(
        checks,
        config=config,
        normalized_profile=normalized_profile,
        enforce_runtime_requirements=enforce_runtime_requirements,
    )
    checks.append(check_dependency("sam2", "sam2"))
    checks.append(check_dependency("natsort", "natsort"))
    checks.append(check_dependency("lightning", "lightning"))
    _append_profiled_check(
        checks,
        enforce=enforce_runtime_requirements,
        advisory_name="import:cosmos_predict2",
        advisory_detail=_advisory_detail(
            normalized_profile,
            "CUDA-only DreamDojo/Cosmos runtime imports are not required for CPU audit runs.",
        ),
        factory=lambda: check_python_import_from_path(
            (
                "cosmos_predict2.action_conditioned.inference",
                "cosmos_predict2.action_conditioned",
            ),
            config.finetune.dreamdojo_repo,
            "import:cosmos_predict2",
        ),
    )

    judge_api_env = config.eval_policy.vlm_judge.api_key_env
    _append_profiled_check(
        checks,
        enforce=enforce_runtime_requirements,
        advisory_name=f"api_key:{judge_api_env}",
        advisory_detail=_advisory_detail(
            normalized_profile,
            f"{judge_api_env} is only enforced for remote judge runtime calls.",
        ),
        factory=lambda: check_api_key(judge_api_env),
    )
    _append_profiled_check(
        checks,
        enforce=enforce_runtime_requirements,
        advisory_name="api_key:eval_spatial",
        advisory_detail=_advisory_detail(
            normalized_profile,
            "Spatial-eval API credentials are only enforced for runtime evaluation.",
        ),
        factory=lambda: check_api_key_for_scope(judge_api_env, "eval_spatial"),
    )
    _append_profiled_check(
        checks,
        enforce=enforce_runtime_requirements,
        advisory_name="api_key:eval_crosssite",
        advisory_detail=_advisory_detail(
            normalized_profile,
            "Cross-site VLM credentials are only enforced for runtime evaluation.",
        ),
        factory=lambda: check_api_key_for_scope(judge_api_env, "eval_crosssite"),
    )

    if config.robot_composite.enabled:
        if config.robot_composite.urdf_path is None:
            checks.append(
                PreflightCheck(
                    name="robot_composite:urdf_path",
                    passed=False,
                    detail="Set robot_composite.urdf_path when robot_composite.enabled=true",
                )
            )
        else:
            checks.append(
                check_path_exists(config.robot_composite.urdf_path, "robot_composite:urdf_path")
            )

    if config.gemini_polish.enabled:
        _append_profiled_check(
            checks,
            enforce=enforce_runtime_requirements,
            advisory_name=f"api_key:{config.gemini_polish.api_key_env}",
            advisory_detail=_advisory_detail(
                normalized_profile,
                f"{config.gemini_polish.api_key_env} is only enforced for Gemini runtime calls.",
            ),
            factory=lambda: check_api_key(config.gemini_polish.api_key_env),
        )

    if config.robosplat.enabled:
        valid_backend = config.robosplat.backend in {"auto", "vendor", "native", "legacy_scan"}
        checks.append(
            PreflightCheck(
                name="robosplat:backend",
                passed=valid_backend,
                detail=config.robosplat.backend,
            )
        )
        valid_mode = config.robosplat.parity_mode in {"hybrid", "strict", "scan_only"}
        checks.append(
            PreflightCheck(
                name="robosplat:parity_mode",
                passed=valid_mode,
                detail=config.robosplat.parity_mode,
            )
        )
        checks.append(
            PreflightCheck(
                name="robosplat:variants_per_input",
                passed=int(config.robosplat.variants_per_input) > 0,
                detail=str(config.robosplat.variants_per_input),
            )
        )
        checks.append(
            PreflightCheck(
                name="robosplat:demo_source",
                passed=config.robosplat.demo_source in {"synthetic", "real", "required_real"},
                detail=config.robosplat.demo_source,
            )
        )
        if config.robosplat.demo_source == "required_real":
            checks.append(
                PreflightCheck(
                    name="robosplat:required_real_demo",
                    passed=False,
                    detail=(
                        "demo_source=required_real requires a real demo manifest; "
                        "not configured in current schema"
                    ),
                )
            )
        vendor_exists = config.robosplat.vendor_repo_path.exists()
        vendor_required = config.robosplat.backend == "vendor" or (
            config.robosplat.parity_mode == "strict" and config.robosplat.backend == "auto"
        )
        checks.append(
            PreflightCheck(
                name="robosplat:vendor_repo",
                passed=vendor_exists if vendor_required else True,
                detail=(
                    str(config.robosplat.vendor_repo_path)
                    if vendor_exists
                    else f"missing (optional): {config.robosplat.vendor_repo_path}"
                ),
            )
        )

    if config.robosplat_scan.enabled:
        checks.append(
            PreflightCheck(
                name="robosplat_scan:temporal_speed_factors",
                passed=bool(config.robosplat_scan.temporal_speed_factors),
                detail=(
                    "Configured"
                    if config.robosplat_scan.temporal_speed_factors
                    else "Provide at least one temporal speed factor"
                ),
            )
        )
        relight_valid = (
            config.robosplat_scan.relight_gain_min > 0
            and config.robosplat_scan.relight_gain_max > 0
        )
        checks.append(
            PreflightCheck(
                name="robosplat_scan:relight_range",
                passed=relight_valid,
                detail=(
                    f"{config.robosplat_scan.relight_gain_min}..{config.robosplat_scan.relight_gain_max}"
                    if relight_valid
                    else "relight_gain_min/max must be > 0"
                ),
            )
        )

    checks.append(check_external_interaction_manifest(config))
    checks.append(check_external_rollout_manifest(config))
    if bool(config.policy_compare.enabled):
        facility_count = len(config.facilities)
        checks.append(
            PreflightCheck(
                name="policy_eval_matrix:mode",
                passed=True,
                detail=(
                    "single_facility_same_site_policy_uplift"
                    if facility_count < 2
                    else f"multi_facility_cross_site ({facility_count} facilities)"
                ),
            )
        )

    if wm_only_scope:
        checks.append(
            PreflightCheck(
                name="policy_pipeline:deferred",
                passed=True,
                detail="OpenVLA stages deferred because eval_policy.headline_scope=wm_only",
            )
        )
    elif adapter_name == "dreamzero":
        dz = config.policy_adapter.dreamzero
        _append_dreamzero_adapter_runtime_checks(
            checks,
            config=config,
            normalized_profile=normalized_profile,
            enforce_runtime_requirements=enforce_runtime_requirements,
        )
        checks.append(
            PreflightCheck(
                name="policy_adapter:dreamzero:action_dim",
                passed=int(dz.policy_action_dim) > 0,
                detail=str(dz.policy_action_dim),
            )
        )
        requires_policy_training = bool(effective_policy_finetune_enabled)
        allow_training_enabled = bool(dz.allow_training)
        allow_training_pass = (not requires_policy_training) or allow_training_enabled
        if requires_policy_training and not allow_training_enabled:
            allow_training_detail = (
                "Effective pipeline requires policy training (policy_finetune enabled directly or via action_boost), "
                "but policy_adapter.dreamzero.allow_training=false. "
                "Set allow_training=true or disable policy_finetune/action_boost auto-enable."
            )
        elif requires_policy_training:
            allow_training_detail = "true (required by effective policy_finetune pipeline)"
        else:
            allow_training_detail = f"{allow_training_enabled} (policy training not required)"
        checks.append(
            PreflightCheck(
                name="policy_adapter:dreamzero:allow_training",
                passed=allow_training_pass,
                detail=allow_training_detail,
            )
        )

    if not wm_only_scope and effective_policy_finetune_enabled:
        if adapter_name == "openvla_oft":
            _append_openvla_policy_runtime_checks(
                checks,
                config=config,
                normalized_profile=normalized_profile,
                enforce_runtime_requirements=enforce_runtime_requirements,
            )
            if config.rollout_dataset.enabled:
                checks.append(check_dependency("tensorflow", "tensorflow"))
                checks.append(check_dependency("tensorflow_datasets", "tensorflow-datasets"))
                checks.append(
                    PreflightCheck(
                        name="policy_finetune:data_root_dir",
                        passed=True,
                        detail="Dataset root generated by Stage 4b/4c pipeline",
                    )
                )
                checks.append(
                    PreflightCheck(
                        name="policy_finetune:dataset_dir",
                        passed=True,
                        detail=f"Dataset '{config.policy_finetune.dataset_name}' produced during run",
                    )
                )
            elif config.policy_finetune.data_root_dir is None:
                checks.append(
                    PreflightCheck(
                        name="policy_finetune:data_root_dir",
                        passed=False,
                        detail="Set policy_finetune.data_root_dir when policy_finetune.enabled=true",
                    )
                )
            else:
                checks.append(
                    check_path_exists(
                        config.policy_finetune.data_root_dir,
                        "policy_finetune:data_root_dir",
                    )
                )
                dataset_dir = (
                    config.policy_finetune.data_root_dir / config.policy_finetune.dataset_name
                )
                checks.append(check_path_exists(dataset_dir, "policy_finetune:dataset_dir"))
        elif adapter_name == "pi05":
            pi05 = config.policy_adapter.pi05
            checks.append(
                PreflightCheck(
                    name="policy_adapter:pi05:profile",
                    passed=pi05.profile in {"pi05_libero", "pi05_droid"},
                    detail=pi05.profile,
                )
            )
            checks.append(
                PreflightCheck(
                    name="policy_adapter:pi05:runtime_mode",
                    passed=pi05.runtime_mode == "inprocess",
                    detail=pi05.runtime_mode,
                )
            )
            checks.append(
                PreflightCheck(
                    name="policy_adapter:pi05:train_backend",
                    passed=pi05.train_backend == "pytorch",
                    detail=pi05.train_backend,
                )
            )
            _append_pi05_policy_runtime_checks(
                checks,
                config=config,
                normalized_profile=normalized_profile,
                enforce_runtime_requirements=enforce_runtime_requirements,
            )
            checks.append(check_dependency("lerobot", "lerobot"))
            if config.rollout_dataset.enabled:
                checks.append(
                    PreflightCheck(
                        name="policy_finetune:data_root_dir",
                        passed=True,
                        detail="Dataset root generated by Stage 4b/4c pipeline",
                    )
                )
            elif config.policy_finetune.data_root_dir is None:
                checks.append(
                    PreflightCheck(
                        name="policy_finetune:data_root_dir",
                        passed=False,
                        detail="Set policy_finetune.data_root_dir when policy_finetune.enabled=true",
                    )
                )
            else:
                checks.append(
                    check_path_exists(
                        config.policy_finetune.data_root_dir,
                        "policy_finetune:data_root_dir",
                    )
                )
                dataset_dir = (
                    config.policy_finetune.data_root_dir / config.policy_finetune.dataset_name
                )
                checks.append(check_path_exists(dataset_dir, "policy_finetune:dataset_dir"))
        elif adapter_name == "dreamzero":
            dz = config.policy_adapter.dreamzero
            if bool(dz.allow_training):
                _append_dreamzero_policy_runtime_checks(
                    checks,
                    config=config,
                    normalized_profile=normalized_profile,
                    enforce_runtime_requirements=enforce_runtime_requirements,
                )
                if config.rollout_dataset.enabled:
                    checks.append(
                        PreflightCheck(
                            name="policy_finetune:data_root_dir",
                            passed=True,
                            detail="Dataset root generated by Stage 4b/4c pipeline",
                        )
                    )
                elif config.policy_finetune.data_root_dir is None:
                    checks.append(
                        PreflightCheck(
                            name="policy_finetune:data_root_dir",
                            passed=False,
                            detail=(
                                "Set policy_finetune.data_root_dir when policy_finetune.enabled=true "
                                "and dreamzero.allow_training=true"
                            ),
                        )
                    )
                else:
                    checks.append(
                        check_path_exists(
                            config.policy_finetune.data_root_dir,
                            "policy_finetune:data_root_dir",
                        )
                    )
                    dataset_dir = (
                        config.policy_finetune.data_root_dir / config.policy_finetune.dataset_name
                    )
                    checks.append(check_path_exists(dataset_dir, "policy_finetune:dataset_dir"))
        else:
            checks.append(
                PreflightCheck(
                    name="policy_adapter:name",
                    passed=False,
                    detail=(
                        f"Unsupported adapter '{config.policy_adapter.name}'. "
                        "Supported: openvla_oft, pi05, dreamzero"
                    ),
                )
            )

    _append_profiled_check(
        checks,
        enforce=enforce_cloud_requirements,
        advisory_name="cloud:budget_enforcement",
        advisory_detail=_advisory_detail(
            normalized_profile,
            "Cloud budget enforcement is only required for runtime-cloud preflight.",
        ),
        factory=lambda: check_cloud_budget_enforcement(config),
    )
    _append_profiled_check(
        checks,
        enforce=enforce_cloud_requirements,
        advisory_name="cloud:auto_shutdown_enforcement",
        advisory_detail=_advisory_detail(
            normalized_profile,
            "Auto-shutdown enforcement is only required for runtime-cloud preflight.",
        ),
        factory=lambda: check_cloud_shutdown_enforcement(config),
    )

    if config.policy_rl_loop.enabled:
        checks.append(
            PreflightCheck(
                name="policy_rl_loop:iterations",
                passed=config.policy_rl_loop.iterations > 0,
                detail=str(config.policy_rl_loop.iterations),
            )
        )
        checks.append(
            PreflightCheck(
                name="policy_rl_loop:reward_mode",
                passed=config.policy_rl_loop.reward_mode
                in {
                    "hybrid",
                    "vlm_only",
                    "heuristic_only",
                },
                detail=config.policy_rl_loop.reward_mode,
            )
        )
        checks.append(
            PreflightCheck(
                name="policy_rl_loop:policy_finetune_enabled",
                passed=config.policy_finetune.enabled,
                detail=(
                    "OK"
                    if config.policy_finetune.enabled
                    else "Enable policy_finetune for RL policy updates"
                ),
            )
        )

    passed = sum(1 for c in checks if c.passed)
    total = len(checks)
    profile_label = normalized_profile.replace("_", "-")
    logger.info("Preflight (%s): %d/%d checks passed", profile_label, passed, total)
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        logger.info("  [%s] %s %s", status, c.name, f"— {c.detail}" if c.detail else "")

    return checks
