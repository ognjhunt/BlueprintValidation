"""Preflight checks for GPU, dependencies, and model weights."""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import List

from .common import PreflightCheck, get_logger
from .config import ValidationConfig
from .enrichment.cosmos_runner import build_controlnet_spec
from .training.dreamdojo_finetune import resolve_dreamdojo_experiment_name

logger = get_logger("preflight")

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
    return key


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


def check_policy_base_reference_for_adapter(
    adapter_name: str,
    model_name: str,
    checkpoint_path: Path,
) -> PreflightCheck:
    if adapter_name != "pi05":
        return check_policy_base_reference(model_name=model_name, checkpoint_path=checkpoint_path)

    ckpt_exists = checkpoint_path.exists() and (
        checkpoint_path.is_file() or (checkpoint_path.is_dir() and any(checkpoint_path.iterdir()))
    )
    if ckpt_exists:
        if _is_openvla_like_reference("", checkpoint_path):
            return PreflightCheck(
                name="policy:base_reference",
                passed=False,
                detail=(
                    "pi05 adapter selected but eval_policy.checkpoint_path points to an OpenVLA-like "
                    f"artifact: {checkpoint_path}. Set a pi05/OpenPI checkpoint instead."
                ),
            )
        return PreflightCheck(
            name="policy:base_reference",
            passed=True,
            detail=f"pi05 checkpoint reference: {checkpoint_path}",
        )

    if model_name and not _is_openvla_like_reference(model_name, Path("")):
        return PreflightCheck(
            name="policy:base_reference",
            passed=True,
            detail=f"pi05 model reference: {model_name}",
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

    spec = build_controlnet_spec(
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
        experiment = resolve_dreamdojo_experiment_name(dreamdojo_repo, configured_experiment)
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


def run_preflight(config: ValidationConfig) -> List[PreflightCheck]:
    """Run all preflight checks and return results."""
    checks: List[PreflightCheck] = []
    adapter_name = _canonical_policy_adapter_name(config.policy_adapter.name)

    # GPU
    checks.append(check_gpu())

    # Core dependencies
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

    # External tools
    checks.append(check_external_tool("ffmpeg"))

    # Facility PLY files
    for fid, fconf in config.facilities.items():
        checks.append(check_facility_ply(fid, fconf.ply_path))
        if fconf.task_hints_path is not None:
            checks.append(
                check_path_exists(
                    fconf.task_hints_path,
                    f"task_hints:{fid}",
                )
            )

    # Model weights
    checks.append(check_model_weights(config.finetune.dreamdojo_checkpoint, "DreamDojo"))
    checks.append(check_model_weights(config.enrich.cosmos_checkpoint, "Cosmos-Transfer-2.5"))
    checks.append(
        check_policy_base_reference_for_adapter(
            adapter_name=adapter_name,
            model_name=config.eval_policy.model_name,
            checkpoint_path=config.eval_policy.checkpoint_path,
        )
    )
    checks.append(check_hf_auth())

    # Runtime repos and scripts required by Stage 2+.
    checks.append(check_path_exists(config.enrich.cosmos_repo, "repo:cosmos_transfer"))
    checks.append(
        check_path_exists_under(
            config.enrich.cosmos_repo,
            "examples/inference.py",
            "repo:cosmos_transfer:inference_script",
        )
    )
    checks.append(check_cosmos_wrapper_contract(config.enrich.cosmos_repo))
    # Cosmos Transfer inference imports SAM2 helpers and natsort at module import time.
    # Require both up-front so Stage 2 cannot fail mid-run on missing dependency.
    checks.append(check_dependency("sam2", "sam2"))
    checks.append(check_dependency("natsort", "natsort"))
    checks.append(check_path_exists(config.finetune.dreamdojo_repo, "repo:dreamdojo"))
    checks.append(
        check_path_exists_under(
            config.finetune.dreamdojo_repo,
            "scripts/train.py",
            "repo:dreamdojo:train_entrypoint",
        )
    )
    checks.append(
        check_path_exists_under(
            config.finetune.dreamdojo_repo,
            "configs",
            "repo:dreamdojo:configs",
        )
    )
    checks.append(
        check_python_import_from_path(
            (
                "cosmos_predict2.action_conditioned.inference",
                "cosmos_predict2.action_conditioned",
            ),
            config.finetune.dreamdojo_repo,
            "import:cosmos_predict2",
        )
    )
    checks.append(
        check_dreamdojo_contract(
            config.finetune.dreamdojo_repo,
            config.finetune.experiment_config,
        )
    )

    # API keys
    checks.append(check_api_key(config.eval_policy.vlm_judge.api_key_env))
    checks.append(check_api_key_for_scope(config.eval_policy.vlm_judge.api_key_env, "eval_spatial"))
    checks.append(
        check_api_key_for_scope(config.eval_policy.vlm_judge.api_key_env, "eval_crosssite")
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
        checks.append(check_api_key(config.gemini_polish.api_key_env))

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

    if config.policy_finetune.enabled:
        if adapter_name == "openvla_oft":
            openvla_backend = config.policy_adapter.openvla
            checks.append(check_external_tool("torchrun"))
            checks.append(
                check_path_exists(
                    openvla_backend.openvla_repo,
                    "policy_finetune:openvla_repo",
                )
            )
            script_path = Path(openvla_backend.finetune_script)
            if not script_path.is_absolute():
                script_path = openvla_backend.openvla_repo / script_path
            checks.append(check_path_exists(script_path, "policy_finetune:finetune_script"))
            checks.append(
                check_openvla_finetune_contract(
                    openvla_backend.openvla_repo,
                    openvla_backend.finetune_script,
                )
            )
            dataset_names = {config.policy_finetune.dataset_name}
            if config.rollout_dataset.enabled:
                dataset_names.add(config.rollout_dataset.baseline_dataset_name)
                dataset_names.add(config.rollout_dataset.adapted_dataset_name)
            checks.append(
                check_openvla_dataset_registry(
                    openvla_backend.openvla_repo,
                    dataset_names,
                )
            )
            # When rollout_dataset is enabled, pair-training stages generate datasets later in-pipeline.
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
                checks.append(
                    check_path_exists(
                        dataset_dir,
                        "policy_finetune:dataset_dir",
                    )
                )
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
            checks.append(
                check_path_exists(
                    pi05.openpi_repo,
                    "policy_adapter:pi05:openpi_repo",
                )
            )
            train_script = Path(pi05.train_script)
            norm_script = Path(pi05.norm_stats_script)
            if not train_script.is_absolute():
                train_script = pi05.openpi_repo / train_script
            if not norm_script.is_absolute():
                norm_script = pi05.openpi_repo / norm_script
            checks.append(check_path_exists(train_script, "policy_adapter:pi05:train_script"))
            checks.append(check_path_exists(norm_script, "policy_adapter:pi05:norm_stats_script"))
            checks.append(check_pi05_train_contract(pi05.openpi_repo, pi05.train_script))
            checks.append(check_pi05_norm_stats_contract(pi05.openpi_repo, pi05.norm_stats_script))
            checks.append(
                check_python_import_from_path(
                    "openpi",
                    pi05.openpi_repo / "src",
                    "import:openpi",
                )
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
                checks.append(
                    check_path_exists(
                        dataset_dir,
                        "policy_finetune:dataset_dir",
                    )
                )
        else:
            checks.append(
                PreflightCheck(
                    name="policy_adapter:name",
                    passed=False,
                    detail=(
                        f"Unsupported adapter '{config.policy_adapter.name}'. "
                        "Supported: openvla_oft, pi05"
                    ),
                )
            )

    checks.append(check_cloud_budget_enforcement(config))
    checks.append(check_cloud_shutdown_enforcement(config))

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

    # Log summary
    passed = sum(1 for c in checks if c.passed)
    total = len(checks)
    logger.info("Preflight: %d/%d checks passed", passed, total)
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        logger.info("  [%s] %s %s", status, c.name, f"— {c.detail}" if c.detail else "")

    return checks
