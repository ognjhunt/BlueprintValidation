"""Preflight checks for GPU, dependencies, and model weights."""

from __future__ import annotations

import ast
import os
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
        return PreflightCheck(name=f"dep:{pkg}", passed=False, detail=f"Cannot import {module_name}")


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


def check_python_import_from_path(module_name: str, extra_path: Path, name: str) -> PreflightCheck:
    """Check if a module can be imported with an additional path on sys.path."""
    if not extra_path.exists():
        return PreflightCheck(name=name, passed=False, detail=f"Path not found: {extra_path}")

    inserted = False
    try:
        path_text = str(extra_path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
            inserted = True
        import_module(module_name)
        return PreflightCheck(name=name, passed=True, detail=f"Importable from {extra_path}")
    except Exception as exc:
        return PreflightCheck(name=name, passed=False, detail=f"Cannot import {module_name}: {exc}")
    finally:
        if inserted:
            try:
                sys.path.remove(str(extra_path))
            except ValueError:
                pass
        for key in list(sys.modules.keys()):
            if key == module_name or key.startswith(module_name + "."):
                sys.modules.pop(key, None)


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
            if isinstance(target, ast.Name) and target.id == dict_name and isinstance(node.value, ast.Dict):
                keys: set[str] = set()
                for key in node.value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        keys.add(key.value)
                return keys
    return set()


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
    registry_path = (
        openvla_repo
        / "prismatic"
        / "vla"
        / "datasets"
        / "rlds"
        / "oxe"
        / "configs.py"
    )
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


def check_dreamdojo_contract(dreamdojo_repo: Path, configured_experiment: str | None) -> PreflightCheck:
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
        check_model_weights(config.eval_policy.openvla_checkpoint, "OpenVLA-OFT base weights")
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
            "cosmos_predict2.action_conditioned.inference",
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
    checks.append(check_api_key_for_scope(config.eval_policy.vlm_judge.api_key_env, "eval_crosssite"))

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
        vendor_required = (
            config.robosplat.backend == "vendor"
            or (config.robosplat.parity_mode == "strict" and config.robosplat.backend == "auto")
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

    # Optional OpenVLA-OFT fine-tuning prerequisites
    if config.policy_finetune.enabled:
        checks.append(check_external_tool("torchrun"))
        checks.append(
            check_path_exists(
                config.policy_finetune.openvla_repo,
                "policy_finetune:openvla_repo",
            )
        )
        script_path = Path(config.policy_finetune.finetune_script)
        if not script_path.is_absolute():
            script_path = config.policy_finetune.openvla_repo / script_path
        checks.append(check_path_exists(script_path, "policy_finetune:finetune_script"))
        checks.append(
            check_openvla_finetune_contract(
                config.policy_finetune.openvla_repo,
                config.policy_finetune.finetune_script,
            )
        )
        dataset_names = {config.policy_finetune.dataset_name}
        if config.rollout_dataset.enabled:
            dataset_names.add(config.rollout_dataset.baseline_dataset_name)
            dataset_names.add(config.rollout_dataset.adapted_dataset_name)
        checks.append(
            check_openvla_dataset_registry(
                config.policy_finetune.openvla_repo,
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
                passed=config.policy_rl_loop.reward_mode in {
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
