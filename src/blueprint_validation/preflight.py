"""Preflight checks for GPU, dependencies, and model weights."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import List

from .common import PreflightCheck, get_logger
from .config import ValidationConfig

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
    if not hf_cli:
        return PreflightCheck(
            name="hf_auth",
            passed=False,
            detail="huggingface-cli not found and HF_TOKEN not set",
        )

    try:
        proc = subprocess.run(
            [hf_cli, "whoami"],
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
    checks.append(check_model_weights(config.eval_policy.openvla_checkpoint, "OpenVLA"))
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
    checks.append(check_path_exists(config.finetune.dreamdojo_repo, "repo:dreamdojo"))
    checks.append(
        check_path_exists_under(
            config.finetune.dreamdojo_repo,
            "launch.sh",
            "repo:dreamdojo:launch",
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

    # Optional OpenVLA fine-tuning prerequisites
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
        # When rollout_dataset is enabled, pair-training stages generate datasets later in-pipeline.
        if config.rollout_dataset.enabled:
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

    # Log summary
    passed = sum(1 for c in checks if c.passed)
    total = len(checks)
    logger.info("Preflight: %d/%d checks passed", passed, total)
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        logger.info("  [%s] %s %s", status, c.name, f"— {c.detail}" if c.detail else "")

    return checks
