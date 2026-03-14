"""Operational helpers for bootstrapping and smoke-testing the NeoVerse runtime."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping
from urllib import parse as urllib_parse

from huggingface_hub import snapshot_download

from blueprint_contracts.site_world_contract import load_site_world_bundle

from .config import ValidationConfig
from .hosted_session import create_session, reset_session, step_session
from .neoverse_runtime_client import NeoVerseRuntimeClient, NeoVerseRuntimeClientConfig
from .preflight import run_preflight


_DEFAULT_NEOVERSE_GIT_URL = "https://github.com/IamCreateAI/NeoVerse.git"
_DEFAULT_NEOVERSE_HF_REPO = "Yuppie1204/NeoVerse"
_DEFAULT_GSPLAT_FALLBACK_VERSION = "1.5.3"
_DEFAULT_GSPLAT_GIT_URL = "https://github.com/nerfstudio-project/gsplat.git"
_DEFAULT_TORCH_SPECS = {
    "cu121": {
        "torch": "2.3.1",
        "torchvision": "0.18.1",
        "torchaudio": "2.3.1",
        "index_url": "https://download.pytorch.org/whl/cu121",
        "scatter_url": "https://data.pyg.org/whl/torch-2.3.1+cu121.html",
    },
    "cu128": {
        "torch": "2.7.1",
        "torchvision": "0.22.1",
        "torchaudio": "2.7.1",
        "index_url": "https://download.pytorch.org/whl/cu128",
        "scatter_url": "https://data.pyg.org/whl/torch-2.7.1+cu128.html",
    },
}


@dataclass(frozen=True)
class NeoVerseBootstrapResult:
    repo_root: Path
    venv_python: Path
    model_root: Path
    checkpoint_path: Path
    env_file: Path
    runner_command: str
    cuda_variant: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_root": str(self.repo_root),
            "venv_python": str(self.venv_python),
            "model_root": str(self.model_root),
            "checkpoint_path": str(self.checkpoint_path),
            "env_file": str(self.env_file),
            "runner_command": self.runner_command,
            "cuda_variant": self.cuda_variant,
        }


def _run(cmd: list[str], *, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> None:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}\n{message[:1000]}")


def _detect_cuda_variant() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "cu121"
    text = f"{completed.stdout}\n{completed.stderr}"
    if "CUDA Version: 12.8" in text or "CUDA Version: 12.7" in text:
        return "cu128"
    return "cu121"


def _ensure_repo(repo_root: Path, *, git_url: str) -> None:
    if (repo_root / ".git").is_dir():
        _run(["git", "pull", "--ff-only"], cwd=repo_root)
        return
    repo_root.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", git_url, str(repo_root)])


def _ensure_venv(repo_root: Path, *, python_bin: str) -> Path:
    venv_dir = repo_root / ".venv"
    venv_python = venv_dir / "bin" / "python"
    if not venv_python.is_file():
        _run([python_bin, "-m", "venv", str(venv_dir)])
    _run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
    return venv_python


def _ensure_system_build_dependencies() -> None:
    if not Path("/usr/bin/apt-get").is_file():
        return
    python_headers = Path("/usr/include/python3.10/Python.h")
    if python_headers.is_file():
        return
    env = dict(os.environ)
    env["DEBIAN_FRONTEND"] = "noninteractive"
    _run(["apt-get", "update"], env=env)
    _run(["apt-get", "install", "-y", "build-essential", "python3.10-dev"], env=env)


def _install_neoverse_requirements(repo_root: Path, venv_python: Path, *, cuda_variant: str) -> None:
    _ensure_system_build_dependencies()
    spec = _DEFAULT_TORCH_SPECS.get(cuda_variant, _DEFAULT_TORCH_SPECS["cu121"])
    _run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            f"torch=={spec['torch']}",
            f"torchvision=={spec['torchvision']}",
            f"torchaudio=={spec['torchaudio']}",
            "--index-url",
            str(spec["index_url"]),
        ]
    )
    _run([str(venv_python), "-m", "pip", "install", "-r", str(repo_root / "requirements.txt")], cwd=repo_root)
    _run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "torch-scatter",
            "-f",
            str(spec["scatter_url"]),
        ],
        cwd=repo_root,
    )
    gsplat_git_cmd = [
        str(venv_python),
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        f"git+{_DEFAULT_GSPLAT_GIT_URL}",
    ]
    try:
        _run(gsplat_git_cmd, cwd=repo_root)
    except RuntimeError as exc:
        fallback_cmd = [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            f"git+{_DEFAULT_GSPLAT_GIT_URL}@v{_DEFAULT_GSPLAT_FALLBACK_VERSION}",
        ]
        try:
            _run(fallback_cmd, cwd=repo_root)
        except RuntimeError as fallback_exc:
            raise RuntimeError(
                "Failed to install gsplat from the default Git HEAD and the pinned fallback "
                f"v{_DEFAULT_GSPLAT_FALLBACK_VERSION}.\n"
                f"default error: {exc}\n"
                f"fallback error: {fallback_exc}"
            ) from fallback_exc


def _download_models(model_root: Path, *, hf_repo: str) -> None:
    model_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=hf_repo, local_dir=str(model_root / "NeoVerse"), local_dir_use_symlinks=False)


def _ensure_neoverse_model_layout(model_root: Path) -> Path:
    nested_root = model_root / "NeoVerse"
    if nested_root.is_dir():
        return nested_root

    flat_markers = [
        model_root / "reconstructor.ckpt",
        model_root / "models_t5_umt5-xxl-enc-bf16.pth",
        model_root / "Wan2.1_VAE.pth",
        model_root / "diffusion_pytorch_model.safetensors.index.json",
    ]
    if not any(path.exists() for path in flat_markers):
        nested_root.mkdir(parents=True, exist_ok=True)
        return nested_root

    nested_root.mkdir(parents=True, exist_ok=True)
    for child in model_root.iterdir():
        if child.name == "NeoVerse":
            continue
        target = nested_root / child.name
        if target.exists():
            continue
        target.symlink_to(child, target_is_directory=child.is_dir())
    return nested_root


def _find_checkpoint(model_root: Path) -> Path:
    preferred = [item for item in model_root.rglob("*") if item.is_file() and item.suffix in {".ckpt", ".pt", ".pth", ".safetensors"}]
    if not preferred:
        raise RuntimeError(f"No checkpoint file found under {model_root}")
    preferred.sort(key=lambda item: ("reconstructor" not in item.name.lower(), len(str(item))))
    return preferred[0]


def bootstrap_neoverse_runtime(
    *,
    repo_root: Path,
    env_file: Path,
    git_url: str = _DEFAULT_NEOVERSE_GIT_URL,
    hf_repo: str = _DEFAULT_NEOVERSE_HF_REPO,
    bootstrap_python: str = sys.executable,
    cuda_variant: str | None = None,
    skip_install: bool = False,
    skip_download: bool = False,
) -> NeoVerseBootstrapResult:
    repo_root = repo_root.expanduser().resolve()
    env_file = env_file.expanduser().resolve()
    selected_cuda = cuda_variant or _detect_cuda_variant()
    _ensure_repo(repo_root, git_url=git_url)
    venv_python = _ensure_venv(repo_root, python_bin=bootstrap_python)
    if not skip_install:
        _install_neoverse_requirements(repo_root, venv_python, cuda_variant=selected_cuda)
    model_root = repo_root / "models"
    if not skip_download:
        _download_models(model_root, hf_repo=hf_repo)
    _ensure_neoverse_model_layout(model_root)
    checkpoint_path = _find_checkpoint(model_root)
    runner_command = f"{sys.executable} -m blueprint_validation.neoverse_runner_wrapper"
    env_payload = {
        "NEOVERSE_REPO_ROOT": str(repo_root),
        "NEOVERSE_PYTHON_BIN": str(venv_python),
        "NEOVERSE_MODEL_ROOT": str(model_root),
        "NEOVERSE_CHECKPOINT_PATH": str(checkpoint_path),
        "NEOVERSE_RUNNER_COMMAND": runner_command,
        "NEOVERSE_TORCH_CUDA_VARIANT": selected_cuda,
        "NEOVERSE_GPU_ENABLED": "true",
    }
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text(
        "".join(f'export {key}="{value}"\n' for key, value in env_payload.items()),
        encoding="utf-8",
    )
    return NeoVerseBootstrapResult(
        repo_root=repo_root,
        venv_python=venv_python,
        model_root=model_root,
        checkpoint_path=checkpoint_path,
        env_file=env_file,
        runner_command=runner_command,
        cuda_variant=selected_cuda,
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _runtime_service_url(env: Mapping[str, str]) -> str:
    public = str(env.get("NEOVERSE_RUNTIME_PUBLIC_BASE_URL") or "").strip()
    if public:
        return public.rstrip("/")
    host = str(env.get("NEOVERSE_RUNTIME_SERVICE_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    port = str(env.get("NEOVERSE_RUNTIME_SERVICE_PORT") or "8787").strip() or "8787"
    return f"http://{host}:{port}".rstrip("/")


def _wait_for_runtime(url: str, *, timeout_seconds: int = 180) -> None:
    client = NeoVerseRuntimeClient(NeoVerseRuntimeClientConfig(service_url=url, api_key="", timeout_seconds=5))
    deadline = time.time() + max(1, int(timeout_seconds))
    last_error = "timeout"
    while time.time() < deadline:
        try:
            health = client.healthcheck()
            if str(health.get("status") or "").strip().lower() == "ok":
                return
        except Exception as exc:  # pragma: no cover - network timing
            last_error = str(exc)
        time.sleep(2.0)
    raise RuntimeError(f"Timed out waiting for runtime service at {url}: {last_error}")


def _service_process_env(base_env: Mapping[str, str], *, runtime_root: Path, service_url: str) -> Dict[str, str]:
    parsed = urllib_parse.urlparse(service_url)
    host = parsed.hostname or "127.0.0.1"
    port = str(parsed.port or 8787)
    env = dict(base_env)
    env["NEOVERSE_RUNTIME_SERVICE_HOST"] = host
    env["NEOVERSE_RUNTIME_SERVICE_PORT"] = port
    env["NEOVERSE_RUNTIME_PUBLIC_BASE_URL"] = f"{parsed.scheme}://{host}:{port}"
    env["NEOVERSE_RUNTIME_ROOT"] = str(runtime_root)
    env["NEOVERSE_RUNTIME_SERVICE_URL"] = f"{parsed.scheme}://{host}:{port}"
    return env


def run_neoverse_runtime_smoke_test(
    *,
    config: ValidationConfig,
    registration_path: Path,
    work_dir: Path,
    session_id: str,
    robot_profile_id: str,
    task_id: str,
    scenario_id: str,
    start_state_id: str,
    boot_service: bool = True,
    service_url: str = "",
) -> Dict[str, Any]:
    bundle = load_site_world_bundle(registration_path, require_spec=True)
    configured_url = service_url.strip() or str(config.scene_memory_runtime.neoverse_service.service_url or "").strip() or str(os.environ.get("NEOVERSE_RUNTIME_SERVICE_URL") or "").strip()
    if not configured_url:
        configured_url = f"http://127.0.0.1:{_free_port()}"
    service_process: subprocess.Popen[str] | None = None
    service_env = _service_process_env(os.environ, runtime_root=work_dir / "runtime-service", service_url=configured_url)
    try:
        if boot_service:
            service_process = subprocess.Popen(
                [sys.executable, "-m", "blueprint_validation.neoverse_runtime_service"],
                cwd=str(Path(__file__).resolve().parents[2]),
                env=service_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            _wait_for_runtime(configured_url)

        effective_config = replace(
            config,
            scene_memory_runtime=replace(
                config.scene_memory_runtime,
                neoverse_service=replace(
                    config.scene_memory_runtime.neoverse_service,
                    service_url=configured_url,
                ),
            ),
        )
        checks = run_preflight(effective_config, site_world_registration=registration_path)
        failed = [check.to_dict() for check in checks if not check.passed]
        if failed:
            raise RuntimeError(f"Runtime preflight failed: {failed}")

        session_work_dir = work_dir / "smoke-session"
        create_payload = create_session(
            config=effective_config,
            session_id=session_id,
            session_work_dir=session_work_dir,
            registration_path=registration_path,
            robot_profile_id=robot_profile_id,
            task_id=task_id,
            scenario_id=scenario_id,
            start_state_id=start_state_id,
        )
        reset_payload = reset_session(
            config=effective_config,
            session_id=session_id,
            session_work_dir=session_work_dir,
        )
        episode_id = str(reset_payload["episode"]["episodeId"])
        step_payload = step_session(
            config=effective_config,
            session_work_dir=session_work_dir,
            episode_id=episode_id,
            action=[0.0] * int(((bundle.resolved.get("robot_profiles") or [{}])[0].get("action_space") or {}).get("dim", 7)),
        )
        session_state_path = session_work_dir / "session_state.json"
        session_state = json.loads(session_state_path.read_text(encoding="utf-8"))
        runtime_client = NeoVerseRuntimeClient(
            NeoVerseRuntimeClientConfig(
                service_url=configured_url,
                api_key="",
                timeout_seconds=max(5, int(config.scene_memory_runtime.neoverse_service.timeout_seconds)),
            )
        )
        render_payload = runtime_client.render_bytes(str(session_state.get("remote_session_id") or session_id))
        if not render_payload:
            raise RuntimeError("Runtime render returned empty bytes.")
        return {
            "service_url": configured_url,
            "runtime_info": dict(runtime_client.runtime_info()),
            "preflight_checks": [check.to_dict() for check in checks],
            "create_payload": create_payload,
            "reset_payload": reset_payload,
            "step_payload": step_payload,
            "render_bytes": len(render_payload),
            "session_work_dir": str(session_work_dir),
        }
    finally:
        if service_process is not None:
            service_process.terminate()
            try:
                service_process.wait(timeout=15)
            except subprocess.TimeoutExpired:  # pragma: no cover - process cleanup
                service_process.kill()
