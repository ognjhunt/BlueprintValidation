"""Configuration loader for runtime-aware Blueprint validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .runtime_backend import DEFAULT_RUNTIME_BACKEND_KIND, RuntimeBackendKind, normalize_runtime_kind


@dataclass(frozen=True)
class NeoVerseServiceConfig:
    service_url: str = ""
    api_key_env: str = "NEOVERSE_RUNTIME_SERVICE_API_KEY"
    timeout_seconds: int = 120


@dataclass(frozen=True)
class NeoVerseModelConfig:
    model_root: str = ""
    checkpoint_path: str = ""
    cache_root: str = ""
    runner_command: str = ""
    device: str = "cuda:0"
    gpu_enabled: bool = True


@dataclass(frozen=True)
class SceneMemoryRuntimeConfig:
    enabled: bool = True
    required_runtime_kind: RuntimeBackendKind = DEFAULT_RUNTIME_BACKEND_KIND
    allow_smoke_fallback: bool = False
    neoverse_service: NeoVerseServiceConfig = field(default_factory=NeoVerseServiceConfig)
    neoverse_model: NeoVerseModelConfig = field(default_factory=NeoVerseModelConfig)


@dataclass(frozen=True)
class ValidationConfig:
    project_name: str = "Blueprint Validation"
    scene_memory_runtime: SceneMemoryRuntimeConfig = field(default_factory=SceneMemoryRuntimeConfig)


def _as_mapping(raw: Any) -> Mapping[str, Any]:
    return raw if isinstance(raw, Mapping) else {}


def load_config(path: str | Path) -> ValidationConfig:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = _as_mapping(payload)
    runtime_raw = _as_mapping(root.get("scene_memory_runtime"))
    service_raw = _as_mapping(runtime_raw.get("neoverse_service"))

    return ValidationConfig(
        project_name=str(root.get("project_name") or "Blueprint Validation"),
        scene_memory_runtime=SceneMemoryRuntimeConfig(
            enabled=bool(runtime_raw.get("enabled", True)),
            required_runtime_kind=normalize_runtime_kind(runtime_raw.get("required_runtime_kind")),
            allow_smoke_fallback=bool(runtime_raw.get("allow_smoke_fallback", False)),
            neoverse_service=NeoVerseServiceConfig(
                service_url=str(service_raw.get("service_url") or "").rstrip("/"),
                api_key_env=str(service_raw.get("api_key_env") or "NEOVERSE_RUNTIME_SERVICE_API_KEY"),
                timeout_seconds=max(1, int(service_raw.get("timeout_seconds", 120) or 120)),
            ),
            neoverse_model=NeoVerseModelConfig(
                model_root=str(_as_mapping(runtime_raw.get("neoverse_model")).get("model_root") or "").strip(),
                checkpoint_path=str(_as_mapping(runtime_raw.get("neoverse_model")).get("checkpoint_path") or "").strip(),
                cache_root=str(_as_mapping(runtime_raw.get("neoverse_model")).get("cache_root") or "").strip(),
                runner_command=str(_as_mapping(runtime_raw.get("neoverse_model")).get("runner_command") or "").strip(),
                device=str(_as_mapping(runtime_raw.get("neoverse_model")).get("device") or "cuda").strip() or "cuda",
                gpu_enabled=bool(_as_mapping(runtime_raw.get("neoverse_model")).get("gpu_enabled", True)),
            ),
        ),
    )
