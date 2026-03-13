"""Minimal configuration loader for the NeoVerse-only runtime path."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class NeoVerseServiceConfig:
    service_url: str = ""
    api_key_env: str = "NEOVERSE_RUNTIME_SERVICE_API_KEY"
    timeout_seconds: int = 120


@dataclass(frozen=True)
class SceneMemoryRuntimeConfig:
    enabled: bool = True
    neoverse_service: NeoVerseServiceConfig = field(default_factory=NeoVerseServiceConfig)


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
            neoverse_service=NeoVerseServiceConfig(
                service_url=str(service_raw.get("service_url") or "").rstrip("/"),
                api_key_env=str(service_raw.get("api_key_env") or "NEOVERSE_RUNTIME_SERVICE_API_KEY"),
                timeout_seconds=max(1, int(service_raw.get("timeout_seconds", 120) or 120)),
            ),
        ),
    )
