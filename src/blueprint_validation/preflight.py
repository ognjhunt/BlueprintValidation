"""Preflight checks for the NeoVerse-only runtime path."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from blueprint_contracts.site_world_contract import SiteWorldIntakeError, load_site_world_bundle

from .common import PreflightCheck
from .config import ValidationConfig
from .neoverse_runtime_client import NeoVerseRuntimeClient, NeoVerseRuntimeClientConfig


def _configured_service_url(config: ValidationConfig) -> str:
    return (
        str(config.scene_memory_runtime.neoverse_service.service_url or "").strip()
        or str(os.environ.get("NEOVERSE_RUNTIME_SERVICE_URL") or "").strip()
    ).rstrip("/")


def _runtime_client(config: ValidationConfig) -> NeoVerseRuntimeClient:
    service_url = _configured_service_url(config)
    api_key_env = config.scene_memory_runtime.neoverse_service.api_key_env
    api_key = str(os.environ.get(api_key_env, "") or "").strip() if api_key_env else ""
    return NeoVerseRuntimeClient(
        NeoVerseRuntimeClientConfig(
            service_url=service_url,
            api_key=api_key,
            timeout_seconds=max(1, int(config.scene_memory_runtime.neoverse_service.timeout_seconds)),
        )
    )


def run_preflight(
    config: ValidationConfig,
    *,
    site_world_registration: str | Path | None = None,
) -> List[PreflightCheck]:
    checks: List[PreflightCheck] = []
    service_url = _configured_service_url(config)
    if not service_url:
        checks.append(
            PreflightCheck(
                name="runtime:service_url",
                passed=False,
                detail="NeoVerse runtime service URL is not configured.",
            )
        )
        return checks

    checks.append(PreflightCheck(name="runtime:service_url", passed=True, detail=service_url))
    client = _runtime_client(config)

    try:
        health = client.healthcheck()
        checks.append(
            PreflightCheck(
                name="runtime:healthz",
                passed=str(health.get("status") or "").strip().lower() == "ok",
                detail=str(health.get("service") or "ok"),
            )
        )
    except Exception as exc:
        checks.append(PreflightCheck(name="runtime:healthz", passed=False, detail=str(exc)))

    try:
        runtime = client.runtime_info()
        capabilities = runtime.get("capabilities") if isinstance(runtime.get("capabilities"), dict) else {}
        missing = [
            key
            for key in ("site_world_registration", "session_reset", "session_step", "session_render", "session_state")
            if not bool(capabilities.get(key))
        ]
        checks.append(
            PreflightCheck(
                name="runtime:capabilities",
                passed=not missing,
                detail="ok" if not missing else f"missing={','.join(missing)}",
            )
        )
    except Exception as exc:
        checks.append(PreflightCheck(name="runtime:capabilities", passed=False, detail=str(exc)))

    if site_world_registration is not None:
        registration_path = Path(site_world_registration).expanduser().resolve()
        try:
            bundle = load_site_world_bundle(registration_path, require_spec=True)
            launchable = bool(bundle.health.get("launchable", False))
            detail = "launchable" if launchable else ",".join(str(item) for item in bundle.health.get("blockers", []))
            checks.append(
                PreflightCheck(
                    name="site_world:bundle",
                    passed=True,
                    detail=str(bundle.registration.get("site_world_id") or registration_path),
                )
            )
            checks.append(
                PreflightCheck(
                    name="site_world:launchable",
                    passed=launchable,
                    detail=detail or "blocked",
                )
            )
        except (SiteWorldIntakeError, OSError, ValueError) as exc:
            checks.append(PreflightCheck(name="site_world:bundle", passed=False, detail=str(exc)))

    return checks
