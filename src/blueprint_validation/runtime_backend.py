"""Compatibility shim for the shared runtime-service contract.

Prefer ``blueprint_contracts.runtime_service_contract`` when available. Keep a
local fallback so this repo remains usable until its pinned dependency catches
up to the extracted shared contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping

try:
    from blueprint_contracts.runtime_service_contract import (
        DEFAULT_RUNTIME_BACKEND_KIND,
        KNOWN_RUNTIME_BACKEND_KINDS,
        RuntimeBackendKind,
        RuntimeMetadata,
        normalize_runtime_kind,
        parse_runtime_metadata,
        runtime_kind_label,
        runtime_kind_matches,
    )
except ImportError:
    RuntimeBackendKind = Literal["smoke_contract", "native_world_model", "neoverse_production"]

    DEFAULT_RUNTIME_BACKEND_KIND: RuntimeBackendKind = "native_world_model"
    KNOWN_RUNTIME_BACKEND_KINDS = {"smoke_contract", "native_world_model", "neoverse_production"}

    @dataclass(frozen=True)
    class RuntimeMetadata:
        runtime_kind: RuntimeBackendKind
        production_grade: bool
        service: str
        version: str
        runtime_base_url: str
        websocket_base_url: str
        engine_identity: Dict[str, Any]
        model_identity: Dict[str, Any]
        checkpoint_identity: Dict[str, Any]
        state_guarantees: Dict[str, Any]
        capabilities: Dict[str, Any]
        readiness: Dict[str, Any]

        def to_dict(self) -> Dict[str, Any]:
            return {
                "service": self.service,
                "version": self.version,
                "api_version": "v1",
                "runtime_kind": self.runtime_kind,
                "production_grade": self.production_grade,
                "runtime_base_url": self.runtime_base_url,
                "websocket_base_url": self.websocket_base_url,
                "engine_identity": dict(self.engine_identity),
                "model_identity": dict(self.model_identity),
                "checkpoint_identity": dict(self.checkpoint_identity),
                "state_guarantees": dict(self.state_guarantees),
                "capabilities": dict(self.capabilities),
                "readiness": dict(self.readiness),
            }

    def normalize_runtime_kind(
        value: object,
        *,
        default: RuntimeBackendKind = DEFAULT_RUNTIME_BACKEND_KIND,
    ) -> RuntimeBackendKind:
        text = str(value or "").strip().lower()
        if text in KNOWN_RUNTIME_BACKEND_KINDS:
            return text  # type: ignore[return-value]
        return default

    def runtime_kind_label(kind: object) -> str:
        normalized = normalize_runtime_kind(kind)
        if normalized == "smoke_contract":
            return "Smoke contract runtime"
        if normalized == "native_world_model":
            return "Native world-model runtime"
        return "NeoVerse production runtime"

    def parse_runtime_metadata(payload: Mapping[str, Any]) -> RuntimeMetadata:
        runtime_base_url = str(payload.get("runtime_base_url") or "").rstrip("/")
        websocket_base_url = str(payload.get("websocket_base_url") or "").rstrip("/")
        return RuntimeMetadata(
            runtime_kind=normalize_runtime_kind(payload.get("runtime_kind")),
            production_grade=bool(payload.get("production_grade", False)),
            service=str(payload.get("service") or "site-world-runtime"),
            version=str(payload.get("version") or ""),
            runtime_base_url=runtime_base_url,
            websocket_base_url=websocket_base_url,
            engine_identity=dict(payload.get("engine_identity") or {}),
            model_identity=dict(payload.get("model_identity") or {}),
            checkpoint_identity=dict(payload.get("checkpoint_identity") or {}),
            state_guarantees=dict(payload.get("state_guarantees") or {}),
            capabilities=dict(payload.get("capabilities") or {}),
            readiness=dict(payload.get("readiness") or {}),
        )

    def runtime_kind_matches(
        runtime: Mapping[str, Any],
        *,
        required_kind: RuntimeBackendKind,
        allow_smoke_fallback: bool,
    ) -> tuple[bool, str]:
        metadata = parse_runtime_metadata(runtime)
        if metadata.runtime_kind == required_kind:
            return True, "ok"
        if allow_smoke_fallback and metadata.runtime_kind == "smoke_contract":
            return True, "smoke_fallback"
        return False, f"expected={required_kind},actual={metadata.runtime_kind}"

__all__ = [
    "DEFAULT_RUNTIME_BACKEND_KIND",
    "KNOWN_RUNTIME_BACKEND_KINDS",
    "RuntimeBackendKind",
    "RuntimeMetadata",
    "normalize_runtime_kind",
    "parse_runtime_metadata",
    "runtime_kind_label",
    "runtime_kind_matches",
]
