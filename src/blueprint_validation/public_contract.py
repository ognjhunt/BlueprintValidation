"""Customer-facing helpers for NeoVerse session/runtime reporting."""

from __future__ import annotations

from typing import Any, Dict, Mapping


_PUBLIC_RUNTIME_LABELS = {
    "neoverse": "NeoVerse runtime",
    "neoverse_service": "NeoVerse runtime service",
}


def public_runtime_label(backend: object) -> str:
    key = str(backend or "").strip().lower()
    if not key:
        return "NeoVerse runtime"
    return _PUBLIC_RUNTIME_LABELS.get(key, key.replace("_", " ").title())


def build_standardized_eval_report(report_data: Mapping[str, Any]) -> Dict[str, Any]:
    session = report_data.get("session") if isinstance(report_data.get("session"), Mapping) else {}
    batch = report_data.get("batch") if isinstance(report_data.get("batch"), Mapping) else {}
    export_data = report_data.get("export") if isinstance(report_data.get("export"), Mapping) else {}
    return {
        "schema_version": "v1",
        "project_name": report_data.get("project_name"),
        "session_id": session.get("session_id"),
        "site_world_id": session.get("site_world_id"),
        "runtime_backend_public": public_runtime_label(session.get("runtime_backend_selected")),
        "session_status": session.get("status"),
        "batch_summary": batch.get("summary"),
        "export_artifacts": export_data.get("artifact_uris", {}),
    }
