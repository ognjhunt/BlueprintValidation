"""Provider preview benchmarking helpers."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


def _score_manifest(manifest: Mapping[str, Any]) -> float:
    status = str(manifest.get("status") or "").strip().lower()
    score = 0.0
    if status == "succeeded":
        score += 60
    elif status in {"submitted", "processing"}:
        score += 30

    if manifest.get("preview_manifest_uri"):
        score += 20
    if manifest.get("artifact_uris"):
        score += 10
    latency_ms = manifest.get("latency_ms")
    if isinstance(latency_ms, (int, float)) and latency_ms > 0:
        score += max(0.0, 10 - min(float(latency_ms) / 1000.0, 10.0))
    return score


def build_provider_benchmark_report(manifests: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    scored = []
    for manifest in manifests:
        entry = dict(manifest)
        entry["preview_usefulness_score"] = round(_score_manifest(manifest), 2)
        scored.append(entry)

    ranked = sorted(
        scored,
        key=lambda item: float(item.get("preview_usefulness_score") or 0.0),
        reverse=True,
    )
    winner = ranked[0] if ranked else None
    return {
        "schema_version": "v1",
        "providers": ranked,
        "recommended_provider": winner.get("provider_name") if winner else None,
        "recommended_preview_status": winner.get("status") if winner else None,
    }
