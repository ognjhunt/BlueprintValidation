from __future__ import annotations

from blueprint_validation.provider_benchmark import build_provider_benchmark_report


def test_provider_benchmark_report_prefers_successful_provider() -> None:
    report = build_provider_benchmark_report(
        [
            {
                "provider_name": "world_labs",
                "status": "succeeded",
                "preview_manifest_uri": "gs://bucket/preview.json",
                "artifact_uris": {"still": "gs://bucket/still.png"},
                "latency_ms": 1200,
            },
            {
                "provider_name": "stub_preview",
                "status": "failed",
                "latency_ms": 50,
            },
        ]
    )

    assert report["recommended_provider"] == "world_labs"
    assert report["providers"][0]["preview_usefulness_score"] > report["providers"][1]["preview_usefulness_score"]
