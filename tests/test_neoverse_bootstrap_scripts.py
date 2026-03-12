from __future__ import annotations

from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_runtime_env_example_mentions_neoverse_service_contract() -> None:
    text = _read("/Users/nijelhunt_1/workspace/BlueprintValidation/scripts/runtime_env.example")
    assert "NEOVERSE_RUNTIME_SERVICE_URL" in text
    assert "NEOVERSE_RUNTIME_SERVICE_TIMEOUT_SECONDS" in text
    assert "NEOVERSE_HOSTED_RUNTIME_MODULE" not in text
    assert "NEOVERSE_HOSTED_RUNTIME_CLASS" not in text


def test_bootstrap_scripts_require_and_persist_neoverse_service() -> None:
    cloud_prepare = _read("/Users/nijelhunt_1/workspace/BlueprintValidation/scripts/cloud_prepare_0787.sh")
    provision = _read("/Users/nijelhunt_1/workspace/BlueprintValidation/scripts/provision_same_facility_claim_runtime.sh")
    for text in (cloud_prepare, provision):
        assert "NeoVerse runtime service URL not configured; set NEOVERSE_RUNTIME_SERVICE_URL." in text
        assert 'persist_neoverse_runtime_env' in text
        assert 'NEOVERSE_RUNTIME_SERVICE_URL' in text
        assert 'NEOVERSE_HOSTED_RUNTIME_MODULE' not in text
        assert 'NEOVERSE_HOSTED_RUNTIME_CLASS' not in text
