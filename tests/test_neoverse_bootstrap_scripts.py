from __future__ import annotations

from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_runtime_env_example_mentions_neoverse_bootstrap_contract() -> None:
    text = _read("/Users/nijelhunt_1/workspace/BlueprintValidation/scripts/runtime_env.example")
    assert "NEOVERSE_REPO_URL" in text
    assert "NEOVERSE_REPO_REF" in text
    assert "NEOVERSE_REPO_PATH" in text
    assert "NEOVERSE_HOSTED_RUNTIME_MODULE" not in text
    assert "NEOVERSE_HOSTED_RUNTIME_CLASS" not in text


def test_runtime_snapshot_build_and_runpod_image_wire_neoverse_clone() -> None:
    snapshot = _read("/Users/nijelhunt_1/workspace/BlueprintValidation/scripts/build_runtime_snapshot.sh")
    dockerfile = _read("/Users/nijelhunt_1/workspace/BlueprintValidation/docker/runpod.Dockerfile")
    assert 'NEOVERSE_REPO_URL="${NEOVERSE_REPO_URL:-}"' in snapshot
    assert '--build-arg NEOVERSE_REPO_URL="$NEOVERSE_REPO_URL"' in snapshot
    assert 'ARG NEOVERSE_REPO_URL=' in dockerfile
    assert 'resolve_vendor_or_clone "neoverse"' in dockerfile


def test_bootstrap_scripts_require_and_persist_neoverse_runtime() -> None:
    cloud_prepare = _read("/Users/nijelhunt_1/workspace/BlueprintValidation/scripts/cloud_prepare_0787.sh")
    provision = _read("/Users/nijelhunt_1/workspace/BlueprintValidation/scripts/provision_same_facility_claim_runtime.sh")
    for text in (cloud_prepare, provision):
        assert "NeoVerse runtime not installed; set NEOVERSE_REPO_URL or preinstall /opt/neoverse." in text
        assert 'persist_neoverse_runtime_env' in text
        assert 'NEOVERSE_REPO_PATH' in text
        assert 'NEOVERSE_HOSTED_RUNTIME_MODULE' not in text
        assert 'NEOVERSE_HOSTED_RUNTIME_CLASS' not in text
