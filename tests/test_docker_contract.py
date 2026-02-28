"""Static checks for RunPod Docker reproducibility contract."""

from __future__ import annotations

from pathlib import Path


def test_runpod_dockerfile_uses_optional_vendor_strategy():
    dockerfile = Path("docker/runpod.Dockerfile").read_text(encoding="utf-8")

    # Build must not hard-require untracked data/vendor trees from local workspaces.
    assert "COPY data/vendor/" not in dockerfile

    # Contract: use vendored repos when present, otherwise clone pinned refs.
    assert "ARG VENDOR_STRATEGY" in dockerfile
    assert "resolve_vendor_or_clone" in dockerfile
    assert "ARG DREAMDOJO_REF" in dockerfile
    assert "ARG COSMOS_REF" in dockerfile
    assert "ARG OPENVLA_REF" in dockerfile
