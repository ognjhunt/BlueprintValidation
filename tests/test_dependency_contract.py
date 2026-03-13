"""Dependency-shape tests for lean installs and explicit extras."""

from __future__ import annotations

import tomllib
from pathlib import Path


def _pyproject() -> dict:
    root = Path(__file__).resolve().parents[1]
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))


def test_default_dependencies_exclude_heavy_render_and_training_packages():
    project = _pyproject()["project"]
    dependencies = set(project["dependencies"])

    unexpected_prefixes = {
        "torch>=",
        "torchvision>=",
        "gsplat>=",
        "torchmetrics[image]>=",
        "lpips>=",
        "transformers>=",
        "accelerate>=",
        "peft>=",
        "opencv-python>=",
    }

    assert not {dep for dep in dependencies if dep in unexpected_prefixes}


def test_heavy_dependencies_live_in_explicit_extras():
    extras = _pyproject()["project"]["optional-dependencies"]

    assert "opencv-python>=4.8.0" in extras["vision"]
    assert "gsplat>=1.0.0" in extras["render"]
    assert "lpips>=0.1.4" in extras["render"]
    assert "transformers>=4.40.0" in extras["training"]
    assert "peft>=0.7.0" in extras["training"]
    assert "torch>=2.2.0" in extras["full"]
    assert "tensorflow==2.15.0" in extras["full"]


def test_uv_sync_dev_group_exists_for_local_ci_workflows():
    groups = _pyproject()["dependency-groups"]

    assert "pytest>=7.0.0" in groups["dev"]
    assert "ruff>=0.6.0" in groups["dev"]
