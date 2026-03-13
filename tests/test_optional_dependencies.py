"""Tests for explicit optional-dependency failure messages."""

from __future__ import annotations

from importlib import import_module

import pytest


def test_require_optional_dependency_returns_imported_module():
    from blueprint_validation.optional_dependencies import require_optional_dependency

    module = require_optional_dependency(
        "json",
        extra="vision",
        purpose="test coverage",
    )
    assert module is import_module("json")


def test_require_optional_dependency_raises_install_hint(monkeypatch):
    from blueprint_validation import optional_dependencies

    def _missing(name: str):
        raise ModuleNotFoundError("No module named 'cv2'", name="cv2")

    monkeypatch.setattr(optional_dependencies, "import_module", _missing)

    with pytest.raises(RuntimeError, match="uv sync --extra vision"):
        optional_dependencies.require_optional_dependency(
            "cv2",
            extra="vision",
            purpose="hosted session image and video handling",
        )


def test_require_optional_dependency_preserves_nested_import_errors(monkeypatch):
    from blueprint_validation import optional_dependencies

    def _nested_missing(name: str):
        raise ModuleNotFoundError("No module named 'pkg_resources'", name="pkg_resources")

    monkeypatch.setattr(optional_dependencies, "import_module", _nested_missing)

    with pytest.raises(ModuleNotFoundError, match="pkg_resources"):
        optional_dependencies.require_optional_dependency(
            "cv2",
            extra="vision",
            purpose="hosted session image and video handling",
        )
