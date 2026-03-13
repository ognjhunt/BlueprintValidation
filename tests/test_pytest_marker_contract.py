"""Tests for pytest marker behavior around integration paths."""

from __future__ import annotations

import importlib.util
from pathlib import Path


class _DummyConfig:
    def __init__(self, **options):
        self._options = options

    def getoption(self, name: str):
        return self._options[name]


class _DummyItem:
    def __init__(self, path: Path, keywords: set[str] | None = None):
        self.path = path
        self.keywords = {key: True for key in (keywords or set())}
        self.markers: list[object] = []

    def add_marker(self, marker):
        self.markers.append(marker)


def _marker_names(item: _DummyItem) -> list[str]:
    names: list[str] = []
    for marker in item.markers:
        mark = getattr(marker, "mark", marker)
        names.append(mark.name)
    return names


def _load_repo_conftest():
    path = Path(__file__).resolve().parent / "conftest.py"
    spec = importlib.util.spec_from_file_location("blueprint_validation_test_conftest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collection_auto_marks_integration_paths():
    repo_conftest = _load_repo_conftest()
    item = _DummyItem(Path("tests/integration/test_stage_smoke.py"))
    config = _DummyConfig(**{"--run-gpu": False, "--run-slow": False, "--run-integration": True})

    repo_conftest.pytest_collection_modifyitems(config, [item])

    assert "integration" in _marker_names(item)


def test_collection_skips_integration_without_opt_in():
    repo_conftest = _load_repo_conftest()
    item = _DummyItem(Path("tests/integration/test_stage_smoke.py"))
    config = _DummyConfig(
        **{"--run-gpu": False, "--run-slow": False, "--run-integration": False}
    )

    repo_conftest.pytest_collection_modifyitems(config, [item])

    names = _marker_names(item)
    assert "integration" in names
    assert "skip" in names


def test_collection_keeps_regular_tests_unmarked():
    repo_conftest = _load_repo_conftest()
    item = _DummyItem(Path("tests/test_config.py"))
    config = _DummyConfig(
        **{"--run-gpu": False, "--run-slow": False, "--run-integration": False}
    )

    repo_conftest.pytest_collection_modifyitems(config, [item])

    assert "integration" not in _marker_names(item)
