"""Helpers for optional dependency imports with actionable install hints."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def require_optional_dependency(module_name: str, *, extra: str, purpose: str) -> ModuleType:
    """Import an optional dependency or raise a clear runtime error."""
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        root_module = module_name.split(".", 1)[0]
        if exc.name not in {module_name, root_module}:
            raise
        raise RuntimeError(
            f"{root_module} is required for {purpose}. "
            f"Install the '{extra}' extra via `uv sync --extra {extra}` "
            f"or `pip install -e .[{extra}]`."
        ) from exc
