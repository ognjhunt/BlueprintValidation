"""Policy adapter registry."""

from __future__ import annotations

from .base import PolicyAdapter
from .openvla_adapter import OpenVLAPolicyAdapter


def get_policy_adapter(name: str) -> PolicyAdapter:
    key = (name or "").strip().lower()
    if key in {"openvla", "open-vla"}:
        return OpenVLAPolicyAdapter()
    raise ValueError(
        f"Unsupported policy adapter: {name}. "
        "Supported adapters: openvla"
    )
