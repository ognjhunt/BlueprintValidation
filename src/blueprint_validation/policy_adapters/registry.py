"""Policy adapter registry."""

from __future__ import annotations

from .base import PolicyAdapter
from .openvla_adapter import OpenVLAPolicyAdapter
from .openvla_oft_adapter import OpenVLAOFTPolicyAdapter


def get_policy_adapter(name: str) -> PolicyAdapter:
    key = (name or "").strip().lower()
    if key in {"openvla", "open-vla"}:
        return OpenVLAPolicyAdapter()
    if key in {"openvla_oft", "openvla-oft", "oft"}:
        return OpenVLAOFTPolicyAdapter()
    raise ValueError(
        f"Unsupported policy adapter: {name}. "
        "Supported adapters: openvla, openvla_oft"
    )
