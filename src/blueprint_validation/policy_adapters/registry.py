"""Policy adapter registry."""

from __future__ import annotations

from .base import PolicyAdapter
from .openvla_oft_adapter import OpenVLAOFTPolicyAdapter


def get_policy_adapter(name: str) -> PolicyAdapter:
    key = (name or "").strip().lower()
    if key in {"openvla_oft", "openvla-oft", "oft", "openvla", "open-vla"}:
        # Keep legacy "openvla" aliases working, but route all execution to OFT.
        return OpenVLAOFTPolicyAdapter()
    raise ValueError(
        f"Unsupported policy adapter: {name}. "
        "Supported adapters: openvla_oft"
    )
