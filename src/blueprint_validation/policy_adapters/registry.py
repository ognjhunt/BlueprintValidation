"""Policy adapter registry."""

from __future__ import annotations

from ..config import PolicyAdapterConfig
from .base import PolicyAdapter
from .openvla_oft_adapter import OpenVLAOFTPolicyAdapter
from .pi05_adapter import Pi05PolicyAdapter


def get_policy_adapter(adapter_config: PolicyAdapterConfig) -> PolicyAdapter:
    key = (adapter_config.name or "").strip().lower()
    if key in {"openvla_oft", "openvla-oft", "oft", "openvla", "open-vla"}:
        # Keep legacy "openvla" aliases working, but route all execution to OFT.
        return OpenVLAOFTPolicyAdapter(adapter_config)
    if key in {"pi05", "pi0.5", "openpi"}:
        return Pi05PolicyAdapter(adapter_config)
    raise ValueError(
        f"Unsupported policy adapter: {adapter_config.name}. Supported adapters: openvla_oft, pi05"
    )
