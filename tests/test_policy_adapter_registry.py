"""Tests for policy adapter registry defaults and aliases."""

from __future__ import annotations


def test_registry_returns_openvla_oft():
    from blueprint_validation.policy_adapters.registry import get_policy_adapter

    adapter = get_policy_adapter("openvla_oft")
    assert adapter.name == "openvla_oft"


def test_registry_supports_oft_alias():
    from blueprint_validation.policy_adapters.registry import get_policy_adapter

    adapter = get_policy_adapter("oft")
    assert adapter.name == "openvla_oft"
