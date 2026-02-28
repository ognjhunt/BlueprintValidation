"""Tests for policy adapter registry defaults and aliases."""

from __future__ import annotations


def test_registry_returns_openvla_oft():
    from blueprint_validation.config import PolicyAdapterConfig
    from blueprint_validation.policy_adapters.registry import get_policy_adapter

    adapter = get_policy_adapter(PolicyAdapterConfig(name="openvla_oft"))
    assert adapter.name == "openvla_oft"


def test_registry_supports_oft_alias():
    from blueprint_validation.config import PolicyAdapterConfig
    from blueprint_validation.policy_adapters.registry import get_policy_adapter

    adapter = get_policy_adapter(PolicyAdapterConfig(name="oft"))
    assert adapter.name == "openvla_oft"


def test_registry_maps_legacy_openvla_alias_to_oft():
    from blueprint_validation.config import PolicyAdapterConfig
    from blueprint_validation.policy_adapters.registry import get_policy_adapter

    adapter = get_policy_adapter(PolicyAdapterConfig(name="openvla"))
    assert adapter.name == "openvla_oft"


def test_registry_maps_pi05_aliases():
    from blueprint_validation.config import PolicyAdapterConfig
    from blueprint_validation.policy_adapters.registry import get_policy_adapter

    for alias in ("pi05", "pi0.5", "openpi"):
        adapter = get_policy_adapter(PolicyAdapterConfig(name=alias))
        assert adapter.name == "pi05"


def test_registry_rejects_unknown_adapter():
    import pytest

    from blueprint_validation.config import PolicyAdapterConfig
    from blueprint_validation.policy_adapters.registry import get_policy_adapter

    with pytest.raises(ValueError, match="Supported adapters: openvla_oft, pi05"):
        get_policy_adapter(PolicyAdapterConfig(name="unknown_policy"))
