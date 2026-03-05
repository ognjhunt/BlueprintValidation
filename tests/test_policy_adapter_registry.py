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


def test_registry_maps_dreamzero_aliases():
    from blueprint_validation.config import PolicyAdapterConfig
    from blueprint_validation.policy_adapters.registry import get_policy_adapter

    for alias in ("dreamzero", "dream-zero", "dz"):
        adapter = get_policy_adapter(PolicyAdapterConfig(name=alias))
        assert adapter.name == "dreamzero"


def test_registry_rejects_unknown_adapter():
    import pytest

    from blueprint_validation.config import PolicyAdapterConfig
    from blueprint_validation.policy_adapters.registry import get_policy_adapter

    with pytest.raises(ValueError, match="Supported adapters: openvla_oft, pi05, dreamzero"):
        get_policy_adapter(PolicyAdapterConfig(name="unknown_policy"))


def test_openvla_adapter_prefers_adapter_owned_base_reference():
    from pathlib import Path

    from blueprint_validation.config import PolicyAdapterConfig, PolicyEvalConfig
    from blueprint_validation.policy_adapters.openvla_oft_adapter import OpenVLAOFTPolicyAdapter

    cfg = PolicyAdapterConfig(name="openvla_oft")
    cfg.openvla.base_model_name = "openvla/custom-7b"
    cfg.openvla.base_checkpoint_path = Path("/tmp/openvla_adapter_ckpt")

    adapter = OpenVLAOFTPolicyAdapter(cfg)
    model_name, checkpoint = adapter.base_model_ref(
        PolicyEvalConfig(
            model_name="openvla/openvla-7b",
            checkpoint_path=Path("/tmp/openvla_eval_ckpt"),
        )
    )

    assert model_name == "openvla/custom-7b"
    assert checkpoint == Path("/tmp/openvla_adapter_ckpt")
