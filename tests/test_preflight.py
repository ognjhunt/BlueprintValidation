"""Tests for preflight checks."""

from __future__ import annotations

import sys
import types


def test_check_gpu_uses_total_memory(monkeypatch):
    from blueprint_validation.preflight import check_gpu

    props = types.SimpleNamespace(total_memory=80 * 1024**3)
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda idx: "Fake H100",
        get_device_properties=lambda idx: props,
    )
    fake_torch = types.SimpleNamespace(cuda=cuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = check_gpu()
    assert result.passed is True
    assert "80GB" in result.detail


def test_check_gpu_fallback_total_mem(monkeypatch):
    from blueprint_validation.preflight import check_gpu

    props = types.SimpleNamespace(total_mem=24 * 1024**3)
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda idx: "Legacy GPU",
        get_device_properties=lambda idx: props,
    )
    fake_torch = types.SimpleNamespace(cuda=cuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = check_gpu()
    assert result.passed is True
    assert "24GB" in result.detail


def test_check_gpu_handles_runtime_errors(monkeypatch):
    from blueprint_validation.preflight import check_gpu

    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda idx: (_ for _ in ()).throw(RuntimeError("driver error")),
        get_device_properties=lambda idx: None,
    )
    fake_torch = types.SimpleNamespace(cuda=cuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = check_gpu()
    assert result.passed is False
    assert "GPU check failed" in result.detail
