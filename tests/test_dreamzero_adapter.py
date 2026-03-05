"""Tests for DreamZero policy adapter action-only integration mode."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_dreamzero_adapter_frame_history_and_action_padding(monkeypatch):
    from blueprint_validation.config import PolicyAdapterConfig
    from blueprint_validation.policy_adapters.dreamzero_adapter import DreamZeroPolicyAdapter

    cfg = PolicyAdapterConfig(name="dreamzero")
    cfg.dreamzero.frame_history = 2
    cfg.dreamzero.policy_action_dim = 4
    cfg.dreamzero.strict_action_contract = False

    call_history_lengths: list[int] = []

    class FakeRuntime:
        def predict_action(self, *, frames, prompt):
            del prompt
            call_history_lengths.append(len(frames))
            return {"action": [1.0, 2.0]}

    adapter = DreamZeroPolicyAdapter(cfg)
    monkeypatch.setattr(adapter, "_instantiate_runtime", lambda **kwargs: FakeRuntime())

    handle = adapter.load_policy("dz-base", checkpoint_path=None, device="cpu")
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    a1 = adapter.predict_action(handle, frame, "task", None, "cpu")
    a2 = adapter.predict_action(handle, frame, "task", None, "cpu")
    a3 = adapter.predict_action(handle, frame, "task", None, "cpu")

    assert call_history_lengths == [1, 2, 2]
    assert a1.shape == (4,)
    assert a2.shape == (4,)
    assert a3.shape == (4,)
    assert np.allclose(a1, np.array([1.0, 2.0, 0.0, 0.0], dtype=np.float32))


def test_dreamzero_adapter_strict_action_contract_raises(monkeypatch):
    from blueprint_validation.config import PolicyAdapterConfig
    from blueprint_validation.policy_adapters.dreamzero_adapter import DreamZeroPolicyAdapter

    cfg = PolicyAdapterConfig(name="dreamzero")
    cfg.dreamzero.policy_action_dim = 7
    cfg.dreamzero.strict_action_contract = True

    class FakeRuntime:
        def predict_action(self, *, frames, prompt):
            del frames, prompt
            return [0.1, 0.2, 0.3]

    adapter = DreamZeroPolicyAdapter(cfg)
    monkeypatch.setattr(adapter, "_instantiate_runtime", lambda **kwargs: FakeRuntime())
    handle = adapter.load_policy("dz-base", checkpoint_path=None, device="cpu")

    with pytest.raises(RuntimeError, match="strict_action_contract"):
        adapter.predict_action(handle, np.zeros((4, 4, 3), dtype=np.uint8), "task", None, "cpu")


def test_dreamzero_adapter_train_policy_disabled_by_default(tmp_path):
    from blueprint_validation.config import PolicyAdapterConfig, PolicyFinetuneConfig
    from blueprint_validation.policy_adapters.dreamzero_adapter import DreamZeroPolicyAdapter

    cfg = PolicyAdapterConfig(name="dreamzero")
    cfg.dreamzero.allow_training = False

    adapter = DreamZeroPolicyAdapter(cfg)
    result = adapter.train_policy(
        base_model_name="dz-base",
        base_checkpoint=None,
        dataset_root=tmp_path / "dataset_root",
        dataset_name="bridge_orig",
        output_dir=tmp_path / "out",
        finetune_config=PolicyFinetuneConfig(),
    )
    assert result.status == "skipped"
    assert "disabled" in result.detail.lower()
    assert result.raw.get("reason") == "dreamzero_training_disabled"
