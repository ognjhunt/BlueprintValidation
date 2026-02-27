"""Tests for OpenVLA runner integration contracts."""

from __future__ import annotations

import numpy as np
import pytest


def test_run_rollout_passes_unnorm_key():
    from blueprint_validation.evaluation.openvla_runner import run_rollout

    class DummyInputs(dict):
        def to(self, *args, **kwargs):
            return self

    class DummyProcessor:
        def __call__(self, prompt, image, return_tensors="pt"):
            return DummyInputs(pixel_values=np.zeros((1,)))

    class DummyModel:
        def __init__(self):
            self.kwargs_seen = []

        def predict_action(self, **kwargs):
            self.kwargs_seen.append(kwargs)
            return np.array([0.1, 0.2, 0.3])

    class DummyWorld:
        def predict_next_frame(self, current_frame, action):
            return current_frame

    model = DummyModel()
    processor = DummyProcessor()
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    run_rollout(
        world_model=DummyWorld(),
        openvla_model=model,
        openvla_processor=processor,
        initial_frame=frame,
        task_prompt="go forward",
        max_steps=2,
        unnorm_key="bridge_orig",
        device="cpu",
    )
    assert model.kwargs_seen
    assert model.kwargs_seen[0]["unnorm_key"] == "bridge_orig"


def test_load_dreamdojo_world_model_no_stub(monkeypatch, tmp_path):
    from blueprint_validation.evaluation.openvla_runner import load_dreamdojo_world_model

    checkpoint = tmp_path / "checkpoints" / "DreamDojo" / "2B_pretrain"
    checkpoint.mkdir(parents=True)

    monkeypatch.delitem(__import__("sys").modules, "cosmos_predict2", raising=False)
    with pytest.raises(RuntimeError, match="not importable"):
        load_dreamdojo_world_model(
            checkpoint_path=checkpoint,
            adapted_checkpoint=None,
            device="cpu",
        )
