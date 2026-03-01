"""Tests for OpenVLA-OFT runner integration contracts."""

from __future__ import annotations

import sys

import numpy as np
import pytest


def test_normalize_action_chunk_pads_to_ratio():
    from blueprint_validation.evaluation.openvla_runner import _normalize_action_chunk

    arr = _normalize_action_chunk(
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        expected_action_dim=3,
        actions_per_latent_frame=4,
    )
    assert arr.shape == (4, 3)
    assert np.allclose(arr[0], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.allclose(arr[-1], np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_normalize_action_chunk_raises_on_dim_mismatch():
    from blueprint_validation.evaluation.openvla_runner import _normalize_action_chunk

    with pytest.raises(RuntimeError, match="Action-space mismatch"):
        _normalize_action_chunk(
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            expected_action_dim=7,
            actions_per_latent_frame=4,
        )


def test_normalize_action_chunk_honors_min_steps():
    from blueprint_validation.evaluation.openvla_runner import _normalize_action_chunk

    arr = _normalize_action_chunk(
        np.array([1.0, 2.0], dtype=np.float32),
        expected_action_dim=2,
        actions_per_latent_frame=4,
        min_action_steps=12,
    )
    assert arr.shape == (12, 2)
    assert np.allclose(arr[0], np.array([1.0, 2.0], dtype=np.float32))
    assert np.allclose(arr[-1], np.array([1.0, 2.0], dtype=np.float32))


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

    monkeypatch.delitem(sys.modules, "cosmos_predict2", raising=False)
    with pytest.raises(RuntimeError, match="not importable"):
        load_dreamdojo_world_model(
            checkpoint_path=checkpoint,
            adapted_checkpoint=None,
            device="cpu",
        )


def test_load_dreamdojo_world_model_repo_path_fallback(monkeypatch, tmp_path):
    from blueprint_validation.evaluation.openvla_runner import load_dreamdojo_world_model

    checkpoint = tmp_path / "checkpoints" / "DreamDojo" / "2B_pretrain"
    checkpoint.mkdir(parents=True)

    repo = tmp_path / "DreamDojo"
    pkg = repo / "cosmos_predict2" / "action_conditioned"
    pkg.mkdir(parents=True)
    (repo / "cosmos_predict2" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "inference.py").write_text(
        """
class ActionConditionedInferenceArguments:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir


class _DummyModel:
    def to(self, _device):
        return self

    def predict_next_frame(self, frame, action):
        return frame


def setup(_args):
    return _DummyModel()
""".strip()
    )

    for key in list(sys.modules):
        if key == "cosmos_predict2" or key.startswith("cosmos_predict2."):
            monkeypatch.delitem(sys.modules, key, raising=False)

    model = load_dreamdojo_world_model(
        checkpoint_path=checkpoint,
        adapted_checkpoint=None,
        dreamdojo_repo=repo,
        device="cpu",
    )
    assert hasattr(model, "predict_next_frame")


def test_load_dreamdojo_world_model_repo_path_module_style(monkeypatch, tmp_path):
    from blueprint_validation.evaluation.openvla_runner import load_dreamdojo_world_model

    checkpoint = tmp_path / "checkpoints" / "DreamDojo" / "2B_pretrain"
    checkpoint.mkdir(parents=True)

    repo = tmp_path / "DreamDojo"
    pkg = repo / "cosmos_predict2"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "action_conditioned.py").write_text(
        """
class ActionConditionedInferenceArguments:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir


class _DummyModel:
    def to(self, _device):
        return self

    def predict_next_frame(self, frame, action):
        return frame


def setup(_args):
    return _DummyModel()
""".strip()
    )

    for key in list(sys.modules):
        if key == "cosmos_predict2" or key.startswith("cosmos_predict2."):
            monkeypatch.delitem(sys.modules, key, raising=False)

    model = load_dreamdojo_world_model(
        checkpoint_path=checkpoint,
        adapted_checkpoint=None,
        dreamdojo_repo=repo,
        device="cpu",
    )
    assert hasattr(model, "predict_next_frame")
