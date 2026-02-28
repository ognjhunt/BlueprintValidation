"""Tests for pi0.5 policy adapter behavior."""

from __future__ import annotations

import json

import numpy as np


def _make_pi05_adapter(tmp_path):
    from blueprint_validation.config import Pi05AdapterBackendConfig, PolicyAdapterConfig
    from blueprint_validation.policy_adapters.pi05_adapter import Pi05PolicyAdapter

    cfg = PolicyAdapterConfig(
        name="pi05",
        pi05=Pi05AdapterBackendConfig(
            openpi_repo=tmp_path / "openpi",
            profile="pi05_libero",
            policy_action_dim=7,
            policy_state_dim=7,
        ),
    )
    return Pi05PolicyAdapter(cfg)


def test_pi05_predict_action_queues_chunk(tmp_path):
    from blueprint_validation.policy_adapters.base import PolicyHandle

    adapter = _make_pi05_adapter(tmp_path)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    calls = {"n": 0}

    class FakeModel:
        def infer(self, observation):
            calls["n"] += 1
            assert "image" in observation
            return np.asarray(
                [
                    [1, 2, 3, 4, 5, 6, 7],
                    [8, 9, 10, 11, 12, 13, 14],
                ],
                dtype=np.float32,
            )

    handle = PolicyHandle(model=FakeModel(), processor=None, metadata={"pending_actions": []})
    a0 = adapter.predict_action(handle, frame, "pick", "bridge_orig", "cpu")
    a1 = adapter.predict_action(handle, frame, "pick", "bridge_orig", "cpu")

    assert calls["n"] == 1
    assert a0.shape == (7,)
    assert a1.shape == (7,)
    assert float(a0[0]) == 1.0
    assert float(a1[0]) == 8.0


def test_pi05_profile_mapping_droid_adds_right_wrist(tmp_path):
    from blueprint_validation.config import Pi05AdapterBackendConfig, PolicyAdapterConfig
    from blueprint_validation.policy_adapters.pi05_adapter import Pi05PolicyAdapter

    adapter = Pi05PolicyAdapter(
        PolicyAdapterConfig(
            name="pi05",
            pi05=Pi05AdapterBackendConfig(
                openpi_repo=tmp_path / "openpi",
                profile="pi05_droid",
                policy_action_dim=7,
                policy_state_dim=9,
            ),
        )
    )
    obs = adapter._build_observation(np.zeros((8, 8, 3), dtype=np.uint8), "task")
    assert "right_wrist_image" in obs
    assert len(obs["state"]) == 9


def test_pi05_dataset_transform_converts_rlds(tmp_path):
    adapter = _make_pi05_adapter(tmp_path)

    source = tmp_path / "source"
    source.mkdir(parents=True)
    image = source / "0000.jpg"
    image.write_bytes(b"x")
    payload = {
        "episode_id": "ep0",
        "task": "Pick object",
        "steps": [
            {
                "observation": {"image_path": str(image)},
                "action": [0.1, 0.2],
                "language_instruction": "Pick object",
            }
        ],
    }
    (source / "episodes.jsonl").write_text(json.dumps(payload) + "\n")

    dataset_dir = adapter.dataset_transform(
        source_dataset_dir=source,
        output_root=tmp_path / "out",
        dataset_name="bridge_orig",
    )
    assert (dataset_dir / "train" / "episodes.jsonl").exists()
