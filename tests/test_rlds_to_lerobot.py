"""Tests for RLDS -> LeRobot conversion utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_minimal_episode_jsonl(path: Path, image_path: Path) -> None:
    payload = {
        "episode_id": "ep0",
        "task": "Pick object",
        "steps": [
            {
                "observation": {"image_path": str(image_path)},
                "action": [0.1, 0.2, 0.3],
                "language_instruction": "Pick object",
                "reward": 0.0,
                "is_first": True,
                "is_last": False,
                "is_terminal": False,
            },
            {
                "observation": {"image_path": str(image_path)},
                "action": [0.4, 0.5, 0.6],
                "language_instruction": "Pick object",
                "reward": 1.0,
                "is_first": False,
                "is_last": True,
                "is_terminal": True,
            },
        ],
    }
    path.write_text(json.dumps(payload) + "\n")


def test_convert_rlds_to_lerobot_minimal(tmp_path):
    from blueprint_validation.training.rlds_to_lerobot import convert_rlds_to_lerobot_dataset

    source = tmp_path / "source"
    source.mkdir(parents=True)
    image = source / "0000.jpg"
    image.write_bytes(b"fake_image")
    _write_minimal_episode_jsonl(source / "episodes.jsonl", image)

    target = convert_rlds_to_lerobot_dataset(
        source_dataset_dir=source,
        output_root=tmp_path / "out",
        dataset_name="bridge_orig",
        profile="pi05_libero",
        policy_state_dim=7,
        policy_action_dim=5,
    )
    assert (target / "train" / "episodes.jsonl").exists()
    summary = json.loads((target / "conversion_summary.json").read_text())
    assert summary["total_episodes"] == 1
    assert summary["total_steps"] == 2
    assert summary["splits"]["train"]["episodes"] == 1


def test_convert_rlds_to_lerobot_fails_on_missing_image(tmp_path):
    from blueprint_validation.training.rlds_to_lerobot import convert_rlds_to_lerobot_dataset

    source = tmp_path / "source"
    source.mkdir(parents=True)
    missing_image = source / "missing.jpg"
    _write_minimal_episode_jsonl(source / "episodes.jsonl", missing_image)

    with pytest.raises(RuntimeError, match="Missing frame image referenced by dataset"):
        convert_rlds_to_lerobot_dataset(
            source_dataset_dir=source,
            output_root=tmp_path / "out",
            dataset_name="bridge_orig",
            profile="pi05_libero",
            policy_state_dim=7,
            policy_action_dim=7,
        )


def test_normalize_action_dim_padding_and_truncation():
    from blueprint_validation.training.rlds_to_lerobot import normalize_action_dim

    assert normalize_action_dim([1.0, 2.0], 4) == [1.0, 2.0, 0.0, 0.0]
    assert normalize_action_dim([1.0, 2.0, 3.0], 2) == [1.0, 2.0]


def test_infer_split_layout_detects_train_eval(tmp_path):
    from blueprint_validation.training.rlds_to_lerobot import infer_split_layout

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    (root / "eval").mkdir(parents=True)
    (root / "train" / "episodes.jsonl").write_text("")
    (root / "eval" / "episodes.jsonl").write_text("")
    assert infer_split_layout(root) == ("eval", "train")
