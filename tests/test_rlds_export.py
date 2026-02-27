"""Tests for rollout RLDS-style export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_tiny_video(path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (16, 16))
    for _ in range(6):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_rollout_success_manipulation_flags():
    from blueprint_validation.training.rlds_export import rollout_success

    entry = {
        "is_manipulation_task": True,
        "grasp_acquired": True,
        "lifted_clear": True,
        "placed_in_target": True,
        "task_score": 1.0,
    }
    assert rollout_success(entry, task_threshold=7.0) is True


def test_export_rollouts_to_rlds_jsonl(tmp_path):
    from blueprint_validation.training.rlds_export import export_rollouts_to_rlds_jsonl

    video = tmp_path / "rollout.mp4"
    _write_tiny_video(video)
    rollouts = [
        {
            "rollout_index": 0,
            "task": "Pick up tote",
            "task_score": 8.0,
            "video_path": str(video),
            "action_sequence": [[0.0] * 7 for _ in range(5)],
            "is_manipulation_task": True,
            "grasp_acquired": True,
            "lifted_clear": True,
            "placed_in_target": True,
            "stable_after_place": True,
        }
    ]
    out_dir = tmp_path / "dataset"
    meta = export_rollouts_to_rlds_jsonl(
        rollouts=rollouts,
        output_dir=out_dir,
        condition="adapted",
        split="train",
        task_threshold=7.0,
        min_steps_per_rollout=4,
        include_failed_rollouts=False,
    )
    assert meta["num_episodes"] == 1
    assert (out_dir / "episodes.jsonl").exists()
    assert (out_dir / "episodes_meta.json").exists()
