"""Tests for Stage 4b rollout dataset export validation boundaries."""

from __future__ import annotations

import json
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


def test_rollout_dataset_stage_fails_on_invalid_policy_scores_manifest(sample_config, tmp_path):
    from blueprint_validation.common import write_json
    from blueprint_validation.stages.s4b_rollout_dataset import RolloutDatasetStage

    sample_config.eval_policy.headline_scope = "dual"
    sample_config.rollout_dataset.enabled = True

    scores_path = tmp_path / "policy_eval" / "vlm_scores.json"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    write_json({"scores": [123]}, scores_path)

    stage = RolloutDatasetStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, {})
    assert result.status == "failed"
    assert "Invalid policy scores manifest" in result.detail
    assert "must be object" in result.detail


def test_rollout_dataset_stage_reserves_heldout_tasks_for_eval_only(sample_config, tmp_path):
    from blueprint_validation.common import write_json
    from blueprint_validation.stages.s4b_rollout_dataset import RolloutDatasetStage

    sample_config.eval_policy.headline_scope = "dual"
    sample_config.rollout_dataset.enabled = True
    sample_config.policy_compare.enabled = True
    sample_config.policy_compare.heldout_tasks = ["Heldout task"]
    sample_config.rollout_dataset.task_score_threshold = 5.0
    sample_config.rollout_dataset.min_steps_per_rollout = 2

    video = tmp_path / "policy_eval" / "rollout.mp4"
    _write_tiny_video(video)

    scores = {
        "scores": [
            {
                "condition": "baseline",
                "task": "Seen task",
                "task_score": 7.0,
                "rollout_index": 0,
                "video_path": str(video),
                "action_sequence": [[0.0] * 7 for _ in range(5)],
            },
            {
                "condition": "adapted",
                "task": "Seen task",
                "task_score": 8.0,
                "rollout_index": 0,
                "video_path": str(video),
                "action_sequence": [[0.0] * 7 for _ in range(5)],
            },
            {
                "condition": "baseline",
                "task": "Seen task",
                "task_score": 7.0,
                "rollout_index": 1,
                "video_path": str(video),
                "action_sequence": [[0.0] * 7 for _ in range(5)],
            },
            {
                "condition": "adapted",
                "task": "Seen task",
                "task_score": 8.0,
                "rollout_index": 1,
                "video_path": str(video),
                "action_sequence": [[0.0] * 7 for _ in range(5)],
            },
            {
                "condition": "baseline",
                "task": "Heldout task",
                "task_score": 7.0,
                "rollout_index": 2,
                "video_path": str(video),
                "action_sequence": [[0.0] * 7 for _ in range(5)],
            },
            {
                "condition": "adapted",
                "task": "Heldout task",
                "task_score": 8.0,
                "rollout_index": 2,
                "video_path": str(video),
                "action_sequence": [[0.0] * 7 for _ in range(5)],
            },
        ]
    }
    scores_path = tmp_path / "policy_eval" / "vlm_scores.json"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(scores, scores_path)

    stage = RolloutDatasetStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, {})
    assert result.status == "success"
    assert result.metrics["num_forced_heldout_pairs"] == 1

    train_jsonl = Path(result.outputs["baseline_dataset_root"]) / "train" / "episodes.jsonl"
    heldout_jsonl = Path(result.outputs["baseline_dataset_root"]) / "heldout" / "episodes.jsonl"

    train_tasks = [json.loads(line)["task"] for line in train_jsonl.read_text().splitlines()]
    heldout_tasks = [json.loads(line)["task"] for line in heldout_jsonl.read_text().splitlines()]
    assert "Heldout task" not in train_tasks
    assert "Heldout task" in heldout_tasks
