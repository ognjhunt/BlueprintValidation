"""Tests for mixed rollout curriculum utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_tiny_video(path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (16, 16))
    for _ in range(4):
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()


def test_bucket_rollout_uses_manipulation_flags():
    from blueprint_validation.training.rollout_curriculum import (
        RolloutBucketThresholds,
        bucket_rollout,
    )

    thr = RolloutBucketThresholds(task_score_threshold=7.0, near_miss_min_task_score=5.0)
    success = {
        "is_manipulation_task": True,
        "grasp_acquired": True,
        "lifted_clear": True,
        "placed_in_target": True,
        "task_score": 1.0,
    }
    near = {
        "is_manipulation_task": True,
        "grasp_acquired": True,
        "lifted_clear": False,
        "placed_in_target": False,
        "task_score": 1.0,
    }
    hard = {"is_manipulation_task": False, "task_score": 2.0}

    assert bucket_rollout(success, thr) == "success"
    assert bucket_rollout(near, thr) == "near_miss"
    assert bucket_rollout(hard, thr) == "hard_negative"


def test_bucket_rollouts_by_quantile_creates_non_hard_mix():
    from blueprint_validation.training.rollout_curriculum import bucket_rollouts_by_quantile

    rows = [
        {"rollout_index": idx, "task": "Pick tote", "task_score": score}
        for idx, score in enumerate([0.0, 1.0, 2.0, 3.0, 4.0])
    ]
    result = bucket_rollouts_by_quantile(
        rows,
        success_quantile=0.8,
        near_miss_quantile=0.4,
    )
    assert result["success"]
    assert result["near_miss"]
    assert result["success_threshold"] is not None
    assert result["near_miss_threshold"] is not None
    assert len(result["success"]) + len(result["near_miss"]) + len(result["hard_negative"]) == len(rows)


def test_sample_policy_curriculum_generates_disjoint_splits(sample_config, tmp_path):
    from blueprint_validation.training.rollout_curriculum import sample_policy_curriculum

    sample_config.rollout_dataset.selection_mode = "success_near_miss_hard"
    sample_config.rollout_dataset.near_miss_target_fraction = 0.3
    sample_config.rollout_dataset.hard_negative_target_fraction = 0.1
    sample_config.rollout_dataset.min_steps_per_rollout = 2

    rows = []
    for i, score in enumerate([8.0, 8.0, 6.0, 6.0, 3.0, 2.0]):
        rows.append(
            {
                "condition": "adapted",
                "task": "Pick tote",
                "rollout_index": i,
                "task_score": score,
                "video_path": str(tmp_path / "video.mp4"),
                "action_sequence": [[0.0] * 7 for _ in range(4)],
                "is_manipulation_task": False,
            }
        )

    result = sample_policy_curriculum(rows, sample_config.rollout_dataset, seed=17)
    train_ids = set(result["train_pair_ids"])
    eval_ids = set(result["eval_pair_ids"])
    assert train_ids
    assert eval_ids
    assert train_ids.isdisjoint(eval_ids)


def test_build_world_refresh_mix_includes_stage2_and_rollouts(sample_config, tmp_path):
    from blueprint_validation.training.rollout_curriculum import build_world_refresh_mix

    stage2_video = tmp_path / "stage2.mp4"
    success_video = tmp_path / "success.mp4"
    near_video = tmp_path / "near.mp4"
    hard_video = tmp_path / "hard.mp4"
    for path in [stage2_video, success_video, near_video, hard_video]:
        _write_tiny_video(path)

    stage2_manifest = {
        "clips": [
            {
                "clip_name": "clip_000",
                "variant_name": "daylight",
                "prompt": "daylight",
                "output_video_path": str(stage2_video),
                "input_video_path": str(stage2_video),
            }
        ]
    }
    selected = [{"rollout_index": 1, "task": "Pick tote", "video_path": str(success_video)}]
    near = [{"rollout_index": 2, "task": "Pick tote", "video_path": str(near_video)}]
    hard = [{"rollout_index": 3, "task": "Pick tote", "video_path": str(hard_video)}]

    mix = build_world_refresh_mix(
        stage2_manifest=stage2_manifest,
        selected_success=selected,
        near_miss=near,
        hard_negative=hard,
        cfg=sample_config.policy_rl_loop,
        seed=17,
    )
    assert mix["clips"]
    metrics = mix["mix_metrics"]
    assert metrics["selected_total"] >= 1
    assert metrics["available_total"] >= 4
