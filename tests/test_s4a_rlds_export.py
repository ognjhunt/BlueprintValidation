"""Tests for Stage 4a: RLDS export stage."""

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


def test_rlds_export_stage_skips_when_disabled(sample_config, tmp_path):
    from blueprint_validation.stages.s4a_rlds_export import RLDSExportStage

    sample_config.rollout_dataset.enabled = False
    stage = RLDSExportStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, {})
    assert result.status == "skipped"


def test_rlds_export_stage_skips_when_policy_finetune_disabled(sample_config, tmp_path):
    from blueprint_validation.stages.s4a_rlds_export import RLDSExportStage

    sample_config.rollout_dataset.enabled = True
    sample_config.policy_finetune.enabled = False
    stage = RLDSExportStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, {})
    assert result.status == "skipped"


def test_rlds_export_stage_fails_without_s4(sample_config, tmp_path):
    from blueprint_validation.stages.s4a_rlds_export import RLDSExportStage

    sample_config.rollout_dataset.enabled = True
    sample_config.policy_finetune.enabled = True
    stage = RLDSExportStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, {})
    assert result.status == "failed"
    assert "Stage 4" in result.detail


def test_rlds_export_stage_succeeds_with_rollouts(sample_config, tmp_path):
    from blueprint_validation.common import StageResult, write_json
    from blueprint_validation.stages.s4a_rlds_export import RLDSExportStage

    sample_config.rollout_dataset.enabled = True
    sample_config.policy_finetune.enabled = True
    sample_config.rollout_dataset.task_score_threshold = 5.0
    sample_config.rollout_dataset.min_steps_per_rollout = 2

    # Create mock S4 scores — need multiple adapted rollouts for train/eval split
    video = tmp_path / "rollout.mp4"
    _write_tiny_video(video)

    adapted_rollouts = []
    for i in range(5):
        adapted_rollouts.append(
            {
                "condition": "adapted",
                "task": "Pick tote",
                "task_score": 8.0,
                "rollout_index": i,
                "video_path": str(video),
                "action_sequence": [[0.0] * 7 for _ in range(5)],
                "is_manipulation_task": True,
                "grasp_acquired": True,
                "lifted_clear": True,
                "placed_in_target": True,
            }
        )

    scores = {
        "scores": adapted_rollouts
        + [
            {
                "condition": "baseline",
                "task": "Pick tote",
                "task_score": 4.0,
                "rollout_index": 0,
                "video_path": str(video),
                "action_sequence": [[0.0] * 7 for _ in range(5)],
                "is_manipulation_task": True,
            },
        ]
    }
    scores_path = tmp_path / "vlm_scores.json"
    write_json(scores, scores_path)

    s4_result = StageResult(
        stage_name="s4_policy_eval",
        status="success",
        elapsed_seconds=0,
        outputs={"scores_path": str(scores_path)},
    )

    stage = RLDSExportStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, {"s4_policy_eval": s4_result})
    assert result.status == "success"
    assert "rlds_dataset_dir" in result.outputs
    assert result.metrics["num_train_episodes"] >= 1
