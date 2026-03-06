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


def test_rollout_dataset_stage_mixes_native_teacher_rows(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.common import write_json
    from blueprint_validation.stages.s4b_rollout_dataset import RolloutDatasetStage

    sample_config.eval_policy.headline_scope = "dual"
    sample_config.eval_policy.claim_protocol = "fixed_same_facility_uplift"
    sample_config.rollout_dataset.enabled = True
    sample_config.native_teacher.enabled = True
    sample_config.native_teacher.generate_corrections = True
    sample_config.policy_compare.enabled = True

    video = tmp_path / "policy_eval" / "rollout.mp4"
    _write_tiny_video(video)

    scores = {
        "scores": [
            {
                "condition": "baseline",
                "task": "Pick up bowl_101 and place it in the target zone",
                "task_score": 7.0,
                "task_success": True,
                "task_success_available": True,
                "eval_cell_id": "eval_train_000",
                "task_spec_id": "task_spec_pick",
                "rollout_index": 0,
                "video_path": str(video),
                "action_sequence": [[0.0] * 7 for _ in range(5)],
            },
            {
                "condition": "adapted",
                "task": "Pick up bowl_101 and place it in the target zone",
                "task_score": 2.0,
                "task_success": False,
                "task_success_available": True,
                "eval_cell_id": "eval_train_000",
                "task_spec_id": "task_spec_pick",
                "rollout_index": 0,
                "video_path": str(video),
                "action_sequence": [[0.0] * 7 for _ in range(5)],
            },
        ]
    }
    scores_path = tmp_path / "policy_eval" / "vlm_scores.json"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(scores, scores_path)
    write_json({"claim_mode": False, "action_contract": {}, "reliability_gate": {}}, tmp_path / "policy_eval" / "policy_eval_report.json")
    write_json(
        {
            "world_snapshot_hash": "world_hash",
            "train_eval_cell_ids": ["eval_train_000"],
            "eval_eval_cell_ids": [],
            "cells": [
                {
                    "eval_cell_id": "eval_train_000",
                    "task_spec_id": "task_spec_pick",
                    "start_clip_id": "clip_000",
                    "start_region_id": "manipulation:101",
                    "rollout_index": 0,
                }
            ],
        },
        tmp_path / "policy_eval" / "claim_split_manifest.json",
    )
    write_json({"world_snapshot_hash": "world_hash"}, tmp_path / "policy_eval" / "claim_manifest.json")

    monkeypatch.setattr(
        "blueprint_validation.stages.s4b_rollout_dataset.generate_teacher_rollouts",
        lambda **_kwargs: (
            [
                {
                    "condition": "adapted",
                    "task": "Pick up bowl_101 and place it in the target zone",
                    "task_score": 10.0,
                    "task_success": True,
                    "task_success_available": True,
                    "source_type": "planner_demo",
                    "sim_backend": "native_benchmark_sim",
                    "success_source": "sim_ground_truth",
                    "eval_cell_id": "eval_train_000",
                    "task_spec_id": "task_spec_pick",
                    "rollout_index": 100000,
                    "video_path": str(video),
                    "action_sequence": [[0.0] * 7 for _ in range(5)],
                }
            ],
            {"num_successful": 1},
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4b_rollout_dataset.generate_correction_rollouts",
        lambda **_kwargs: (
            [
                {
                    "condition": "adapted",
                    "task": "Pick up bowl_101 and place it in the target zone",
                    "task_score": 10.0,
                    "task_success": True,
                    "task_success_available": True,
                    "source_type": "planner_correction",
                    "sim_backend": "native_benchmark_sim",
                    "success_source": "sim_ground_truth",
                    "parent_rollout_id": "eval_train_000",
                    "eval_cell_id": "eval_train_000",
                    "task_spec_id": "task_spec_pick",
                    "rollout_index": 300000,
                    "video_path": str(video),
                    "action_sequence": [[0.0] * 7 for _ in range(5)],
                }
            ],
            {"num_successful_corrections": 1},
        ),
    )

    stage = RolloutDatasetStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, {})
    assert result.status == "success"
    assert result.metrics["generic_control_train_episodes"] >= 1
    summary = json.loads(Path(result.outputs["summary_path"]).read_text())
    assert summary["native_teacher"]["num_site_teacher_rows"] == 1
    assert summary["native_teacher"]["num_site_correction_rows"] == 1


def test_leave_one_facility_out_generic_pool_excludes_target(sample_config, tmp_path):
    from blueprint_validation.config import FacilityConfig
    from blueprint_validation.stages.s4b_rollout_dataset import _load_leave_one_facility_out_generic_pool

    sample_config.eval_policy.claim_protocol = "fixed_same_facility_uplift"
    sample_config.claim_portfolio.min_facilities = 3
    for facility_id in ("facility_b", "facility_c"):
        sample_config.facilities[facility_id] = FacilityConfig(
            name=facility_id,
            ply_path=tmp_path / f"{facility_id}.ply",
            claim_benchmark_path=tmp_path / f"{facility_id}_benchmark.json",
        )
        sample_config.facilities[facility_id].ply_path.write_text("")
        sample_config.facilities[facility_id].claim_benchmark_path.write_text('{"version": 1, "task_specs": [], "assignments": []}')

    work_root = tmp_path / "outputs"
    (work_root / "facility_b" / "native_teacher" / "site_teacher").mkdir(parents=True, exist_ok=True)
    (work_root / "facility_c" / "native_teacher" / "site_corrections").mkdir(parents=True, exist_ok=True)
    (work_root / "facility_b" / "native_teacher" / "site_teacher" / "adapted_site_teacher_rows.json").write_text(
        json.dumps({"rows": [{"task_spec_id": "task_b", "rollout_index": 10, "video_path": "b.mp4", "source_type": "planner_demo"}]})
    )
    (work_root / "facility_c" / "native_teacher" / "site_corrections" / "adapted_site_correction_rows.json").write_text(
        json.dumps({"rows": [{"task_spec_id": "task_c", "rollout_index": 11, "video_path": "c.mp4", "source_type": "planner_correction"}]})
    )

    rows = _load_leave_one_facility_out_generic_pool(
        config=sample_config,
        target_facility_id="test_facility",
        target_teacher_count=1,
        target_correction_count=1,
        pipeline_work_root=work_root,
    )
    assert len(rows) == 2
    assert all(row["task_spec_id"] in {"task_b", "task_c"} for row in rows)
