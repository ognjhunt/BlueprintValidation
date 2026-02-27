"""Tests for Stage 4e: Trained policy evaluation stage."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_trained_eval_skips_without_s3b(sample_config, tmp_path):
    from blueprint_validation.stages.s4e_trained_eval import TrainedPolicyEvalStage

    stage = TrainedPolicyEvalStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, {})
    assert result.status == "skipped"
    assert "S3b" in result.detail


def test_trained_eval_skips_when_s3b_failed(sample_config, tmp_path):
    from blueprint_validation.common import StageResult
    from blueprint_validation.stages.s4e_trained_eval import TrainedPolicyEvalStage

    s3b = StageResult(
        stage_name="s3b_policy_finetune",
        status="failed",
        elapsed_seconds=0,
    )
    stage = TrainedPolicyEvalStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, {"s3b_policy_finetune": s3b})
    assert result.status == "skipped"


def test_trained_eval_fails_missing_checkpoint(sample_config, tmp_path):
    from blueprint_validation.common import StageResult
    from blueprint_validation.stages.s4e_trained_eval import TrainedPolicyEvalStage

    s3b = StageResult(
        stage_name="s3b_policy_finetune",
        status="success",
        elapsed_seconds=0,
        outputs={"adapted_openvla_checkpoint": "/nonexistent/path"},
    )
    stage = TrainedPolicyEvalStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, {"s3b_policy_finetune": s3b})
    assert result.status == "failed"
    assert "checkpoint" in result.detail.lower()


def test_trained_eval_stage_name():
    from blueprint_validation.stages.s4e_trained_eval import TrainedPolicyEvalStage

    stage = TrainedPolicyEvalStage()
    assert stage.name == "s4e_trained_eval"


def test_build_pairwise_metrics():
    from blueprint_validation.stages.s4e_trained_eval import _build_pairwise_metrics

    scores = [
        {"condition": "baseline", "task_score": 4.0},
        {"condition": "baseline", "task_score": 5.0},
        {"condition": "adapted", "task_score": 6.0},
        {"condition": "adapted", "task_score": 7.0},
        {"condition": "trained", "task_score": 8.0},
        {"condition": "trained", "task_score": 9.0},
    ]
    result = _build_pairwise_metrics(scores)

    assert "adapted_vs_trained" in result or "baseline_vs_trained" in result
    assert "baseline_vs_adapted" in result

    ba = result["baseline_vs_adapted"]
    assert ba["baseline_mean"] == 4.5
    assert ba["adapted_mean"] == 6.5
    assert ba["improvement_pct"] > 0
    assert ba["win_rate"] == 1.0


def test_manipulation_success_rate():
    from blueprint_validation.stages.s4e_trained_eval import _manipulation_success_rate

    scores = [
        {"grasp_acquired": True, "lifted_clear": True, "placed_in_target": True},
        {"grasp_acquired": True, "lifted_clear": True, "placed_in_target": False},
        {"grasp_acquired": None, "task_score": 8.0},  # fallback to score
        {"grasp_acquired": None, "task_score": 3.0},  # below threshold
    ]
    rate = _manipulation_success_rate(scores)
    assert rate == 0.5  # 2 out of 4


def test_build_rollout_plan():
    from blueprint_validation.stages.s4e_trained_eval import _build_rollout_plan

    plan = _build_rollout_plan(["a", "b"], 5)
    assert len(plan) == 5
    assert plan == ["a", "b", "a", "b", "a"]

    assert _build_rollout_plan([], 5) == []
    assert _build_rollout_plan(["a"], 0) == []
