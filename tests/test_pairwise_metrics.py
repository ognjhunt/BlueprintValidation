"""Tests for pairwise metric computation in S4 policy eval."""

import pytest


def test_build_pairwise_metrics_two_conditions():
    from blueprint_validation.stages.s4_policy_eval import _build_pairwise_metrics

    scores = [
        {"condition": "baseline", "task_score": 3.0},
        {"condition": "baseline", "task_score": 5.0},
        {"condition": "adapted", "task_score": 6.0},
        {"condition": "adapted", "task_score": 8.0},
    ]
    result = _build_pairwise_metrics(scores, ["baseline", "adapted"])

    assert "baseline_vs_adapted" in result
    pair = result["baseline_vs_adapted"]
    assert pair["baseline_mean"] == 4.0
    assert pair["adapted_mean"] == 7.0
    assert pair["improvement_pct"] == 75.0
    assert pair["win_rate"] == 1.0


def test_build_pairwise_metrics_three_conditions():
    from blueprint_validation.stages.s4_policy_eval import _build_pairwise_metrics

    scores = [
        {"condition": "baseline", "task_score": 4.0},
        {"condition": "adapted", "task_score": 6.0},
        {"condition": "trained", "task_score": 8.0},
    ]
    result = _build_pairwise_metrics(scores, ["baseline", "adapted", "trained"])

    assert len(result) == 3
    assert "baseline_vs_adapted" in result
    assert "baseline_vs_trained" in result
    assert "adapted_vs_trained" in result


def test_build_pairwise_metrics_empty():
    from blueprint_validation.stages.s4_policy_eval import _build_pairwise_metrics

    result = _build_pairwise_metrics([], ["baseline", "adapted"])
    assert result == {}


def test_build_pairwise_metrics_single_condition():
    from blueprint_validation.stages.s4_policy_eval import _build_pairwise_metrics

    scores = [{"condition": "baseline", "task_score": 5.0}]
    result = _build_pairwise_metrics(scores, ["baseline"])
    assert result == {}


def test_manipulation_success_rate():
    from blueprint_validation.stages.s4_policy_eval import _manipulation_success_rate

    scores = [
        {"grasp_acquired": True, "lifted_clear": True, "placed_in_target": True},
        {"grasp_acquired": True, "lifted_clear": False, "placed_in_target": False},
    ]
    rate = _manipulation_success_rate(scores)
    assert rate == 0.5


def test_manipulation_success_rate_empty():
    from blueprint_validation.stages.s4_policy_eval import _manipulation_success_rate

    assert _manipulation_success_rate([]) == 0.0
