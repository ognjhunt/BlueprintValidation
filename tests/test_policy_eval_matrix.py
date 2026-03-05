"""Tests for policy eval matrix artifact generation."""

from __future__ import annotations

from pathlib import Path


def _write_pair_eval_fixture(
    facility_dir: Path,
    *,
    base_mean: float,
    site_mean: float,
    abs_diff: float,
    p_value: float,
    rows: list[dict],
) -> None:
    from blueprint_validation.common import write_json

    write_json(
        {
            "stage_name": "s4d_policy_pair_eval",
            "status": "success",
            "elapsed_seconds": 0.0,
            "metrics": {
                "num_pairs": len(rows) // 2,
                "policy_base_mean_task_score": base_mean,
                "policy_site_mean_task_score": site_mean,
                "task_score_absolute_difference": abs_diff,
                "p_value_task_score": p_value,
            },
        },
        facility_dir / "s4d_policy_pair_eval_result.json",
    )
    write_json({"scores": rows}, facility_dir / "policy_pair_eval" / "pair_scores.json")


def test_policy_eval_matrix_artifact_written(sample_config, tmp_path):
    from blueprint_validation.config import FacilityConfig
    from blueprint_validation.evaluation.policy_eval_matrix import (
        build_policy_eval_matrix_artifact,
    )

    sample_config.facilities["novel_facility"] = FacilityConfig(
        name="Novel Facility",
        ply_path=tmp_path / "novel.ply",
    )
    sample_config.facilities["novel_facility"].ply_path.write_text("")
    sample_config.policy_compare.heldout_tasks = ["seen_task", "unseen_obj_task"]

    rows_primary = [
        {"episode_id": "e1", "policy": "policy_base", "task": "seen_task", "task_score": 6.0},
        {"episode_id": "e1", "policy": "policy_site", "task": "seen_task", "task_score": 8.0},
        {
            "episode_id": "e2",
            "policy": "policy_base",
            "task": "unseen_obj_task",
            "task_score": 5.5,
        },
        {
            "episode_id": "e2",
            "policy": "policy_site",
            "task": "unseen_obj_task",
            "task_score": 7.0,
        },
    ]
    rows_novel = [
        {"episode_id": "e1", "policy": "policy_base", "task": "seen_task", "task_score": 5.8},
        {"episode_id": "e1", "policy": "policy_site", "task": "seen_task", "task_score": 6.8},
        {
            "episode_id": "e2",
            "policy": "policy_base",
            "task": "unseen_obj_task",
            "task_score": 5.2,
        },
        {
            "episode_id": "e2",
            "policy": "policy_site",
            "task": "unseen_obj_task",
            "task_score": 6.0,
        },
    ]

    _write_pair_eval_fixture(
        tmp_path / "test_facility",
        base_mean=5.75,
        site_mean=7.5,
        abs_diff=1.75,
        p_value=0.01,
        rows=rows_primary,
    )
    _write_pair_eval_fixture(
        tmp_path / "novel_facility",
        base_mean=5.5,
        site_mean=6.4,
        abs_diff=0.9,
        p_value=0.04,
        rows=rows_novel,
    )

    output_path = build_policy_eval_matrix_artifact(sample_config, tmp_path)
    assert output_path is not None
    assert output_path.exists()

    import json

    payload = json.loads(output_path.read_text())
    assert payload["axes"]["seen_task_seen_env"]["available"] is True
    assert payload["axes"]["unseen_object_seen_env"]["available"] is True
    assert payload["axes"]["seen_task_novel_env"]["available"] is True
    assert isinstance(payload["forgetting_ratio"], float)


def test_policy_eval_matrix_returns_none_without_s4d(sample_config, tmp_path):
    from blueprint_validation.evaluation.policy_eval_matrix import (
        build_policy_eval_matrix_artifact,
    )

    output_path = build_policy_eval_matrix_artifact(sample_config, tmp_path)
    assert output_path is None
