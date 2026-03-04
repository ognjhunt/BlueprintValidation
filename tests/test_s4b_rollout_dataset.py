"""Tests for Stage 4b rollout dataset export validation boundaries."""

from __future__ import annotations


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
