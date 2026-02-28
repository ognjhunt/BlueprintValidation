"""Tests for policy RL loop helper logic."""

from __future__ import annotations

from pathlib import Path


def test_select_rollouts_returns_top_and_near_miss():
    from blueprint_validation.training.policy_rl_loop import _select_rollouts

    rows = []
    for i, reward in enumerate([0.9, 0.8, 0.6, 0.5, 0.3, 0.1]):
        rows.append({"task": "pick", "rl_reward": reward, "rollout_index": i})

    selected, near_miss = _select_rollouts(
        rollout_rows=rows,
        group_size=2,
        top_quantile=0.30,
        near_miss_min_quantile=0.30,
        near_miss_max_quantile=0.60,
    )
    assert selected
    assert near_miss
    assert all(r["task"] == "pick" for r in selected)
    assert all("advantage" in r for r in selected + near_miss)


def test_score_rollout_reward_heuristic_only(sample_config, tmp_path):
    from blueprint_validation.training.policy_rl_loop import _score_rollout_reward

    sample_config.policy_rl_loop.reward_mode = "heuristic_only"
    sample_config.policy_rl_loop.vlm_reward_fraction = 0.0
    fac = list(sample_config.facilities.values())[0]
    video_path = tmp_path / "dummy.mp4"
    video_path.write_bytes(b"x")

    result = _score_rollout_reward(
        config=sample_config,
        facility=fac,
        task="Pick up the tote",
        rollout_index=0,
        video_path=video_path,
        action_sequence=[[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]],
    )
    assert result["vlm_reward"] == 0.0
    assert result["heuristic_reward"] > 0.0
    assert result["rl_reward"] > 0.0


def test_policy_rl_stage_skips_when_disabled(sample_config, tmp_path):
    from blueprint_validation.stages.s3c_policy_rl_loop import PolicyRLLoopStage

    sample_config.policy_rl_loop.enabled = False
    fac = list(sample_config.facilities.values())[0]
    stage = PolicyRLLoopStage()
    result = stage.run(sample_config, fac, tmp_path, {})
    assert result.status == "skipped"
