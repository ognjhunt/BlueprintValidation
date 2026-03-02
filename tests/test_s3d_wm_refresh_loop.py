from __future__ import annotations

from pathlib import Path

from blueprint_validation.common import StageResult, write_json
from blueprint_validation.stages.s3d_wm_refresh_loop import WorldModelRefreshLoopStage


def test_wm_refresh_loop_skips_when_disabled(sample_config, tmp_path):
    sample_config.wm_refresh_loop.enabled = False
    stage = WorldModelRefreshLoopStage()
    fac = sample_config.facilities["test_facility"]
    result = stage.execute(sample_config, fac, tmp_path / "fac", {})
    assert result.status == "skipped"
    assert "wm_refresh_loop.enabled=false" in result.detail


def test_wm_refresh_loop_success_with_mocked_refresh(sample_config, tmp_path, monkeypatch):
    sample_config.eval_policy.headline_scope = "wm_only"
    sample_config.wm_refresh_loop.enabled = True
    sample_config.wm_refresh_loop.iterations = 1
    sample_config.wm_refresh_loop.source_condition = "adapted"
    sample_config.wm_refresh_loop.min_non_hard_rollouts = 1
    fac = sample_config.facilities["test_facility"]
    work_dir = tmp_path / "fac"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Mock Stage 4 score/report artifacts.
    video = work_dir / "rollout.mp4"
    video.write_bytes(b"00")
    scores_path = work_dir / "policy_eval_scores.json"
    write_json(
        {
            "scores": [
                {
                    "condition": "adapted",
                    "task": "pick object",
                    "rollout_index": 0,
                    "video_path": str(video),
                    "task_score": 8.0,
                    "is_manipulation_task": True,
                    "grasp_acquired": True,
                    "lifted_clear": True,
                    "placed_in_target": True,
                },
                {
                    "condition": "adapted",
                    "task": "pick object",
                    "rollout_index": 1,
                    "video_path": str(video),
                    "task_score": 5.5,
                    "is_manipulation_task": False,
                },
            ]
        },
        scores_path,
    )
    report_path = work_dir / "policy_eval_report.json"
    write_json({"absolute_point_differential": 1.23}, report_path)

    current_ckpt = work_dir / "finetune" / "adapted_checkpoint"
    current_ckpt.mkdir(parents=True, exist_ok=True)
    next_ckpt = work_dir / "wm_refresh_loop" / "iter_00" / "next_ckpt"
    next_ckpt.mkdir(parents=True, exist_ok=True)

    def _fake_refresh(**kwargs):
        manifest = kwargs["output_dir"] / "refresh_manifest.json"
        write_json({"clips": []}, manifest)
        return {
            "status": "success",
            "adapted_checkpoint_path": str(next_ckpt),
            "refresh_manifest_path": str(manifest),
            "mix_metrics": {"selected_total": 2},
        }

    monkeypatch.setattr(
        "blueprint_validation.stages.s3d_wm_refresh_loop.refresh_world_model_from_bucketed_rollouts",
        _fake_refresh,
    )

    prev = {
        "s4_policy_eval": StageResult(
            stage_name="s4_policy_eval",
            status="success",
            elapsed_seconds=0.0,
            outputs={"scores_path": str(scores_path), "report_path": str(report_path)},
        ),
        "s3_finetune": StageResult(
            stage_name="s3_finetune",
            status="success",
            elapsed_seconds=0.0,
            outputs={"adapted_checkpoint_path": str(current_ckpt)},
        ),
    }
    stage = WorldModelRefreshLoopStage()
    result = stage.execute(sample_config, fac, work_dir, prev)
    assert result.status == "success"
    assert Path(result.outputs["final_adapted_checkpoint_path"]).exists()
    assert result.metrics["iterations_completed"] == 1
    assert result.metrics["absolute_point_differential_pre_refresh"] == 1.23


def test_wm_refresh_loop_uses_quantile_fallback_when_thresholds_have_no_positives(
    sample_config, tmp_path, monkeypatch
):
    sample_config.eval_policy.headline_scope = "wm_only"
    sample_config.wm_refresh_loop.enabled = True
    sample_config.wm_refresh_loop.iterations = 1
    sample_config.wm_refresh_loop.source_condition = "adapted"
    sample_config.wm_refresh_loop.quantile_fallback_enabled = True
    sample_config.wm_refresh_loop.min_non_hard_rollouts = 2
    sample_config.wm_refresh_loop.fail_on_degenerate_mix = True
    fac = sample_config.facilities["test_facility"]
    work_dir = tmp_path / "fac"
    work_dir.mkdir(parents=True, exist_ok=True)

    video = work_dir / "rollout.mp4"
    video.write_bytes(b"00")
    scores_path = work_dir / "policy_eval_scores.json"
    write_json(
        {
            "scores": [
                {
                    "condition": "adapted",
                    "task": "pick object",
                    "rollout_index": idx,
                    "video_path": str(video),
                    "task_score": score,
                    "is_manipulation_task": False,
                }
                for idx, score in enumerate([0.0, 1.0, 2.0, 3.0])
            ]
        },
        scores_path,
    )
    report_path = work_dir / "policy_eval_report.json"
    write_json({"absolute_point_differential": 0.0}, report_path)

    current_ckpt = work_dir / "finetune" / "adapted_checkpoint"
    current_ckpt.mkdir(parents=True, exist_ok=True)
    next_ckpt = work_dir / "wm_refresh_loop" / "iter_00" / "next_ckpt"
    next_ckpt.mkdir(parents=True, exist_ok=True)

    def _fake_refresh(**kwargs):
        assert len(kwargs["selected_success_rows"]) + len(kwargs["near_miss_rows"]) >= 2
        manifest = kwargs["output_dir"] / "refresh_manifest.json"
        write_json({"clips": []}, manifest)
        return {
            "status": "success",
            "adapted_checkpoint_path": str(next_ckpt),
            "refresh_manifest_path": str(manifest),
            "mix_metrics": {"selected_total": 4},
        }

    monkeypatch.setattr(
        "blueprint_validation.stages.s3d_wm_refresh_loop.refresh_world_model_from_bucketed_rollouts",
        _fake_refresh,
    )

    prev = {
        "s4_policy_eval": StageResult(
            stage_name="s4_policy_eval",
            status="success",
            elapsed_seconds=0.0,
            outputs={"scores_path": str(scores_path), "report_path": str(report_path)},
        ),
        "s3_finetune": StageResult(
            stage_name="s3_finetune",
            status="success",
            elapsed_seconds=0.0,
            outputs={"adapted_checkpoint_path": str(current_ckpt)},
        ),
    }
    stage = WorldModelRefreshLoopStage()
    result = stage.execute(sample_config, fac, work_dir, prev)
    assert result.status == "success"
    assert result.metrics["bucketing_strategy"] == "quantile_fallback"
    assert result.metrics["num_positive_rollouts"] >= 2


def test_wm_refresh_loop_fails_when_mix_is_degenerate(sample_config, tmp_path):
    sample_config.eval_policy.headline_scope = "wm_only"
    sample_config.wm_refresh_loop.enabled = True
    sample_config.wm_refresh_loop.iterations = 1
    sample_config.wm_refresh_loop.source_condition = "adapted"
    sample_config.wm_refresh_loop.quantile_fallback_enabled = False
    sample_config.wm_refresh_loop.fail_on_degenerate_mix = True
    sample_config.wm_refresh_loop.min_non_hard_rollouts = 1
    fac = sample_config.facilities["test_facility"]
    work_dir = tmp_path / "fac"
    work_dir.mkdir(parents=True, exist_ok=True)

    video = work_dir / "rollout.mp4"
    video.write_bytes(b"00")
    scores_path = work_dir / "policy_eval_scores.json"
    write_json(
        {
            "scores": [
                {
                    "condition": "adapted",
                    "task": "pick object",
                    "rollout_index": 0,
                    "video_path": str(video),
                    "task_score": 0.0,
                    "is_manipulation_task": True,
                    "grasp_acquired": False,
                    "lifted_clear": False,
                    "placed_in_target": False,
                }
            ]
        },
        scores_path,
    )
    report_path = work_dir / "policy_eval_report.json"
    write_json({"absolute_point_differential": -0.2}, report_path)

    current_ckpt = work_dir / "finetune" / "adapted_checkpoint"
    current_ckpt.mkdir(parents=True, exist_ok=True)
    prev = {
        "s4_policy_eval": StageResult(
            stage_name="s4_policy_eval",
            status="success",
            elapsed_seconds=0.0,
            outputs={"scores_path": str(scores_path), "report_path": str(report_path)},
        ),
        "s3_finetune": StageResult(
            stage_name="s3_finetune",
            status="success",
            elapsed_seconds=0.0,
            outputs={"adapted_checkpoint_path": str(current_ckpt)},
        ),
    }
    stage = WorldModelRefreshLoopStage()
    result = stage.execute(sample_config, fac, work_dir, prev)
    assert result.status == "failed"
    assert "degenerate rollout mix" in result.detail.lower()
