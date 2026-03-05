"""Tests for Stage 4e: Trained policy evaluation stage."""

from __future__ import annotations

from pathlib import Path
import json
from types import SimpleNamespace


def test_trained_eval_skips_without_s3b(sample_config, tmp_path):
    from blueprint_validation.stages.s4e_trained_eval import TrainedPolicyEvalStage

    stage = TrainedPolicyEvalStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, {})
    assert result.status == "skipped"
    assert "S3c" in result.detail or "S3b" in result.detail


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
        outputs={"adapted_policy_checkpoint": "/nonexistent/path"},
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

    # Conditions sorted alphabetically: adapted < baseline < trained
    assert "adapted_vs_trained" in result
    assert "baseline_vs_trained" in result
    assert "adapted_vs_baseline" in result

    ab = result["adapted_vs_baseline"]
    assert ab["adapted_mean"] == 6.5
    assert ab["baseline_mean"] == 4.5


def test_claim_pair_selection_prefers_world_fixed_comparison():
    from blueprint_validation.stages.s4e_trained_eval import _select_claim_pairwise_comparison

    pairwise = {
        "baseline_vs_trained": {"absolute_difference": 1.2},
        "adapted_vs_trained": {"absolute_difference": 0.8},
    }
    key, pair, world_fixed = _select_claim_pairwise_comparison(pairwise)
    assert key == "adapted_vs_trained"
    assert pair == pairwise["adapted_vs_trained"]
    assert world_fixed is True


def test_claim_pair_selection_falls_back_to_non_isolated_when_needed():
    from blueprint_validation.stages.s4e_trained_eval import _select_claim_pairwise_comparison

    pairwise = {"baseline_vs_trained": {"absolute_difference": 1.2}}
    key, pair, world_fixed = _select_claim_pairwise_comparison(pairwise)
    assert key == "baseline_vs_trained"
    assert pair == pairwise["baseline_vs_trained"]
    assert world_fixed is False


def test_trained_uplift_abs_diff_handles_reverse_pair_key():
    from blueprint_validation.stages.s4e_trained_eval import _trained_uplift_abs_diff

    forward = _trained_uplift_abs_diff("adapted_vs_trained", {"absolute_difference": 0.7})
    reverse = _trained_uplift_abs_diff("trained_vs_adapted", {"absolute_difference": -0.7})
    assert forward == 0.7
    assert reverse == 0.7


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


def test_resolve_trained_checkpoint_prefers_s3c(tmp_path):
    from blueprint_validation.common import StageResult
    from blueprint_validation.stages.s4e_trained_eval import _resolve_trained_checkpoint

    s3c_ckpt = tmp_path / "s3c_ckpt"
    s3c_ckpt.mkdir(parents=True)
    s3b_ckpt = tmp_path / "s3b_ckpt"
    s3b_ckpt.mkdir(parents=True)

    previous = {
        "s3c_policy_rl_loop": StageResult(
            stage_name="s3c_policy_rl_loop",
            status="success",
            elapsed_seconds=0,
            outputs={"adapted_policy_checkpoint_rl": str(s3c_ckpt)},
        ),
        "s3b_policy_finetune": StageResult(
            stage_name="s3b_policy_finetune",
            status="success",
            elapsed_seconds=0,
            outputs={"adapted_policy_checkpoint": str(s3b_ckpt)},
        ),
    }
    resolved = _resolve_trained_checkpoint(previous, tmp_path)
    assert resolved == s3c_ckpt


def test_trained_eval_build_task_list_includes_task_hints(tmp_path):
    from blueprint_validation.config import FacilityConfig, ValidationConfig
    from blueprint_validation.stages.s4e_trained_eval import _build_task_list

    hints = tmp_path / "task_targets.json"
    hints.write_text(
        json.dumps(
            {
                "tasks": [{"task_id": "pick_place_manipulation"}],
                "manipulation_candidates": [{"label": "bowl", "instance_id": "101"}],
                "articulation_hints": [{"label": "door", "instance_id": "7"}],
            }
        )
    )

    cfg = ValidationConfig()
    cfg.eval_policy.tasks = ["Navigate forward through the corridor"]
    fac = FacilityConfig(name="A", ply_path=Path("/tmp/a.ply"), task_hints_path=hints)

    tasks, hint_count = _build_task_list(cfg, fac)
    assert hint_count > 0
    assert "Navigate forward through the corridor" in tasks
    assert "Pick up bowl_101 and place it in the target zone" in tasks
    assert "Open and close door_7" in tasks


def test_resolve_eval_world_checkpoint_prefers_s3c(tmp_path):
    from blueprint_validation.common import StageResult
    from blueprint_validation.stages.s4e_trained_eval import _resolve_eval_world_checkpoint

    s3_world = tmp_path / "s3_world"
    s3_world.mkdir(parents=True)
    s3c_world = tmp_path / "s3c_world"
    s3c_world.mkdir(parents=True)
    previous = {
        "s3_finetune": StageResult(
            stage_name="s3_finetune",
            status="success",
            elapsed_seconds=0,
            outputs={"adapted_checkpoint_path": str(s3_world)},
        ),
        "s3c_policy_rl_loop": StageResult(
            stage_name="s3c_policy_rl_loop",
            status="success",
            elapsed_seconds=0,
            outputs={"adapted_world_checkpoint_rl": str(s3c_world)},
        ),
    }
    path, source = _resolve_eval_world_checkpoint(previous, tmp_path)
    assert path == s3c_world
    assert source == "s3c_rl"


def test_resolve_split_manifest_path_prefers_s4a_output(tmp_path):
    from blueprint_validation.common import StageResult, write_json
    from blueprint_validation.stages.s4e_trained_eval import _resolve_split_manifest_path

    split = tmp_path / "split_manifest.json"
    write_json({"train_pair_ids": [], "eval_pair_ids": []}, split)
    previous = {
        "s4a_rlds_export": StageResult(
            stage_name="s4a_rlds_export",
            status="success",
            elapsed_seconds=0,
            outputs={"split_manifest_path": str(split)},
        )
    }
    resolved = _resolve_split_manifest_path(previous, tmp_path)
    assert resolved == split


def test_trained_eval_fails_on_invalid_prior_policy_scores_manifest(
    sample_config, tmp_path, monkeypatch
):
    from blueprint_validation.common import StageResult, write_json
    from blueprint_validation.stages.s4e_trained_eval import TrainedPolicyEvalStage

    sample_config.eval_policy.headline_scope = "dual"
    sample_config.eval_policy.num_rollouts = 1
    sample_config.action_boost.enabled = False

    trained_ckpt = tmp_path / "trained_ckpt"
    adapted_ckpt = tmp_path / "adapted_ckpt"
    trained_ckpt.mkdir(parents=True, exist_ok=True)
    adapted_ckpt.mkdir(parents=True, exist_ok=True)

    render_manifest_path = tmp_path / "renders" / "render_manifest.json"
    render_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json({"clips": [{"clip_index": 0, "clip_name": "clip_000"}]}, render_manifest_path)

    rollout_video = tmp_path / "trained_eval" / "trained_rollouts" / "trained_clip.mp4"
    rollout_video.parent.mkdir(parents=True, exist_ok=True)
    rollout_video.write_bytes(b"x")

    invalid_scores_path = tmp_path / "policy_eval" / "vlm_scores.json"
    invalid_scores_path.parent.mkdir(parents=True, exist_ok=True)
    write_json({"scores": [{"condition": "baseline", "video_path": ""}]}, invalid_scores_path)

    monkeypatch.setattr(
        "blueprint_validation.stages.s4e_trained_eval._resolve_trained_checkpoint",
        lambda **_kwargs: trained_ckpt,
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4e_trained_eval._resolve_eval_world_checkpoint",
        lambda *_args, **_kwargs: (adapted_ckpt, "s3"),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4e_trained_eval._build_task_list",
        lambda *_args, **_kwargs: (["Navigate forward"], 0),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4e_trained_eval.load_shared_task_start_manifest",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4e_trained_eval.build_task_start_assignments",
        lambda **_kwargs: [
            {
                "rollout_index": 0,
                "task": "Navigate forward",
                "clip_index": 0,
                "clip_name": "clip_000",
            }
        ],
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4e_trained_eval.save_shared_task_start_manifest",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4e_trained_eval.load_initial_frames_for_assignments",
        lambda *_args, **_kwargs: {0: object()},
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4e_trained_eval.load_dreamdojo_world_model",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4e_trained_eval.run_rollout_with_adapter",
        lambda **_kwargs: SimpleNamespace(
            video_path=rollout_video,
            num_steps=2,
            action_sequence=[[0.0] * 7 for _ in range(2)],
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4e_trained_eval.score_rollout",
        lambda **_kwargs: SimpleNamespace(
            task_score=8.0,
            visual_score=8.0,
            spatial_score=8.0,
            reasoning="ok",
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4e_trained_eval.score_rollout_manipulation",
        lambda **_kwargs: SimpleNamespace(
            task_score=8.0,
            visual_score=8.0,
            spatial_score=8.0,
            reasoning="ok",
            grasp_acquired=True,
            lifted_clear=True,
            placed_in_target=True,
            stable_after_place=True,
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4e_trained_eval.write_judge_audit_csv",
        lambda scores, path: path.write_text(str(len(scores)), encoding="utf-8"),
    )

    class _FakeAdapter:
        def base_model_ref(self, _eval_policy):
            return "base_model", None

        def load_policy(self, **_kwargs):
            return object()

    monkeypatch.setattr(
        "blueprint_validation.stages.s4e_trained_eval.get_policy_adapter",
        lambda *_args, **_kwargs: _FakeAdapter(),
    )

    stage = TrainedPolicyEvalStage()
    fac = list(sample_config.facilities.values())[0]
    previous_results = {
        "s3b_policy_finetune": StageResult(
            stage_name="s3b_policy_finetune",
            status="success",
            elapsed_seconds=0,
            outputs={"adapted_policy_checkpoint": str(trained_ckpt)},
        ),
        "s4_policy_eval": StageResult(
            stage_name="s4_policy_eval",
            status="success",
            elapsed_seconds=0,
            outputs={"scores_path": str(invalid_scores_path)},
        ),
    }
    result = stage.run(sample_config, fac, tmp_path, previous_results)
    assert result.status == "failed"
    assert "Invalid policy scores manifest" in result.detail
