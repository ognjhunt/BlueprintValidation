from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from blueprint_validation.common import StageResult
from blueprint_validation.evaluation.vlm_judge import ManipulationJudgeScore
from blueprint_validation.stages.s4_policy_eval import PolicyEvalStage


def test_s4_reliability_enforcement_fails_on_scoring_failure_rate(sample_config, tmp_path, monkeypatch):
    sample_config.eval_policy.mode = "research"
    sample_config.eval_policy.headline_scope = "wm_only"
    sample_config.eval_policy.num_rollouts = 1
    sample_config.eval_policy.tasks = ["Pick up bowl_101 and place it in the target zone"]
    sample_config.eval_policy.manipulation_tasks = []
    sample_config.eval_policy.rollout_driver = "scripted"
    sample_config.eval_policy.reliability.enforce_stage_success = True
    sample_config.eval_policy.reliability.max_scoring_failure_rate = 0.0
    sample_config.eval_policy.reliability.min_replay_pass_rate = 0.0
    sample_config.eval_policy.reliability.min_controllability_pass_rate = 0.0
    sample_config.finetune.eval_world_experiment = (
        "cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320"
    )

    fac = sample_config.facilities["test_facility"]
    work_dir = tmp_path / "outputs" / "test_facility"
    (work_dir / "finetune" / "adapted_checkpoint").mkdir(parents=True, exist_ok=True)
    render_dir = work_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    (render_dir / "render_manifest.json").write_text('{"clips":[]}')

    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval.build_task_start_assignments",
        lambda **kwargs: [
            {
                "rollout_index": 0,
                "task": sample_config.eval_policy.tasks[0],
                "clip_index": 0,
                "clip_name": "clip_000",
                "path_type": "manipulation",
                "target_label": "bowl",
                "target_instance_id": "101",
                "target_grounded": True,
                "assignment_quality_score": 1.0,
                "assignment_reject_reason": None,
                "video_orientation_fix": "none",
            }
        ],
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval.load_initial_frames_for_assignments",
        lambda assignments: {0: frame},
    )

    class FakeWorldModel:
        expected_action_dim = 7

        def predict_next_frame(self, current_frame, action):
            return current_frame

    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval.load_dreamdojo_world_model",
        lambda *args, **kwargs: FakeWorldModel(),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval.build_scripted_trace_manifest",
        lambda assignments, action_dim, max_steps: {
            f"{a['rollout_index']}::{a['clip_index']}::{a['task']}": {
                "trace_id": "trace_0",
                "trace_type": "manipulation_scripted",
                "seed": 7,
                "action_sequence": [[0.02] * action_dim for _ in range(4)],
            }
            for a in assignments
        },
    )

    def _fake_run_scripted_rollout(**kwargs):
        video_path = kwargs["output_dir"] / f"{kwargs['clip_name']}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"x")
        return SimpleNamespace(
            video_path=video_path,
            action_sequence=kwargs["action_sequence"],
            num_steps=len(kwargs["action_sequence"]),
            driver_type="scripted",
            action_contract={
                "policy_dim": None,
                "world_dim": 7,
                "dataset_dim": 7,
                "compliant": True,
                "reason": "",
            },
        )

    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval.run_scripted_rollout",
        _fake_run_scripted_rollout,
    )

    def _fake_score_rollout_manipulation(*, video_path: Path, **kwargs):
        if "adapted_" in str(video_path):
            raise RuntimeError("invalid json response")
        return ManipulationJudgeScore(
            task_score=8.0,
            visual_score=8.0,
            spatial_score=8.0,
            grasp_acquired=True,
            lifted_clear=True,
            placed_in_target=True,
            stable_after_place=True,
            reasoning="ok",
            raw_response="{}",
        )

    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval.score_rollout_manipulation",
        _fake_score_rollout_manipulation,
    )

    result = PolicyEvalStage().execute(
        sample_config,
        fac,
        work_dir,
        {
            "s3_finetune": StageResult(
                stage_name="s3_finetune",
                status="success",
                elapsed_seconds=0.0,
                outputs={"adapted_checkpoint_path": str(work_dir / "finetune" / "adapted_checkpoint")},
            )
        },
    )
    assert result.status == "failed"
    assert result.metrics["num_scoring_failures"] >= 1
    assert result.metrics["reliability_gate"]["passed"] is False
