from __future__ import annotations

import json
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
        "blueprint_validation.stages.s4_policy_eval.get_policy_adapter",
        lambda *_args, **_kwargs: SimpleNamespace(
            base_model_ref=lambda _eval_policy: ("fake_model", None),
            resolve_latest_checkpoint=lambda _run_root: None,
        ),
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


def test_s4_short_rollout_guard_fails_stage(sample_config, tmp_path, monkeypatch):
    sample_config.eval_policy.mode = "research"
    sample_config.eval_policy.headline_scope = "wm_only"
    sample_config.eval_policy.num_rollouts = 1
    sample_config.eval_policy.tasks = ["Pick up bowl_101 and place it in the target zone"]
    sample_config.eval_policy.manipulation_tasks = []
    sample_config.eval_policy.rollout_driver = "scripted"
    sample_config.eval_policy.reliability.fail_on_short_rollout = True
    sample_config.eval_policy.reliability.min_rollout_frames = 13
    sample_config.eval_policy.reliability.enforce_stage_success = True
    sample_config.eval_policy.reliability.max_scoring_failure_rate = 1.0
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
    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval._video_frame_count",
        lambda *_args, **_kwargs: 3,
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
    assert result.metrics["num_short_rollouts"] > 0
    assert result.metrics["min_rollout_frames_required"] == 13


def test_s4_short_step_rollout_guard_fails_stage(sample_config, tmp_path, monkeypatch):
    sample_config.eval_policy.mode = "research"
    sample_config.eval_policy.headline_scope = "wm_only"
    sample_config.eval_policy.num_rollouts = 1
    sample_config.eval_policy.tasks = ["Pick up bowl_101 and place it in the target zone"]
    sample_config.eval_policy.manipulation_tasks = []
    sample_config.eval_policy.rollout_driver = "scripted"
    sample_config.eval_policy.reliability.fail_on_short_rollout = True
    sample_config.eval_policy.reliability.min_rollout_frames = 3
    sample_config.eval_policy.reliability.min_rollout_steps = 12
    sample_config.eval_policy.reliability.enforce_stage_success = True
    sample_config.eval_policy.reliability.max_scoring_failure_rate = 1.0
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
                "action_sequence": [[0.02] * action_dim for _ in range(2)],
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
    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval._video_frame_count",
        lambda *_args, **_kwargs: 16,
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
    assert result.metrics["num_short_step_rollouts"] > 0
    assert result.metrics["min_rollout_steps_required"] == 12


def test_s4_fixed_claim_protocol_does_not_fail_on_supporting_task_score_thresholds(
    sample_config, tmp_path, monkeypatch
):
    sample_config.eval_policy.claim_protocol = "fixed_same_facility_uplift"
    sample_config.eval_policy.primary_endpoint = "task_success"
    sample_config.eval_policy.freeze_world_snapshot = True
    sample_config.eval_policy.split_strategy = "disjoint_tasks_and_starts"
    sample_config.eval_policy.claim_strictness.min_eval_task_specs = 1
    sample_config.eval_policy.claim_strictness.min_eval_start_clips = 1
    sample_config.eval_policy.claim_strictness.min_common_eval_cells = 1
    sample_config.eval_policy.mode = "claim"
    sample_config.eval_policy.headline_scope = "wm_only"
    sample_config.eval_policy.num_rollouts = 4
    sample_config.eval_policy.tasks = [
        "Pick up bowl_101 and place it in the target zone",
        "Pick up mug_102 and place it in the target zone",
    ]
    sample_config.eval_policy.rollout_driver = "scripted"
    sample_config.eval_policy.reliability.max_scoring_failure_rate = 1.0
    sample_config.eval_policy.reliability.min_replay_pass_rate = 0.0
    sample_config.eval_policy.reliability.min_controllability_pass_rate = 0.0
    sample_config.finetune.eval_world_experiment = (
        "cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320"
    )

    fac = sample_config.facilities["test_facility"]
    work_dir = tmp_path / "outputs" / "test_facility"
    adapted_ckpt = work_dir / "finetune" / "adapted_checkpoint"
    adapted_ckpt.mkdir(parents=True, exist_ok=True)
    (adapted_ckpt / "weights.bin").write_bytes(b"world")
    render_dir = work_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    (render_dir / "render_manifest.json").write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "clip_index": 0,
                        "clip_name": "clip_000",
                        "path_type": "manipulation",
                        "video_path": str(render_dir / "clip_000.mp4"),
                        "initial_camera": {"position": [0.0, 0.0, 1.0]},
                        "path_context": {"approach_point": [0.0, 0.0, 0.8]},
                    },
                    {
                        "clip_index": 1,
                        "clip_name": "clip_001",
                        "path_type": "manipulation",
                        "video_path": str(render_dir / "clip_001.mp4"),
                        "initial_camera": {"position": [0.5, 0.0, 1.0]},
                        "path_context": {"approach_point": [0.5, 0.0, 0.8]},
                    },
                ]
            }
        )
    )
    benchmark_path = tmp_path / "claim_benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "version": 1,
                "task_specs": [
                    {
                        "task_spec_id": "task_spec_bowl",
                        "task_prompt": sample_config.eval_policy.tasks[0],
                        "task_family": "manipulation",
                        "target_instance_id": "101",
                        "target_label": "bowl",
                        "goal_region_id": "region::target_zone",
                        "success_predicate": {
                            "type": "manipulation_pick_place_stable",
                            "target_instance_id": "101",
                            "goal_region_id": "region::target_zone",
                            "require_stable_after_place": True,
                        },
                    },
                    {
                        "task_spec_id": "task_spec_mug",
                        "task_prompt": sample_config.eval_policy.tasks[1],
                        "task_family": "manipulation",
                        "target_instance_id": "102",
                        "target_label": "mug",
                        "goal_region_id": "region::target_zone",
                        "success_predicate": {
                            "type": "manipulation_pick_place_stable",
                            "target_instance_id": "102",
                            "goal_region_id": "region::target_zone",
                            "require_stable_after_place": True,
                        },
                    },
                ],
                "assignments": [
                    {
                        "rollout_index": 0,
                        "task_spec_id": "task_spec_bowl",
                        "clip_name": "clip_000",
                        "start_clip_id": "clip_000",
                        "start_region_id": "manip:101_a",
                        "target_instance_id": "101",
                        "target_label": "bowl",
                    },
                    {
                        "rollout_index": 1,
                        "task_spec_id": "task_spec_mug",
                        "clip_name": "clip_000",
                        "start_clip_id": "clip_000",
                        "start_region_id": "manip:102_a",
                        "target_instance_id": "102",
                        "target_label": "mug",
                    },
                    {
                        "rollout_index": 2,
                        "task_spec_id": "task_spec_bowl",
                        "clip_name": "clip_001",
                        "start_clip_id": "clip_001",
                        "start_region_id": "manip:101_b",
                        "target_instance_id": "101",
                        "target_label": "bowl",
                    },
                    {
                        "rollout_index": 3,
                        "task_spec_id": "task_spec_mug",
                        "clip_name": "clip_001",
                        "start_clip_id": "clip_001",
                        "start_region_id": "manip:102_b",
                        "target_instance_id": "102",
                        "target_label": "mug",
                    },
                ],
            }
        )
    )
    fac.claim_benchmark_path = benchmark_path

    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval.load_initial_frames_for_assignments",
        lambda assignments: {0: frame, 1: frame},
    )

    class FakeWorldModel:
        expected_action_dim = 7

        def predict_next_frame(self, current_frame, action):
            return current_frame

        def capture_rollout_state(self, **kwargs):
            return {
                "grasp_acquired": True,
                "lifted_clear": True,
                "placed_in_target": True,
                "stable_after_place": True,
            }

    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval.load_dreamdojo_world_model",
        lambda *args, **kwargs: FakeWorldModel(),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval.build_scripted_trace_manifest",
        lambda assignments, action_dim, max_steps: {
            f"{a['rollout_index']}::{a['clip_index']}::{a['task']}": {
                "trace_id": f"trace_{a['rollout_index']}",
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
            state_trace=[
                {"grasp_acquired": True},
                {"lifted_clear": True},
                {"placed_in_target": True},
                {"stable_after_place": True},
            ],
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
    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval.overlay_scripted_trace_on_video",
        lambda **kwargs: kwargs["input_video_path"],
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval.score_rollout_manipulation",
        lambda **_kwargs: ManipulationJudgeScore(
            task_score=5.0,
            visual_score=8.0,
            spatial_score=8.0,
            grasp_acquired=True,
            lifted_clear=True,
            placed_in_target=True,
            stable_after_place=True,
            reasoning="ok",
            raw_response="{}",
        ),
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
                outputs={"adapted_checkpoint_path": str(adapted_ckpt)},
            )
        },
    )
    assert result.status == "success"
    assert result.metrics["absolute_difference"] == 0.0
    assert result.metrics["manipulation_success_delta_pp"] == 0.0


def test_s4_frozen_policy_eval_ignores_adapted_policy_checkpoints(
    sample_config, tmp_path, monkeypatch
):
    from blueprint_validation.common import StageResult
    from blueprint_validation.stages.s4_policy_eval import PolicyEvalStage

    sample_config.eval_policy.mode = "research"
    sample_config.eval_policy.headline_scope = "dual"
    sample_config.eval_policy.num_rollouts = 1
    sample_config.eval_policy.tasks = ["Pick up bowl_101 and place it in the target zone"]
    sample_config.eval_policy.rollout_driver = "scripted"
    sample_config.eval_policy.reliability.max_scoring_failure_rate = 1.0
    sample_config.eval_policy.reliability.min_replay_pass_rate = 0.0
    sample_config.eval_policy.reliability.min_controllability_pass_rate = 0.0
    sample_config.finetune.eval_world_experiment = (
        "cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320"
    )

    fac = sample_config.facilities["test_facility"]
    work_dir = tmp_path / "outputs" / "test_facility"
    adapted_ckpt = work_dir / "finetune" / "adapted_checkpoint"
    adapted_ckpt.mkdir(parents=True, exist_ok=True)
    (adapted_ckpt / "weights.bin").write_bytes(b"world")
    trained_policy_ckpt = tmp_path / "trained_policy_ckpt"
    trained_policy_ckpt.mkdir(parents=True, exist_ok=True)

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
    loaded_checkpoints = []

    class FakeAdapter:
        def base_model_ref(self, _eval_policy):
            return "fake_model", None

        def load_policy(self, *, model_name, checkpoint_path, device):
            loaded_checkpoints.append(checkpoint_path)
            return {"checkpoint_path": checkpoint_path}

    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval.get_policy_adapter",
        lambda *_args, **_kwargs: FakeAdapter(),
    )

    def _fake_run_rollout(**kwargs):
        video_path = kwargs["output_dir"] / f"{kwargs['clip_name']}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"x")
        return SimpleNamespace(
            video_path=video_path,
            action_sequence=[[0.02] * 7 for _ in range(4)],
            num_steps=4,
            action_contract={
                "policy_dim": 7,
                "world_dim": 7,
                "dataset_dim": 7,
                "compliant": True,
                "reason": "",
            },
        )

    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval.run_rollout_with_adapter",
        _fake_run_rollout,
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval.score_rollout_manipulation",
        lambda **_kwargs: ManipulationJudgeScore(
            task_score=8.0,
            visual_score=8.0,
            spatial_score=8.0,
            grasp_acquired=True,
            lifted_clear=True,
            placed_in_target=True,
            stable_after_place=True,
            reasoning="ok",
            raw_response="{}",
        ),
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
                outputs={"adapted_checkpoint_path": str(adapted_ckpt)},
            ),
            "s3b_policy_finetune": StageResult(
                stage_name="s3b_policy_finetune",
                status="success",
                elapsed_seconds=0.0,
                outputs={"adapted_policy_checkpoint": str(trained_policy_ckpt)},
            ),
        },
    )
    assert result.status == "success"
    assert result.metrics["used_adapted_policy_checkpoint"] is False
    assert loaded_checkpoints == [None, None]


def test_s4_fixed_claim_protocol_fails_early_on_benchmark_render_mismatch(sample_config, tmp_path):
    sample_config.eval_policy.claim_protocol = "fixed_same_facility_uplift"
    sample_config.eval_policy.primary_endpoint = "task_success"
    sample_config.eval_policy.freeze_world_snapshot = True
    sample_config.eval_policy.split_strategy = "disjoint_tasks_and_starts"
    sample_config.eval_policy.claim_strictness.min_eval_task_specs = 1
    sample_config.eval_policy.claim_strictness.min_eval_start_clips = 1
    sample_config.eval_policy.claim_strictness.min_common_eval_cells = 1
    sample_config.eval_policy.mode = "claim"
    sample_config.eval_policy.headline_scope = "wm_only"

    fac = sample_config.facilities["test_facility"]
    work_dir = tmp_path / "outputs" / "test_facility"
    render_dir = work_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    (render_dir / "render_manifest.json").write_text(
        json.dumps({"clips": [{"clip_index": 0, "clip_name": "clip_000"}]})
    )
    benchmark_path = tmp_path / "claim_benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "version": 1,
                "task_specs": [
                    {
                        "task_spec_id": "task_spec_a",
                        "task_prompt": "Navigate to region A",
                        "task_family": "navigation",
                    }
                ],
                "assignments": [
                    {
                        "rollout_index": 0,
                        "task_spec_id": "task_spec_a",
                        "clip_name": "clip_999_missing",
                        "start_clip_id": "clip_999_missing",
                        "start_region_id": "region_start::a",
                    }
                ],
            }
        )
    )
    fac.claim_benchmark_path = benchmark_path

    result = PolicyEvalStage().execute(sample_config, fac, work_dir, {})

    assert result.status == "failed"
    assert "clip_999_missing" in result.detail


def test_s4_fixed_claim_protocol_fails_early_on_undersized_benchmark(sample_config, tmp_path):
    sample_config.eval_policy.claim_protocol = "fixed_same_facility_uplift"
    sample_config.eval_policy.primary_endpoint = "task_success"
    sample_config.eval_policy.freeze_world_snapshot = True
    sample_config.eval_policy.split_strategy = "disjoint_tasks_and_starts"
    sample_config.eval_policy.claim_strictness.min_eval_task_specs = 2
    sample_config.eval_policy.claim_strictness.min_eval_start_clips = 2
    sample_config.eval_policy.claim_strictness.min_common_eval_cells = 4
    sample_config.eval_policy.mode = "claim"
    sample_config.eval_policy.headline_scope = "wm_only"

    fac = sample_config.facilities["test_facility"]
    work_dir = tmp_path / "outputs" / "test_facility"
    render_dir = work_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    (render_dir / "render_manifest.json").write_text(
        json.dumps({"clips": [{"clip_index": 0, "clip_name": "clip_000"}]})
    )
    benchmark_path = tmp_path / "claim_benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "version": 1,
                "task_specs": [
                    {
                        "task_spec_id": "task_spec_a",
                        "task_prompt": "Navigate to region A",
                        "task_family": "navigation",
                    }
                ],
                "assignments": [
                    {
                        "rollout_index": 0,
                        "task_spec_id": "task_spec_a",
                        "clip_name": "clip_000",
                        "start_clip_id": "clip_000",
                        "start_region_id": "region_start::a",
                    }
                ],
            }
        )
    )
    fac.claim_benchmark_path = benchmark_path

    result = PolicyEvalStage().execute(sample_config, fac, work_dir, {})

    assert result.status == "failed"
    assert "unique_task_specs=1 < min_eval_task_specs=2" in result.detail
