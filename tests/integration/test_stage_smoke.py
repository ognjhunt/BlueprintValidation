"""Integration-smoke tests with mocked model/tool boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_validation.common import write_json


def test_stage3b_policy_finetune_skips_when_disabled(sample_config, tmp_path):
    from blueprint_validation.stages.s3b_policy_finetune import PolicyFinetuneStage

    fac = sample_config.facilities["test_facility"]
    work_dir = tmp_path / "outputs" / "test_facility"
    stage = PolicyFinetuneStage()
    result = stage.execute(sample_config, fac, work_dir, {})
    assert result.status == "skipped"


def test_stage2_to_stage3_handoff(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.enrichment.cosmos_runner import CosmosOutput
    from blueprint_validation.stages.s2_enrich import EnrichStage
    from blueprint_validation.stages.s3_finetune import FinetuneStage

    fac = sample_config.facilities["test_facility"]
    work_dir = tmp_path / "outputs" / "test_facility"
    render_dir = work_dir / "renders"
    render_dir.mkdir(parents=True)

    input_video = render_dir / "clip_000_orbit.mp4"
    depth_video = render_dir / "clip_000_orbit_depth.mp4"
    input_video.write_bytes(b"x")
    depth_video.write_bytes(b"x")
    write_json(
        {
            "clips": [
                {
                    "clip_name": "clip_000_orbit",
                    "video_path": str(input_video),
                    "depth_video_path": str(depth_video),
                }
            ]
        },
        render_dir / "render_manifest.json",
    )

    def fake_enrich_clip(**kwargs):
        out = work_dir / "enriched" / "clip_000_orbit_daylight.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x")
        return [
            CosmosOutput(
                variant_name="daylight",
                prompt="daylight",
                output_video_path=out,
                input_video_path=input_video,
                depth_video_path=depth_video,
            )
        ]

    monkeypatch.setattr("blueprint_validation.stages.s2_enrich.enrich_clip", fake_enrich_clip)

    s2 = EnrichStage()
    s2_result = s2.execute(sample_config, fac, work_dir, {})
    assert s2_result.status == "success"
    enriched_manifest = work_dir / "enriched" / "enriched_manifest.json"
    assert enriched_manifest.exists()

    dataset_dir = work_dir / "finetune" / "dreamdojo_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "blueprint_validation.stages.s3_finetune.build_dreamdojo_dataset",
        lambda **kwargs: dataset_dir,
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s3_finetune.run_dreamdojo_finetune",
        lambda **kwargs: {
            "status": "success",
            "checkpoint_dir": str(work_dir / "finetune" / "adapted_checkpoint"),
            "elapsed_seconds": 1.0,
            "loss_history": [{"loss": 0.123}],
        },
    )

    s3 = FinetuneStage()
    s3_result = s3.execute(sample_config, fac, work_dir, {})
    assert s3_result.status == "success"
    assert s3_result.outputs["dataset_dir"] == str(dataset_dir)


def test_stage4_policy_eval_deterministic_metrics(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.evaluation.vlm_judge import JudgeScore
    from blueprint_validation.common import StageResult
    from blueprint_validation.stages.s4_policy_eval import PolicyEvalStage

    sample_config.eval_policy.num_rollouts = 5
    sample_config.eval_policy.tasks = ["task_a", "task_b"]

    fac = sample_config.facilities["test_facility"]
    work_dir = tmp_path / "outputs" / "test_facility"
    (work_dir / "finetune" / "adapted_checkpoint").mkdir(parents=True)
    adapted_policy_dir = work_dir / "policy_finetune" / "adapters"
    adapted_policy_dir.mkdir(parents=True)
    (adapted_policy_dir / "adapter_model.safetensors").write_bytes(b"x")
    (work_dir / "renders").mkdir(parents=True)
    write_json({"clips": []}, work_dir / "renders" / "render_manifest.json")

    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval.build_task_start_assignments",
        lambda **kwargs: [
            {
                "rollout_index": i,
                "task": sample_config.eval_policy.tasks[i % len(sample_config.eval_policy.tasks)],
                "clip_index": 0,
                "clip_name": "clip_000",
                "path_type": "orbit",
            }
            for i in range(sample_config.eval_policy.num_rollouts)
        ],
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval.load_initial_frames_for_assignments",
        lambda assignments: {0: frame},
    )
    load_calls = []

    class FakeAdapter:
        name = "openvla_oft"

        def base_model_ref(self, eval_config):
            return eval_config.model_name, eval_config.checkpoint_path

        def load_policy(self, model_name, checkpoint_path, device):
            load_calls.append((model_name, checkpoint_path, device))
            return type("Handle", (), {"model": "model", "processor": "processor"})()

    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval.get_policy_adapter",
        lambda name: FakeAdapter(),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval.load_dreamdojo_world_model",
        lambda *args, **kwargs: "world",
    )

    class FakeRollout:
        def __init__(self, video_path: Path):
            self.video_path = video_path
            self.num_steps = 3

    def fake_run_rollout_with_adapter(*, output_dir, clip_name, **kwargs):
        video_path = output_dir / f"{clip_name}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"x")
        return FakeRollout(video_path)

    def fake_score_rollout(*, video_path, **kwargs):
        if "adapted" in video_path.name:
            return JudgeScore(8, 7, 7, "adapted better", "{}")
        return JudgeScore(5, 6, 6, "baseline", "{}")

    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval.run_rollout_with_adapter",
        fake_run_rollout_with_adapter,
    )
    monkeypatch.setattr("blueprint_validation.stages.s4_policy_eval.score_rollout", fake_score_rollout)

    previous = {
        "s3b_policy_finetune": StageResult(
            stage_name="s3b_policy_finetune",
            status="success",
            elapsed_seconds=1.0,
            outputs={"adapted_policy_checkpoint": str(adapted_policy_dir)},
        )
    }
    stage = PolicyEvalStage()
    result = stage.execute(sample_config, fac, work_dir, previous)
    assert result.status == "success"
    assert result.metrics["num_rollouts_baseline"] == 5
    assert result.metrics["num_rollouts_adapted"] == 5
    assert result.metrics["improvement_pct"] > 0
    assert result.metrics["used_adapted_policy_checkpoint"] is True
    assert len(load_calls) == 2
    assert load_calls[1][1] == adapted_policy_dir


def _write_tiny_video(path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (16, 16))
    for _ in range(6):
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()


def test_stage4b_rollout_dataset_exports(sample_config, tmp_path):
    from blueprint_validation.stages.s4b_rollout_dataset import RolloutDatasetStage

    fac = sample_config.facilities["test_facility"]
    work_dir = tmp_path / "outputs" / "test_facility"
    score_dir = work_dir / "policy_eval"
    score_dir.mkdir(parents=True)
    vid_a = score_dir / "a.mp4"
    vid_b = score_dir / "b.mp4"
    _write_tiny_video(vid_a)
    _write_tiny_video(vid_b)
    write_json(
        {
            "scores": [
                {
                    "condition": "baseline",
                    "rollout_index": 0,
                    "task": "Pick up tote",
                    "task_score": 8.0,
                    "video_path": str(vid_a),
                    "action_sequence": [[0.0] * 7 for _ in range(5)],
                    "is_manipulation_task": True,
                    "grasp_acquired": True,
                    "lifted_clear": True,
                    "placed_in_target": True,
                    "stable_after_place": True,
                },
                {
                    "condition": "adapted",
                    "rollout_index": 0,
                    "task": "Pick up tote",
                    "task_score": 9.0,
                    "video_path": str(vid_b),
                    "action_sequence": [[0.1] * 7 for _ in range(5)],
                    "is_manipulation_task": True,
                    "grasp_acquired": True,
                    "lifted_clear": True,
                    "placed_in_target": True,
                    "stable_after_place": True,
                },
            ]
        },
        score_dir / "vlm_scores.json",
    )

    stage = RolloutDatasetStage()
    result = stage.execute(sample_config, fac, work_dir, {})
    assert result.status == "success"
    assert Path(result.outputs["summary_path"]).exists()


def test_stage4c_policy_pair_train_smoke(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.stages.s4c_policy_pair_train import PolicyPairTrainStage

    sample_config.policy_finetune.enabled = True
    fac = sample_config.facilities["test_facility"]
    work_dir = tmp_path / "outputs" / "test_facility"
    ds_root = sample_config.rollout_dataset.export_dir / "test_facility"
    (ds_root / "baseline" / "train").mkdir(parents=True, exist_ok=True)
    (ds_root / "adapted" / "train").mkdir(parents=True, exist_ok=True)
    write_json({"ok": True}, ds_root / "dataset_export_summary.json")

    class FakeTrainResult:
        def __init__(self, path):
            self.status = "success"
            self.adapted_checkpoint_path = path
            self.elapsed_seconds = 1.0
            self.detail = ""

    class FakeAdapter:
        name = "openvla_oft"

        def base_model_ref(self, eval_config):
            return eval_config.model_name, eval_config.checkpoint_path

        def dataset_transform(self, source_dataset_dir, output_root, dataset_name):
            out = output_root / dataset_name
            out.mkdir(parents=True, exist_ok=True)
            return out

        def train_policy(
            self,
            base_model_name,
            base_checkpoint,
            dataset_root,
            dataset_name,
            output_dir,
            finetune_config,
        ):
            out = output_dir / "adapter"
            out.mkdir(parents=True, exist_ok=True)
            return FakeTrainResult(out)

    monkeypatch.setattr(
        "blueprint_validation.stages.s4c_policy_pair_train.get_policy_adapter",
        lambda name: FakeAdapter(),
    )

    stage = PolicyPairTrainStage()
    result = stage.execute(sample_config, fac, work_dir, {})
    assert result.status == "success"
    assert Path(result.outputs["policy_base_checkpoint"]).exists()
    assert Path(result.outputs["policy_site_checkpoint"]).exists()


def test_stage4d_policy_pair_eval_smoke(sample_config, tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    from blueprint_validation.evaluation.vlm_judge import JudgeScore, ManipulationJudgeScore
    from blueprint_validation.stages.s4d_policy_pair_eval import PolicyPairEvalStage

    fac = sample_config.facilities["test_facility"]
    work_dir = tmp_path / "outputs" / "test_facility"
    sample_config.policy_compare.enabled = True
    sample_config.policy_compare.heldout_num_rollouts = 1
    sample_config.policy_compare.eval_world_model = "baseline"

    base_ckpt = work_dir / "policy_pair_train" / "policy_base_ckpt"
    site_ckpt = work_dir / "policy_pair_train" / "policy_site_ckpt"
    base_ckpt.mkdir(parents=True, exist_ok=True)
    site_ckpt.mkdir(parents=True, exist_ok=True)
    write_json(
        {
            "policy_base": {"adapted_checkpoint_path": str(base_ckpt)},
            "policy_site": {"adapted_checkpoint_path": str(site_ckpt)},
        },
        work_dir / "policy_pair_train" / "policy_pair_train_summary.json",
    )

    ds_root = sample_config.rollout_dataset.export_dir / "test_facility" / "adapted" / "heldout"
    ds_root.mkdir(parents=True, exist_ok=True)
    img = ds_root / "0000.jpg"
    cv2.imwrite(str(img), np.zeros((16, 16, 3), dtype=np.uint8))
    episode = {
        "episode_id": "ep0",
        "steps": [
            {"observation": {"image_path": str(img)}, "language_instruction": "Pick up tote"}
        ],
    }
    (ds_root / "episodes.jsonl").write_text(f"{json.dumps(episode)}\n")

    class FakeWorld:
        def predict_next_frame(self, frame, action):
            return frame.copy()

    class FakeHandle:
        pass

    class FakeAdapter:
        name = "openvla_oft"

        def base_model_ref(self, eval_config):
            return eval_config.model_name, eval_config.checkpoint_path

        def load_policy(self, model_name, checkpoint_path, device):
            return FakeHandle()

        def predict_action(self, handle, frame, task_prompt, unnorm_key, device):
            return np.zeros(7, dtype=np.float32)

    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.get_policy_adapter",
        lambda name: FakeAdapter(),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.load_dreamdojo_world_model",
        lambda **kwargs: FakeWorld(),
    )

    def fake_manip(*args, **kwargs):
        vp = str(kwargs["video_path"])
        if "policy_site" in vp:
            return ManipulationJudgeScore(
                task_score=9,
                visual_score=8,
                spatial_score=8,
                grasp_acquired=True,
                lifted_clear=True,
                placed_in_target=True,
                stable_after_place=True,
                reasoning="ok",
                raw_response="{}",
            )
        return ManipulationJudgeScore(
            task_score=6,
            visual_score=8,
            spatial_score=8,
            grasp_acquired=False,
            lifted_clear=False,
            placed_in_target=False,
            stable_after_place=False,
            reasoning="base",
            raw_response="{}",
        )

    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.score_rollout_manipulation",
        fake_manip,
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.score_rollout",
        lambda **kwargs: JudgeScore(5, 7, 7, "ok", "{}"),
    )

    stage = PolicyPairEvalStage()
    result = stage.execute(sample_config, fac, work_dir, {})
    assert result.status == "success"
    assert result.metrics["task_score_improvement_pct"] > 0
