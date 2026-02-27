"""Integration-smoke tests with mocked model/tool boundaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
        "blueprint_validation.stages.s4_policy_eval._extract_initial_frames",
        lambda manifest: [frame],
    )
    load_calls = []

    def fake_load_openvla(model_name, checkpoint_path, device):
        load_calls.append((model_name, checkpoint_path, device))
        return ("model", "processor")

    monkeypatch.setattr("blueprint_validation.stages.s4_policy_eval.load_openvla", fake_load_openvla)
    monkeypatch.setattr(
        "blueprint_validation.stages.s4_policy_eval.load_dreamdojo_world_model",
        lambda *args, **kwargs: "world",
    )

    class FakeRollout:
        def __init__(self, video_path: Path):
            self.video_path = video_path
            self.num_steps = 3

    def fake_run_rollout(*, output_dir, clip_name, **kwargs):
        video_path = output_dir / f"{clip_name}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"x")
        return FakeRollout(video_path)

    def fake_score_rollout(*, video_path, **kwargs):
        if "adapted" in video_path.name:
            return JudgeScore(8, 7, 7, "adapted better", "{}")
        return JudgeScore(5, 6, 6, "baseline", "{}")

    monkeypatch.setattr("blueprint_validation.stages.s4_policy_eval.run_rollout", fake_run_rollout)
    monkeypatch.setattr("blueprint_validation.stages.s4_policy_eval.score_rollout", fake_score_rollout)

    previous = {
        "s3b_policy_finetune": StageResult(
            stage_name="s3b_policy_finetune",
            status="success",
            elapsed_seconds=1.0,
            outputs={"adapted_openvla_checkpoint": str(adapted_policy_dir)},
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
