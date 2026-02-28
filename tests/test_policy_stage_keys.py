"""Tests for adapter-neutral policy stage output keys."""

from __future__ import annotations

from pathlib import Path

from blueprint_validation.common import StageResult


def test_s3b_emits_canonical_and_legacy_checkpoint_keys(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.policy_adapters.base import PolicyTrainingResult
    from blueprint_validation.stages.s3b_policy_finetune import PolicyFinetuneStage

    sample_config.policy_finetune.enabled = True
    fac = sample_config.facilities["test_facility"]
    work_dir = tmp_path / "outputs" / "test_facility"
    source = work_dir / "rlds" / "train"
    source.mkdir(parents=True, exist_ok=True)
    (source / "episodes.jsonl").write_text("")

    class FakeAdapter:
        name = "openvla_oft"

        def dataset_transform(self, source_dataset_dir, output_root, dataset_name):
            out = output_root / dataset_name
            out.mkdir(parents=True, exist_ok=True)
            return out

        def base_model_ref(self, eval_config):
            return eval_config.model_name, eval_config.checkpoint_path

        def train_policy(self, **kwargs):
            out = kwargs["output_dir"] / "adapter"
            out.mkdir(parents=True, exist_ok=True)
            return PolicyTrainingResult(
                status="success",
                adapted_checkpoint_path=out,
                elapsed_seconds=1.0,
                detail="",
                raw={"returncode": 0},
            )

    monkeypatch.setattr(
        "blueprint_validation.stages.s3b_policy_finetune.get_policy_adapter",
        lambda cfg: FakeAdapter(),
    )

    prev = {
        "s4a_rlds_export": StageResult(
            stage_name="s4a_rlds_export",
            status="success",
            elapsed_seconds=0,
            outputs={"train_jsonl": str(source / "episodes.jsonl"), "dataset_name": "bridge_orig"},
        )
    }
    result = PolicyFinetuneStage().run(sample_config, fac, work_dir, prev)
    assert result.status == "success"
    assert result.outputs["adapted_policy_checkpoint"]
    assert result.outputs["adapted_openvla_checkpoint"]


def test_s3c_emits_canonical_and_legacy_checkpoint_keys(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.stages.s3c_policy_rl_loop import PolicyRLLoopStage

    sample_config.policy_finetune.enabled = True
    sample_config.policy_rl_loop.enabled = True
    fac = sample_config.facilities["test_facility"]
    work_dir = tmp_path / "outputs" / "test_facility"
    adapted_world = work_dir / "finetune" / "adapted_checkpoint"
    adapted_world.mkdir(parents=True, exist_ok=True)

    class FakeAdapter:
        def resolve_latest_checkpoint(self, run_root_dir: Path):
            return None

    monkeypatch.setattr(
        "blueprint_validation.stages.s3c_policy_rl_loop.get_policy_adapter",
        lambda cfg: FakeAdapter(),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s3c_policy_rl_loop.run_policy_rl_iterations",
        lambda **kwargs: {
            "status": "success",
            "final_policy_checkpoint": str(tmp_path / "ckpt"),
            "final_world_checkpoint": str(adapted_world),
            "iterations_completed": 1,
        },
    )

    prev = {
        "s3_finetune": StageResult(
            stage_name="s3_finetune",
            status="success",
            elapsed_seconds=0,
            outputs={"adapted_checkpoint_path": str(adapted_world)},
        )
    }
    result = PolicyRLLoopStage().run(sample_config, fac, work_dir, prev)
    assert result.status == "success"
    assert result.outputs["adapted_policy_checkpoint_rl"]
    assert result.outputs["adapted_openvla_checkpoint_rl"]
