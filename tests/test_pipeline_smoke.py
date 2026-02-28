"""Smoke test for the full pipeline with mocked stages."""

import json
from pathlib import Path


def test_pipeline_instantiation(sample_config, tmp_path):
    from blueprint_validation.pipeline import ValidationPipeline

    work_dir = tmp_path / "outputs"
    pipeline = ValidationPipeline(sample_config, work_dir)
    assert pipeline.config == sample_config
    assert work_dir.exists()


def test_stage_result_save(tmp_path):
    from blueprint_validation.common import StageResult, read_json

    result = StageResult(
        stage_name="test_stage",
        status="success",
        elapsed_seconds=5.0,
        outputs={"path": "/tmp/out"},
        metrics={"score": 0.95},
    )

    path = tmp_path / "result.json"
    result.save(path)
    assert path.exists()

    data = read_json(path)
    assert data["stage_name"] == "test_stage"
    assert data["status"] == "success"
    assert data["metrics"]["score"] == 0.95


def test_stage_base_execute_timing():
    """Test that the base execute() wrapper tracks timing."""
    from blueprint_validation.common import StageResult
    from blueprint_validation.stages.base import PipelineStage
    from blueprint_validation.config import FacilityConfig, ValidationConfig

    class DummyStage(PipelineStage):
        @property
        def name(self):
            return "dummy"

        @property
        def description(self):
            return "A test stage"

        def run(self, config, facility, work_dir, previous_results):
            return StageResult(
                stage_name="dummy",
                status="success",
                elapsed_seconds=0,
                metrics={"test": True},
            )

    stage = DummyStage()
    config = ValidationConfig()
    fac = FacilityConfig(name="test", ply_path=Path("/tmp/test.ply"))

    result = stage.execute(config, fac, Path("/tmp"), {})
    assert result.status == "success"
    assert result.elapsed_seconds >= 0
    assert result.metrics["test"] is True


def test_stage_execute_catches_errors():
    """Test that execute() catches exceptions and returns failed result."""
    from blueprint_validation.stages.base import PipelineStage
    from blueprint_validation.config import FacilityConfig, ValidationConfig

    class FailingStage(PipelineStage):
        @property
        def name(self):
            return "failing"

        @property
        def description(self):
            return "A failing stage"

        def run(self, config, facility, work_dir, previous_results):
            raise RuntimeError("Something broke")

    stage = FailingStage()
    config = ValidationConfig()
    fac = FacilityConfig(name="test", ply_path=Path("/tmp/test.ply"))

    result = stage.execute(config, fac, Path("/tmp"), {})
    assert result.status == "failed"
    assert "Something broke" in result.detail


def test_rollout_plan_honors_exact_count():
    from blueprint_validation.stages.s4_policy_eval import _build_rollout_plan

    tasks = ["a", "b", "c"]
    plan = _build_rollout_plan(tasks, 5)
    assert len(plan) == 5
    assert plan == ["a", "b", "c", "a", "b"]


def _patch_pipeline_stages_with_dummies(monkeypatch, call_counts):
    import blueprint_validation.pipeline as pipeline_mod
    from blueprint_validation.common import StageResult
    from blueprint_validation.stages.base import PipelineStage

    class DummyStage(PipelineStage):
        def __init__(self, stage_name: str):
            self._stage_name = stage_name

        @property
        def name(self) -> str:
            return self._stage_name

        @property
        def description(self) -> str:
            return f"Dummy stage {self._stage_name}"

        def run(self, config, facility, work_dir, previous_results):
            del config, facility, work_dir, previous_results
            call_counts[self._stage_name] = call_counts.get(self._stage_name, 0) + 1
            return StageResult(
                stage_name=self._stage_name,
                status="success",
                elapsed_seconds=0.0,
                outputs={"dummy_stage": self._stage_name},
            )

    stage_map = {
        "TaskHintsBootstrapStage": "s0_task_hints_bootstrap",
        "RenderStage": "s1_render",
        "RobotCompositeStage": "s1b_robot_composite",
        "GeminiPolishStage": "s1c_gemini_polish",
        "GaussianAugmentStage": "s1d_gaussian_augment",
        "SplatSimInteractionStage": "s1e_splatsim_interaction",
        "EnrichStage": "s2_enrich",
        "FinetuneStage": "s3_finetune",
        "PolicyEvalStage": "s4_policy_eval",
        "RLDSExportStage": "s4a_rlds_export",
        "PolicyFinetuneStage": "s3b_policy_finetune",
        "PolicyRLLoopStage": "s3c_policy_rl_loop",
        "TrainedPolicyEvalStage": "s4e_trained_eval",
        "RolloutDatasetStage": "s4b_rollout_dataset",
        "PolicyPairTrainStage": "s4c_policy_pair_train",
        "PolicyPairEvalStage": "s4d_policy_pair_eval",
        "VisualFidelityStage": "s5_visual_fidelity",
        "SpatialAccuracyStage": "s6_spatial_accuracy",
        "CrossSiteStage": "s7_cross_site",
    }

    for class_name, stage_name in stage_map.items():
        monkeypatch.setattr(
            pipeline_mod,
            class_name,
            lambda stage_name=stage_name: DummyStage(stage_name),
        )

    return pipeline_mod


def test_pipeline_resume_reuses_success_results(sample_config, tmp_path, monkeypatch):
    call_counts = {}
    pipeline_mod = _patch_pipeline_stages_with_dummies(monkeypatch, call_counts)
    sample_config.cloud.max_cost_usd = 0
    work_dir = tmp_path / "outputs"
    pipeline = pipeline_mod.ValidationPipeline(sample_config, work_dir)

    first = pipeline.run_all(resume_from_results=False)
    assert first["test_facility/s6_spatial_accuracy"].status == "success"
    assert call_counts["s1_render"] == 1
    assert call_counts["s6_spatial_accuracy"] == 1

    call_counts.clear()
    second = pipeline.run_all(resume_from_results=True)
    assert second["test_facility/s6_spatial_accuracy"].status == "success"

    # Resume mode should not re-execute already successful stages.
    assert call_counts == {}


def test_pipeline_resume_reruns_failed_result(sample_config, tmp_path, monkeypatch):
    call_counts = {}
    pipeline_mod = _patch_pipeline_stages_with_dummies(monkeypatch, call_counts)
    sample_config.cloud.max_cost_usd = 0
    work_dir = tmp_path / "outputs"
    pipeline = pipeline_mod.ValidationPipeline(sample_config, work_dir)
    pipeline.run_all(resume_from_results=False)

    stage_file = work_dir / "test_facility" / "s2_enrich_result.json"
    payload = json.loads(stage_file.read_text())
    payload["status"] = "failed"
    stage_file.write_text(json.dumps(payload))

    call_counts.clear()
    pipeline.run_all(resume_from_results=True)

    assert call_counts.get("s2_enrich", 0) == 1
    assert call_counts.get("s1_render", 0) == 0


def test_pipeline_post_stage_sync_hook(sample_config, tmp_path, monkeypatch):
    call_counts = {}
    pipeline_mod = _patch_pipeline_stages_with_dummies(monkeypatch, call_counts)
    sample_config.cloud.max_cost_usd = 0
    work_dir = tmp_path / "outputs"

    hook_log = tmp_path / "hook.log"
    hook_script = tmp_path / "hook.sh"
    hook_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'echo "${BLUEPRINT_SYNC_STAGE_KEY}|${BLUEPRINT_SYNC_STAGE_STATUS}" >> "$HOOK_LOG_PATH"\n'
    )
    hook_script.chmod(0o755)

    monkeypatch.setenv("HOOK_LOG_PATH", str(hook_log))
    monkeypatch.setenv("BLUEPRINT_POST_STAGE_SYNC_CMD", str(hook_script))

    pipeline = pipeline_mod.ValidationPipeline(sample_config, work_dir)
    pipeline.run_all(resume_from_results=False)

    lines = hook_log.read_text().strip().splitlines()
    assert len(lines) == 18
    assert any(line.startswith("test_facility/s1_render|success") for line in lines)
