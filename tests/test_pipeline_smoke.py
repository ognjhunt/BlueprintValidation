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


def test_stage_execute_fails_on_stage_preflight():
    from blueprint_validation.common import PreflightCheck
    from blueprint_validation.stages.base import PipelineStage
    from blueprint_validation.config import FacilityConfig, ValidationConfig

    class PreflightFailingStage(PipelineStage):
        @property
        def name(self):
            return "preflight_failing"

        @property
        def description(self):
            return "A stage with failing preflight"

        def preflight(self, config):
            del config
            return [PreflightCheck(name="fixture", passed=False, detail="blocked")]

        def run(self, config, facility, work_dir, previous_results):
            raise AssertionError("run() should not execute when stage preflight fails")

    stage = PreflightFailingStage()
    config = ValidationConfig()
    fac = FacilityConfig(name="test", ply_path=Path("/tmp/test.ply"))

    result = stage.execute(config, fac, Path("/tmp"), {})
    assert result.status == "failed"
    assert "Stage preflight failed" in result.detail
    assert "fixture" in result.detail


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
        "ScenePackageStage": "s0a_scene_package",
        "SceneMemoryRuntimeStage": "s0b_scene_memory_runtime",
        "IsaacRenderStage": "s1_isaac_render",
        "RenderStage": "s1_render",
        "RobotCompositeStage": "s1b_robot_composite",
        "GeminiPolishStage": "s1c_gemini_polish",
        "GaussianAugmentStage": "s1d_gaussian_augment",
        "ExternalInteractionIngestStage": "s1f_external_interaction_ingest",
        "ExternalRolloutIngestStage": "s1g_external_rollout_ingest",
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
    assert first["test_facility/s1g_external_rollout_ingest"].status == "success"
    assert call_counts["s0a_scene_package"] == 1
    assert call_counts["s0b_scene_memory_runtime"] == 1
    assert call_counts["s1_isaac_render"] == 1
    assert call_counts["s1_render"] == 1
    assert call_counts["s1g_external_rollout_ingest"] == 1

    call_counts.clear()
    second = pipeline.run_all(resume_from_results=True)
    assert second["test_facility/s1g_external_rollout_ingest"].status == "success"

    # Resume mode should not re-execute already successful stages.
    assert call_counts == {}


def test_pipeline_resume_reruns_failed_result(sample_config, tmp_path, monkeypatch):
    call_counts = {}
    pipeline_mod = _patch_pipeline_stages_with_dummies(monkeypatch, call_counts)
    sample_config.cloud.max_cost_usd = 0
    work_dir = tmp_path / "outputs"
    pipeline = pipeline_mod.ValidationPipeline(sample_config, work_dir)
    pipeline.run_all(resume_from_results=False)

    stage_file = work_dir / "test_facility" / "s1_render_result.json"
    payload = json.loads(stage_file.read_text())
    payload["status"] = "failed"
    stage_file.write_text(json.dumps(payload))

    call_counts.clear()
    pipeline.run_all(resume_from_results=True)

    assert call_counts.get("s1_render", 0) == 1
    assert call_counts.get("s1_isaac_render", 0) == 0


def test_pipeline_resume_fails_fast_on_corrupt_stage_result(sample_config, tmp_path, monkeypatch):
    call_counts = {}
    pipeline_mod = _patch_pipeline_stages_with_dummies(monkeypatch, call_counts)
    sample_config.cloud.max_cost_usd = 0
    work_dir = tmp_path / "outputs"
    pipeline = pipeline_mod.ValidationPipeline(sample_config, work_dir)
    pipeline.run_all(resume_from_results=False)

    corrupt_path = work_dir / "test_facility" / "s1_render_result.json"
    corrupt_path.write_text("{not valid json")

    call_counts.clear()
    results = pipeline.run_all(resume_from_results=True)
    assert results["test_facility/s1_render"].status == "failed"
    assert "Corrupt resume artifact" in results["test_facility/s1_render"].detail

    summary = json.loads((work_dir / "pipeline_summary.json").read_text())
    provenance = summary["stages"]["test_facility/s1_render"]["provenance"]
    assert provenance["source"] == "resume_corrupt"
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
    assert len(lines) == 10
    assert any(line.startswith("test_facility/s0a_scene_package|success") for line in lines)
    assert any(line.startswith("test_facility/s0b_scene_memory_runtime|success") for line in lines)
    assert any(line.startswith("test_facility/s1_isaac_render|success") for line in lines)
    assert any(line.startswith("test_facility/s1_render|success") for line in lines)

def test_auto_shutdown_uses_argv_without_shell(sample_config, tmp_path, monkeypatch):
    import blueprint_validation.pipeline as pipeline_mod

    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return None

    sample_config.cloud.auto_shutdown = True
    monkeypatch.setenv("BLUEPRINT_AUTO_SHUTDOWN_CMD", "echo safe; touch /tmp/pwned")
    monkeypatch.setattr(pipeline_mod.subprocess, "run", _fake_run)

    pipeline = pipeline_mod.ValidationPipeline(sample_config, tmp_path / "outputs")
    pipeline._maybe_trigger_auto_shutdown("budget triggered")

    assert captured["cmd"] == ["echo", "safe;", "touch", "/tmp/pwned"]
    assert captured["kwargs"]["shell"] is False


def test_pipeline_summary_includes_run_metadata_and_stage_provenance(
    sample_config, tmp_path, monkeypatch
):
    import json

    call_counts = {}
    pipeline_mod = _patch_pipeline_stages_with_dummies(monkeypatch, call_counts)
    sample_config.cloud.max_cost_usd = 0
    work_dir = tmp_path / "outputs"
    pipeline = pipeline_mod.ValidationPipeline(sample_config, work_dir)
    pipeline.run_all(resume_from_results=False)

    summary = json.loads((work_dir / "pipeline_summary.json").read_text())
    assert summary["run_mode"] == "fresh"
    assert summary["run_started_at"]
    assert summary["run_finished_at"]
    for stage_payload in summary["stages"].values():
        provenance = stage_payload.get("provenance", {})
        assert provenance.get("source") == "executed"
        assert provenance.get("result_path")


def test_pipeline_summary_marks_resumed_stage_provenance(sample_config, tmp_path, monkeypatch):
    import json

    call_counts = {}
    pipeline_mod = _patch_pipeline_stages_with_dummies(monkeypatch, call_counts)
    sample_config.cloud.max_cost_usd = 0
    work_dir = tmp_path / "outputs"
    pipeline = pipeline_mod.ValidationPipeline(sample_config, work_dir)
    pipeline.run_all(resume_from_results=False)
    pipeline.run_all(resume_from_results=True)

    summary = json.loads((work_dir / "pipeline_summary.json").read_text())
    assert summary["run_mode"] == "resume"
    assert any(
        stage_payload.get("provenance", {}).get("source") == "resumed"
        for stage_payload in summary["stages"].values()
    )
