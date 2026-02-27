"""Smoke test for the full pipeline with mocked stages."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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
