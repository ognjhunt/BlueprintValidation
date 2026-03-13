"""Tests for pipeline stage ordering and integration."""

from __future__ import annotations


def test_pipeline_stage_smoke_imports(sample_config, tmp_path):
    """Verify core stage classes import and expose stable names."""
    from blueprint_validation.pipeline import ValidationPipeline
    from blueprint_validation.stages.s0_task_hints_bootstrap import TaskHintsBootstrapStage
    from blueprint_validation.stages.s0a_scene_package import ScenePackageStage
    from blueprint_validation.stages.s0b_scene_memory_runtime import SceneMemoryRuntimeStage
    from blueprint_validation.stages.s1_isaac_render import IsaacRenderStage
    from blueprint_validation.stages.s1_render import RenderStage
    from blueprint_validation.stages.s1b_robot_composite import RobotCompositeStage
    from blueprint_validation.stages.s1c_gemini_polish import GeminiPolishStage
    from blueprint_validation.stages.s1d_gaussian_augment import GaussianAugmentStage
    from blueprint_validation.stages.s1f_external_interaction_ingest import (
        ExternalInteractionIngestStage,
    )
    from blueprint_validation.stages.s1g_external_rollout_ingest import (
        ExternalRolloutIngestStage,
    )

    # Ensure the pipeline can be instantiated with the shared fixture config.
    ValidationPipeline(sample_config, tmp_path / "outputs")

    assert TaskHintsBootstrapStage().name == "s0_task_hints_bootstrap"
    assert ScenePackageStage().name == "s0a_scene_package"
    assert SceneMemoryRuntimeStage().name == "s0b_scene_memory_runtime"
    assert IsaacRenderStage().name == "s1_isaac_render"
    assert RenderStage().name == "s1_render"
    assert RobotCompositeStage().name == "s1b_robot_composite"
    assert GeminiPolishStage().name == "s1c_gemini_polish"
    assert GaussianAugmentStage().name == "s1d_gaussian_augment"
    assert ExternalInteractionIngestStage().name == "s1f_external_interaction_ingest"
    assert ExternalRolloutIngestStage().name == "s1g_external_rollout_ingest"


def test_stage_names_are_unique():
    """All stage classes in the pipeline must have unique names."""
    from blueprint_validation.stages.s0_task_hints_bootstrap import TaskHintsBootstrapStage
    from blueprint_validation.stages.s0a_scene_package import ScenePackageStage
    from blueprint_validation.stages.s0b_scene_memory_runtime import SceneMemoryRuntimeStage
    from blueprint_validation.stages.s1_isaac_render import IsaacRenderStage
    from blueprint_validation.stages.s1_render import RenderStage
    from blueprint_validation.stages.s1b_robot_composite import RobotCompositeStage
    from blueprint_validation.stages.s1c_gemini_polish import GeminiPolishStage
    from blueprint_validation.stages.s1d_gaussian_augment import GaussianAugmentStage
    from blueprint_validation.stages.s1f_external_interaction_ingest import (
        ExternalInteractionIngestStage,
    )
    from blueprint_validation.stages.s1g_external_rollout_ingest import ExternalRolloutIngestStage

    stages = [
        TaskHintsBootstrapStage(),
        ScenePackageStage(),
        SceneMemoryRuntimeStage(),
        IsaacRenderStage(),
        RenderStage(),
        RobotCompositeStage(),
        GeminiPolishStage(),
        GaussianAugmentStage(),
        ExternalInteractionIngestStage(),
        ExternalRolloutIngestStage(),
    ]
    names = [stage.name for stage in stages]
    assert len(names) == len(set(names)), f"Duplicate stage names: {names}"


def test_pipeline_executes_runtime_first_stages_in_order(sample_config, tmp_path, monkeypatch):
    import blueprint_validation.pipeline as pipeline_mod
    from blueprint_validation.common import StageResult
    from blueprint_validation.stages.base import PipelineStage

    execution_order: list[str] = []

    class OrderedStage(PipelineStage):
        def __init__(self, stage_name: str):
            self._stage_name = stage_name

        @property
        def name(self):
            return self._stage_name

        @property
        def description(self):
            return self._stage_name

        def run(self, config, facility, work_dir, previous_results):
            del config, facility, work_dir, previous_results
            execution_order.append(self._stage_name)
            return StageResult(
                stage_name=self._stage_name,
                status="success",
                elapsed_seconds=0.0,
            )

    monkeypatch.setattr(pipeline_mod, "TaskHintsBootstrapStage", lambda: OrderedStage("s0_task_hints_bootstrap"))
    monkeypatch.setattr(pipeline_mod, "ScenePackageStage", lambda: OrderedStage("s0a_scene_package"))
    monkeypatch.setattr(
        pipeline_mod,
        "SceneMemoryRuntimeStage",
        lambda: OrderedStage("s0b_scene_memory_runtime"),
    )
    monkeypatch.setattr(pipeline_mod, "IsaacRenderStage", lambda: OrderedStage("s1_isaac_render"))
    monkeypatch.setattr(pipeline_mod, "RenderStage", lambda: OrderedStage("s1_render"))
    monkeypatch.setattr(pipeline_mod, "RobotCompositeStage", lambda: OrderedStage("s1b_robot_composite"))
    monkeypatch.setattr(pipeline_mod, "GeminiPolishStage", lambda: OrderedStage("s1c_gemini_polish"))
    monkeypatch.setattr(pipeline_mod, "GaussianAugmentStage", lambda: OrderedStage("s1d_gaussian_augment"))
    monkeypatch.setattr(
        pipeline_mod,
        "ExternalInteractionIngestStage",
        lambda: OrderedStage("s1f_external_interaction_ingest"),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "ExternalRolloutIngestStage",
        lambda: OrderedStage("s1g_external_rollout_ingest"),
    )

    sample_config.cloud.max_cost_usd = 0
    pipeline = pipeline_mod.ValidationPipeline(sample_config, tmp_path / "outputs")
    pipeline.run_all(resume_from_results=False)

    assert execution_order.index("s0_task_hints_bootstrap") < execution_order.index("s0b_scene_memory_runtime")
    assert execution_order.index("s0b_scene_memory_runtime") < execution_order.index("s0a_scene_package")
    assert execution_order.index("s0a_scene_package") < execution_order.index("s1_isaac_render")
    assert execution_order.index("s1_isaac_render") < execution_order.index("s1_render")
    assert execution_order.index("s1_render") < execution_order.index("s1b_robot_composite")
    assert execution_order.index("s1b_robot_composite") < execution_order.index("s1c_gemini_polish")
    assert execution_order.index("s1c_gemini_polish") < execution_order.index("s1d_gaussian_augment")
    assert execution_order.index("s1d_gaussian_augment") < execution_order.index("s1f_external_interaction_ingest")
    assert execution_order.index("s1f_external_interaction_ingest") < execution_order.index("s1g_external_rollout_ingest")
