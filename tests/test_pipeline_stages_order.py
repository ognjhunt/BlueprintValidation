"""Tests for pipeline stage ordering and integration."""

from __future__ import annotations


def test_pipeline_stage_smoke_imports(sample_config, tmp_path):
    """Verify core stage classes import and expose stable names."""
    from blueprint_validation.pipeline import ValidationPipeline
    from blueprint_validation.stages.s0_task_hints_bootstrap import TaskHintsBootstrapStage
    from blueprint_validation.stages.s1e_splatsim_interaction import SplatSimInteractionStage
    from blueprint_validation.stages.s3b_policy_finetune import PolicyFinetuneStage
    from blueprint_validation.stages.s3c_policy_rl_loop import PolicyRLLoopStage
    from blueprint_validation.stages.s3d_wm_refresh_loop import WorldModelRefreshLoopStage
    from blueprint_validation.stages.s4a_rlds_export import RLDSExportStage
    from blueprint_validation.stages.s4e_trained_eval import TrainedPolicyEvalStage

    # Ensure the pipeline can be instantiated with the shared fixture config.
    ValidationPipeline(sample_config, tmp_path / "outputs")

    assert TaskHintsBootstrapStage().name == "s0_task_hints_bootstrap"
    assert SplatSimInteractionStage().name == "s1e_splatsim_interaction"
    assert PolicyFinetuneStage().name == "s3b_policy_finetune"
    assert PolicyRLLoopStage().name == "s3c_policy_rl_loop"
    assert WorldModelRefreshLoopStage().name == "s3d_wm_refresh_loop"
    assert RLDSExportStage().name == "s4a_rlds_export"
    assert TrainedPolicyEvalStage().name == "s4e_trained_eval"


def test_stage_names_are_unique():
    """All stage classes in the pipeline must have unique names."""
    from blueprint_validation.stages.s0_task_hints_bootstrap import TaskHintsBootstrapStage
    from blueprint_validation.stages.s1_render import RenderStage
    from blueprint_validation.stages.s1b_robot_composite import RobotCompositeStage
    from blueprint_validation.stages.s1c_gemini_polish import GeminiPolishStage
    from blueprint_validation.stages.s1d_gaussian_augment import GaussianAugmentStage
    from blueprint_validation.stages.s1e_splatsim_interaction import SplatSimInteractionStage
    from blueprint_validation.stages.s2_enrich import EnrichStage
    from blueprint_validation.stages.s3_finetune import FinetuneStage
    from blueprint_validation.stages.s3b_policy_finetune import PolicyFinetuneStage
    from blueprint_validation.stages.s3c_policy_rl_loop import PolicyRLLoopStage
    from blueprint_validation.stages.s3d_wm_refresh_loop import WorldModelRefreshLoopStage
    from blueprint_validation.stages.s4_policy_eval import PolicyEvalStage
    from blueprint_validation.stages.s4a_rlds_export import RLDSExportStage
    from blueprint_validation.stages.s4b_rollout_dataset import RolloutDatasetStage
    from blueprint_validation.stages.s4c_policy_pair_train import PolicyPairTrainStage
    from blueprint_validation.stages.s4d_policy_pair_eval import PolicyPairEvalStage
    from blueprint_validation.stages.s4e_trained_eval import TrainedPolicyEvalStage
    from blueprint_validation.stages.s5_visual_fidelity import VisualFidelityStage
    from blueprint_validation.stages.s6_spatial_accuracy import SpatialAccuracyStage
    from blueprint_validation.stages.s7_cross_site import CrossSiteStage

    stages = [
        TaskHintsBootstrapStage(),
        RenderStage(),
        RobotCompositeStage(),
        GeminiPolishStage(),
        GaussianAugmentStage(),
        SplatSimInteractionStage(),
        EnrichStage(),
        FinetuneStage(),
        PolicyEvalStage(),
        RLDSExportStage(),
        PolicyFinetuneStage(),
        PolicyRLLoopStage(),
        WorldModelRefreshLoopStage(),
        TrainedPolicyEvalStage(),
        RolloutDatasetStage(),
        PolicyPairTrainStage(),
        PolicyPairEvalStage(),
        VisualFidelityStage(),
        SpatialAccuracyStage(),
        CrossSiteStage(),
    ]
    names = [stage.name for stage in stages]
    assert len(names) == len(set(names)), f"Duplicate stage names: {names}"


def test_validation_config_exposes_action_boost_defaults():
    from blueprint_validation.config import ValidationConfig

    cfg = ValidationConfig()
    assert cfg.action_boost.enabled is True
    assert cfg.action_boost.require_full_pipeline is True
    assert cfg.action_boost.compute_profile == "standard"


def test_pipeline_reruns_s4_after_successful_s3d(sample_config, tmp_path, monkeypatch):
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
    monkeypatch.setattr(pipeline_mod, "RenderStage", lambda: OrderedStage("s1_render"))
    monkeypatch.setattr(pipeline_mod, "RobotCompositeStage", lambda: OrderedStage("s1b_robot_composite"))
    monkeypatch.setattr(pipeline_mod, "GeminiPolishStage", lambda: OrderedStage("s1c_gemini_polish"))
    monkeypatch.setattr(pipeline_mod, "GaussianAugmentStage", lambda: OrderedStage("s1d_gaussian_augment"))
    monkeypatch.setattr(pipeline_mod, "SplatSimInteractionStage", lambda: OrderedStage("s1e_splatsim_interaction"))
    monkeypatch.setattr(pipeline_mod, "EnrichStage", lambda: OrderedStage("s2_enrich"))
    monkeypatch.setattr(pipeline_mod, "FinetuneStage", lambda: OrderedStage("s3_finetune"))
    monkeypatch.setattr(pipeline_mod, "PolicyEvalStage", lambda: OrderedStage("s4_policy_eval"))
    monkeypatch.setattr(
        pipeline_mod, "WorldModelRefreshLoopStage", lambda: OrderedStage("s3d_wm_refresh_loop")
    )
    monkeypatch.setattr(pipeline_mod, "RLDSExportStage", lambda: OrderedStage("s4a_rlds_export"))
    monkeypatch.setattr(pipeline_mod, "PolicyFinetuneStage", lambda: OrderedStage("s3b_policy_finetune"))
    monkeypatch.setattr(pipeline_mod, "PolicyRLLoopStage", lambda: OrderedStage("s3c_policy_rl_loop"))
    monkeypatch.setattr(pipeline_mod, "TrainedPolicyEvalStage", lambda: OrderedStage("s4e_trained_eval"))
    monkeypatch.setattr(pipeline_mod, "RolloutDatasetStage", lambda: OrderedStage("s4b_rollout_dataset"))
    monkeypatch.setattr(pipeline_mod, "PolicyPairTrainStage", lambda: OrderedStage("s4c_policy_pair_train"))
    monkeypatch.setattr(pipeline_mod, "PolicyPairEvalStage", lambda: OrderedStage("s4d_policy_pair_eval"))
    monkeypatch.setattr(pipeline_mod, "VisualFidelityStage", lambda: OrderedStage("s5_visual_fidelity"))
    monkeypatch.setattr(pipeline_mod, "SpatialAccuracyStage", lambda: OrderedStage("s6_spatial_accuracy"))
    monkeypatch.setattr(pipeline_mod, "CrossSiteStage", lambda: OrderedStage("s7_cross_site"))

    sample_config.cloud.max_cost_usd = 0
    sample_config.eval_policy.headline_scope = "dual"
    sample_config.eval_policy.reliability.enforce_stage_success = False
    pipeline = pipeline_mod.ValidationPipeline(sample_config, tmp_path / "outputs")
    pipeline.run_all(resume_from_results=False)

    first_s4 = execution_order.index("s4_policy_eval")
    s3d = execution_order.index("s3d_wm_refresh_loop")
    second_s4 = execution_order.index("s4_policy_eval", first_s4 + 1)
    s4a = execution_order.index("s4a_rlds_export")
    assert first_s4 < s3d < second_s4 < s4a
