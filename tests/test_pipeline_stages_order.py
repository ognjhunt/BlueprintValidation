"""Tests for pipeline stage ordering and integration."""

from __future__ import annotations


def test_pipeline_stage_smoke_imports(sample_config, tmp_path):
    """Verify core stage classes import and expose stable names."""
    from blueprint_validation.pipeline import ValidationPipeline
    from blueprint_validation.stages.s0_task_hints_bootstrap import TaskHintsBootstrapStage
    from blueprint_validation.stages.s1e_splatsim_interaction import SplatSimInteractionStage
    from blueprint_validation.stages.s3b_policy_finetune import PolicyFinetuneStage
    from blueprint_validation.stages.s3c_policy_rl_loop import PolicyRLLoopStage
    from blueprint_validation.stages.s4a_rlds_export import RLDSExportStage
    from blueprint_validation.stages.s4e_trained_eval import TrainedPolicyEvalStage

    # Ensure the pipeline can be instantiated with the shared fixture config.
    ValidationPipeline(sample_config, tmp_path / "outputs")

    assert TaskHintsBootstrapStage().name == "s0_task_hints_bootstrap"
    assert SplatSimInteractionStage().name == "s1e_splatsim_interaction"
    assert PolicyFinetuneStage().name == "s3b_policy_finetune"
    assert PolicyRLLoopStage().name == "s3c_policy_rl_loop"
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
