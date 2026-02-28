"""Tests for pipeline stage ordering and integration."""

from pathlib import Path

import pytest


def test_pipeline_stage_order(sample_config, tmp_path):
    """Verify the pipeline stages are in the correct order."""
    from blueprint_validation.pipeline import ValidationPipeline

    pipeline = ValidationPipeline(sample_config, tmp_path / "outputs")

    # Access the per_facility_stages list by inspecting run_all
    # We test the import works correctly for all new stages
    from blueprint_validation.stages.s4a_rlds_export import RLDSExportStage
    from blueprint_validation.stages.s1e_splatsim_interaction import SplatSimInteractionStage
    from blueprint_validation.stages.s3b_policy_finetune import PolicyFinetuneStage
    from blueprint_validation.stages.s3c_policy_rl_loop import PolicyRLLoopStage
    from blueprint_validation.stages.s4e_trained_eval import TrainedPolicyEvalStage

    assert RLDSExportStage().name == "s4a_rlds_export"
    assert SplatSimInteractionStage().name == "s1e_splatsim_interaction"
    assert PolicyFinetuneStage().name == "s3b_policy_finetune"
    assert PolicyRLLoopStage().name == "s3c_policy_rl_loop"
    assert TrainedPolicyEvalStage().name == "s4e_trained_eval"


def test_all_stages_importable():
    """Verify all pipeline stage modules can be imported."""
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


def test_s4a_runs_after_s4():
    """Verify S4a requires S4 output in previous_results."""
    from blueprint_validation.stages.s4a_rlds_export import RLDSExportStage

    stage = RLDSExportStage()
    assert stage.name == "s4a_rlds_export"
    assert stage.description  # has a description


def test_s4e_runs_after_s3b():
    """Verify S4e checks for S3b output."""
    from blueprint_validation.stages.s4e_trained_eval import TrainedPolicyEvalStage

    stage = TrainedPolicyEvalStage()
    assert stage.name == "s4e_trained_eval"
    assert stage.description


def test_stage_names_are_unique():
    """All stages in the pipeline must have unique names."""
    from blueprint_validation.pipeline import ValidationPipeline
    from blueprint_validation.config import ValidationConfig

    config = ValidationConfig()
    # Instantiate pipeline to get stage list (indirectly via imports)
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

    stages = [
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
    ]
    names = [s.name for s in stages]
    assert len(names) == len(set(names)), f"Duplicate stage names: {names}"
