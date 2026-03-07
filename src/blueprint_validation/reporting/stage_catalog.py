"""Canonical stage result names used by report collection."""

from __future__ import annotations


FACILITY_STAGE_RESULT_NAMES = [
    "s0_task_hints_bootstrap",
    "s1_render",
    "s1b_robot_composite",
    "s1c_gemini_polish",
    "s1d_gaussian_augment",
    "s1f_external_interaction_ingest",
    "s2_enrich",
    "s3_finetune",
    "s3d_wm_refresh_loop",
    "s4_policy_eval",
    "s4a_rlds_export",
    "s3b_policy_finetune",
    "s3c_policy_rl_loop",
    "s4e_trained_eval",
    "s4b_rollout_dataset",
    "s4c_policy_pair_train",
    "s4d_policy_pair_eval",
    "s5_visual_fidelity",
    "s6_spatial_accuracy",
]
