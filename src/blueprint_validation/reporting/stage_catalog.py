"""Canonical stage result names used by report collection."""

from __future__ import annotations


FACILITY_STAGE_RESULT_NAMES = [
    "s0_task_hints_bootstrap",
    "s0b_scene_memory_runtime",
    "s0a_scene_package",
    "s1_isaac_render",
    "s1_render",
    "s1b_robot_composite",
    "s1c_gemini_polish",
    "s1d_gaussian_augment",
    "s1f_external_interaction_ingest",
    "s1g_external_rollout_ingest",
]
