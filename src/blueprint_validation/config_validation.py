"""Key-level validation for YAML config files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


_TOP_LEVEL_KEYS = {
    "schema_version",
    "project_name",
    "facilities",
    "render",
    "robot_composite",
    "gemini_polish",
    "enrich",
    "finetune",
    "eval_policy",
    "eval_polaris",
    "scene_builder",
    "policy_finetune",
    "policy_adapter",
    "robosplat",
    "robosplat_scan",
    "external_interaction",
    "external_rollouts",
    "native_teacher",
    "claim_portfolio",
    "action_boost",
    "policy_rl_loop",
    "wm_refresh_loop",
    "rollout_dataset",
    "policy_compare",
    "eval_visual",
    "eval_spatial",
    "eval_crosssite",
    "cloud",
}

_FACILITY_KEYS = {
    "name",
    "ply_path",
    "scene_package_path",
    "task_hints_path",
    "claim_benchmark_path",
    "description",
    "landmarks",
    "floor_height_m",
    "ceiling_height_m",
    "manipulation_zones",
    "up_axis",
    "scene_rotation_deg",
    "video_orientation_fix",
}

_MANIPULATION_ZONE_KEYS = {
    "name",
    "approach_point",
    "target_point",
    "camera_height_m",
    "camera_look_down_deg",
    "arc_radius_m",
}

_CAMERA_PATH_KEYS = {
    "type",
    "radius_m",
    "num_orbits",
    "length_m",
    "path",
    "height_override_m",
    "look_down_override_deg",
    "approach_point",
    "arc_radius_m",
    "arc_span_deg",
    "arc_phase_offset_deg",
    "source_tag",
    "target_instance_id",
    "target_label",
    "target_category",
    "target_role",
    "target_extents_m",
    "locked_eye_point",
    "locked_look_at_point",
    "locked_probe_motion_radius_m",
}

_RENDER_KEYS = {
    "backend",
    "resolution",
    "fps",
    "num_frames",
    "camera_height_m",
    "camera_look_down_deg",
    "camera_paths",
    "num_clips_per_path",
    "scene_aware",
    "collision_check",
    "voxel_size_m",
    "density_threshold",
    "min_clearance_m",
    "vlm_fallback",
    "vlm_fallback_model",
    "vlm_fallback_num_views",
    "task_scoped_scene_aware",
    "task_scoped_max_specs",
    "task_scoped_context_per_target",
    "task_scoped_overview_specs",
    "task_scoped_fallback_specs",
    "task_scoped_profile",
    "preserve_num_frames_after_collision_filter",
    "task_scoped_num_clips_per_path",
    "task_scoped_num_frames_override",
    "stage1_coverage_gate_enabled",
    "stage1_coverage_min_visible_frame_ratio",
    "stage1_coverage_min_approach_angle_bins",
    "stage1_coverage_angle_bin_deg",
    "stage1_coverage_blur_laplacian_min",
    "stage1_coverage_blur_sample_every_n_frames",
    "stage1_coverage_blur_max_samples_per_clip",
    "stage1_coverage_min_center_band_ratio",
    "stage1_coverage_center_band_x",
    "stage1_coverage_center_band_y",
    "stage1_quality_planner_enabled",
    "stage1_quality_candidate_budget",
    "stage1_quality_autoretry_enabled",
    "stage1_quality_max_regen_attempts",
    "stage1_quality_min_clip_score",
    "stage1_strict_require_task_hints",
    "stage1_active_perception_enabled",
    "stage1_active_perception_scope",
    "stage1_active_perception_max_loops",
    "stage1_active_perception_fail_closed",
    "stage1_probe_frames_override",
    "stage1_probe_resolution_scale",
    "stage1_probe_min_viable_pose_ratio",
    "stage1_probe_min_unique_positions",
    "stage1_probe_dedupe_enabled",
    "stage1_probe_dedupe_max_regen_attempts",
    "stage1_probe_dedupe_center_dist_m",
    "stage1_probe_consensus_votes",
    "stage1_probe_consensus_high_variance_delta",
    "stage1_probe_tiebreak_extra_votes",
    "stage1_probe_tiebreak_spread_threshold",
    "stage1_probe_primary_model_only",
    "stage1_vlm_min_task_score",
    "stage1_vlm_min_visual_score",
    "stage1_vlm_min_spatial_score",
    "stage1_keep_probe_videos",
    "stage1_repeat_dedupe_enabled",
    "stage1_repeat_dedupe_max_regen_attempts",
    "stage1_repeat_min_xy_jitter_m",
    "stage1_repeat_similarity_ssim_threshold",
    "scene_locked_profile",
    "orientation_autocorrect_enabled",
    "orientation_autocorrect_mode",
    "manipulation_random_xy_offset_m",
    "non_manipulation_random_xy_offset_m",
    "manipulation_target_z_bias_m",
}

_ENRICH_KEYS = {
    "cosmos_model",
    "cosmos_checkpoint",
    "cosmos_repo",
    "disable_guardrails",
    "controlnet_inputs",
    "num_variants_per_render",
    "variants",
    "guidance",
    "dynamic_variants",
    "dynamic_variants_model",
    "allow_dynamic_variant_fallback",
    "context_frame_index",
    "context_frame_mode",
    "max_input_frames",
    "max_source_clips",
    "min_source_clips",
    "min_valid_outputs",
    "max_blur_reject_rate",
    "green_frame_ratio_max",
    "enable_visual_collapse_gate",
    "vlm_quality_gate_enabled",
    "vlm_quality_fail_closed",
    "vlm_quality_autoretry_enabled",
    "vlm_quality_max_regen_attempts",
    "vlm_quality_min_task_score",
    "vlm_quality_min_visual_score",
    "vlm_quality_min_spatial_score",
    "vlm_quality_require_reasoning_consistency",
    "vlm_quality_retry_context_frame_stride",
    "vlm_quality_disable_depth_on_final_retry",
    "source_clip_selection_mode",
    "source_clip_selection_fail_closed",
    "source_clip_task",
    "source_clip_name",
    "multi_view_context_enabled",
    "multi_view_context_offsets",
    "scene_index_enabled",
    "scene_index_k",
    "scene_index_sample_every_n_frames",
    "cosmos_output_quality",
    "min_frame0_ssim",
    "delete_rejected_outputs",
}

_VARIANT_KEYS = {"name", "prompt"}

_ROBOT_COMPOSITE_KEYS = {
    "enabled",
    "urdf_path",
    "end_effector_link",
    "base_xyz",
    "base_rpy",
    "start_joint_positions",
    "end_joint_positions",
    "min_visible_joint_ratio",
    "min_consistency_score",
    "line_color_bgr",
    "line_thickness",
}

_GEMINI_POLISH_KEYS = {
    "enabled",
    "model",
    "api_key_env",
    "prompt",
    "sample_every_n_frames",
}

_FINETUNE_KEYS = {
    "dreamdojo_repo",
    "dreamdojo_checkpoint",
    "python_executable",
    "experiment_config",
    "eval_world_experiment",
    "model_size",
    "video_dataset_backend",
    "probe_dataloader_sample",
    "use_lora",
    "lora_rank",
    "lora_alpha",
    "lora_target_modules",
    "learning_rate",
    "num_epochs",
    "batch_size",
    "gradient_accumulation_steps",
    "warmup_steps",
    "save_every_n_epochs",
    "max_training_hours",
    "dataset_quality",
}

_DATASET_QUALITY_KEYS = {
    "strict_manifest_validation",
    "quarantine_rejections",
    "fail_on_rejections",
    "max_reject_fraction",
    "enable_duplicate_detection",
    "enable_leakage_detection",
    "prompt_lint",
    "temporal_gates",
    "distribution",
}

_DATASET_PROMPT_LINT_KEYS = {
    "enabled",
    "min_chars",
    "min_tokens",
    "min_unique_token_ratio",
    "allow_generic_substrings",
}

_DATASET_TEMPORAL_KEYS = {
    "enabled",
    "min_frames_for_check",
    "max_frames_to_sample",
    "min_mean_interframe_delta",
    "max_freeze_ratio",
    "max_abrupt_cut_ratio",
    "max_blockiness_score",
}

_DATASET_DISTRIBUTION_KEYS = {
    "enabled",
    "min_total_clips_for_caps",
    "min_unique_variants",
    "min_unique_source_clips",
    "max_single_variant_fraction",
    "max_single_source_clip_fraction",
    "max_prompt_dominance_fraction",
}

_EVAL_POLICY_KEYS = {
    "model_name",
    "checkpoint_path",
    "unnorm_key",
    "num_rollouts",
    "max_steps_per_rollout",
    "tasks",
    "manipulation_tasks",
    "conditions",
    "headline_scope",
    "rollout_driver",
    "scripted_rollouts_per_task",
    "mode",
    "required_action_dim",
    "manip_eval_mode",
    "min_assignment_quality_score",
    "require_object_grounded_manip_tasks",
    "min_absolute_difference",
    "min_manip_success_delta_pp",
    "require_native_action_compat",
    "claim_protocol",
    "primary_endpoint",
    "freeze_world_snapshot",
    "split_strategy",
    "min_practical_success_lift_pp",
    "replication",
    "claim_strictness",
    "reliability",
    "vlm_judge",
    "openvla_model",
    "openvla_checkpoint",
}

_EVAL_POLARIS_KEYS = {
    "enabled",
    "repo_path",
    "hub_path",
    "environment_mode",
    "environment_name",
    "default_as_primary_gate",
    "use_for_claim_gate",
    "num_rollouts",
    "device",
    "policy_client",
    "observation_mode",
    "action_mode",
    "export_dir",
    "require_scene_package",
    "require_success_correlation_metadata",
}

_SCENE_BUILDER_KEYS = {
    "enabled",
    "source_ply_path",
    "output_scene_root",
    "static_collision_mode",
    "asset_manifest_path",
    "robot_type",
    "task_template",
    "emit_isaac_lab",
    "emit_polaris_metadata",
}

_CLAIM_REPLICATION_KEYS = {"training_seeds"}
_CLAIM_STRICTNESS_KEYS = {
    "min_eval_task_specs",
    "min_eval_start_clips",
    "min_common_eval_cells",
    "min_positive_training_seeds",
    "p_value_threshold",
    "require_site_specific_advantage",
    "site_vs_generic_min_lift_pp",
}
_POLICY_EVAL_RELIABILITY_KEYS = {
    "max_horizon_steps",
    "keyframe_reanchor_every",
    "min_replay_pass_rate",
    "min_controllability_pass_rate",
    "enforce_stage_success",
    "max_scoring_failure_rate",
    "fail_on_short_rollout",
    "min_rollout_frames",
    "min_rollout_steps",
}
_VLM_JUDGE_KEYS = {
    "model",
    "fallback_models",
    "api_key_env",
    "enable_agentic_vision",
    "video_metadata_fps",
    "scoring_prompt",
}

_POLICY_FINETUNE_KEYS = {
    "enabled",
    "openvla_repo",
    "finetune_script",
    "data_root_dir",
    "dataset_name",
    "run_root_dir",
    "adapter_tmp_dir",
    "lora_rank",
    "batch_size",
    "grad_accumulation_steps",
    "learning_rate",
    "save_steps",
    "max_steps",
    "image_aug",
    "nproc_per_node",
    "wandb_project",
    "wandb_entity",
    "recipe",
    "action_chunk_size",
    "use_continuous_actions",
    "use_l1_regression",
    "parallel_decoding",
    "seed",
    "extra_args",
}

_POLICY_ADAPTER_KEYS = {"name", "openvla", "pi05", "dreamzero"}
_OPENVLA_ADAPTER_KEYS = {
    "openvla_repo",
    "finetune_script",
    "base_model_name",
    "base_checkpoint_path",
    "policy_action_dim",
    "extra_train_args",
}
_PI05_ADAPTER_KEYS = {
    "openpi_repo",
    "profile",
    "runtime_mode",
    "train_backend",
    "train_script",
    "norm_stats_script",
    "policy_action_dim",
    "policy_state_dim",
    "strict_action_contract",
    "allow_synthetic_state_for_eval",
    "extra_train_args",
}
_DREAMZERO_ADAPTER_KEYS = {
    "repo_path",
    "base_model_name",
    "checkpoint_path",
    "inference_module",
    "inference_class",
    "policy_action_dim",
    "frame_history",
    "strict_action_contract",
    "strict_action_min",
    "strict_action_max",
    "train_script",
    "extra_train_args",
    "allow_training",
}

_ROBOSPLAT_KEYS = {
    "enabled",
    "backend",
    "parity_mode",
    "runtime_preset",
    "variants_per_input",
    "object_source_priority",
    "demo_source",
    "demo_manifest_path",
    "min_successful_demos",
    "demo_success_task_score_threshold",
    "require_manipulation_success_flags",
    "world_model_bootstrap_enabled",
    "bootstrap_if_missing_demo",
    "bootstrap_num_rollouts",
    "bootstrap_horizon_steps",
    "bootstrap_tasks_limit",
    "quality_gate_enabled",
    "min_variants_required_per_clip",
    "fallback_to_legacy_scan",
    "fallback_on_backend_error",
    "persist_scene_variants",
    "vendor_repo_path",
    "vendor_ref",
}

_ROBOSPLAT_SCAN_KEYS = {
    "enabled",
    "num_augmented_clips_per_input",
    "yaw_jitter_deg",
    "pitch_jitter_deg",
    "camera_height_jitter_m",
    "relight_gain_min",
    "relight_gain_max",
    "color_temp_shift",
    "temporal_speed_factors",
}

_EXTERNAL_INTERACTION_KEYS = {"enabled", "manifest_path", "source_name"}
_EXTERNAL_ROLLOUTS_KEYS = {"enabled", "manifest_path", "source_name", "mode"}
_NATIVE_TEACHER_KEYS = {
    "enabled",
    "include_generic_control",
    "generate_corrections",
    "planner_horizon_steps",
}
_CLAIM_PORTFOLIO_KEYS = {
    "min_facilities",
    "min_mean_site_vs_frozen_lift_pp",
    "min_mean_site_vs_generic_lift_pp",
    "max_negative_task_family_delta_pp",
    "require_manipulation_nonzero",
}
_ACTION_BOOST_KEYS = {
    "enabled",
    "require_full_pipeline",
    "auto_switch_headline_scope_to_dual",
    "auto_enable_rollout_dataset",
    "auto_enable_policy_finetune",
    "auto_enable_policy_rl_loop",
    "compute_profile",
    "strict_disjoint_eval",
}
_POLICY_RL_LOOP_KEYS = {
    "enabled",
    "iterations",
    "horizon_steps",
    "rollouts_per_task",
    "group_size",
    "reward_mode",
    "vlm_reward_fraction",
    "top_quantile",
    "near_miss_min_quantile",
    "near_miss_max_quantile",
    "policy_refine_steps_per_iter",
    "policy_refine_near_miss_fraction",
    "policy_refine_hard_negative_fraction",
    "world_model_refresh_enabled",
    "world_model_refresh_mix_with_stage2",
    "world_model_refresh_require_stage2_vlm_pass",
    "world_model_refresh_stage2_fraction",
    "world_model_refresh_success_fraction",
    "world_model_refresh_near_miss_fraction",
    "world_model_refresh_min_total_clips",
    "world_model_refresh_max_total_clips",
    "world_model_refresh_seed",
    "world_model_refresh_epochs",
    "world_model_refresh_learning_rate",
}
_WM_REFRESH_LOOP_KEYS = {
    "enabled",
    "iterations",
    "source_condition",
    "fail_if_refresh_fails",
    "fail_on_degenerate_mix",
    "min_non_hard_rollouts",
    "max_hard_negative_fraction",
    "require_valid_video_decode",
    "enforce_vlm_quality_floor",
    "min_refresh_task_score",
    "min_refresh_visual_score",
    "min_refresh_spatial_score",
    "fail_on_reasoning_conflict",
    "backfill_from_stage2_vlm_passed",
    "quantile_fallback_enabled",
    "quantile_success_threshold",
    "quantile_near_miss_threshold",
}
_ROLLOUT_DATASET_KEYS = {
    "enabled",
    "seed",
    "train_split",
    "min_steps_per_rollout",
    "task_score_threshold",
    "include_failed_rollouts",
    "selection_mode",
    "near_miss_min_task_score",
    "near_miss_max_task_score",
    "near_miss_target_fraction",
    "hard_negative_target_fraction",
    "per_task_max_episodes",
    "max_action_delta_norm",
    "require_consistent_action_dim",
    "baseline_dataset_name",
    "adapted_dataset_name",
    "export_dir",
}
_POLICY_COMPARE_KEYS = {
    "enabled",
    "heldout_num_rollouts",
    "heldout_seed",
    "eval_world_model",
    "heldout_tasks",
    "control_arms",
    "task_score_success_threshold",
    "manipulation_task_keywords",
    "require_grasp_for_manipulation",
    "require_lift_for_manipulation",
    "require_place_for_manipulation",
}
_EVAL_VISUAL_KEYS = {"metrics", "lpips_backbone"}
_EVAL_SPATIAL_KEYS = {"num_sample_frames", "vlm_model", "min_valid_samples", "fail_on_reasoning_conflict"}
_EVAL_CROSSSITE_KEYS = {"num_clips_per_model", "vlm_model"}
_CLOUD_KEYS = {"provider", "gpu_type", "num_gpus", "max_cost_usd", "auto_shutdown"}


def _format_path(prefix: str, key: object) -> str:
    token = str(key)
    return f"{prefix}.{token}" if prefix else token


def _expect_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section '{label}' must be a mapping")
    return value


def _collect_unknown_keys(
    mapping: Mapping[str, object],
    allowed_keys: set[str],
    prefix: str,
    unknown_keys: list[str],
) -> None:
    for key in mapping.keys():
        if str(key) not in allowed_keys:
            unknown_keys.append(_format_path(prefix, key))


def _validate_mapping_section(
    raw: Mapping[str, object],
    key: str,
    allowed_keys: set[str],
    unknown_keys: list[str],
) -> Mapping[str, object] | None:
    if key not in raw:
        return None
    mapping = _expect_mapping(raw[key], key)
    _collect_unknown_keys(mapping, allowed_keys, key, unknown_keys)
    return mapping


def validate_config_keys(raw: Mapping[str, object], *, config_path: Path) -> None:
    unknown_keys: list[str] = []
    _collect_unknown_keys(raw, _TOP_LEVEL_KEYS, "", unknown_keys)

    facilities = _expect_mapping(raw["facilities"], "facilities") if "facilities" in raw else None
    if facilities is not None:
        for facility_id, facility_raw in facilities.items():
            facility = _expect_mapping(facility_raw, f"facilities.{facility_id}")
            facility_prefix = _format_path("facilities", facility_id)
            _collect_unknown_keys(facility, _FACILITY_KEYS, facility_prefix, unknown_keys)
            zones = facility.get("manipulation_zones")
            if zones is not None:
                if not isinstance(zones, list):
                    raise ValueError(f"Config section '{facility_prefix}.manipulation_zones' must be a list")
                for idx, zone_raw in enumerate(zones):
                    zone = _expect_mapping(zone_raw, f"{facility_prefix}.manipulation_zones[{idx}]")
                    _collect_unknown_keys(
                        zone,
                        _MANIPULATION_ZONE_KEYS,
                        f"{facility_prefix}.manipulation_zones[{idx}]",
                        unknown_keys,
                    )

    render = _validate_mapping_section(raw, "render", _RENDER_KEYS, unknown_keys)
    if render is not None and render.get("camera_paths") is not None:
        camera_paths = render.get("camera_paths")
        if not isinstance(camera_paths, list):
            raise ValueError("Config section 'render.camera_paths' must be a list")
        for idx, path_raw in enumerate(camera_paths):
            path_mapping = _expect_mapping(path_raw, f"render.camera_paths[{idx}]")
            _collect_unknown_keys(
                path_mapping,
                _CAMERA_PATH_KEYS,
                f"render.camera_paths[{idx}]",
                unknown_keys,
            )

    enrich = _validate_mapping_section(raw, "enrich", _ENRICH_KEYS, unknown_keys)
    if enrich is not None and enrich.get("variants") is not None:
        variants = enrich.get("variants")
        if not isinstance(variants, list):
            raise ValueError("Config section 'enrich.variants' must be a list")
        for idx, variant_raw in enumerate(variants):
            variant = _expect_mapping(variant_raw, f"enrich.variants[{idx}]")
            _collect_unknown_keys(variant, _VARIANT_KEYS, f"enrich.variants[{idx}]", unknown_keys)

    _validate_mapping_section(raw, "robot_composite", _ROBOT_COMPOSITE_KEYS, unknown_keys)
    _validate_mapping_section(raw, "gemini_polish", _GEMINI_POLISH_KEYS, unknown_keys)

    finetune = _validate_mapping_section(raw, "finetune", _FINETUNE_KEYS, unknown_keys)
    if finetune is not None and finetune.get("dataset_quality") is not None:
        dataset_quality = _expect_mapping(finetune["dataset_quality"], "finetune.dataset_quality")
        _collect_unknown_keys(dataset_quality, _DATASET_QUALITY_KEYS, "finetune.dataset_quality", unknown_keys)
        if dataset_quality.get("prompt_lint") is not None:
            prompt_lint = _expect_mapping(
                dataset_quality["prompt_lint"],
                "finetune.dataset_quality.prompt_lint",
            )
            _collect_unknown_keys(
                prompt_lint,
                _DATASET_PROMPT_LINT_KEYS,
                "finetune.dataset_quality.prompt_lint",
                unknown_keys,
            )
        if dataset_quality.get("temporal_gates") is not None:
            temporal = _expect_mapping(
                dataset_quality["temporal_gates"],
                "finetune.dataset_quality.temporal_gates",
            )
            _collect_unknown_keys(
                temporal,
                _DATASET_TEMPORAL_KEYS,
                "finetune.dataset_quality.temporal_gates",
                unknown_keys,
            )
        if dataset_quality.get("distribution") is not None:
            distribution = _expect_mapping(
                dataset_quality["distribution"],
                "finetune.dataset_quality.distribution",
            )
            _collect_unknown_keys(
                distribution,
                _DATASET_DISTRIBUTION_KEYS,
                "finetune.dataset_quality.distribution",
                unknown_keys,
            )

    eval_policy = _validate_mapping_section(raw, "eval_policy", _EVAL_POLICY_KEYS, unknown_keys)
    if eval_policy is not None:
        for nested_key, allowed in (
            ("replication", _CLAIM_REPLICATION_KEYS),
            ("claim_strictness", _CLAIM_STRICTNESS_KEYS),
            ("reliability", _POLICY_EVAL_RELIABILITY_KEYS),
            ("vlm_judge", _VLM_JUDGE_KEYS),
        ):
            if eval_policy.get(nested_key) is not None:
                nested_mapping = _expect_mapping(eval_policy[nested_key], f"eval_policy.{nested_key}")
                _collect_unknown_keys(
                    nested_mapping,
                    allowed,
                    f"eval_policy.{nested_key}",
                    unknown_keys,
                )

    _validate_mapping_section(raw, "eval_polaris", _EVAL_POLARIS_KEYS, unknown_keys)
    _validate_mapping_section(raw, "scene_builder", _SCENE_BUILDER_KEYS, unknown_keys)
    _validate_mapping_section(raw, "policy_finetune", _POLICY_FINETUNE_KEYS, unknown_keys)
    policy_adapter = _validate_mapping_section(raw, "policy_adapter", _POLICY_ADAPTER_KEYS, unknown_keys)
    if policy_adapter is not None:
        for nested_key, allowed in (
            ("openvla", _OPENVLA_ADAPTER_KEYS),
            ("pi05", _PI05_ADAPTER_KEYS),
            ("dreamzero", _DREAMZERO_ADAPTER_KEYS),
        ):
            if policy_adapter.get(nested_key) is not None:
                nested_mapping = _expect_mapping(policy_adapter[nested_key], f"policy_adapter.{nested_key}")
                _collect_unknown_keys(
                    nested_mapping,
                    allowed,
                    f"policy_adapter.{nested_key}",
                    unknown_keys,
                )

    for section, allowed in (
        ("robosplat", _ROBOSPLAT_KEYS),
        ("robosplat_scan", _ROBOSPLAT_SCAN_KEYS),
        ("external_interaction", _EXTERNAL_INTERACTION_KEYS),
        ("external_rollouts", _EXTERNAL_ROLLOUTS_KEYS),
        ("native_teacher", _NATIVE_TEACHER_KEYS),
        ("claim_portfolio", _CLAIM_PORTFOLIO_KEYS),
        ("action_boost", _ACTION_BOOST_KEYS),
        ("policy_rl_loop", _POLICY_RL_LOOP_KEYS),
        ("wm_refresh_loop", _WM_REFRESH_LOOP_KEYS),
        ("rollout_dataset", _ROLLOUT_DATASET_KEYS),
        ("policy_compare", _POLICY_COMPARE_KEYS),
        ("eval_visual", _EVAL_VISUAL_KEYS),
        ("eval_spatial", _EVAL_SPATIAL_KEYS),
        ("eval_crosssite", _EVAL_CROSSSITE_KEYS),
        ("cloud", _CLOUD_KEYS),
    ):
        _validate_mapping_section(raw, section, allowed, unknown_keys)

    if unknown_keys:
        joined = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown config key(s) in {config_path}: {joined}")
