"""Tests for configuration loading and validation."""

import json
from pathlib import Path

import pytest


def test_load_config_from_yaml(sample_config_yaml):
    from blueprint_validation.config import load_config

    config = load_config(sample_config_yaml)
    assert config.project_name == "Test"
    assert "test_a" in config.facilities
    assert config.facilities["test_a"].name == "Test A"
    assert config.render.resolution == (120, 160)
    assert config.render.fps == 5


def test_load_config_rejects_unknown_top_level_key(tmp_path):
    from blueprint_validation.config import load_config

    config_path = tmp_path / "unknown_top_level.yaml"
    config_path.write_text(
        """
project_name: Test
facilities:
  a:
    name: A
    ply_path: /tmp/a.ply
totally_unknown: true
""".strip()
    )

    with pytest.raises(ValueError, match="totally_unknown"):
        load_config(config_path)


def test_load_config_rejects_unknown_nested_key(tmp_path):
    from blueprint_validation.config import load_config

    config_path = tmp_path / "unknown_nested.yaml"
    config_path.write_text(
        """
project_name: Test
facilities:
  a:
    name: A
    ply_path: /tmp/a.ply
render:
  resolution: [64, 64]
  num_framez: 8
""".strip()
    )

    with pytest.raises(ValueError, match=r"render\.num_framez"):
        load_config(config_path)


def test_load_config_rejects_removed_splatsim_key(tmp_path):
    from blueprint_validation.config import load_config

    config_path = tmp_path / "removed_splatsim.yaml"
    config_path.write_text(
        """
project_name: Test
facilities:
  a:
    name: A
    ply_path: /tmp/a.ply
splatsim:
  enabled: false
""".strip()
    )

    with pytest.raises(ValueError, match="splatsim"):
        load_config(config_path)


def test_load_config_rejects_invalid_render_backend(tmp_path):
    from blueprint_validation.config import load_config

    config_path = tmp_path / "bad_render_backend.yaml"
    config_path.write_text(
        """
project_name: Test
facilities:
  a:
    name: A
    ply_path: /tmp/a.ply
render:
  backend: usd_magic
""".strip()
    )

    with pytest.raises(ValueError, match="render.backend"):
        load_config(config_path)


def test_config_defaults():
    from blueprint_validation.config import ValidationConfig

    config = ValidationConfig()
    assert config.schema_version == "v1"
    assert config.render.backend == "auto"
    assert config.render.resolution == (480, 640)
    assert config.render.num_frames == 49
    assert config.render.camera_height_m == 1.2
    assert config.render.task_scoped_scene_aware is False
    assert config.render.task_scoped_max_specs == 40
    assert config.render.preserve_num_frames_after_collision_filter is True
    assert config.render.task_scoped_num_clips_per_path == 1
    assert config.render.task_scoped_num_frames_override == 0
    assert config.render.stage1_coverage_gate_enabled is True
    assert config.render.stage1_coverage_min_visible_frame_ratio == pytest.approx(0.35)
    assert config.render.stage1_coverage_min_approach_angle_bins == 2
    assert config.render.stage1_coverage_min_center_band_ratio == pytest.approx(0.4)
    assert config.render.stage1_coverage_center_band_x == [0.2, 0.8]
    assert config.render.stage1_coverage_center_band_y == [0.2, 0.8]
    assert config.render.vlm_fallback is False
    assert config.render.stage1_quality_planner_enabled is True
    assert config.render.stage1_quality_candidate_budget == "medium"
    assert config.render.stage1_quality_autoretry_enabled is True
    assert config.render.stage1_quality_max_regen_attempts == 2
    assert config.render.stage1_quality_min_clip_score == pytest.approx(0.55)
    assert config.render.stage1_strict_require_task_hints is False
    assert config.render.stage1_active_perception_enabled is True
    assert config.render.stage1_active_perception_scope == "all"
    assert config.render.stage1_active_perception_max_loops == 2
    assert config.render.stage1_active_perception_fail_closed is True
    assert config.render.stage1_probe_frames_override == 0
    assert config.render.stage1_probe_resolution_scale == pytest.approx(0.0)
    assert config.render.stage1_probe_min_viable_pose_ratio == pytest.approx(0.55)
    assert config.render.stage1_probe_min_unique_positions == 8
    assert config.render.stage1_probe_dedupe_enabled is True
    assert config.render.stage1_probe_dedupe_max_regen_attempts == 2
    assert config.render.stage1_probe_dedupe_center_dist_m == pytest.approx(0.08)
    assert config.render.stage1_vlm_min_task_score == pytest.approx(7.0)
    assert config.render.stage1_vlm_min_visual_score == pytest.approx(7.0)
    assert config.render.stage1_vlm_min_spatial_score == pytest.approx(6.0)
    assert config.render.stage1_probe_tiebreak_extra_votes == 2
    assert config.render.stage1_probe_tiebreak_spread_threshold == pytest.approx(3.0)
    assert config.render.stage1_keep_probe_videos is False
    assert config.render.scene_locked_profile == "auto"
    assert config.render.orientation_autocorrect_enabled is True
    assert config.render.orientation_autocorrect_mode == "auto"
    assert config.render.manipulation_random_xy_offset_m == pytest.approx(0.0)
    assert config.render.non_manipulation_random_xy_offset_m == pytest.approx(1.0)
    assert config.render.manipulation_target_z_bias_m == pytest.approx(0.0)
    assert config.finetune.num_epochs == 50
    assert config.finetune.use_lora is True
    assert config.finetune.lora_rank == 32
    assert config.finetune.video_dataset_backend == "opencv"
    assert config.finetune.probe_dataloader_sample is True
    assert config.eval_policy.vlm_judge.model == "gemini-3-flash-preview"
    assert config.eval_policy.vlm_judge.fallback_models == ["gemini-2.5-flash"]
    assert config.eval_policy.model_name == "openvla/openvla-7b"
    assert str(config.eval_policy.checkpoint_path).endswith("data/checkpoints/openvla-7b")
    assert config.eval_policy.unnorm_key == "bridge_orig"
    assert config.eval_policy.headline_scope == "wm_only"
    assert config.eval_policy.rollout_driver == "scripted"
    assert config.eval_policy.scripted_rollouts_per_task == 12
    assert config.eval_policy.manip_eval_mode == "overlay_marker"
    assert config.eval_policy.min_assignment_quality_score == pytest.approx(0.0)
    assert config.eval_policy.require_object_grounded_manip_tasks is True
    assert config.eval_policy.reliability.enforce_stage_success is False
    assert config.eval_policy.reliability.max_scoring_failure_rate == pytest.approx(0.02)
    assert config.eval_policy.reliability.fail_on_short_rollout is False
    assert config.eval_policy.reliability.min_rollout_frames == 13
    assert config.eval_policy.reliability.min_rollout_steps == 12
    assert config.eval_policy.min_absolute_difference == pytest.approx(1.0)
    assert config.eval_policy.min_manip_success_delta_pp == pytest.approx(15.0)
    assert config.eval_policy.vlm_judge.enable_agentic_vision is True
    assert config.eval_policy.vlm_judge.video_metadata_fps == pytest.approx(10.0)
    assert config.eval_polaris.enabled is False
    assert str(config.eval_polaris.repo_path).endswith("opt/PolaRiS")
    assert config.eval_polaris.environment_mode == "scene_package_bridge"
    assert config.eval_polaris.default_as_primary_gate is True
    assert config.eval_polaris.observation_mode == "external_only"
    assert config.eval_polaris.action_mode == "native"
    assert config.enrich.max_input_frames == 0
    assert config.enrich.max_source_clips == 0
    assert config.enrich.min_source_clips == 8
    assert config.enrich.min_valid_outputs == 8
    assert config.enrich.max_blur_reject_rate == pytest.approx(0.30)
    assert config.enrich.green_frame_ratio_max == pytest.approx(0.10)
    assert config.enrich.enable_visual_collapse_gate is True
    assert config.enrich.vlm_quality_gate_enabled is False
    assert config.enrich.vlm_quality_fail_closed is True
    assert config.enrich.vlm_quality_autoretry_enabled is True
    assert config.enrich.vlm_quality_max_regen_attempts == 2
    assert config.enrich.vlm_quality_min_task_score == pytest.approx(7.0)
    assert config.enrich.vlm_quality_min_visual_score == pytest.approx(7.0)
    assert config.enrich.vlm_quality_min_spatial_score == pytest.approx(6.0)
    assert config.enrich.vlm_quality_require_reasoning_consistency is True
    assert config.enrich.vlm_quality_retry_context_frame_stride == 6
    assert config.enrich.vlm_quality_disable_depth_on_final_retry is True
    assert config.enrich.source_clip_selection_mode == "all"
    assert config.enrich.source_clip_selection_fail_closed is True
    assert config.enrich.source_clip_task is None
    assert config.enrich.source_clip_name is None
    assert config.enrich.multi_view_context_enabled is False
    assert config.enrich.multi_view_context_offsets == [-12, 0, 12]
    assert config.enrich.scene_index_enabled is False
    assert config.enrich.scene_index_k == 3
    assert config.enrich.scene_index_sample_every_n_frames == 8
    assert config.enrich.cosmos_output_quality == 5
    assert config.enrich.context_frame_mode == "target_centered"
    assert config.policy_adapter.name == "openvla_oft"
    assert str(config.policy_adapter.openvla.openvla_repo).endswith("opt/openvla-oft")
    assert config.policy_adapter.pi05.profile == "pi05_libero"
    assert config.external_interaction.enabled is True
    assert config.external_rollouts.enabled is False
    assert config.rollout_dataset.enabled is True
    assert config.rollout_dataset.selection_mode == "success_near_miss"
    assert config.rollout_dataset.near_miss_min_task_score == pytest.approx(5.0)
    assert config.rollout_dataset.near_miss_max_task_score == pytest.approx(6.99)
    assert config.rollout_dataset.near_miss_target_fraction == pytest.approx(0.30)
    assert config.rollout_dataset.hard_negative_target_fraction == pytest.approx(0.0)
    assert config.rollout_dataset.per_task_max_episodes == 0
    assert config.action_boost.enabled is True
    assert config.action_boost.require_full_pipeline is True
    assert config.action_boost.auto_switch_headline_scope_to_dual is True
    assert config.action_boost.compute_profile == "standard"
    assert config.action_boost.strict_disjoint_eval is True
    assert config.policy_compare.enabled is False
    assert config.gemini_polish.enabled is False
    assert config.enrich.dynamic_variants is False
    assert config.policy_finetune.enabled is True
    assert config.policy_finetune.dataset_name == "bridge_orig"
    assert config.policy_finetune.recipe == "oft"
    assert config.robosplat.enabled is True
    assert config.robosplat.backend == "auto"
    assert config.robosplat_scan.enabled is True
    assert config.policy_rl_loop.enabled is False
    assert config.policy_rl_loop.policy_refine_near_miss_fraction == pytest.approx(0.30)
    assert config.policy_rl_loop.policy_refine_hard_negative_fraction == pytest.approx(0.10)
    assert config.policy_rl_loop.world_model_refresh_mix_with_stage2 is True
    assert config.policy_rl_loop.world_model_refresh_require_stage2_vlm_pass is True
    assert config.policy_rl_loop.world_model_refresh_stage2_fraction == pytest.approx(0.60)
    assert config.policy_rl_loop.world_model_refresh_success_fraction == pytest.approx(0.25)
    assert config.policy_rl_loop.world_model_refresh_near_miss_fraction == pytest.approx(0.15)
    assert config.policy_rl_loop.world_model_refresh_min_total_clips == 128
    assert config.policy_rl_loop.world_model_refresh_max_total_clips == 512
    assert config.policy_rl_loop.world_model_refresh_seed == 17
    assert config.wm_refresh_loop.enabled is True
    assert config.wm_refresh_loop.iterations == 1
    assert config.wm_refresh_loop.source_condition == "adapted"
    assert config.wm_refresh_loop.fail_on_degenerate_mix is True
    assert config.wm_refresh_loop.min_non_hard_rollouts == 8
    assert config.wm_refresh_loop.max_hard_negative_fraction == pytest.approx(0.75)
    assert config.wm_refresh_loop.require_valid_video_decode is True
    assert config.wm_refresh_loop.enforce_vlm_quality_floor is True
    assert config.wm_refresh_loop.min_refresh_task_score == pytest.approx(7.0)
    assert config.wm_refresh_loop.min_refresh_visual_score == pytest.approx(7.0)
    assert config.wm_refresh_loop.min_refresh_spatial_score == pytest.approx(6.0)
    assert config.wm_refresh_loop.fail_on_reasoning_conflict is True
    assert config.wm_refresh_loop.backfill_from_stage2_vlm_passed is True
    assert config.wm_refresh_loop.quantile_fallback_enabled is True
    assert config.wm_refresh_loop.quantile_success_threshold == pytest.approx(0.85)
    assert config.wm_refresh_loop.quantile_near_miss_threshold == pytest.approx(0.50)
    assert config.eval_spatial.min_valid_samples == 3
    assert config.eval_spatial.fail_on_reasoning_conflict is True


def test_facility_config():
    from blueprint_validation.config import FacilityConfig

    fac = FacilityConfig(
        name="Test",
        ply_path=Path("/tmp/test.ply"),
        landmarks=["a", "b"],
    )
    assert fac.name == "Test"
    assert fac.floor_height_m == 0.0
    assert len(fac.landmarks) == 2
    assert fac.up_axis == "auto"
    assert fac.scene_rotation_deg == [0.0, 0.0, 0.0]
    assert fac.video_orientation_fix == "none"
    assert fac.scene_package_path is None


def test_config_parses_polaris_block_and_scene_package(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    scene_root = tmp_path / "scene_pkg"
    (scene_root / "assets").mkdir(parents=True)
    (scene_root / "usd").mkdir(parents=True)
    (scene_root / "assets" / "scene_manifest.json").write_text("{}")
    (scene_root / "usd" / "scene.usda").write_text("#usda 1.0")
    config_path = tmp_path / "polaris.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "polaris",
                "facilities": {
                    "a": {
                        "name": "A",
                        "ply_path": "/tmp/a.ply",
                        "scene_package_path": str(scene_root),
                    }
                },
                "eval_polaris": {
                    "enabled": True,
                    "repo_path": "/tmp/polaris",
                    "hub_path": "/tmp/polaris-hub",
                    "environment_mode": "native_bundle",
                    "environment_name": "DROID-TestScene",
                    "default_as_primary_gate": True,
                    "use_for_claim_gate": False,
                    "num_rollouts": 8,
                    "device": "cpu",
                    "policy_client": "OpenVLA",
                    "observation_mode": "external_wrist_stitched",
                    "action_mode": "joint_position_bridge",
                    "export_dir": "/tmp/polaris-outputs",
                    "require_scene_package": True,
                    "require_success_correlation_metadata": False,
                },
            }
        )
    )

    cfg = load_config(config_path)
    assert cfg.facilities["a"].scene_package_path == scene_root.resolve()
    assert cfg.eval_polaris.enabled is True
    assert cfg.eval_polaris.environment_mode == "native_bundle"
    assert cfg.eval_polaris.environment_name == "DROID-TestScene"
    assert cfg.eval_polaris.observation_mode == "external_wrist_stitched"
    assert cfg.eval_polaris.action_mode == "joint_position_bridge"


def test_config_parses_scene_builder_block(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "scene_builder.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "scene-builder",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "scene_builder": {
                    "enabled": True,
                    "source_ply_path": "/tmp/source.ply",
                    "output_scene_root": str(tmp_path / "scene_pkg"),
                    "static_collision_mode": "simple",
                    "asset_manifest_path": str(tmp_path / "assets.json"),
                    "scene_edit_manifest_path": str(tmp_path / "scene_edit.json"),
                    "task_hints_path": str(tmp_path / "task_hints.json"),
                    "robot_type": "franka",
                    "task_template": "pick_place_v1",
                    "emit_isaac_lab": True,
                    "emit_polaris_metadata": True,
                    "fail_on_physics_qc": True,
                },
            }
        )
    )
    cfg = load_config(config_path)
    assert cfg.scene_builder.enabled is True
    assert cfg.scene_builder.source_ply_path == Path("/tmp/source.ply")
    assert cfg.scene_builder.output_scene_root == (tmp_path / "scene_pkg").resolve()
    assert cfg.scene_builder.asset_manifest_path == (tmp_path / "assets.json").resolve()
    assert cfg.scene_builder.scene_edit_manifest_path == (tmp_path / "scene_edit.json").resolve()
    assert cfg.scene_builder.task_hints_path == (tmp_path / "task_hints.json").resolve()
    assert cfg.scene_builder.fail_on_physics_qc is True


def test_config_parses_camera_path_target_metadata(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "camera_path_metadata.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "CameraPath Metadata",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "render": {
                    "camera_paths": [
                        {
                            "type": "manipulation",
                            "approach_point": [0.0, 0.0, 0.5],
                            "target_instance_id": "101",
                            "target_label": "bowl",
                            "target_category": "manipulation",
                            "target_role": "targets",
                        }
                    ]
                },
            }
        )
    )
    config = load_config(config_path)
    path_spec = config.render.camera_paths[0]
    assert path_spec.target_instance_id == "101"
    assert path_spec.target_label == "bowl"
    assert path_spec.target_category == "manipulation"
    assert path_spec.target_role == "targets"


def test_config_with_all_sections(tmp_path):
    from blueprint_validation.config import load_config

    config_data = {
        "schema_version": "v1",
        "project_name": "Full Test",
        "facilities": {
            "a": {
                "name": "A",
                "ply_path": "/tmp/a.ply",
                "task_hints_path": "/tmp/a_tasks.json",
                "video_orientation_fix": "rotate180",
            },
            "b": {"name": "B", "ply_path": "/tmp/b.ply"},
        },
        "render": {
            "backend": "auto",
            "resolution": [240, 320],
            "num_frames": 10,
            "task_scoped_scene_aware": True,
            "task_scoped_max_specs": 35,
            "task_scoped_context_per_target": 1,
            "task_scoped_overview_specs": 4,
            "task_scoped_fallback_specs": 10,
            "task_scoped_profile": "dreamdojo",
            "preserve_num_frames_after_collision_filter": True,
            "task_scoped_num_clips_per_path": 2,
            "task_scoped_num_frames_override": 97,
            "stage1_coverage_gate_enabled": True,
            "stage1_coverage_min_visible_frame_ratio": 0.4,
            "stage1_coverage_min_approach_angle_bins": 3,
            "stage1_coverage_angle_bin_deg": 30.0,
            "stage1_coverage_blur_laplacian_min": 25.0,
            "stage1_coverage_blur_sample_every_n_frames": 3,
            "stage1_coverage_blur_max_samples_per_clip": 9,
            "stage1_coverage_min_center_band_ratio": 0.5,
            "stage1_coverage_center_band_x": [0.25, 0.75],
            "stage1_coverage_center_band_y": [0.3, 0.7],
            "stage1_quality_planner_enabled": True,
            "stage1_quality_candidate_budget": "high",
            "stage1_quality_autoretry_enabled": True,
            "stage1_quality_max_regen_attempts": 3,
            "stage1_quality_min_clip_score": 0.62,
            "stage1_strict_require_task_hints": True,
            "stage1_active_perception_enabled": True,
            "stage1_active_perception_scope": "targeted",
            "stage1_active_perception_max_loops": 1,
            "stage1_active_perception_fail_closed": False,
            "stage1_probe_frames_override": 11,
            "stage1_probe_resolution_scale": 0.6,
            "stage1_probe_min_viable_pose_ratio": 0.45,
            "stage1_probe_min_unique_positions": 6,
            "stage1_probe_dedupe_enabled": False,
            "stage1_probe_dedupe_max_regen_attempts": 4,
            "stage1_probe_dedupe_center_dist_m": 0.12,
            "stage1_probe_tiebreak_extra_votes": 1,
            "stage1_probe_tiebreak_spread_threshold": 2.5,
            "stage1_vlm_min_task_score": 7.5,
            "stage1_vlm_min_visual_score": 7.2,
            "stage1_vlm_min_spatial_score": 6.5,
            "stage1_keep_probe_videos": True,
            "orientation_autocorrect_enabled": True,
            "orientation_autocorrect_mode": "warn_only",
            "manipulation_random_xy_offset_m": 0.0,
            "non_manipulation_random_xy_offset_m": 0.6,
            "manipulation_target_z_bias_m": -0.05,
        },
        "robot_composite": {"enabled": True, "urdf_path": "/tmp/arm.urdf"},
        "gemini_polish": {"enabled": True, "model": "gemini-3.1-flash-image-preview"},
        "enrich": {
            "cosmos_model": "test-model",
            "num_variants_per_render": 3,
            "context_frame_index": 7,
            "max_input_frames": 17,
            "min_source_clips": 2,
            "min_valid_outputs": 2,
            "max_blur_reject_rate": 0.4,
            "green_frame_ratio_max": 0.2,
            "enable_visual_collapse_gate": False,
            "vlm_quality_gate_enabled": False,
            "vlm_quality_fail_closed": False,
            "vlm_quality_autoretry_enabled": False,
            "vlm_quality_max_regen_attempts": 1,
            "vlm_quality_min_task_score": 6.5,
            "vlm_quality_min_visual_score": 6.0,
            "vlm_quality_min_spatial_score": 5.5,
            "vlm_quality_require_reasoning_consistency": False,
            "vlm_quality_retry_context_frame_stride": 4,
            "vlm_quality_disable_depth_on_final_retry": False,
            "min_frame0_ssim": 0.8,
            "delete_rejected_outputs": True,
            "context_frame_mode": "fixed",
            "max_source_clips": 1,
            "source_clip_selection_mode": "task_targeted",
            "source_clip_selection_fail_closed": False,
            "source_clip_task": "Pick up trash_can_157 and place it in the target zone",
            "source_clip_name": "clip_001_manipulation",
            "multi_view_context_enabled": True,
            "multi_view_context_offsets": [-9, 0, 9],
            "scene_index_enabled": True,
            "scene_index_k": 4,
            "scene_index_sample_every_n_frames": 5,
            "cosmos_output_quality": 8,
        },
        "finetune": {
            "num_epochs": 10,
            "lora_rank": 16,
            "video_dataset_backend": "vendor",
            "probe_dataloader_sample": False,
        },
        "eval_policy": {
            "num_rollouts": 10,
            "mode": "research",
            "tasks": ["go forward"],
            "headline_scope": "dual",
            "rollout_driver": "both",
            "scripted_rollouts_per_task": 9,
            "manip_eval_mode": "raw",
            "min_assignment_quality_score": 0.25,
            "require_object_grounded_manip_tasks": False,
            "min_absolute_difference": 1.25,
            "min_manip_success_delta_pp": 20,
            "reliability": {
                "enforce_stage_success": True,
                "max_scoring_failure_rate": 0.01,
                "fail_on_short_rollout": True,
                "min_rollout_frames": 17,
                "min_rollout_steps": 11,
            },
            "vlm_judge": {
                "model": "gemini-3-flash",
                "fallback_models": ["gemini-2.5-flash", "gemini-1.5-flash"],
                "enable_agentic_vision": True,
                "video_metadata_fps": 12.0,
            },
        },
        "policy_finetune": {
            "enabled": True,
            "openvla_repo": "/tmp/openvla-oft",
            "data_root_dir": "/tmp/data",
            "dataset_name": "bridge_orig",
            "max_steps": 100,
        },
        "policy_adapter": {"name": "openvla_oft"},
        "policy_rl_loop": {
            "enabled": True,
            "policy_refine_near_miss_fraction": 0.25,
            "policy_refine_hard_negative_fraction": 0.05,
            "world_model_refresh_mix_with_stage2": True,
            "world_model_refresh_require_stage2_vlm_pass": False,
            "world_model_refresh_stage2_fraction": 0.5,
            "world_model_refresh_success_fraction": 0.3,
            "world_model_refresh_near_miss_fraction": 0.2,
            "world_model_refresh_min_total_clips": 64,
            "world_model_refresh_max_total_clips": 256,
            "world_model_refresh_seed": 23,
        },
        "action_boost": {
            "enabled": True,
            "require_full_pipeline": True,
            "auto_switch_headline_scope_to_dual": True,
            "auto_enable_rollout_dataset": True,
            "auto_enable_policy_finetune": True,
            "auto_enable_policy_rl_loop": True,
            "compute_profile": "aggressive",
            "strict_disjoint_eval": True,
        },
        "rollout_dataset": {
            "seed": 99,
            "train_split": 0.7,
            "selection_mode": "success_near_miss_hard",
            "near_miss_min_task_score": 4.0,
            "near_miss_max_task_score": 6.5,
            "near_miss_target_fraction": 0.2,
            "hard_negative_target_fraction": 0.1,
            "per_task_max_episodes": 3,
        },
        "wm_refresh_loop": {
            "enabled": True,
            "iterations": 2,
            "source_condition": "baseline",
            "fail_if_refresh_fails": True,
            "fail_on_degenerate_mix": True,
            "min_non_hard_rollouts": 11,
            "max_hard_negative_fraction": 0.7,
            "require_valid_video_decode": False,
            "enforce_vlm_quality_floor": False,
            "min_refresh_task_score": 6.8,
            "min_refresh_visual_score": 6.4,
            "min_refresh_spatial_score": 5.9,
            "fail_on_reasoning_conflict": False,
            "backfill_from_stage2_vlm_passed": False,
            "quantile_fallback_enabled": True,
            "quantile_success_threshold": 0.90,
            "quantile_near_miss_threshold": 0.55,
        },
        "policy_compare": {
            "heldout_num_rollouts": 8,
            "heldout_tasks": ["Pick up the tote from the shelf"],
        },
        "eval_visual": {"metrics": ["psnr", "ssim"]},
        "eval_spatial": {
            "num_sample_frames": 12,
            "vlm_model": "gemini-3-flash-preview",
            "min_valid_samples": 5,
            "fail_on_reasoning_conflict": False,
        },
        "eval_crosssite": {"num_clips_per_model": 5},
    }

    config_path = tmp_path / "full.yaml"
    import yaml

    config_path.write_text(yaml.dump(config_data))

    config = load_config(config_path)
    assert len(config.facilities) == 2
    assert config.render.backend == "auto"
    assert config.render.num_frames == 10
    assert config.render.task_scoped_scene_aware is True
    assert config.render.task_scoped_max_specs == 35
    assert config.render.task_scoped_context_per_target == 1
    assert config.render.preserve_num_frames_after_collision_filter is True
    assert config.render.task_scoped_num_clips_per_path == 2
    assert config.render.task_scoped_num_frames_override == 97
    assert config.render.stage1_coverage_gate_enabled is True
    assert config.render.stage1_coverage_min_visible_frame_ratio == pytest.approx(0.4)
    assert config.render.stage1_coverage_min_approach_angle_bins == 3
    assert config.render.stage1_coverage_angle_bin_deg == pytest.approx(30.0)
    assert config.render.stage1_coverage_blur_laplacian_min == pytest.approx(25.0)
    assert config.render.stage1_coverage_blur_sample_every_n_frames == 3
    assert config.render.stage1_coverage_blur_max_samples_per_clip == 9
    assert config.render.stage1_coverage_min_center_band_ratio == pytest.approx(0.5)
    assert config.render.stage1_coverage_center_band_x == [0.25, 0.75]
    assert config.render.stage1_coverage_center_band_y == [0.3, 0.7]
    assert config.render.stage1_quality_planner_enabled is True
    assert config.render.stage1_quality_candidate_budget == "high"
    assert config.render.stage1_quality_autoretry_enabled is True
    assert config.render.stage1_quality_max_regen_attempts == 3
    assert config.render.stage1_quality_min_clip_score == pytest.approx(0.62)
    assert config.render.stage1_strict_require_task_hints is True
    assert config.render.stage1_active_perception_enabled is True
    assert config.render.stage1_active_perception_scope == "targeted"
    assert config.render.stage1_active_perception_max_loops == 1
    assert config.render.stage1_active_perception_fail_closed is False
    assert config.render.stage1_probe_frames_override == 11
    assert config.render.stage1_probe_resolution_scale == pytest.approx(0.6)
    assert config.render.stage1_probe_min_viable_pose_ratio == pytest.approx(0.45)
    assert config.render.stage1_probe_min_unique_positions == 6
    assert config.render.stage1_probe_dedupe_enabled is False
    assert config.render.stage1_probe_dedupe_max_regen_attempts == 4
    assert config.render.stage1_probe_dedupe_center_dist_m == pytest.approx(0.12)
    assert config.render.stage1_probe_tiebreak_extra_votes == 1
    assert config.render.stage1_probe_tiebreak_spread_threshold == pytest.approx(2.5)
    assert config.render.stage1_vlm_min_task_score == pytest.approx(7.5)
    assert config.render.stage1_vlm_min_visual_score == pytest.approx(7.2)
    assert config.render.stage1_vlm_min_spatial_score == pytest.approx(6.5)
    assert config.render.stage1_keep_probe_videos is True
    assert config.render.orientation_autocorrect_enabled is True
    assert config.render.orientation_autocorrect_mode == "warn_only"
    assert config.render.manipulation_random_xy_offset_m == pytest.approx(0.0)
    assert config.render.non_manipulation_random_xy_offset_m == pytest.approx(0.6)
    assert config.render.manipulation_target_z_bias_m == pytest.approx(-0.05)
    assert config.enrich.num_variants_per_render == 3
    assert config.enrich.context_frame_index == 7
    assert config.enrich.context_frame_mode == "fixed"
    assert config.enrich.max_input_frames == 17
    assert config.enrich.max_source_clips == 1
    assert config.enrich.min_source_clips == 2
    assert config.enrich.min_valid_outputs == 2
    assert config.enrich.max_blur_reject_rate == pytest.approx(0.4)
    assert config.enrich.green_frame_ratio_max == pytest.approx(0.2)
    assert config.enrich.enable_visual_collapse_gate is False
    assert config.enrich.vlm_quality_gate_enabled is False
    assert config.enrich.vlm_quality_fail_closed is False
    assert config.enrich.vlm_quality_autoretry_enabled is False
    assert config.enrich.vlm_quality_max_regen_attempts == 1
    assert config.enrich.vlm_quality_min_task_score == pytest.approx(6.5)
    assert config.enrich.vlm_quality_min_visual_score == pytest.approx(6.0)
    assert config.enrich.vlm_quality_min_spatial_score == pytest.approx(5.5)
    assert config.enrich.vlm_quality_require_reasoning_consistency is False
    assert config.enrich.vlm_quality_retry_context_frame_stride == 4
    assert config.enrich.vlm_quality_disable_depth_on_final_retry is False
    assert config.enrich.source_clip_selection_mode == "task_targeted"
    assert config.enrich.source_clip_selection_fail_closed is False
    assert config.enrich.source_clip_task == "Pick up trash_can_157 and place it in the target zone"
    assert config.enrich.source_clip_name == "clip_001_manipulation"
    assert config.enrich.multi_view_context_enabled is True
    assert config.enrich.multi_view_context_offsets == [-9, 0, 9]
    assert config.enrich.scene_index_enabled is True
    assert config.enrich.scene_index_k == 4
    assert config.enrich.scene_index_sample_every_n_frames == 5
    assert config.enrich.cosmos_output_quality == 8
    assert config.enrich.min_frame0_ssim == pytest.approx(0.8)
    assert config.enrich.delete_rejected_outputs is True
    assert config.finetune.num_epochs == 10
    assert config.finetune.lora_rank == 16
    assert config.finetune.video_dataset_backend == "vendor"
    assert config.finetune.probe_dataloader_sample is False
    assert config.eval_policy.num_rollouts == 10
    assert config.eval_policy.headline_scope == "dual"
    assert config.eval_policy.rollout_driver == "both"
    assert config.eval_policy.scripted_rollouts_per_task == 9
    assert config.eval_policy.manip_eval_mode == "raw"
    assert config.eval_policy.min_assignment_quality_score == pytest.approx(0.25)
    assert config.eval_policy.require_object_grounded_manip_tasks is False
    assert config.eval_policy.reliability.enforce_stage_success is True
    assert config.eval_policy.reliability.max_scoring_failure_rate == pytest.approx(0.01)
    assert config.eval_policy.reliability.fail_on_short_rollout is True
    assert config.eval_policy.reliability.min_rollout_frames == 17
    assert config.eval_policy.reliability.min_rollout_steps == 11
    assert config.eval_policy.min_absolute_difference == pytest.approx(1.25)
    assert config.eval_policy.min_manip_success_delta_pp == pytest.approx(20.0)
    assert config.eval_policy.vlm_judge.fallback_models == [
        "gemini-2.5-flash",
        "gemini-1.5-flash",
    ]
    assert config.robot_composite.enabled is True
    assert config.gemini_polish.enabled is True
    assert config.eval_policy.vlm_judge.enable_agentic_vision is True
    assert config.eval_policy.vlm_judge.video_metadata_fps == pytest.approx(12.0)
    assert config.policy_finetune.enabled is True
    assert config.policy_finetune.max_steps == 100
    assert config.rollout_dataset.seed == 99
    assert config.rollout_dataset.selection_mode == "success_near_miss_hard"
    assert config.rollout_dataset.near_miss_min_task_score == pytest.approx(4.0)
    assert config.rollout_dataset.near_miss_max_task_score == pytest.approx(6.5)
    assert config.rollout_dataset.near_miss_target_fraction == pytest.approx(0.2)
    assert config.rollout_dataset.hard_negative_target_fraction == pytest.approx(0.1)
    assert config.rollout_dataset.per_task_max_episodes == 3
    assert config.policy_rl_loop.enabled is True
    assert config.policy_rl_loop.policy_refine_near_miss_fraction == pytest.approx(0.25)
    assert config.policy_rl_loop.policy_refine_hard_negative_fraction == pytest.approx(0.05)
    assert config.policy_rl_loop.world_model_refresh_mix_with_stage2 is True
    assert config.policy_rl_loop.world_model_refresh_require_stage2_vlm_pass is False
    assert config.policy_rl_loop.world_model_refresh_stage2_fraction == pytest.approx(0.5)
    assert config.policy_rl_loop.world_model_refresh_success_fraction == pytest.approx(0.3)
    assert config.policy_rl_loop.world_model_refresh_near_miss_fraction == pytest.approx(0.2)
    assert config.policy_rl_loop.world_model_refresh_min_total_clips == 64
    assert config.policy_rl_loop.world_model_refresh_max_total_clips == 256
    assert config.policy_rl_loop.world_model_refresh_seed == 23
    assert config.wm_refresh_loop.enabled is True
    assert config.wm_refresh_loop.iterations == 2
    assert config.wm_refresh_loop.source_condition == "baseline"
    assert config.wm_refresh_loop.fail_on_degenerate_mix is True
    assert config.wm_refresh_loop.min_non_hard_rollouts == 11
    assert config.wm_refresh_loop.max_hard_negative_fraction == pytest.approx(0.7)
    assert config.wm_refresh_loop.require_valid_video_decode is False
    assert config.wm_refresh_loop.enforce_vlm_quality_floor is False
    assert config.wm_refresh_loop.min_refresh_task_score == pytest.approx(6.8)
    assert config.wm_refresh_loop.min_refresh_visual_score == pytest.approx(6.4)
    assert config.wm_refresh_loop.min_refresh_spatial_score == pytest.approx(5.9)
    assert config.wm_refresh_loop.fail_on_reasoning_conflict is False
    assert config.wm_refresh_loop.backfill_from_stage2_vlm_passed is False
    assert config.wm_refresh_loop.quantile_fallback_enabled is True
    assert config.wm_refresh_loop.quantile_success_threshold == pytest.approx(0.90)
    assert config.wm_refresh_loop.quantile_near_miss_threshold == pytest.approx(0.55)
    assert config.action_boost.enabled is True
    assert config.action_boost.compute_profile == "aggressive"
    assert config.policy_compare.heldout_num_rollouts == 8
    assert config.policy_compare.heldout_tasks == ["Pick up the tote from the shelf"]
    assert str(config.facilities["a"].task_hints_path) == "/tmp/a_tasks.json"
    assert config.facilities["a"].video_orientation_fix == "rotate180"
    assert "psnr" in config.eval_visual.metrics
    assert config.eval_spatial.min_valid_samples == 5
    assert config.eval_spatial.fail_on_reasoning_conflict is False


def test_legacy_robosplat_scan_maps_to_robosplat(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_data = {
        "schema_version": "v1",
        "project_name": "Legacy RoboSplat",
        "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
        "robosplat_scan": {
            "enabled": True,
            "num_augmented_clips_per_input": 3,
        },
    }
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(yaml.dump(config_data))
    config = load_config(config_path)
    assert config.robosplat.enabled is True
    assert config.robosplat.backend == "legacy_scan"
    assert config.robosplat.variants_per_input == 3


def test_config_resolves_relative_paths(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    rel_ply = "./data/facilities/a/splat.ply"
    rel_checkpoint = "./data/checkpoints/openvla-7b"

    config_data = {
        "project_name": "Path Resolution",
        "facilities": {
            "a": {
                "name": "A",
                "ply_path": rel_ply,
                "task_hints_path": "./runs/latest/task_targets.json",
            },
        },
        "enrich": {"cosmos_checkpoint": "./data/checkpoints/cosmos"},
        "finetune": {"dreamdojo_repo": "./vendor/DreamDojo"},
        "eval_policy": {"openvla_checkpoint": rel_checkpoint},
        "policy_finetune": {
            "openvla_repo": "./vendor/openvla-oft",
            "data_root_dir": "./data/openvla",
        },
    }
    config_path = cfg_dir / "validation.yaml"
    config_path.write_text(yaml.dump(config_data))

    config = load_config(config_path)
    assert config.facilities["a"].ply_path.is_absolute()
    assert str(config.facilities["a"].ply_path).endswith("cfg/data/facilities/a/splat.ply")
    assert config.facilities["a"].task_hints_path is not None
    assert config.facilities["a"].task_hints_path.is_absolute()
    assert str(config.facilities["a"].task_hints_path).endswith("cfg/runs/latest/task_targets.json")
    assert config.enrich.cosmos_checkpoint.is_absolute()
    assert config.finetune.dreamdojo_repo.is_absolute()
    assert config.eval_policy.openvla_checkpoint.is_absolute()
    assert config.eval_policy.checkpoint_path.is_absolute()
    assert config.policy_finetune.openvla_repo.is_absolute()
    assert config.policy_finetune.data_root_dir.is_absolute()


def test_eval_policy_legacy_aliases_map_to_generic_fields(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_data = {
        "project_name": "Legacy Eval Alias",
        "facilities": {
            "a": {"name": "A", "ply_path": "/tmp/a.ply"},
        },
        "eval_policy": {
            "openvla_model": "org/custom-openvla",
            "openvla_checkpoint": "./checkpoints/custom",
        },
    }
    config_path = tmp_path / "legacy_eval.yaml"
    config_path.write_text(yaml.dump(config_data))

    config = load_config(config_path)
    assert config.eval_policy.model_name == "org/custom-openvla"
    assert config.eval_policy.openvla_model == "org/custom-openvla"
    assert str(config.eval_policy.checkpoint_path).endswith("checkpoints/custom")


def test_same_facility_claim_config_uses_repo_output_root():
    from blueprint_validation.config import load_config

    config = load_config(Path("configs/same_facility_policy_uplift_openvla.yaml"))
    assert str(config.rollout_dataset.export_dir).endswith("data/outputs/policy_datasets")
    assert "/configs/data/outputs/" not in str(config.rollout_dataset.export_dir)


def test_policy_adapter_pi05_block_parses(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_data = {
        "project_name": "pi05 Config",
        "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
        "policy_adapter": {
            "name": "pi05",
            "pi05": {
                "openpi_repo": "./vendor/openpi",
                "profile": "pi05_droid",
                "runtime_mode": "inprocess",
                "train_backend": "pytorch",
                "train_script": "scripts/train_pytorch.py",
                "norm_stats_script": "scripts/compute_norm_stats.py",
                "policy_action_dim": 14,
                "policy_state_dim": 16,
            },
        },
    }
    config_path = tmp_path / "pi05.yaml"
    config_path.write_text(yaml.dump(config_data))

    config = load_config(config_path)
    assert config.policy_adapter.name == "pi05"
    assert config.policy_adapter.pi05.profile == "pi05_droid"
    assert config.policy_adapter.pi05.runtime_mode == "inprocess"
    assert config.policy_adapter.pi05.train_backend == "pytorch"
    assert config.policy_adapter.pi05.policy_action_dim == 14
    assert config.policy_adapter.pi05.policy_state_dim == 16
    assert config.policy_adapter.pi05.openpi_repo.is_absolute()


def test_config_parses_scene_orientation_fields(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "scene_orientation.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Scene Orientation",
                "facilities": {
                    "a": {
                        "name": "A",
                        "ply_path": "/tmp/a.ply",
                        "up_axis": "y",
                        "scene_rotation_deg": [10, 20, 30],
                    },
                },
            }
        )
    )

    config = load_config(config_path)
    fac = config.facilities["a"]
    assert fac.up_axis == "y"
    assert fac.scene_rotation_deg == [10.0, 20.0, 30.0]


def test_config_rejects_invalid_scene_rotation_length(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_scene_orientation.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Scene Orientation",
                "facilities": {
                    "a": {
                        "name": "A",
                        "ply_path": "/tmp/a.ply",
                        "scene_rotation_deg": [0, 90],
                    },
                },
            }
        )
    )

    with pytest.raises(ValueError, match="scene_rotation_deg"):
        load_config(config_path)


def test_config_rejects_invalid_action_boost_compute_profile(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_action_boost.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Action Boost",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "action_boost": {"compute_profile": "ultra"},
            }
        )
    )

    with pytest.raises(ValueError, match="action_boost.compute_profile"):
        load_config(config_path)


def test_config_rejects_invalid_rollout_selection_mode(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_selection_mode.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Selection Mode",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "rollout_dataset": {"selection_mode": "all"},
            }
        )
    )

    with pytest.raises(ValueError, match="rollout_dataset.selection_mode"):
        load_config(config_path)


def test_config_rejects_invalid_manip_eval_mode(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_manip_eval_mode.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Manip Eval Mode",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "eval_policy": {"manip_eval_mode": "overlay_magic"},
            }
        )
    )

    with pytest.raises(ValueError, match="eval_policy.manip_eval_mode"):
        load_config(config_path)


def test_config_rejects_invalid_scoring_failure_rate(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_scoring_failure_rate.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Reliability",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "eval_policy": {"reliability": {"max_scoring_failure_rate": 1.5}},
            }
        )
    )

    with pytest.raises(ValueError, match="max_scoring_failure_rate"):
        load_config(config_path)


def test_config_rejects_invalid_wm_refresh_quantiles(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_wm_refresh_quantiles.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad WM Refresh Quantiles",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "wm_refresh_loop": {
                    "quantile_success_threshold": 0.4,
                    "quantile_near_miss_threshold": 0.6,
                },
            }
        )
    )

    with pytest.raises(ValueError, match="quantile_success_threshold"):
        load_config(config_path)


def test_config_rejects_invalid_spatial_min_valid_samples(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_spatial_min_valid_samples.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Spatial Config",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "eval_spatial": {"min_valid_samples": 0},
            }
        )
    )

    with pytest.raises(ValueError, match="eval_spatial.min_valid_samples"):
        load_config(config_path)


def test_config_rejects_invalid_min_rollout_frames(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_min_rollout_frames.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Min Rollout Frames",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "eval_policy": {"reliability": {"min_rollout_frames": 1}},
            }
        )
    )

    with pytest.raises(ValueError, match="min_rollout_frames"):
        load_config(config_path)


def test_config_rejects_short_rollout_horizon_when_guard_enabled(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_short_rollout_guard.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Short Rollout Guard",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "eval_policy": {
                    "mode": "research",
                    "max_steps_per_rollout": 4,
                    "reliability": {
                        "fail_on_short_rollout": True,
                        "min_rollout_frames": 13,
                    },
                },
            }
        )
    )

    with pytest.raises(ValueError, match="max_steps_per_rollout"):
        load_config(config_path)


def test_config_rejects_invalid_enrich_blur_reject_rate(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_blur_reject_rate.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Blur Reject Rate",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "enrich": {"max_blur_reject_rate": 1.2},
            }
        )
    )

    with pytest.raises(ValueError, match="max_blur_reject_rate"):
        load_config(config_path)


def test_config_rejects_invalid_enrich_context_frame_mode(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_context_frame_mode.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Context Frame Mode",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "enrich": {"context_frame_mode": "auto_center"},
            }
        )
    )

    with pytest.raises(ValueError, match="enrich.context_frame_mode"):
        load_config(config_path)


def test_config_rejects_fixed_context_mode_without_index(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "fixed_context_missing_index.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Fixed Context",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "enrich": {"context_frame_mode": "fixed"},
            }
        )
    )

    with pytest.raises(ValueError, match="context_frame_index must be set"):
        load_config(config_path)


def test_config_rejects_invalid_enrich_min_frame0_ssim(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_min_frame0_ssim.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Min Frame0 SSIM",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "enrich": {"min_frame0_ssim": 1.2},
            }
        )
    )

    with pytest.raises(ValueError, match="enrich.min_frame0_ssim"):
        load_config(config_path)


def test_config_rejects_invalid_enrich_max_input_frames(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_max_input_frames.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Max Input Frames",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "enrich": {"max_input_frames": -1},
            }
        )
    )

    with pytest.raises(ValueError, match="enrich.max_input_frames"):
        load_config(config_path)


def test_config_rejects_empty_multi_view_context_offsets(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_multi_view_offsets_empty.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Multi-View Offsets",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "enrich": {"multi_view_context_offsets": []},
            }
        )
    )

    with pytest.raises(ValueError, match="multi_view_context_offsets must contain at least one"):
        load_config(config_path)


def test_config_rejects_multi_view_offsets_without_zero_anchor(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_multi_view_offsets_no_zero.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Multi-View Anchor",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "enrich": {
                    "multi_view_context_enabled": True,
                    "multi_view_context_offsets": [-8, 8],
                },
            }
        )
    )

    with pytest.raises(ValueError, match="must include 0"):
        load_config(config_path)


def test_config_rejects_invalid_enrich_scene_index_k(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_scene_index_k.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Scene Index K",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "enrich": {"scene_index_k": -1},
            }
        )
    )

    with pytest.raises(ValueError, match="enrich.scene_index_k"):
        load_config(config_path)


def test_config_rejects_invalid_enrich_scene_index_sample_every_n_frames(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_scene_index_stride.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Scene Index Stride",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "enrich": {"scene_index_sample_every_n_frames": 0},
            }
        )
    )

    with pytest.raises(ValueError, match="enrich.scene_index_sample_every_n_frames"):
        load_config(config_path)


def test_config_rejects_invalid_min_rollout_steps(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_min_rollout_steps.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Min Rollout Steps",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "eval_policy": {"reliability": {"min_rollout_steps": 0}},
            }
        )
    )

    with pytest.raises(ValueError, match="min_rollout_steps"):
        load_config(config_path)


def test_config_rejects_invalid_enrich_vlm_retry_stride(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_enrich_vlm_retry_stride.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Enrich Retry Stride",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "enrich": {"vlm_quality_retry_context_frame_stride": 0},
            }
        )
    )

    with pytest.raises(ValueError, match="vlm_quality_retry_context_frame_stride"):
        load_config(config_path)


def test_config_rejects_invalid_enrich_vlm_min_score(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_enrich_vlm_min_score.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Enrich VLM Min Score",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "enrich": {"vlm_quality_min_visual_score": 11.0},
            }
        )
    )

    with pytest.raises(ValueError, match="vlm_quality_min_visual_score"):
        load_config(config_path)


def test_config_rejects_invalid_stage1_quality_candidate_budget(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_stage1_budget.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Stage1 Budget",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "render": {"stage1_quality_candidate_budget": "ultra"},
            }
        )
    )

    with pytest.raises(ValueError, match="stage1_quality_candidate_budget"):
        load_config(config_path)


def test_config_rejects_invalid_stage1_quality_min_clip_score(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_stage1_quality_min_score.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Stage1 Min Score",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "render": {"stage1_quality_min_clip_score": 1.2},
            }
        )
    )

    with pytest.raises(ValueError, match="stage1_quality_min_clip_score"):
        load_config(config_path)


def test_config_rejects_invalid_stage1_active_perception_scope(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_stage1_active_scope.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Stage1 Active Scope",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "render": {"stage1_active_perception_scope": "everything"},
            }
        )
    )

    with pytest.raises(ValueError, match="stage1_active_perception_scope"):
        load_config(config_path)


def test_config_rejects_invalid_scene_locked_profile(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_scene_locked_profile.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Scene Locked Profile",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "render": {"scene_locked_profile": "always"},
            }
        )
    )

    with pytest.raises(ValueError, match="scene_locked_profile"):
        load_config(config_path)


def test_config_rejects_invalid_vlm_video_metadata_fps(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_vlm_video_fps.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad VLM Video FPS",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "eval_policy": {"vlm_judge": {"video_metadata_fps": 30.0}},
            }
        )
    )

    with pytest.raises(ValueError, match="video_metadata_fps"):
        load_config(config_path)


def test_config_rejects_short_rollout_step_horizon_when_guard_enabled(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_short_rollout_step_guard.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Short Rollout Step Guard",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "eval_policy": {
                    "mode": "research",
                    "max_steps_per_rollout": 4,
                    "reliability": {
                        "fail_on_short_rollout": True,
                        "min_rollout_frames": 5,
                        "min_rollout_steps": 8,
                    },
                },
            }
        )
    )

    with pytest.raises(ValueError, match="min_rollout_steps"):
        load_config(config_path)


def test_config_rejects_invalid_wm_refresh_hard_negative_fraction(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_wm_refresh_hard_negative_fraction.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad WM Refresh Hard Negative Fraction",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "wm_refresh_loop": {"max_hard_negative_fraction": 1.5},
            }
        )
    )

    with pytest.raises(ValueError, match="max_hard_negative_fraction"):
        load_config(config_path)


def test_config_rejects_invalid_wm_refresh_vlm_floor_score(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_wm_refresh_vlm_floor_score.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad WM Refresh VLM Floor Score",
                "facilities": {"a": {"name": "A", "ply_path": "/tmp/a.ply"}},
                "wm_refresh_loop": {"min_refresh_spatial_score": 12.0},
            }
        )
    )

    with pytest.raises(ValueError, match="min_refresh_spatial_score"):
        load_config(config_path)


def test_config_parses_dreamzero_and_external_interaction(tmp_path):
    import yaml

    from blueprint_validation.config import load_config

    manifest_path = tmp_path / "external_manifest.json"
    manifest_path.write_text("{}")
    checkpoint = tmp_path / "dreamzero_ckpt"
    checkpoint.mkdir(parents=True, exist_ok=True)
    repo = tmp_path / "dreamzero_repo"
    repo.mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "dreamzero_external.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "DreamZero External Test",
                "facilities": {"a": {"name": "A", "ply_path": str(tmp_path / "a.ply")}},
                "policy_adapter": {
                    "name": "dreamzero",
                    "dreamzero": {
                        "repo_path": str(repo),
                        "base_model_name": "dreamzero/base",
                        "checkpoint_path": str(checkpoint),
                        "inference_module": "dreamzero.inference",
                        "inference_class": "DreamZeroInference",
                        "policy_action_dim": 7,
                        "frame_history": 4,
                        "allow_training": False,
                    },
                },
                "external_interaction": {
                    "enabled": True,
                    "manifest_path": str(manifest_path),
                    "source_name": "polaris",
                },
            }
        )
    )

    cfg = load_config(config_path)
    assert cfg.policy_adapter.name == "dreamzero"
    assert cfg.policy_adapter.dreamzero.repo_path == repo.resolve()
    assert cfg.policy_adapter.dreamzero.base_model_name == "dreamzero/base"
    assert cfg.policy_adapter.dreamzero.checkpoint_path == checkpoint.resolve()
    assert cfg.policy_adapter.dreamzero.strict_action_min == -1.0
    assert cfg.policy_adapter.dreamzero.strict_action_max == 1.0
    assert cfg.external_interaction.enabled is True
    assert cfg.external_interaction.manifest_path == manifest_path.resolve()
    assert cfg.external_interaction.source_name == "polaris"


def test_config_parses_openvla_adapter_owned_base_reference(tmp_path):
    import yaml

    from blueprint_validation.config import load_config

    checkpoint = tmp_path / "openvla_ckpt"
    checkpoint.mkdir(parents=True, exist_ok=True)
    repo = tmp_path / "openvla_repo"
    repo.mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "openvla_adapter_base.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "OpenVLA Adapter Ref Test",
                "facilities": {"a": {"name": "A", "ply_path": str(tmp_path / "a.ply")}},
                "policy_adapter": {
                    "name": "openvla_oft",
                    "openvla": {
                        "openvla_repo": str(repo),
                        "base_model_name": "openvla/custom-7b",
                        "base_checkpoint_path": str(checkpoint),
                    },
                },
            }
        )
    )

    cfg = load_config(config_path)
    assert cfg.policy_adapter.openvla.base_model_name == "openvla/custom-7b"
    assert cfg.policy_adapter.openvla.base_checkpoint_path == checkpoint.resolve()


def test_config_rejects_invalid_dreamzero_action_bounds(tmp_path):
    import yaml

    from blueprint_validation.config import load_config

    config_path = tmp_path / "bad_dreamzero_bounds.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad DreamZero Bounds",
                "facilities": {"a": {"name": "A", "ply_path": str(tmp_path / "a.ply")}},
                "policy_adapter": {
                    "name": "dreamzero",
                    "dreamzero": {
                        "repo_path": str(tmp_path / "repo"),
                        "checkpoint_path": str(tmp_path / "ckpt"),
                        "strict_action_min": 1.0,
                        "strict_action_max": 1.0,
                    },
                },
            }
        )
    )

    with pytest.raises(ValueError, match="strict_action_min"):
        load_config(config_path)


def test_config_rejects_fixed_claim_protocol_without_disjoint_split_strategy(tmp_path):
    import yaml

    from blueprint_validation.config import load_config

    config_path = tmp_path / "bad_claim_split.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Claim Split",
                "facilities": {
                    "a": {
                        "name": "A",
                        "ply_path": str(tmp_path / "a.ply"),
                        "claim_benchmark_path": str(tmp_path / "claim_benchmark.json"),
                    }
                },
                "eval_policy": {
                    "claim_protocol": "fixed_same_facility_uplift",
                    "primary_endpoint": "task_success",
                    "freeze_world_snapshot": True,
                    "replication": {"training_seeds": [0, 1, 2, 3, 4, 5]},
                    "split_strategy": "legacy",
                },
                "policy_compare": {
                    "enabled": True,
                    "control_arms": ["frozen_baseline", "site_trained", "generic_control"],
                },
            }
        )
    )

    with pytest.raises(ValueError, match="split_strategy"):
        load_config(config_path)


def test_config_rejects_fixed_claim_protocol_without_required_control_arms(tmp_path):
    import yaml

    from blueprint_validation.config import load_config

    config_path = tmp_path / "bad_claim_arms.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Claim Arms",
                "facilities": {
                    "a": {
                        "name": "A",
                        "ply_path": str(tmp_path / "a.ply"),
                        "claim_benchmark_path": str(tmp_path / "claim_benchmark.json"),
                    }
                },
                "eval_policy": {
                    "claim_protocol": "fixed_same_facility_uplift",
                    "primary_endpoint": "task_success",
                    "freeze_world_snapshot": True,
                    "split_strategy": "disjoint_tasks_and_starts",
                    "replication": {"training_seeds": [0, 1, 2, 3, 4, 5]},
                },
                "policy_compare": {
                    "enabled": True,
                    "control_arms": ["frozen_baseline", "site_trained"],
                },
            }
        )
    )

    with pytest.raises(ValueError, match="generic_control"):
        load_config(config_path)


def test_config_rejects_fixed_claim_protocol_with_too_few_training_seeds(tmp_path):
    import yaml

    from blueprint_validation.config import load_config

    config_path = tmp_path / "bad_claim_seeds.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Claim Seeds",
                "facilities": {
                    "a": {
                        "name": "A",
                        "ply_path": str(tmp_path / "a.ply"),
                        "claim_benchmark_path": str(tmp_path / "claim_benchmark.json"),
                    }
                },
                "eval_policy": {
                    "claim_protocol": "fixed_same_facility_uplift",
                    "primary_endpoint": "task_success",
                    "freeze_world_snapshot": True,
                    "split_strategy": "disjoint_tasks_and_starts",
                    "replication": {"training_seeds": [0, 1, 2]},
                },
                "policy_compare": {
                    "enabled": True,
                    "control_arms": ["frozen_baseline", "site_trained", "generic_control"],
                },
            }
        )
    )

    with pytest.raises(ValueError, match="at least 6 seeds"):
        load_config(config_path)


def test_same_facility_policy_uplift_configs_load():
    from pathlib import Path

    from blueprint_validation.config import load_config

    for relpath, adapter_name in (
        ("configs/same_facility_policy_uplift_openvla.yaml", "openvla_oft"),
        ("configs/same_facility_policy_uplift_dreamzero.yaml", "dreamzero"),
    ):
        cfg = load_config(Path(relpath))
        assert len(cfg.facilities) == 1
        assert cfg.eval_policy.mode == "claim"
        assert cfg.eval_policy.headline_scope == "wm_uplift"
        assert cfg.eval_policy.claim_protocol == "fixed_same_facility_uplift"
        assert cfg.eval_policy.primary_endpoint == "task_success"
        assert cfg.eval_policy.freeze_world_snapshot is True
        assert cfg.eval_policy.split_strategy == "disjoint_tasks_and_starts"
        assert len(cfg.eval_policy.claim_replication.training_seeds) >= 6
        assert cfg.policy_finetune.enabled is True
        assert cfg.rollout_dataset.enabled is True
        assert cfg.policy_compare.enabled is True
        assert cfg.policy_adapter.name == adapter_name


def test_config_auto_enables_fixed_world_claim_dependencies(tmp_path):
    import yaml

    from blueprint_validation.config import load_config

    config_path = tmp_path / "auto_claim_defaults.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Auto Claim Defaults",
                "facilities": {
                    "a": {
                        "name": "A",
                        "ply_path": str(tmp_path / "a.ply"),
                        "claim_benchmark_path": str(tmp_path / "claim_benchmark.json"),
                    }
                },
                "eval_policy": {
                    "claim_protocol": "fixed_same_facility_uplift",
                    "primary_endpoint": "task_success",
                    "freeze_world_snapshot": True,
                    "split_strategy": "disjoint_tasks_and_starts",
                    "replication": {"training_seeds": [0, 1, 2, 3, 4, 5]},
                },
            }
        )
    )

    cfg = load_config(config_path)
    assert cfg.eval_policy.headline_scope == "wm_uplift"
    assert cfg.policy_compare.enabled is True
    assert cfg.rollout_dataset.enabled is True
    assert cfg.policy_finetune.enabled is True


def test_config_rejects_claim_strictness_with_zero_min_eval_cells(tmp_path):
    import yaml

    from blueprint_validation.config import load_config

    config_path = tmp_path / "bad_claim_strictness_zero.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Claim Strictness",
                "facilities": {
                    "a": {
                        "name": "A",
                        "ply_path": str(tmp_path / "a.ply"),
                        "claim_benchmark_path": str(tmp_path / "claim_benchmark.json"),
                    }
                },
                "eval_policy": {
                    "claim_protocol": "fixed_same_facility_uplift",
                    "primary_endpoint": "task_success",
                    "freeze_world_snapshot": True,
                    "split_strategy": "disjoint_tasks_and_starts",
                    "replication": {"training_seeds": [0, 1, 2, 3, 4, 5]},
                    "claim_strictness": {"min_common_eval_cells": 0},
                },
                "policy_compare": {
                    "enabled": True,
                    "control_arms": ["frozen_baseline", "site_trained", "generic_control"],
                },
            }
        )
    )

    with pytest.raises(ValueError, match="min_common_eval_cells"):
        load_config(config_path)


def test_config_rejects_claim_strictness_positive_seed_requirement_above_seed_count(tmp_path):
    import yaml

    from blueprint_validation.config import load_config

    config_path = tmp_path / "bad_claim_positive_seed_requirement.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Claim Positive Seed Requirement",
                "facilities": {
                    "a": {
                        "name": "A",
                        "ply_path": str(tmp_path / "a.ply"),
                        "claim_benchmark_path": str(tmp_path / "claim_benchmark.json"),
                    }
                },
                "eval_policy": {
                    "claim_protocol": "fixed_same_facility_uplift",
                    "primary_endpoint": "task_success",
                    "freeze_world_snapshot": True,
                    "split_strategy": "disjoint_tasks_and_starts",
                    "replication": {"training_seeds": [0, 1, 2, 3, 4, 5]},
                    "claim_strictness": {"min_positive_training_seeds": 7},
                },
                "policy_compare": {
                    "enabled": True,
                    "control_arms": ["frozen_baseline", "site_trained", "generic_control"],
                },
            }
        )
    )

    with pytest.raises(ValueError, match="min_positive_training_seeds"):
        load_config(config_path)


def test_config_rejects_fixed_claim_protocol_without_benchmark_manifest(tmp_path):
    import yaml

    from blueprint_validation.config import load_config

    config_path = tmp_path / "bad_claim_benchmark.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "project_name": "Bad Claim Benchmark",
                "facilities": {"a": {"name": "A", "ply_path": str(tmp_path / "a.ply")}},
                "eval_policy": {
                    "claim_protocol": "fixed_same_facility_uplift",
                    "primary_endpoint": "task_success",
                    "freeze_world_snapshot": True,
                    "split_strategy": "disjoint_tasks_and_starts",
                    "replication": {"training_seeds": [0, 1, 2, 3, 4, 5]},
                },
                "policy_compare": {
                    "enabled": True,
                    "control_arms": ["frozen_baseline", "site_trained", "generic_control"],
                },
            }
        )
    )

    with pytest.raises(ValueError, match="claim_benchmark_path"):
        load_config(config_path)


def test_load_config_accepts_qualified_opportunities_with_handoff_and_geometry_bundle(tmp_path):
    import yaml

    from blueprint_validation.config import load_config

    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir(parents=True)
    (bundle_root / "3dgs_compressed.ply").write_text("ply\n")
    (bundle_root / "labels.json").write_text("{}")
    (bundle_root / "structure.json").write_text("{}")
    (bundle_root / "task_targets.synthetic.json").write_text("{}")

    handoff_path = tmp_path / "opportunity_handoff.json"
    handoff_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "site_submission_id": "site-sub-001",
                "opportunity_id": "opp-001",
                "qualification_state": "ready",
                "downstream_evaluation_eligibility": True,
                "operator_approved_summary": "Qualified warehouse tote-pick lane",
                "scoped_task_definition": {
                    "task_id": "task-001",
                    "scoped_task_statement": "Pick tote from shelf bay 3",
                    "success_criteria": ["grasp tote", "clear shelf", "stage at handoff point"],
                    "in_scope_zone": "bay-3",
                },
                "site_constraints": {
                    "operating_constraints": ["night shift only"],
                    "privacy_security_constraints": ["no worker faces"],
                    "known_blockers": ["reflective wrap on two pallets"],
                },
                "target_robot_team": {
                    "team_name_or_id": "team-alpha",
                    "robot_platform": "franka_panda",
                    "embodiment_notes": "fixed-base arm on mobile cart",
                },
            }
        )
    )

    config_path = tmp_path / "qualified.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "schema_version": "v1",
                "project_name": "Qualified Intake",
                "qualified_opportunities": {
                    "opp_a": {
                        "opportunity_handoff_path": str(handoff_path),
                        "geometry_bundle_path": str(bundle_root),
                        "claim_benchmark_path": str(tmp_path / "claim.json"),
                    }
                },
            }
        )
    )
    (tmp_path / "claim.json").write_text('{"version": 1, "task_specs": [], "assignments": []}')

    config = load_config(config_path)

    target = config.facilities["opp_a"]
    assert config.qualified_opportunities is config.facilities
    assert target.uses_qualified_handoff is True
    assert target.opportunity_id == "opp-001"
    assert target.site_submission_id == "site-sub-001"
    assert target.ply_path == bundle_root / "3dgs_compressed.ply"
    assert target.task_hints_path == bundle_root / "task_targets.synthetic.json"
    assert target.labels_path == bundle_root / "labels.json"
    assert target.structure_path == bundle_root / "structure.json"
    assert target.qualification_state == "ready"
    assert target.downstream_evaluation_eligibility is True


def test_load_config_rejects_conflicting_facilities_and_qualified_opportunities(tmp_path):
    import yaml

    from blueprint_validation.config import load_config

    config_path = tmp_path / "conflict.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "schema_version": "v1",
                "facilities": {
                    "shared": {
                        "name": "Legacy Target",
                        "ply_path": str(tmp_path / "legacy.ply"),
                    }
                },
                "qualified_opportunities": {
                    "shared": {
                        "name": "Qualified Target",
                        "ply_path": str(tmp_path / "qualified.ply"),
                    }
                },
            }
        )
    )

    with pytest.raises(ValueError, match="defined differently"):
        load_config(config_path)


def test_load_config_prefers_explicit_ply_path_over_bundle_default(tmp_path):
    import yaml

    from blueprint_validation.config import load_config

    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir(parents=True)
    explicit_ply = tmp_path / "custom_scene.ply"
    explicit_ply.write_text("ply\n")

    handoff_path = tmp_path / "opportunity_handoff.json"
    handoff_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "site_submission_id": "site-sub-002",
                "opportunity_id": "opp-002",
                "qualification_state": "ready",
                "downstream_evaluation_eligibility": True,
                "operator_approved_summary": "Qualified scene",
                "scoped_task_definition": {
                    "task_id": "task-002",
                    "scoped_task_statement": "Inspect handoff station",
                    "success_criteria": ["capture station"],
                    "in_scope_zone": "handoff-station",
                },
                "site_constraints": {
                    "operating_constraints": ["weekends only"],
                    "privacy_security_constraints": ["badge-only area"],
                    "known_blockers": ["none"],
                },
                "target_robot_team": {
                    "team_name_or_id": "team-beta",
                    "robot_platform": "openvla",
                    "embodiment_notes": "camera-first validation",
                },
            }
        )
    )

    config_path = tmp_path / "explicit_ply.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "schema_version": "v1",
                "qualified_opportunities": {
                    "opp_b": {
                        "opportunity_handoff_path": str(handoff_path),
                        "geometry_bundle_path": str(bundle_root),
                        "ply_path": str(explicit_ply),
                    }
                }
            }
        )
    )

    config = load_config(config_path)
    assert config.facilities["opp_b"].ply_path == explicit_ply.resolve()


def test_load_config_accepts_capture_pipeline_handoff_and_infers_bundle(tmp_path):
    import yaml

    from blueprint_validation.config import load_config

    pipeline_dir = tmp_path / "scenes" / "scene_demo" / "captures" / "capture_demo" / "pipeline"
    bundle_root = pipeline_dir / "advanced_geometry"
    bundle_root.mkdir(parents=True)
    (bundle_root / "3dgs_compressed.ply").write_text("ply\n")
    (bundle_root / "labels.json").write_text("{}")
    (bundle_root / "structure.json").write_text("{}")
    (bundle_root / "task_targets.synthetic.json").write_text("{}")

    handoff_path = pipeline_dir / "opportunity_handoff.json"
    handoff_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene_demo",
                "capture_id": "capture_demo",
                "readiness_state": "ready",
                "match_ready": True,
                "summary": "Capture pipeline handoff",
            }
        )
    )

    config_path = tmp_path / "capture_pipeline.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "schema_version": "v1",
                "qualified_opportunities": {
                    "opp_capture": {
                        "opportunity_handoff_path": str(handoff_path),
                    }
                },
            }
        )
    )

    config = load_config(config_path)
    target = config.facilities["opp_capture"]

    assert target.intake_mode == "qualified_opportunity"
    assert target.opportunity_id == "scene_demo"
    assert target.site_submission_id == "capture_demo"
    assert target.geometry_bundle_path == bundle_root.resolve()
    assert target.ply_path == bundle_root.resolve() / "3dgs_compressed.ply"
    assert target.task_hints_path == bundle_root.resolve() / "task_targets.synthetic.json"
    assert target.labels_path == bundle_root.resolve() / "labels.json"
    assert target.structure_path == bundle_root.resolve() / "structure.json"


def test_load_config_leaves_task_hints_unset_when_bundle_has_only_labels_and_structure(tmp_path):
    import yaml

    from blueprint_validation.config import load_config

    pipeline_dir = tmp_path / "pipeline"
    bundle_root = pipeline_dir / "advanced_geometry"
    bundle_root.mkdir(parents=True)
    (bundle_root / "3dgs_compressed.ply").write_text("ply\n")
    (bundle_root / "labels.json").write_text("{}")
    (bundle_root / "structure.json").write_text("{}")

    handoff_path = pipeline_dir / "opportunity_handoff.json"
    handoff_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene_demo",
                "capture_id": "capture_demo",
                "readiness_state": "ready",
                "match_ready": True,
            }
        )
    )

    config_path = tmp_path / "missing_synthetic.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "schema_version": "v1",
                "qualified_opportunities": {
                    "opp_capture": {
                        "opportunity_handoff_path": str(handoff_path),
                    }
                },
            }
        )
    )

    config = load_config(config_path)
    target = config.facilities["opp_capture"]

    assert target.geometry_bundle_path == bundle_root.resolve()
    assert target.ply_path == bundle_root.resolve() / "3dgs_compressed.ply"
    assert target.labels_path == bundle_root.resolve() / "labels.json"
    assert target.structure_path == bundle_root.resolve() / "structure.json"
    assert target.task_hints_path is None
