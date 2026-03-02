"""Tests for configuration loading and validation."""

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


def test_config_defaults():
    from blueprint_validation.config import ValidationConfig

    config = ValidationConfig()
    assert config.schema_version == "v1"
    assert config.render.resolution == (480, 640)
    assert config.render.num_frames == 49
    assert config.render.camera_height_m == 1.2
    assert config.render.task_scoped_scene_aware is False
    assert config.render.task_scoped_max_specs == 40
    assert config.render.preserve_num_frames_after_collision_filter is True
    assert config.render.task_scoped_num_clips_per_path == 1
    assert config.render.task_scoped_num_frames_override == 0
    assert config.render.stage1_coverage_gate_enabled is False
    assert config.render.stage1_coverage_min_visible_frame_ratio == pytest.approx(0.35)
    assert config.render.stage1_coverage_min_approach_angle_bins == 2
    assert config.render.stage1_coverage_min_center_band_ratio == pytest.approx(0.4)
    assert config.render.stage1_coverage_center_band_x == [0.2, 0.8]
    assert config.render.stage1_coverage_center_band_y == [0.2, 0.8]
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
    assert config.eval_policy.min_absolute_difference == pytest.approx(1.0)
    assert config.eval_policy.min_manip_success_delta_pp == pytest.approx(15.0)
    assert config.eval_policy.vlm_judge.enable_agentic_vision is True
    assert config.enrich.max_input_frames == 0
    assert config.enrich.max_source_clips == 0
    assert config.enrich.source_clip_selection_mode == "all"
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
    assert config.policy_finetune.enabled is True
    assert config.policy_finetune.dataset_name == "bridge_orig"
    assert config.policy_finetune.recipe == "oft"
    assert config.robosplat.enabled is True
    assert config.robosplat.backend == "auto"
    assert config.robosplat_scan.enabled is True
    assert config.splatsim.enabled is False
    assert config.splatsim.mode == "hybrid"
    assert config.policy_rl_loop.enabled is False
    assert config.policy_rl_loop.policy_refine_near_miss_fraction == pytest.approx(0.30)
    assert config.policy_rl_loop.policy_refine_hard_negative_fraction == pytest.approx(0.10)
    assert config.policy_rl_loop.world_model_refresh_mix_with_stage2 is True
    assert config.policy_rl_loop.world_model_refresh_stage2_fraction == pytest.approx(0.60)
    assert config.policy_rl_loop.world_model_refresh_success_fraction == pytest.approx(0.25)
    assert config.policy_rl_loop.world_model_refresh_near_miss_fraction == pytest.approx(0.15)
    assert config.policy_rl_loop.world_model_refresh_min_total_clips == 128
    assert config.policy_rl_loop.world_model_refresh_max_total_clips == 512
    assert config.policy_rl_loop.world_model_refresh_seed == 17
    assert config.wm_refresh_loop.enabled is False
    assert config.wm_refresh_loop.iterations == 1
    assert config.wm_refresh_loop.source_condition == "adapted"
    assert config.wm_refresh_loop.fail_on_degenerate_mix is True
    assert config.wm_refresh_loop.min_non_hard_rollouts == 8
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
            "min_frame0_ssim": 0.8,
            "delete_rejected_outputs": True,
            "context_frame_mode": "fixed",
            "max_source_clips": 1,
            "source_clip_selection_mode": "task_targeted",
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
            },
            "vlm_judge": {
                "model": "gemini-3-flash",
                "fallback_models": ["gemini-2.5-flash", "gemini-1.5-flash"],
                "enable_agentic_vision": True,
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
        "splatsim": {
            "enabled": True,
            "mode": "strict",
            "per_zone_rollouts": 3,
            "horizon_steps": 40,
            "min_successful_rollouts_per_zone": 2,
            "fallback_to_prior_manifest": False,
        },
        "policy_rl_loop": {
            "enabled": True,
            "policy_refine_near_miss_fraction": 0.25,
            "policy_refine_hard_negative_fraction": 0.05,
            "world_model_refresh_mix_with_stage2": True,
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
    assert config.enrich.source_clip_selection_mode == "task_targeted"
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
    assert config.eval_policy.min_absolute_difference == pytest.approx(1.25)
    assert config.eval_policy.min_manip_success_delta_pp == pytest.approx(20.0)
    assert config.eval_policy.vlm_judge.fallback_models == [
        "gemini-2.5-flash",
        "gemini-1.5-flash",
    ]
    assert config.robot_composite.enabled is True
    assert config.gemini_polish.enabled is True
    assert config.eval_policy.vlm_judge.enable_agentic_vision is True
    assert config.policy_finetune.enabled is True
    assert config.policy_finetune.max_steps == 100
    assert config.splatsim.enabled is True
    assert config.splatsim.mode == "strict"
    assert config.splatsim.per_zone_rollouts == 3
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
