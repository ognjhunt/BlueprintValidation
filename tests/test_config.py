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


def test_config_defaults():
    from blueprint_validation.config import ValidationConfig

    config = ValidationConfig()
    assert config.schema_version == "v1"
    assert config.render.resolution == (480, 640)
    assert config.render.num_frames == 49
    assert config.render.camera_height_m == 1.2
    assert config.finetune.num_epochs == 50
    assert config.finetune.use_lora is True
    assert config.finetune.lora_rank == 32
    assert config.eval_policy.vlm_judge.model == "gemini-3-flash-preview"
    assert config.eval_policy.unnorm_key == "bridge_orig"
    assert config.eval_policy.vlm_judge.enable_agentic_vision is True
    assert config.policy_adapter.name == "openvla_oft"
    assert config.rollout_dataset.enabled is True
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


def test_config_with_all_sections(tmp_path):
    from blueprint_validation.config import load_config

    config_data = {
        "schema_version": "v1",
        "project_name": "Full Test",
        "facilities": {
            "a": {"name": "A", "ply_path": "/tmp/a.ply", "task_hints_path": "/tmp/a_tasks.json"},
            "b": {"name": "B", "ply_path": "/tmp/b.ply"},
        },
        "render": {"resolution": [240, 320], "num_frames": 10},
        "robot_composite": {"enabled": True, "urdf_path": "/tmp/arm.urdf"},
        "gemini_polish": {"enabled": True, "model": "gemini-3.1-flash-image-preview"},
        "enrich": {"cosmos_model": "test-model", "num_variants_per_render": 3},
        "finetune": {"num_epochs": 10, "lora_rank": 16},
        "eval_policy": {
            "num_rollouts": 10,
            "tasks": ["go forward"],
            "vlm_judge": {"model": "gemini-3-flash", "enable_agentic_vision": True},
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
        "rollout_dataset": {"seed": 99, "train_split": 0.7},
        "policy_compare": {
            "heldout_num_rollouts": 8,
            "heldout_tasks": ["Pick up the tote from the shelf"],
        },
        "eval_visual": {"metrics": ["psnr", "ssim"]},
        "eval_crosssite": {"num_clips_per_model": 5},
    }

    config_path = tmp_path / "full.yaml"
    import yaml
    config_path.write_text(yaml.dump(config_data))

    config = load_config(config_path)
    assert len(config.facilities) == 2
    assert config.render.num_frames == 10
    assert config.enrich.num_variants_per_render == 3
    assert config.finetune.num_epochs == 10
    assert config.finetune.lora_rank == 16
    assert config.eval_policy.num_rollouts == 10
    assert config.robot_composite.enabled is True
    assert config.gemini_polish.enabled is True
    assert config.eval_policy.vlm_judge.enable_agentic_vision is True
    assert config.policy_finetune.enabled is True
    assert config.policy_finetune.max_steps == 100
    assert config.splatsim.enabled is True
    assert config.splatsim.mode == "strict"
    assert config.splatsim.per_zone_rollouts == 3
    assert config.rollout_dataset.seed == 99
    assert config.policy_compare.heldout_num_rollouts == 8
    assert config.policy_compare.heldout_tasks == ["Pick up the tote from the shelf"]
    assert str(config.facilities["a"].task_hints_path) == "/tmp/a_tasks.json"
    assert "psnr" in config.eval_visual.metrics


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
    assert config.policy_finetune.openvla_repo.is_absolute()
    assert config.policy_finetune.data_root_dir.is_absolute()


def test_config_parses_scene_orientation_fields(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "scene_orientation.yaml"
    config_path.write_text(yaml.dump({
        "project_name": "Scene Orientation",
        "facilities": {
            "a": {
                "name": "A",
                "ply_path": "/tmp/a.ply",
                "up_axis": "y",
                "scene_rotation_deg": [10, 20, 30],
            },
        },
    }))

    config = load_config(config_path)
    fac = config.facilities["a"]
    assert fac.up_axis == "y"
    assert fac.scene_rotation_deg == [10.0, 20.0, 30.0]


def test_config_rejects_invalid_scene_rotation_length(tmp_path):
    from blueprint_validation.config import load_config
    import yaml

    config_path = tmp_path / "bad_scene_orientation.yaml"
    config_path.write_text(yaml.dump({
        "project_name": "Bad Scene Orientation",
        "facilities": {
            "a": {
                "name": "A",
                "ply_path": "/tmp/a.ply",
                "scene_rotation_deg": [0, 90],
            },
        },
    }))

    with pytest.raises(ValueError, match="scene_rotation_deg"):
        load_config(config_path)
