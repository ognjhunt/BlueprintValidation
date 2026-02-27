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


def test_config_with_all_sections(tmp_path):
    from blueprint_validation.config import load_config

    config_data = {
        "schema_version": "v1",
        "project_name": "Full Test",
        "facilities": {
            "a": {"name": "A", "ply_path": "/tmp/a.ply"},
            "b": {"name": "B", "ply_path": "/tmp/b.ply"},
        },
        "render": {"resolution": [240, 320], "num_frames": 10},
        "enrich": {"cosmos_model": "test-model", "num_variants_per_render": 3},
        "finetune": {"num_epochs": 10, "lora_rank": 16},
        "eval_policy": {
            "num_rollouts": 10,
            "tasks": ["go forward"],
            "vlm_judge": {"model": "gemini-3-flash", "enable_agentic_vision": True},
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
    assert config.eval_policy.vlm_judge.enable_agentic_vision is True
    assert "psnr" in config.eval_visual.metrics


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
            "a": {"name": "A", "ply_path": rel_ply},
        },
        "enrich": {"cosmos_checkpoint": "./data/checkpoints/cosmos"},
        "finetune": {"dreamdojo_repo": "./vendor/DreamDojo"},
        "eval_policy": {"openvla_checkpoint": rel_checkpoint},
    }
    config_path = cfg_dir / "validation.yaml"
    config_path.write_text(yaml.dump(config_data))

    config = load_config(config_path)
    assert config.facilities["a"].ply_path.is_absolute()
    assert str(config.facilities["a"].ply_path).endswith("cfg/data/facilities/a/splat.ply")
    assert config.enrich.cosmos_checkpoint.is_absolute()
    assert config.finetune.dreamdojo_repo.is_absolute()
    assert config.eval_policy.openvla_checkpoint.is_absolute()
