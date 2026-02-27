"""Tests for manipulation-related config extensions."""

import json
from pathlib import Path

import pytest


def test_manipulation_zone_config_defaults():
    from blueprint_validation.config import ManipulationZoneConfig

    zone = ManipulationZoneConfig(name="test_zone")
    assert zone.name == "test_zone"
    assert zone.camera_height_m == 0.6
    assert zone.camera_look_down_deg == 45.0
    assert zone.arc_radius_m == 0.4
    assert zone.approach_point == [0.0, 0.0, 0.0]
    assert zone.target_point == [0.0, 0.0, 0.0]


def test_facility_config_with_manipulation_zones():
    from blueprint_validation.config import FacilityConfig, ManipulationZoneConfig

    zones = [
        ManipulationZoneConfig(
            name="shelf_pick",
            approach_point=[2.1, 0.5, 0.8],
            target_point=[2.1, 1.5, 0.3],
        ),
        ManipulationZoneConfig(name="conveyor_place"),
    ]
    fac = FacilityConfig(
        name="Test",
        ply_path=Path("/tmp/test.ply"),
        manipulation_zones=zones,
    )
    assert len(fac.manipulation_zones) == 2
    assert fac.manipulation_zones[0].name == "shelf_pick"
    assert fac.manipulation_zones[0].approach_point == [2.1, 0.5, 0.8]


def test_policy_eval_config_conditions():
    from blueprint_validation.config import PolicyEvalConfig

    cfg = PolicyEvalConfig()
    assert cfg.conditions == ["baseline", "adapted"]
    assert cfg.manipulation_tasks == []


def test_policy_eval_config_custom_conditions():
    from blueprint_validation.config import PolicyEvalConfig

    cfg = PolicyEvalConfig(
        conditions=["baseline", "adapted", "trained"],
        manipulation_tasks=["Pick tote", "Place box"],
    )
    assert len(cfg.conditions) == 3
    assert "trained" in cfg.conditions
    assert len(cfg.manipulation_tasks) == 2


def test_camera_path_spec_manipulation_overrides():
    from blueprint_validation.config import CameraPathSpec

    spec = CameraPathSpec(
        type="manipulation",
        height_override_m=0.6,
        look_down_override_deg=45.0,
        approach_point=[1.0, 2.0, 0.5],
        arc_radius_m=0.3,
    )
    assert spec.type == "manipulation"
    assert spec.height_override_m == 0.6
    assert spec.look_down_override_deg == 45.0
    assert spec.approach_point == [1.0, 2.0, 0.5]
    assert spec.arc_radius_m == 0.3


def test_load_config_with_manipulation_zones(tmp_path):
    import yaml
    from blueprint_validation.config import load_config

    config_data = {
        "project_name": "Manip Test",
        "facilities": {
            "wh_a": {
                "name": "Warehouse A",
                "ply_path": str(tmp_path / "a.ply"),
                "manipulation_zones": [
                    {
                        "name": "shelf_pick",
                        "approach_point": [2.0, 0.5, 0.8],
                        "target_point": [2.0, 1.5, 0.3],
                        "camera_height_m": 0.7,
                    }
                ],
            }
        },
        "eval_policy": {
            "conditions": ["baseline", "adapted"],
            "manipulation_tasks": ["Pick the tote", "Place the box"],
        },
    }
    config_path = tmp_path / "manip.yaml"
    config_path.write_text(yaml.dump(config_data))

    config = load_config(config_path)
    assert len(config.facilities["wh_a"].manipulation_zones) == 1
    zone = config.facilities["wh_a"].manipulation_zones[0]
    assert zone.name == "shelf_pick"
    assert zone.camera_height_m == 0.7
    assert config.eval_policy.conditions == ["baseline", "adapted"]
    assert len(config.eval_policy.manipulation_tasks) == 2


def test_load_config_with_manipulation_camera_path(tmp_path):
    import yaml
    from blueprint_validation.config import load_config

    config_data = {
        "project_name": "Camera Test",
        "facilities": {
            "a": {"name": "A", "ply_path": str(tmp_path / "a.ply")},
        },
        "render": {
            "camera_paths": [
                {"type": "orbit", "radius_m": 2.0},
                {
                    "type": "manipulation",
                    "height_override_m": 0.6,
                    "look_down_override_deg": 45.0,
                },
            ]
        },
    }
    config_path = tmp_path / "cam.yaml"
    config_path.write_text(yaml.dump(config_data))

    config = load_config(config_path)
    paths = config.render.camera_paths
    assert len(paths) == 2
    assert paths[1].type == "manipulation"
    assert paths[1].height_override_m == 0.6
