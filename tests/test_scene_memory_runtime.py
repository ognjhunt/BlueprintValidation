from __future__ import annotations

import json

from blueprint_validation.scene_memory_runtime import resolve_scene_memory_runtime_plan
from blueprint_validation.stages.s0b_scene_memory_runtime import SceneMemoryRuntimeStage
from blueprint_validation.config import load_config


def _write_adapter_manifest(path, *, family: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "adapter_id": path.stem,
                "family": family,
                "status": "ready_for_local_experiment",
            }
        )
    )


def test_scene_memory_runtime_prefers_neoverse_then_gen3c(sample_config, tmp_path):
    facility = sample_config.facilities["test_facility"]
    scene_memory_dir = tmp_path / "scene_memory"
    adapter_dir = scene_memory_dir / "adapter_manifests"
    preview_dir = tmp_path / "preview_simulation"
    preview_dir.mkdir(parents=True, exist_ok=True)
    (preview_dir / "preview_simulation_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "supported_backends": ["gen3c", "neoverse", "cosmos_transfer"],
            }
        )
    )
    _write_adapter_manifest(adapter_dir / "gen3c.json", family="GEN3C")
    _write_adapter_manifest(adapter_dir / "neoverse.json", family="NeoVerse")
    _write_adapter_manifest(adapter_dir / "cosmos_transfer.json", family="Cosmos Transfer")

    facility.scene_memory_bundle_path = scene_memory_dir
    facility.preview_simulation_path = preview_dir
    facility.scene_memory_adapter_manifests = {
        "gen3c": adapter_dir / "gen3c.json",
        "neoverse": adapter_dir / "neoverse.json",
        "cosmos_transfer": adapter_dir / "cosmos_transfer.json",
    }

    plan = resolve_scene_memory_runtime_plan(sample_config, facility)

    assert plan["selected_backend"] == "neoverse"
    assert plan["secondary_backend"] == "gen3c"
    assert plan["fallback_backend"] == "cosmos_transfer"
    assert plan["available_backends"] == ["neoverse", "gen3c", "cosmos_transfer"]
    assert plan["default_runtime_policy"]["watchlist_only"] == ["3dsceneprompt"]


def test_scene_memory_runtime_stage_skips_watchlist_only_backends(sample_config, tmp_path):
    facility = sample_config.facilities["test_facility"]
    scene_memory_dir = tmp_path / "scene_memory"
    adapter_dir = scene_memory_dir / "adapter_manifests"
    watchlist_manifest = adapter_dir / "3dsceneprompt.json"
    _write_adapter_manifest(watchlist_manifest, family="3DScenePrompt")

    facility.scene_memory_bundle_path = scene_memory_dir
    facility.scene_memory_adapter_manifests = {
        "3dsceneprompt": watchlist_manifest,
    }

    result = SceneMemoryRuntimeStage().run(
        sample_config,
        facility,
        tmp_path / "work",
        previous_results={},
    )

    assert result.status == "skipped"
    assert result.outputs["selected_backend"] is None
    assert result.outputs["skipped_watchlist_backends"] == ["3dsceneprompt"]


def test_load_config_parses_scene_memory_runtime_block(tmp_path):
    config_path = tmp_path / "validation.yaml"
    vendor_root = tmp_path / "vendor"
    config_path.write_text(
        f"""
schema_version: v1
qualified_opportunities:
  facility_a:
    name: Facility A
    scene_memory_bundle_path: ./scene_memory
scene_memory_runtime:
  preferred_backends: [neoverse, gen3c, cosmos_transfer]
  watchlist_backends: [3dsceneprompt]
  neoverse:
    allow_runtime_execution: true
    repo_path: {vendor_root / "neoverse"}
    python_executable: /usr/bin/python3
    inference_script: scripts/inference.py
  gen3c:
    allow_runtime_execution: true
    repo_path: {vendor_root / "gen3c"}
    inference_script: launch/inference.py
"""
    )

    config = load_config(config_path)

    assert config.scene_memory_runtime.preferred_backends == [
        "neoverse",
        "gen3c",
        "cosmos_transfer",
    ]
    assert config.scene_memory_runtime.watchlist_backends == ["3dsceneprompt"]
    assert config.scene_memory_runtime.neoverse.allow_runtime_execution is True
    assert str(config.scene_memory_runtime.neoverse.repo_path).endswith("/vendor/neoverse")
    assert config.scene_memory_runtime.gen3c.inference_script == "launch/inference.py"
