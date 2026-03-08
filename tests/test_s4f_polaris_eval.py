"""Tests for the PolaRiS evaluation stage."""

from __future__ import annotations

import json
from pathlib import Path


def _write_scene_package(root: Path) -> None:
    (root / "assets").mkdir(parents=True, exist_ok=True)
    (root / "usd").mkdir(parents=True, exist_ok=True)
    (root / "geniesim").mkdir(parents=True, exist_ok=True)
    (root / "isaac_lab" / "scene_task").mkdir(parents=True, exist_ok=True)
    (root / "assets" / "scene_manifest.json").write_text("{}")
    (root / "usd" / "scene.usda").write_text("#usda 1.0")
    (root / "geniesim" / "task_config.json").write_text(
        json.dumps(
            {
                "scene_id": "test_scene",
                "suggested_tasks": [
                    {"description_hint": "Pick up the mug and place it on the table"}
                ],
            }
        )
    )
    (root / "isaac_lab" / "scene_task" / "__init__.py").write_text("")
    (root / "isaac_lab" / "scene_task" / "blueprint_runtime.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "runtime_kind": "blueprint_scene_env",
                "task_package": "scene_task",
                "env_factory": "scene_task.create_env",
                "env_cfg_class": "TeleopEnvCfg",
                "action_dim": 7,
                "camera_keys": ["wrist_rgb"],
                "state_keys": ["policy"],
            }
        )
    )


def test_s4f_polaris_eval_stage_uses_fake_backend(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.common import StageResult
    from blueprint_validation.stages.s4f_polaris_eval import PolarisEvalStage

    monkeypatch.setenv("BLUEPRINT_POLARIS_FAKE_BACKEND", "1")
    scene_root = tmp_path / "scene_pkg"
    _write_scene_package(scene_root)
    adapted_ckpt = tmp_path / "adapted"
    adapted_ckpt.mkdir()

    sample_config.eval_polaris.enabled = True
    sample_config.eval_polaris.default_as_primary_gate = True
    sample_config.facilities["test_facility"].scene_package_path = scene_root
    previous = {
        "s3b_policy_finetune": StageResult(
            stage_name="s3b_policy_finetune",
            status="success",
            elapsed_seconds=0.0,
            outputs={"adapted_policy_checkpoint": str(adapted_ckpt)},
        )
    }

    result = PolarisEvalStage().execute(
        config=sample_config,
        facility=sample_config.facilities["test_facility"],
        work_dir=tmp_path / "facility_work",
        previous_results=previous,
    )
    assert result.status == "success"
    assert result.metrics["winner"] in {"frozen_openvla", "adapted_openvla"}
    assert Path(result.outputs["polaris_summary_path"]).exists()


def test_s4f_polaris_eval_fails_without_primary_eligible_scene(sample_config, tmp_path):
    from blueprint_validation.common import StageResult
    from blueprint_validation.stages.s4f_polaris_eval import PolarisEvalStage

    adapted_ckpt = tmp_path / "adapted"
    adapted_ckpt.mkdir()
    sample_config.eval_polaris.enabled = True
    sample_config.eval_polaris.default_as_primary_gate = True

    previous = {
        "s3b_policy_finetune": StageResult(
            stage_name="s3b_policy_finetune",
            status="success",
            elapsed_seconds=0.0,
            outputs={"adapted_policy_checkpoint": str(adapted_ckpt)},
        )
    }
    result = PolarisEvalStage().execute(
        config=sample_config,
        facility=sample_config.facilities["test_facility"],
        work_dir=tmp_path / "facility_work",
        previous_results=previous,
    )
    assert result.status == "failed"
    assert "primary-eligible" in result.detail
