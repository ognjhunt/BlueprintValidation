from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_hosted_session_cli_flow(monkeypatch, tmp_path, sample_config):
    import blueprint_validation.cli as cli_module
    import cv2
    import numpy as np

    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: v1\n", encoding="utf-8")
    monkeypatch.setattr(cli_module, "load_config", lambda _path: sample_config)
    monkeypatch.setenv("BLUEPRINT_ALLOW_MOCK_POLICY_ADAPTER", "1")

    runtime_dir = tmp_path / "runtime"
    task_anchor_path = runtime_dir / "task_anchor_manifest.json"
    task_run_path = runtime_dir / "task_run_manifest.json"
    scene_memory_manifest_path = runtime_dir / "scene_memory_manifest.json"
    conditioning_bundle_path = runtime_dir / "conditioning_bundle.json"
    conditioning_input_path = runtime_dir / "conditioning_input.png"
    conditioning_input_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(conditioning_input_path), np.full((48, 64, 3), 96, dtype=np.uint8))

    _write_json(
        task_anchor_path,
        {
            "schema_version": "v1",
            "tasks": [
                {
                    "task_id": "task-1",
                    "task_text": "Walk to shelf staging and pick the blue tote",
                }
            ],
        },
    )
    _write_json(task_run_path, {"schema_version": "v1"})
    _write_json(scene_memory_manifest_path, {"schema_version": "v1"})
    _write_json(
        conditioning_bundle_path,
        {
            "schema_version": "v1",
            "keyframe_uri": str(conditioning_input_path),
        },
    )
    runtime_manifest_path = runtime_dir / "hosted_session_runtime_manifest.json"
    _write_json(
        runtime_manifest_path,
        {
            "schema_version": "v1",
            "scene_id": "scene-1",
            "capture_id": "cap-1",
            "site_submission_id": "site-sub-1",
            "pipeline_prefix": "scenes/scene-1/captures/cap-1/pipeline",
            "scene_memory_manifest_uri": str(scene_memory_manifest_path),
            "conditioning_bundle_uri": str(conditioning_bundle_path),
            "conditioning_input_path": str(conditioning_input_path),
            "preview_simulation_manifest_uri": None,
            "task_anchor_manifest_uri": str(task_anchor_path),
            "task_run_manifest_uri": str(task_run_path),
            "available_backends": ["neoverse"],
            "default_backend": "neoverse",
            "task_ids": ["task-1"],
            "task_texts": ["Walk to shelf staging and pick the blue tote"],
            "task_catalog": [
                {
                    "id": "task-1",
                    "task_id": "task-1",
                    "task_text": "Walk to shelf staging and pick the blue tote",
                    "task_category": "pick",
                }
            ],
            "start_states": ["Dock-side tote stack"],
            "start_state_catalog": [
                {
                    "id": "start-dock",
                    "name": "Dock-side tote stack",
                    "task_id": "task-1",
                }
            ],
            "scenario_variants": ["Normal lighting"],
            "scenario_catalog": [{"id": "scenario-normal", "name": "Normal lighting"}],
            "robot_profiles": [
                {
                    "id": "mobile_manipulator_rgb_v1",
                    "display_name": "Mobile manipulator",
                    "embodiment_type": "mobile_manipulator",
                    "action_space": {"name": "ee_delta_pose_gripper", "dim": 7, "labels": []},
                    "observation_cameras": [
                        {"id": "head_rgb", "role": "head", "required": True, "default_enabled": True},
                        {"id": "wrist_rgb", "role": "wrist", "required": False, "default_enabled": True},
                    ],
                    "base_semantics": "holonomic_mobile_base",
                    "gripper_semantics": "parallel_jaw_gripper",
                    "allowed_policy_adapters": ["openvla_oft", "pi05", "dreamzero", "mock"],
                    "default_policy_adapter": "openvla_oft",
                }
            ],
            "launchable": True,
            "launch_blockers": [],
        },
    )

    work_dir = tmp_path / "session-work"
    runner = CliRunner()
    create = runner.invoke(
        cli_module.cli,
        [
            "--config",
            str(config_path),
            "session",
            "create",
            "--session-id",
            "session-1",
            "--session-work-dir",
            str(work_dir),
            "--runtime-manifest",
            str(runtime_manifest_path),
            "--robot-profile-id",
            "mobile_manipulator_rgb_v1",
            "--task-id",
            "task-1",
            "--scenario-id",
            "scenario-normal",
            "--start-state-id",
            "start-dock",
            "--policy-json",
            json.dumps({"adapter_name": "mock", "model_name": "mock-policy"}),
            "--export-mode",
            "raw_bundle",
            "--export-mode",
            "rlds_dataset",
        ],
    )
    assert create.exit_code == 0
    create_payload = json.loads(create.output)
    assert create_payload["runtime_backend_selected"] == "neoverse"

    reset = runner.invoke(
        cli_module.cli,
        [
            "--config",
            str(config_path),
            "session",
            "reset",
            "--session-id",
            "session-1",
            "--session-work-dir",
            str(work_dir),
            "--task-id",
            "task-1",
            "--scenario-id",
            "scenario-normal",
            "--start-state-id",
            "start-dock",
        ],
    )
    assert reset.exit_code == 0
    reset_payload = json.loads(reset.output)
    assert reset_payload["episode"]["episodeId"].startswith("episode-")
    assert reset_payload["episode"]["startStateId"] == "start-dock"

    step = runner.invoke(
        cli_module.cli,
        [
            "--config",
            str(config_path),
            "session",
            "step",
            "--session-id",
            "session-1",
            "--session-work-dir",
            str(work_dir),
            "--episode-id",
            reset_payload["episode"]["episodeId"],
        ],
    )
    assert step.exit_code == 0
    step_payload = json.loads(step.output)
    assert step_payload["episode"]["stepIndex"] == 1
    assert step_payload["episode"]["observation"]["cameraFrames"][0]["available"] is True

    batch = runner.invoke(
        cli_module.cli,
        [
            "--config",
            str(config_path),
            "session",
            "run-batch",
            "--session-id",
            "session-1",
            "--session-work-dir",
            str(work_dir),
            "--num-episodes",
            "3",
            "--task-id",
            "task-1",
            "--scenario-id",
            "scenario-normal",
            "--start-state-id",
            "start-dock",
        ],
    )
    assert batch.exit_code == 0
    batch_payload = json.loads(batch.output)
    assert batch_payload["summary"]["numEpisodes"] == 3

    export = runner.invoke(
        cli_module.cli,
        [
            "--config",
            str(config_path),
            "session",
            "export",
            "--session-id",
            "session-1",
            "--session-work-dir",
            str(work_dir),
        ],
    )
    assert export.exit_code == 0
    export_payload = json.loads(export.output)
    assert export_payload["artifact_uris"]["export_manifest"].endswith("export_manifest.json")
    assert export_payload["artifact_uris"]["rlds_dataset"].endswith("rlds_manifest.json")
