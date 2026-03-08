"""Tests for Stage 1g external rollout ingest."""

from __future__ import annotations

from pathlib import Path

from blueprint_validation.common import write_json


def _write_valid_teleop_manifest(path: Path, tmp_path: Path) -> None:
    video = tmp_path / "wrist.mp4"
    video.write_bytes(b"video")
    calib = tmp_path / "wrist_calibration.json"
    write_json({"fx": 1.0}, calib)
    lerobot_root = tmp_path / "lerobot"
    lerobot_root.mkdir()
    action_path = tmp_path / "actions.json"
    state_path = tmp_path / "states.json"
    write_json([[0.0] * 7 for _ in range(5)], action_path)
    write_json([{"joint_positions": [0.0] * 7} for _ in range(5)], state_path)
    write_json(
        {
            "schema_version": "v1",
            "source_name": "teleop",
            "sessions": [
                {
                    "session_id": "scene_a::pick::000",
                    "scene_id": "scene_a",
                    "task_id": "pick",
                    "task_text": "Pick up the mug",
                    "demo_index": 0,
                    "success": True,
                    "sim_backend": "isaac_sim",
                    "teleop_device": "spacemouse",
                    "robot_type": "franka",
                    "robot_asset_ref": "robot/franka/franka.usd",
                    "action_space": "ee_delta_pose_gripper",
                    "action_dim": 7,
                    "joint_names": [f"joint_{i}" for i in range(7)],
                    "state_keys": ["joint_positions"],
                    "camera_ids": ["wrist"],
                    "video_paths": {"wrist": str(video)},
                    "calibration_refs": {"wrist": str(calib)},
                    "lerobot_root": str(lerobot_root),
                    "episode_ref": "episode_000000",
                    "start_state_hash": "abc123",
                    "action_sequence_path": str(action_path),
                    "state_sequence_path": str(state_path),
                }
            ],
        },
        path,
    )


def test_s1g_external_rollout_ingest_success(sample_config, tmp_path: Path) -> None:
    from blueprint_validation.stages.s1g_external_rollout_ingest import ExternalRolloutIngestStage

    fac = sample_config.facilities["test_facility"]
    manifest = tmp_path / "teleop_manifest.json"
    _write_valid_teleop_manifest(manifest, tmp_path)

    sample_config.external_rollouts.enabled = True
    sample_config.external_rollouts.manifest_path = manifest
    sample_config.external_rollouts.source_name = "teleop"
    sample_config.external_rollouts.mode = "wm_and_policy"

    result = ExternalRolloutIngestStage().run(sample_config, fac, tmp_path, {})
    assert result.status == "success"
    rows_path = Path(result.outputs["rollout_rows_path"])
    assert rows_path.exists()
    payload = rows_path.read_text()
    assert "external_teleop" in payload


def test_s1g_external_rollout_ingest_missing_manifest_fails(sample_config, tmp_path: Path) -> None:
    from blueprint_validation.stages.s1g_external_rollout_ingest import ExternalRolloutIngestStage

    fac = sample_config.facilities["test_facility"]
    sample_config.external_rollouts.enabled = True
    sample_config.external_rollouts.manifest_path = tmp_path / "missing.json"
    sample_config.external_rollouts.mode = "policy_only"

    result = ExternalRolloutIngestStage().run(sample_config, fac, tmp_path, {})
    assert result.status == "failed"
    assert "not found" in result.detail.lower()
