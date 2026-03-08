from __future__ import annotations

import json
from pathlib import Path

from blueprint_validation.common import write_json


def _write_scene_package(root: Path) -> None:
    (root / "assets").mkdir(parents=True, exist_ok=True)
    (root / "usd").mkdir(parents=True, exist_ok=True)
    write_json({"scene_id": "scene_a"}, root / "assets" / "scene_manifest.json")
    (root / "usd" / "scene.usda").write_text("#usda 1.0\n")


def test_validate_scene_package_success(tmp_path: Path) -> None:
    from blueprint_validation.teleop import load_and_validate_scene_package

    _write_scene_package(tmp_path)
    payload = load_and_validate_scene_package(tmp_path)
    assert payload["scene_manifest"]["scene_id"] == "scene_a"
    assert payload["usd_scene_path"].endswith("scene.usda")


def test_write_teleop_manifests_success(tmp_path: Path) -> None:
    from blueprint_validation.teleop import load_and_validate_teleop_manifest, write_teleop_manifests

    video = tmp_path / "wrist.mp4"
    video.write_bytes(b"video")
    calib = tmp_path / "wrist_calibration.json"
    write_json({"fx": 1.0}, calib)
    lerobot_root = tmp_path / "lerobot"
    lerobot_root.mkdir()
    action_path = tmp_path / "actions.json"
    state_path = tmp_path / "states.json"
    write_json([[0.0] * 7 for _ in range(4)], action_path)
    write_json([{"joint_positions": [0.0] * 7} for _ in range(4)], state_path)

    outputs = write_teleop_manifests(
        output_dir=tmp_path / "out",
        source_name="teleop",
        sessions=[
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
    )
    teleop_payload = load_and_validate_teleop_manifest(outputs["teleop_manifest_path"])
    assert teleop_payload["sessions"][0]["robot_type"] == "franka"
    stage1_payload = json.loads(outputs["stage1_source_manifest_path"].read_text())
    assert stage1_payload["num_clips"] == 1
    assert stage1_payload["clips"][0]["camera_id"] == "wrist"
