from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_validation.common import write_json


def _write_scene_package(root: Path) -> None:
    (root / "assets").mkdir(parents=True, exist_ok=True)
    (root / "usd").mkdir(parents=True, exist_ok=True)
    write_json({"scene_id": "scene_a"}, root / "assets" / "scene_manifest.json")
    (root / "usd" / "scene.usda").write_text("#usda 1.0\n")
    (root / "isaac_lab").mkdir(parents=True, exist_ok=True)


def test_keyboard_command_to_action_maps_expected_axes() -> None:
    from blueprint_validation.teleop.runtime import keyboard_command_to_action

    action = keyboard_command_to_action(
        "wdg",
        action_dim=7,
        translation_step_m=0.02,
        rotation_step_rad=0.12,
        gripper_step=1.0,
    )
    assert action.tolist() == pytest.approx([0.02, 0.02, 0.0, 0.0, 0.0, 0.0, 1.0])


def test_record_teleop_session_with_fake_backend(tmp_path: Path) -> None:
    from blueprint_validation.teleop.runtime import (
        RecordedSession,
        TeleopRecorderConfig,
        record_teleop_session,
    )

    scene_root = tmp_path / "scene"
    _write_scene_package(scene_root)
    output_dir = tmp_path / "out"

    class FakeBackend:
        def record(self, config: TeleopRecorderConfig) -> RecordedSession:
            session_dir = config.output_dir / "pick_000"
            session_dir.mkdir(parents=True, exist_ok=True)
            video_dir = session_dir / "videos"
            calib_dir = session_dir / "calibrations"
            lerobot_root = session_dir / "lerobot"
            video_dir.mkdir()
            calib_dir.mkdir()
            lerobot_root.mkdir()
            video = video_dir / "wrist.mp4"
            video.write_bytes(b"video")
            calib = calib_dir / "wrist_calibration.json"
            write_json({"fx": 1.0}, calib)
            actions = session_dir / "actions.json"
            states = session_dir / "states.json"
            write_json([[0.0] * 7 for _ in range(3)], actions)
            write_json([{"joint_positions": [0.0] * 7} for _ in range(3)], states)
            return RecordedSession(
                session_dir=session_dir,
                video_paths={"wrist": video},
                calibration_refs={"wrist": calib},
                action_sequence_path=actions,
                state_sequence_path=states,
                lerobot_root=lerobot_root,
                episode_ref="episode_000000",
                state_keys=["joint_positions"],
                joint_names=[f"joint_{i}" for i in range(7)],
                num_steps=3,
            )

    outputs = record_teleop_session(
        TeleopRecorderConfig(
            scene_root=scene_root,
            output_dir=output_dir,
            task_id="pick",
            task_text="Pick up the mug",
            success=True,
        ),
        backend=FakeBackend(),
    )
    assert outputs["teleop_manifest_path"].exists()
    assert outputs["stage1_source_manifest_path"].exists()
    assert outputs["quality_report_path"].exists()


def test_extract_camera_frames_auto_discovers_camera_like_keys() -> None:
    from blueprint_validation.teleop.runtime import extract_camera_frames

    obs = {
        "policy": [0.1] * 7,
        "sensors": {
            "wrist_rgb": [[[255, 0, 0], [255, 0, 0]], [[255, 0, 0], [255, 0, 0]]],
            "head_camera": [[[0, 255, 0], [0, 255, 0]], [[0, 255, 0], [0, 255, 0]]],
        },
    }
    frames = extract_camera_frames(obs, requested_keys=[])
    assert "wrist" in frames
    assert "head" in frames


def test_spacemouse_state_to_action_maps_expected_axes() -> None:
    from blueprint_validation.teleop.runtime import spacemouse_state_to_action

    class State:
        x = 0.5
        y = -0.5
        z = 0.0
        roll = 0.25
        pitch = -0.25
        yaw = 0.0
        buttons = [1, 0]

    action = spacemouse_state_to_action(
        State(),
        action_dim=7,
        deadzone=0.05,
        translation_scale=0.03,
        rotation_scale=0.18,
        gripper_step=1.0,
    )
    assert action.tolist() == pytest.approx([0.015, -0.015, 0.0, 0.045, -0.045, 0.0, 1.0])


def test_vision_pro_packet_to_action_accepts_direct_action() -> None:
    from blueprint_validation.teleop.runtime import vision_pro_packet_to_action

    action = vision_pro_packet_to_action(
        {"action": [0.1, 0.0, -0.1, 0.2, 0.0, -0.2, 1.0]},
        action_dim=7,
    )
    assert action.tolist() == pytest.approx([0.1, 0.0, -0.1, 0.2, 0.0, -0.2, 1.0])


def test_vision_pro_packet_to_action_accepts_pose_plus_gripper() -> None:
    from blueprint_validation.teleop.runtime import vision_pro_packet_to_action

    action = vision_pro_packet_to_action(
        {"ee_delta_pose": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06], "gripper_delta": -1.0},
        action_dim=7,
    )
    assert action.tolist() == pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, -1.0])


def test_vision_pro_packet_to_action_accepts_pose_like_packet() -> None:
    from blueprint_validation.teleop.vision_pro_relay import normalize_vision_pro_packet

    packet = normalize_vision_pro_packet(
        {
            "right_hand": {
                "translation": [0.01, -0.02, 0.03],
                "rotation_rpy": [0.1, 0.0, -0.1],
            },
            "gestures": {"pinch": True},
        },
        action_dim=7,
        translation_scale=1.0,
        rotation_scale=1.0,
        gripper_open_value=1.0,
        gripper_close_value=-1.0,
    )
    assert packet["action"] == pytest.approx([0.01, -0.02, 0.03, 0.1, 0.0, -0.1, -1.0])


def test_normalize_vision_pro_packet_accepts_done_packet() -> None:
    from blueprint_validation.teleop.vision_pro_relay import normalize_vision_pro_packet

    packet = normalize_vision_pro_packet(
        {"done": True},
        action_dim=7,
        translation_scale=1.0,
        rotation_scale=1.0,
        gripper_open_value=1.0,
        gripper_close_value=-1.0,
    )
    assert packet == {"done": True}


def test_record_teleop_session_retries_until_confirmed_success(tmp_path: Path) -> None:
    from blueprint_validation.teleop.runtime import (
        RecordedSession,
        TeleopRecorderConfig,
        record_teleop_session,
    )

    scene_root = tmp_path / "scene"
    _write_scene_package(scene_root)
    output_dir = tmp_path / "out"

    calls = {"count": 0}

    class FakeBackend:
        def record(self, config: TeleopRecorderConfig) -> RecordedSession:
            calls["count"] += 1
            session_dir = config.output_dir / "pick_000"
            session_dir.mkdir(parents=True, exist_ok=True)
            video_dir = session_dir / "videos"
            calib_dir = session_dir / "calibrations"
            lerobot_root = session_dir / "lerobot"
            video_dir.mkdir()
            calib_dir.mkdir()
            lerobot_root.mkdir()
            video = video_dir / "wrist.mp4"
            video.write_bytes(b"video")
            calib = calib_dir / "wrist_calibration.json"
            write_json({"fx": 1.0}, calib)
            actions = session_dir / "actions.json"
            states = session_dir / "states.json"
            write_json([[0.0] * 7 for _ in range(3)], actions)
            write_json([{"joint_positions": [0.0] * 7} for _ in range(3)], states)
            return RecordedSession(
                session_dir=session_dir,
                video_paths={"wrist": video},
                calibration_refs={"wrist": calib},
                action_sequence_path=actions,
                state_sequence_path=states,
                lerobot_root=lerobot_root,
                episode_ref="episode_000000",
                state_keys=["joint_positions"],
                joint_names=[f"joint_{i}" for i in range(7)],
                num_steps=3,
            )

    decisions = iter([False, True])

    outputs = record_teleop_session(
        TeleopRecorderConfig(
            scene_root=scene_root,
            output_dir=output_dir,
            task_id="pick",
            task_text="Pick up the mug",
            success=None,
            max_attempts=2,
            confirm_success=True,
        ),
        backend=FakeBackend(),
        confirm_fn=lambda *_args: next(decisions),
    )

    assert calls["count"] == 2
    assert outputs["teleop_manifest_path"].exists()
