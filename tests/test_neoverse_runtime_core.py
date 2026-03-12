from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from blueprint_validation.neoverse_runtime_core import NeoVerseRuntimeStore


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_runtime_store_builds_and_steps_site_world(tmp_path: Path) -> None:
    raw_dir = tmp_path / "capture" / "raw" / "arkit"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "poses.jsonl").write_text("{}\n", encoding="utf-8")
    (raw_dir / "intrinsics.json").write_text("{}", encoding="utf-8")
    walkthrough = tmp_path / "capture" / "raw" / "walkthrough.mov"
    frame_path = tmp_path / "capture" / "raw" / "keyframe.png"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frame_path), np.full((48, 64, 3), 96, dtype=np.uint8))
    walkthrough.write_bytes(b"mov")

    spec = {
        "schema_version": "v1",
        "scene_id": "scene-a",
        "capture_id": "cap-a",
        "site_submission_id": "site-sub-1",
        "qualification_state": "ready",
        "downstream_evaluation_eligibility": True,
        "capture_source": "iphone",
        "processing_profile": "pose_assisted",
        "conditioning": {
            "sensor_availability": {
                "arkit_poses": True,
                "arkit_intrinsics": True,
            },
            "local_paths": {
                "raw_video_path": str(walkthrough),
                "keyframe_path": str(frame_path),
                "arkit_poses_path": str(raw_dir / "poses.jsonl"),
                "arkit_intrinsics_path": str(raw_dir / "intrinsics.json"),
                "object_index_path": str(tmp_path / "capture" / "raw" / "object_index.json"),
            },
        },
        "task_catalog": [{"id": "task-1", "task_id": "task-1", "task_text": "Open the fridge"}],
        "scenario_catalog": [{"id": "scenario-default", "name": "default"}],
        "start_state_catalog": [{"id": "start-default", "name": "default_start_state"}],
        "robot_profiles": [
            {
                "id": "mobile_manipulator_rgb_v1",
                "observation_cameras": [
                    {"id": "head_rgb", "role": "head", "required": True},
                    {"id": "wrist_rgb", "role": "wrist", "required": False},
                ],
            }
        ],
    }

    store = NeoVerseRuntimeStore(tmp_path / "runtime", base_url="http://runtime.local")
    registration = store.build_site_world(spec)
    assert registration["status"] == "ready"

    session = store.create_session(
        str(registration["site_world_id"]),
        session_id="session-1",
        robot_profile_id="mobile_manipulator_rgb_v1",
        task_id="task-1",
        scenario_id="scenario-default",
        start_state_id="start-default",
    )
    assert session["session_id"] == "session-1"

    reset = store.reset_session("session-1")
    assert reset["episode"]["stepIndex"] == 0
    assert reset["episode"]["observation"]["frame_path"].startswith("http://runtime.local")

    step = store.step_session("session-1", action=[0.2, 0.0, 0.1, 0, 0, 0, 1])
    assert step["episode"]["stepIndex"] == 1
    assert step["episode"]["status"] == "running"

    render = store.render_bytes("session-1", "head_rgb")
    assert render.startswith(b"\x89PNG")
