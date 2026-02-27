"""Tests for URDF robot compositing utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _write_test_video(path: Path, n: int = 5):
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 32))
    for _ in range(n):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()


def test_load_urdf_chain():
    from blueprint_validation.synthetic.robot_compositor import load_urdf_chain

    urdf = Path("/Users/nijelhunt_1/workspace/BlueprintValidation/configs/robots/sample_6dof_arm.urdf")
    chain = load_urdf_chain(urdf)
    assert len(chain) >= 3
    assert chain[0].name == "joint_1"


def test_composite_robot_arm_into_clip(tmp_path):
    from blueprint_validation.synthetic.robot_compositor import composite_robot_arm_into_clip

    video = tmp_path / "in.mp4"
    _write_test_video(video)
    cam_json = tmp_path / "cam.json"
    cam_json.write_text(
        json.dumps(
            {
                "camera_path": [
                    {"camera_to_world": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1.0, 0, 0, 0, 1], "fov": 60}
                    for _ in range(5)
                ]
            }
        )
    )
    out_video = tmp_path / "out.mp4"
    urdf = Path("/Users/nijelhunt_1/workspace/BlueprintValidation/configs/robots/sample_6dof_arm.urdf")
    metrics = composite_robot_arm_into_clip(
        input_video=video,
        output_video=out_video,
        camera_path_json=cam_json,
        urdf_path=urdf,
        base_xyz=[0, 0, 0],
        base_rpy=[0, 0, 0],
        start_joints=[0, 0, 0, 0, 0, 0],
        end_joints=[0.1, 0.2, -0.1, 0.1, 0.0, 0.2],
        min_visible_joint_ratio=0.0,
        min_consistency_score=0.0,
    )
    assert out_video.exists()
    assert metrics.clip_name == "in"
