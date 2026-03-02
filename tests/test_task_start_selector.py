"""Tests for task-conditioned start frame selection helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_build_task_start_assignments_prefers_task_relevant_clip(tmp_path):
    from blueprint_validation.evaluation.task_start_selector import build_task_start_assignments

    hints = tmp_path / "task_targets.synthetic.json"
    hints.write_text(
        json.dumps(
            {
                "manipulation_candidates": [
                    {
                        "instance_id": "101",
                        "label": "bowl",
                        "boundingBox": {
                            "center": [0.0, 0.0, 1.0],
                            "extents": [0.1, 0.1, 0.1],
                            "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        },
                    }
                ],
                "articulation_hints": [],
                "navigation_hints": [],
            }
        )
    )

    render_manifest = {
        "clips": [
            {
                "clip_index": 0,
                "clip_name": "clip_000_orbit",
                "path_type": "orbit",
                "video_path": str(tmp_path / "clip_000.mp4"),
                "initial_camera": {
                    "position": [5.0, 5.0, 1.0],
                    "forward": [1.0, 0.0, 0.0],
                },
            },
            {
                "clip_index": 1,
                "clip_name": "clip_001_manipulation",
                "path_type": "manipulation",
                "video_path": str(tmp_path / "clip_001.mp4"),
                "initial_camera": {
                    "position": [0.2, 0.0, 1.0],
                    "forward": [-1.0, 0.0, 0.0],
                },
            },
        ]
    }

    tasks = ["Pick up bowl_101 and place it in the target zone"]
    assignments = build_task_start_assignments(
        tasks=tasks,
        num_rollouts=1,
        render_manifest=render_manifest,
        task_hints_path=hints,
    )

    assert len(assignments) == 1
    assert assignments[0]["clip_index"] == 1
    assert assignments[0]["target_instance_id"] == "101"


def test_shared_manifest_compatibility(tmp_path):
    from blueprint_validation.evaluation.task_start_selector import (
        save_shared_task_start_manifest,
        shared_manifest_is_compatible,
    )

    render_manifest_path = tmp_path / "renders" / "render_manifest.json"
    render_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    render_manifest_path.write_text("{}")

    render_manifest = {"clips": [{"clip_index": 0, "clip_name": "clip_000"}]}
    tasks = ["Pick up bowl_101 and place it in the target zone"]
    manifest_path = tmp_path / "shared_task_start_manifest.json"
    save_shared_task_start_manifest(
        path=manifest_path,
        facility_name="Kitchen A",
        render_manifest_path=render_manifest_path,
        task_profile="dreamdojo",
        requested_rollouts=1,
        planned_rollouts=1,
        tasks=tasks,
        assignments=[{"rollout_index": 0}],
        render_manifest=render_manifest,
        video_orientation_fix="rotate180",
        selector_config={
            "min_assignment_quality_score": 0.1,
            "require_object_grounded_manip_tasks": True,
        },
    )
    manifest = json.loads(manifest_path.read_text())
    assert shared_manifest_is_compatible(
        manifest,
        facility_name="Kitchen A",
        render_manifest_path=render_manifest_path,
        render_manifest=render_manifest,
        tasks=tasks,
        video_orientation_fix="rotate180",
        selector_config={
            "min_assignment_quality_score": 0.1,
            "require_object_grounded_manip_tasks": True,
        },
    )
    assert not shared_manifest_is_compatible(
        manifest,
        facility_name="Kitchen B",
        render_manifest_path=render_manifest_path,
    )
    assert not shared_manifest_is_compatible(
        manifest,
        facility_name="Kitchen A",
        render_manifest_path=render_manifest_path,
        render_manifest=render_manifest,
        tasks=["Pick up cup_202 and place it in the target zone"],
        video_orientation_fix="rotate180",
        selector_config={
            "min_assignment_quality_score": 0.1,
            "require_object_grounded_manip_tasks": True,
        },
    )


def test_build_task_start_assignments_penalizes_ceiling_off_target(tmp_path):
    from blueprint_validation.evaluation.task_start_selector import build_task_start_assignments

    hints = tmp_path / "task_targets.synthetic.json"
    hints.write_text(
        json.dumps(
            {
                "manipulation_candidates": [
                    {
                        "instance_id": "157",
                        "label": "trash can",
                        "boundingBox": {
                            "center": [1.0, 1.0, 0.7],
                            "extents": [0.2, 0.2, 0.3],
                            "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        },
                    }
                ],
                "articulation_hints": [],
                "navigation_hints": [],
            }
        )
    )

    render_manifest = {
        "clips": [
            {
                "clip_index": 0,
                "clip_name": "clip_000_manip_bad",
                "path_type": "manipulation",
                "video_path": str(tmp_path / "clip_000.mp4"),
                "ceiling_dominance_flag": True,
                "initial_camera": {"position": [6.0, 6.0, 2.5], "forward": [1.0, 0.0, 0.0]},
                "path_context": {"type": "manipulation"},
            },
            {
                "clip_index": 1,
                "clip_name": "clip_001_manip_good",
                "path_type": "manipulation",
                "video_path": str(tmp_path / "clip_001.mp4"),
                "initial_camera": {"position": [1.2, 1.2, 0.9], "forward": [-0.7, -0.7, -0.1]},
                "path_context": {"type": "manipulation", "approach_point": [1.0, 1.0, 0.7]},
                "target_visibility_ratio": 0.8,
                "target_center_band_ratio": 0.7,
                "clip_quality_score": 0.9,
            },
        ]
    }
    tasks = ["Pick up trash_can_157 and place it in the target zone"]
    assignments = build_task_start_assignments(
        tasks=tasks,
        num_rollouts=1,
        render_manifest=render_manifest,
        task_hints_path=hints,
        min_assignment_quality_score=0.0,
        require_object_grounded_manip_tasks=True,
    )
    assert len(assignments) == 1
    assert assignments[0]["clip_index"] == 1
    assert assignments[0]["target_grounded"] is True
    assert assignments[0]["assignment_reject_reason"] is None


def test_load_initial_frames_applies_assignment_orientation_fix(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from blueprint_validation.evaluation.task_start_selector import load_initial_frames_for_assignments

    video_path = tmp_path / "clip.mp4"
    h, w = 24, 32
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (w, h),
    )
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[0:4, 0:4] = [0, 0, 255]  # BGR red in top-left.
    writer.write(frame)
    writer.release()

    assignments = [
        {
            "rollout_index": 0,
            "clip_index": 0,
            "clip_name": "clip_000_orbit",
            "video_path": str(video_path),
            "video_orientation_fix": "rotate180",
        }
    ]
    frames = load_initial_frames_for_assignments(assignments)
    assert 0 in frames
    rotated = frames[0]
    # Converted to RGB during decode; red should land near bottom-right after rotation.
    assert rotated[h - 2, w - 2, 0] > 200
    assert assignments[0]["start_frame_orientation_fix_applied"] == "rotate180"
