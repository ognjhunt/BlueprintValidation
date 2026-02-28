"""Tests for task-conditioned start frame selection helpers."""

from __future__ import annotations

import json
from pathlib import Path


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
    from blueprint_validation.evaluation.task_start_selector import shared_manifest_is_compatible

    render_manifest_path = tmp_path / "renders" / "render_manifest.json"
    render_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    render_manifest_path.write_text("{}")

    manifest = {
        "facility": "Kitchen A",
        "render_manifest_path": str(render_manifest_path),
        "assignments": [{"rollout_index": 0}],
    }
    assert shared_manifest_is_compatible(
        manifest,
        facility_name="Kitchen A",
        render_manifest_path=render_manifest_path,
    )
    assert not shared_manifest_is_compatible(
        manifest,
        facility_name="Kitchen B",
        render_manifest_path=render_manifest_path,
    )

