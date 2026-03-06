"""Tests for deterministic scripted world-model rollout traces."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_scripted_trace_manifest_is_deterministic():
    from blueprint_validation.evaluation.scripted_rollout_driver import (
        build_scripted_trace_manifest,
    )

    assignments = [
        {"rollout_index": 0, "clip_index": 3, "task": "Navigate forward"},
        {"rollout_index": 1, "clip_index": 3, "task": "Pick up tote"},
    ]
    first = build_scripted_trace_manifest(assignments, action_dim=7, max_steps=5)
    second = build_scripted_trace_manifest(assignments, action_dim=7, max_steps=5)
    assert first == second
    assert len(first) == 2


def test_scripted_trace_shapes_match_action_dim():
    from blueprint_validation.evaluation.scripted_rollout_driver import (
        build_scripted_action_trace,
    )

    nav_actions, nav_type = build_scripted_action_trace(
        task="Navigate to loading dock",
        action_dim=7,
        max_steps=6,
        seed=123,
    )
    manip_actions, manip_type = build_scripted_action_trace(
        task="Pick and place the tote",
        action_dim=12,
        max_steps=4,
        seed=456,
    )

    assert nav_type == "navigation_scripted"
    assert manip_type == "manipulation_scripted"
    assert len(nav_actions) == 6
    assert len(manip_actions) == 4
    assert all(len(v) == 7 for v in nav_actions)
    assert all(len(v) == 12 for v in manip_actions)


def test_scripted_rollout_falls_back_to_deterministic_state_proxy(tmp_path, monkeypatch):
    from blueprint_validation.evaluation.scripted_rollout_driver import run_scripted_rollout

    pytest.importorskip("cv2")

    class _FakeWriter:
        def write(self, _frame):
            return None

        def release(self):
            return None

    def _fake_open_mp4_writer(*, output_path, **_kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"x")
        return _FakeWriter()

    monkeypatch.setattr(
        "blueprint_validation.evaluation.scripted_rollout_driver.open_mp4_writer",
        _fake_open_mp4_writer,
    )
    monkeypatch.setattr(
        "blueprint_validation.evaluation.scripted_rollout_driver.ensure_h264_video",
        lambda *, input_path, **_kwargs: SimpleNamespace(path=input_path),
    )

    class _WorldModel:
        expected_action_dim = 7

        def predict_next_frame(self, current_frame, action):
            del action
            return current_frame

    rollout = run_scripted_rollout(
        world_model=_WorldModel(),
        initial_frame=np.zeros((8, 8, 3), dtype=np.uint8),
        action_sequence=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        output_dir=tmp_path,
        clip_name="claim_rollout",
        trace_id="trace_0",
        rollout_context={
            "start_region_id": "manipulation:101",
            "target_instance_id": "101",
            "target_label": "bowl",
            "initial_camera": {
                "position": [0.0, 0.0, 0.8],
                "forward": [1.0, 0.0, 0.0],
                "right": [0.0, 1.0, 0.0],
                "up": [0.0, 0.0, 1.0],
            },
            "path_context": {"approach_point": [0.35, 0.0, 0.65]},
        },
        task_prompt="Pick up bowl_101 and place it in the target zone",
        task_spec={
            "task_family": "manipulation",
            "target_instance_id": "101",
            "target_label": "bowl",
            "goal_region_id": "target_zone",
            "target_center_xyz": [0.35, 0.0, 0.65],
            "goal_point_xyz": [0.35, 0.0, 0.98],
            "success_predicate": {
                "type": "manipulation_pick_place_stable",
                "target_instance_id": "101",
                "goal_region_id": "target_zone",
                "target_center_xyz": [0.35, 0.0, 0.65],
                "goal_point_xyz": [0.35, 0.0, 0.98],
            },
        },
    )

    assert rollout.state_trace
    assert any(bool(row.get("grasp_acquired")) for row in rollout.state_trace)
