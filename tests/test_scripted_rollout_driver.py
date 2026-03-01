"""Tests for deterministic scripted world-model rollout traces."""

from __future__ import annotations


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

