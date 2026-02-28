"""Tests for task hint ingestion from BlueprintCapturePipeline artifacts."""

from __future__ import annotations

import json


def test_tasks_from_task_hints_maps_task_ids_and_labels(tmp_path):
    from blueprint_validation.evaluation.task_hints import tasks_from_task_hints

    path = tmp_path / "task_targets.json"
    payload = {
        "tasks": [
            {"task_id": "pick_place_manipulation"},
            {"task_id": "open_close_access_points"},
            {"task_id": "open_close_articulation"},
            {"task_id": "navigate_to_target"},
            {"task_id": "Pick up the coffee mug"},
        ],
        "manipulation_candidates": [
            {"label": "tote"},
            {"label": "box"},
            {"label": "staging_lane", "category": "navigation"},
        ],
        "articulation_hints": [
            {"label": "door"},
        ],
        "navigation_hints": [
            {"label": "charging_station"},
        ],
    }
    path.write_text(json.dumps(payload))

    tasks = tasks_from_task_hints(path)
    assert "Pick up a target object and place it in the target zone" in tasks
    assert "Open and close a nearby door or cabinet" in tasks
    assert "Navigate to the target region while avoiding obstacles" in tasks
    assert "Pick up the coffee mug" in tasks
    assert "Pick up the tote and place it in the target zone" in tasks
    assert "Open and close the door" in tasks
    assert "Navigate to the charging station" in tasks
    assert "Pick up the staging lane and place it in the target zone" not in tasks


def test_tasks_from_task_hints_dedupes(tmp_path):
    from blueprint_validation.evaluation.task_hints import tasks_from_task_hints

    path = tmp_path / "task_targets.json"
    payload = {
        "tasks": [{"task_id": "pick_place_manipulation"}],
        "manipulation_candidates": [{"label": "object"}, {"label": "unknown"}, {"label": "tote"}],
        "articulation_hints": [],
    }
    path.write_text(json.dumps(payload))
    tasks = tasks_from_task_hints(path)
    assert tasks.count("Pick up a target object and place it in the target zone") == 1
    assert "Pick up the tote and place it in the target zone" in tasks
