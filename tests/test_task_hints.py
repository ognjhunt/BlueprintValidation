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
            {"label": "tote", "instance_id": "101"},
            {"label": "box"},
            {"label": "staging_lane", "category": "navigation"},
        ],
        "articulation_hints": [
            {"label": "door", "instance_id": "7"},
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
    assert "Pick up tote_101 and place it in the target zone" in tasks
    assert "Open and close door_7" in tasks
    assert "Pick up the box and place it in the target zone" in tasks
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


def test_tasks_from_task_hints_balances_families(tmp_path):
    from blueprint_validation.evaluation.task_hints import tasks_from_task_hints

    path = tmp_path / "task_targets.json"
    payload = {
        "tasks": (
            [{"task_id": "pick_place_manipulation"}, {"task_id": "open_close_access_points"}]
            + [{"task_id": f"Pick up bowl_{i} and place it in the target zone"} for i in range(1, 10)]
            + [
                {"task_id": "Pick up cup_1 and place it in the target zone"},
                {"task_id": "Pick up kettle_1 and place it in the target zone"},
                {"task_id": "Pick up spoon_1 and place it in the target zone"},
                {"task_id": "Open and close door_1"},
                {"task_id": "Open and close door_2"},
                {"task_id": "Open and close drawer_1"},
                {"task_id": "Open and close cabinet_1"},
                {"task_id": "Navigate to the sink"},
            ]
        ),
        "manipulation_candidates": [],
        "articulation_hints": [],
        "navigation_hints": [],
    }
    path.write_text(json.dumps(payload))

    tasks = tasks_from_task_hints(path, profile="dreamdojo")
    bowl_tasks = [t for t in tasks if "bowl_" in t.lower()]
    assert len(bowl_tasks) <= 2
    assert len(tasks) <= 18
    assert any(t.lower().startswith("open and close ") for t in tasks)
    assert any(t.lower().startswith("navigate ") for t in tasks)


def test_recommended_rollouts_per_condition():
    from blueprint_validation.evaluation.task_hints import recommended_rollouts_per_condition

    dream = recommended_rollouts_per_condition(num_unique_tasks=15, requested=50, profile="dreamdojo")
    policy = recommended_rollouts_per_condition(num_unique_tasks=24, requested=50, profile="policy")
    assert 80 <= dream <= 125
    assert 80 <= policy <= 200
