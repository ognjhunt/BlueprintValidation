import pytest

from blueprint_validation.evaluation.claim_protocol import build_claim_split_payload, build_task_specs


def test_build_claim_split_payload_holds_out_both_tasks_and_starts(sample_config, tmp_path):
    hints = tmp_path / "task_targets.json"
    hints.write_text(
        """
        {
          "manipulation_candidates": [{"label": "bowl", "instance_id": "101"}],
          "navigation_hints": [{"label": "sink", "instance_id": "201"}],
          "articulation_hints": []
        }
        """.strip()
    )
    facility = sample_config.facilities["test_facility"]
    facility.task_hints_path = hints
    tasks = [
        "Pick up bowl_101 and place it in the target zone",
        "Navigate to the sink",
    ]
    task_specs = build_task_specs(config=sample_config, facility=facility, tasks=tasks)
    assignments = [
        {"rollout_index": 0, "task": tasks[0], "clip_index": 0, "clip_name": "clip_000", "path_type": "manipulation", "target_instance_id": "101"},
        {"rollout_index": 1, "task": tasks[1], "clip_index": 0, "clip_name": "clip_000", "path_type": "navigation", "target_instance_id": "201"},
        {"rollout_index": 2, "task": tasks[0], "clip_index": 1, "clip_name": "clip_001", "path_type": "manipulation", "target_instance_id": "101"},
        {"rollout_index": 3, "task": tasks[1], "clip_index": 1, "clip_name": "clip_001", "path_type": "navigation", "target_instance_id": "201"},
    ]
    split = build_claim_split_payload(
        task_specs=task_specs,
        assignments=assignments,
        world_snapshot_hash="world_hash",
        train_split=0.5,
        split_strategy="disjoint_tasks_and_starts",
    )
    train_ids = set(split["train_eval_cell_ids"])
    eval_ids = set(split["eval_eval_cell_ids"])
    assert train_ids
    assert eval_ids
    assert train_ids.isdisjoint(eval_ids)
    eval_cells = [cell for cell in split["cells"] if cell["eval_cell_id"] in eval_ids]
    train_cells = [cell for cell in split["cells"] if cell["eval_cell_id"] in train_ids]
    assert {cell["task_spec_id"] for cell in eval_cells}.isdisjoint(
        {cell["task_spec_id"] for cell in train_cells}
    )
    assert {cell["start_clip_id"] for cell in eval_cells}.isdisjoint(
        {cell["start_clip_id"] for cell in train_cells}
    )


def test_build_task_specs_preserves_requested_articulation_sequence(sample_config, tmp_path):
    hints = tmp_path / "task_targets.json"
    hints.write_text(
        """
        {
          "articulation_hints": [{"label": "cabinet", "instance_id": "7"}]
        }
        """.strip()
    )
    facility = sample_config.facilities["test_facility"]
    facility.task_hints_path = hints
    specs = build_task_specs(
        config=sample_config,
        facility=facility,
        tasks=["Open cabinet", "Close cabinet", "Open and close cabinet"],
    )
    by_prompt = {spec["task_prompt"]: spec for spec in specs}
    assert by_prompt["Open cabinet"]["success_predicate"]["sequence"] == ["open"]
    assert by_prompt["Close cabinet"]["success_predicate"]["sequence"] == ["close"]
    assert by_prompt["Open and close cabinet"]["success_predicate"]["sequence"] == ["open", "close"]


def test_build_claim_split_payload_rejects_non_disjoint_protocol_split(sample_config, tmp_path):
    hints = tmp_path / "task_targets.json"
    hints.write_text(
        """
        {
          "navigation_hints": [{"label": "sink", "instance_id": "201"}]
        }
        """.strip()
    )
    facility = sample_config.facilities["test_facility"]
    facility.task_hints_path = hints
    tasks = ["Navigate to the sink"]
    task_specs = build_task_specs(config=sample_config, facility=facility, tasks=tasks)
    assignments = [
        {
            "rollout_index": 0,
            "task": tasks[0],
            "clip_index": 0,
            "clip_name": "clip_000",
            "path_type": "navigation",
            "target_instance_id": "201",
        },
        {
            "rollout_index": 1,
            "task": tasks[0],
            "clip_index": 1,
            "clip_name": "clip_001",
            "path_type": "navigation",
            "target_instance_id": "201",
        },
    ]
    with pytest.raises(ValueError, match="two disjoint task specs"):
        build_claim_split_payload(
            task_specs=task_specs,
            assignments=assignments,
            world_snapshot_hash="world_hash",
            train_split=0.5,
            split_strategy="disjoint_tasks_and_starts",
        )


def test_build_claim_split_payload_expands_holdouts_to_meet_eval_cell_floor():
    task_specs = [
        {
            "task_spec_id": f"task_spec_{idx:02d}",
            "task_prompt": f"Navigate to region {idx:02d}",
            "task_family": "navigation",
        }
        for idx in range(12)
    ]
    assignments = []
    rollout_index = 0
    for task_idx, spec in enumerate(task_specs):
        for start_idx in range(12):
            assignments.append(
                {
                    "rollout_index": rollout_index,
                    "task": spec["task_prompt"],
                    "clip_index": start_idx,
                    "clip_name": f"clip_{start_idx:03d}",
                    "path_type": "navigation",
                    "target_instance_id": f"{200 + task_idx}",
                }
            )
            rollout_index += 1

    split = build_claim_split_payload(
        task_specs=task_specs,
        assignments=assignments,
        world_snapshot_hash="world_hash",
        train_split=0.8,
        split_strategy="disjoint_tasks_and_starts",
        min_eval_task_specs=3,
        min_eval_start_clips=3,
        min_common_eval_cells=30,
    )

    eval_ids = set(split["eval_eval_cell_ids"])
    eval_cells = [cell for cell in split["cells"] if cell["eval_cell_id"] in eval_ids]
    assert len(eval_cells) >= 30
    assert len({cell["task_spec_id"] for cell in eval_cells}) >= 3
    assert len({cell["start_clip_id"] for cell in eval_cells}) >= 3
