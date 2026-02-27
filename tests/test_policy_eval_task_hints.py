"""Unit tests for Stage 4 task list assembly with task hints."""

from __future__ import annotations

import json
from pathlib import Path


def test_build_task_list_includes_task_hints(tmp_path):
    from blueprint_validation.config import FacilityConfig, ValidationConfig
    from blueprint_validation.stages.s4_policy_eval import _build_task_list

    hints = tmp_path / "task_targets.json"
    hints.write_text(
        json.dumps(
            {
                "tasks": [{"task_id": "pick_place_manipulation"}],
                "manipulation_candidates": [{"label": "tote"}],
                "articulation_hints": [{"label": "door"}],
            }
        )
    )

    cfg = ValidationConfig()
    cfg.eval_policy.tasks = ["Navigate forward through the corridor"]
    fac = FacilityConfig(name="A", ply_path=Path("/tmp/a.ply"), task_hints_path=hints)

    tasks, hint_count = _build_task_list(cfg, fac)
    assert hint_count > 0
    assert "Navigate forward through the corridor" in tasks
    assert "Pick up a target object and place it in the target zone" in tasks
    assert "Pick up the tote and place it in the target zone" in tasks
