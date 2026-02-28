"""Tests for task-scoped scene-aware OBB selection in Stage 1 render."""

from __future__ import annotations

import numpy as np


def _obb(instance_id: str, label: str, center: tuple[float, float, float], category: str, confidence: float = 1.0):
    from blueprint_validation.rendering.scene_geometry import OrientedBoundingBox

    return OrientedBoundingBox(
        instance_id=instance_id,
        label=label,
        center=np.asarray(center, dtype=np.float64),
        extents=np.asarray([0.3, 0.3, 0.3], dtype=np.float64),
        axes=np.eye(3, dtype=np.float64),
        confidence=confidence,
        category=category,
    )


def test_task_scoped_selection_targets_context_and_overview():
    from blueprint_validation.stages.s1_render import _select_task_scoped_obbs

    obbs = [
        _obb("1", "mug", (0.0, 0.0, 0.0), "manipulation"),
        _obb("2", "bowl", (0.2, 0.0, 0.0), "manipulation"),
        _obb("3", "plate", (0.4, 0.0, 0.0), "manipulation"),
        _obb("4", "fridge", (5.0, 0.0, 0.0), "articulation"),
        _obb("5", "hallway", (8.0, 0.0, 0.0), "navigation"),
        _obb("6", "cabinet", (6.0, 0.0, 0.0), "articulation"),
    ]
    selected, stats = _select_task_scoped_obbs(
        obbs=obbs,
        tasks=["Pick up mug_1 and place it on the counter"],
        max_specs=4,
        context_per_target=2,
        overview_specs=2,
        fallback_specs=3,
    )

    assert len(selected) == 4
    assert selected[0].instance_id == "1"  # primary task target first
    assert stats["targets"] == 1
    assert stats["context"] == 2
    assert stats["overview"] == 1
    assert stats["fallback"] == 0
    selected_ids = {o.instance_id for o in selected}
    assert {"1", "2", "3"}.issubset(selected_ids)


def test_task_scoped_selection_fallback_when_no_task_targets_match():
    from blueprint_validation.stages.s1_render import _select_task_scoped_obbs

    obbs = [
        _obb("10", "hallway", (8.0, 0.0, 0.0), "navigation", confidence=1.0),
        _obb("11", "mug", (0.0, 0.0, 0.0), "manipulation", confidence=0.8),
        _obb("12", "bowl", (1.0, 0.0, 0.0), "manipulation", confidence=0.9),
        _obb("13", "cabinet", (2.0, 0.0, 0.0), "articulation", confidence=1.0),
    ]
    selected, stats = _select_task_scoped_obbs(
        obbs=obbs,
        tasks=["Do something unrelated"],
        max_specs=3,
        context_per_target=1,
        overview_specs=1,
        fallback_specs=3,
    )

    assert len(selected) == 3
    assert stats["fallback"] == 3
    assert stats["targets"] == 0
    # Fallback prioritizes manipulation/articulation before navigation.
    assert selected[0].category == "manipulation"
    assert selected[1].category in {"manipulation", "articulation"}


def test_task_scoped_selection_resolves_label_instance_token():
    from blueprint_validation.stages.s1_render import _select_task_scoped_obbs

    obbs = [
        _obb("101", "bowl", (0.0, 0.0, 0.0), "manipulation"),
        _obb("102", "cup", (1.0, 0.0, 0.0), "manipulation"),
        _obb("103", "cabinet", (2.0, 0.0, 0.0), "articulation"),
    ]
    selected, stats = _select_task_scoped_obbs(
        obbs=obbs,
        tasks=["Pick up bowl_101 and place it in the sink"],
        max_specs=2,
        context_per_target=0,
        overview_specs=0,
        fallback_specs=2,
    )

    assert len(selected) == 1
    assert selected[0].instance_id == "101"
    assert stats["targets"] == 1
