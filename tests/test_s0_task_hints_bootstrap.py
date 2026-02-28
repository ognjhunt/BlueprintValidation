"""Tests for Stage 0 synthetic task-hints bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blueprint_validation.common import read_json
from blueprint_validation.config import CameraPathSpec, FacilityConfig, ValidationConfig
from blueprint_validation.rendering.vlm_scene_detector import (
    DetectedRegion,
    SceneDetectionResult,
)
from blueprint_validation.stages.s0_task_hints_bootstrap import (
    TaskHintsBootstrapStage,
    _derive_tasks_from_detections,
    _infer_category_from_label,
    _obb_from_corners,
    _obb_from_position_size,
    ingest_interiorgs,
)


def test_bootstrap_skips_when_hints_already_exist(tmp_path):
    hints = tmp_path / "existing_task_targets.json"
    hints.write_text(json.dumps({"tasks": [], "manipulation_candidates": []}))

    config = ValidationConfig()
    fac = FacilityConfig(name="A", ply_path=Path("/tmp/a.ply"), task_hints_path=hints)
    stage = TaskHintsBootstrapStage()

    result = stage.execute(config, fac, tmp_path / "outputs" / "a", {})
    assert result.status == "skipped"
    assert result.outputs["task_hints_path"] == str(hints)


def test_bootstrap_writes_synthetic_hints_from_vlm(sample_ply, tmp_path, monkeypatch):
    config = ValidationConfig()
    fac = FacilityConfig(name="A", ply_path=sample_ply)
    stage = TaskHintsBootstrapStage()

    vlm_result = SceneDetectionResult(
        specs=[
            CameraPathSpec(type="manipulation", approach_point=[1.0, 2.0, 0.5], arc_radius_m=0.6),
            CameraPathSpec(type="manipulation", approach_point=[2.0, 3.0, 0.8], arc_radius_m=0.7),
            CameraPathSpec(type="manipulation", approach_point=[4.0, 1.5, 0.2], arc_radius_m=0.9),
        ],
        detections=[
            DetectedRegion(
                label="coffee_mug",
                center_3d=np.array([1.0, 2.0, 0.5], dtype=np.float32),
                extents_3d=np.array([0.12, 0.08, 0.15], dtype=np.float32),
                category="manipulation",
            ),
            DetectedRegion(
                label="cabinet_door",
                center_3d=np.array([2.0, 3.0, 0.8], dtype=np.float32),
                extents_3d=np.array([0.60, 0.04, 0.80], dtype=np.float32),
                category="articulation",
            ),
            DetectedRegion(
                label="loading_dock",
                center_3d=np.array([4.0, 1.5, 0.2], dtype=np.float32),
                extents_3d=np.array([1.50, 1.20, 0.20], dtype=np.float32),
                category="navigation",
            ),
        ],
        scene_type="kitchen",
        suggested_tasks=[
            {"suggested_task": "Pick up the coffee mug"},
            {"suggested_task": "Open the cabinet door"},
        ],
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s0_task_hints_bootstrap.detect_and_generate_specs",
        lambda **kwargs: vlm_result,
    )

    work_dir = tmp_path / "outputs" / "a"
    result = stage.execute(config, fac, work_dir, {})
    assert result.status == "success"
    assert result.metrics["source"] == "vlm"
    assert result.metrics["num_candidates"] == 1
    assert result.metrics["num_articulation"] == 1
    assert result.metrics["num_navigation"] == 1
    assert result.metrics["scene_type"] == "kitchen"

    hints_path = Path(result.outputs["task_hints_path"])
    assert hints_path.exists()
    assert fac.task_hints_path == hints_path

    payload = read_json(hints_path)
    assert payload["bootstrap_generated"] is True
    assert payload["scene_type"] == "kitchen"

    manip = payload["manipulation_candidates"][0]
    assert manip["label"] == "coffee_mug"
    np.testing.assert_allclose(manip["boundingBox"]["extents"], [0.12, 0.08, 0.15], atol=1e-6)
    np.testing.assert_allclose(manip["boundingBox"]["center"], [1.0, 2.0, 0.5], atol=1e-10)

    artic = payload["articulation_hints"][0]
    assert artic["label"] == "cabinet_door"
    np.testing.assert_allclose(artic["boundingBox"]["extents"], [0.60, 0.04, 0.80], atol=1e-6)
    nav = payload["navigation_hints"][0]
    assert nav["label"] == "loading_dock"
    np.testing.assert_allclose(nav["boundingBox"]["extents"], [1.50, 1.20, 0.20], atol=1e-6)

    task_ids = {t["task_id"] for t in payload["tasks"]}
    assert "Pick up the coffee mug" in task_ids
    assert "Open the cabinet door" in task_ids

    # Later runs should deterministically reuse/skip existing hints.
    result2 = stage.execute(config, fac, work_dir, {})
    assert result2.status == "skipped"


def test_bootstrap_fails_when_vlm_empty_requires_manual_analysis(sample_ply, tmp_path, monkeypatch):
    config = ValidationConfig()
    fac = FacilityConfig(name="A", ply_path=sample_ply)
    stage = TaskHintsBootstrapStage()

    monkeypatch.setattr(
        "blueprint_validation.stages.s0_task_hints_bootstrap.detect_and_generate_specs",
        lambda **kwargs: SceneDetectionResult(specs=[], detections=[]),
    )

    result = stage.execute(config, fac, tmp_path / "outputs" / "a", {})
    assert result.status == "failed"
    assert "manual analysis is required" in (result.detail or "").lower()


def test_bootstrap_writes_centers_in_original_frame_for_y_up(sample_ply, tmp_path, monkeypatch):
    config = ValidationConfig()
    fac = FacilityConfig(name="A", ply_path=sample_ply, up_axis="y")
    stage = TaskHintsBootstrapStage()

    vlm_result = SceneDetectionResult(
        specs=[
            CameraPathSpec(type="manipulation", approach_point=[0.0, 0.0, 5.0], arc_radius_m=0.8)
        ],
        detections=[
            DetectedRegion(
                label="target",
                center_3d=np.array([0.0, 0.0, 5.0], dtype=np.float32),
                extents_3d=np.array([0.2, 0.4, 0.6], dtype=np.float32),
                category="manipulation",
            )
        ],
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s0_task_hints_bootstrap.detect_and_generate_specs",
        lambda **kwargs: vlm_result,
    )

    result = stage.execute(config, fac, tmp_path / "outputs" / "a", {})
    assert result.status == "success"
    payload = read_json(Path(result.outputs["task_hints_path"]))
    center = payload["manipulation_candidates"][0]["boundingBox"]["center"]
    extents = payload["manipulation_candidates"][0]["boundingBox"]["extents"]
    np.testing.assert_allclose(center, [0.0, 5.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(extents, [0.2, 0.6, 0.4], atol=1e-8)


def test_derive_tasks_from_detections_prefers_suggestions_then_categories():
    detections = [
        DetectedRegion(label="mug", center_3d=np.zeros(3), category="manipulation"),
        DetectedRegion(label="door", center_3d=np.ones(3), category="articulation"),
    ]
    suggested = [{"suggested_task": "Pick up the mug"}]

    tasks = _derive_tasks_from_detections(detections, suggested, scene_type="office")
    task_ids = {t["task_id"] for t in tasks}

    assert "Pick up the mug" in task_ids
    assert "open_close_access_points" in task_ids


# ---------------------------------------------------------------------------
# InteriorGS direct ingestion unit tests
# ---------------------------------------------------------------------------


def _make_box_corners(center, size):
    """Generate 8 corners of an AABB given center + [w, l, h]."""
    cx, cy, cz = center
    hw, hl, hh = size[0] / 2, size[1] / 2, size[2] / 2
    return [
        [cx - hw, cy - hl, cz - hh],
        [cx + hw, cy - hl, cz - hh],
        [cx + hw, cy + hl, cz - hh],
        [cx - hw, cy + hl, cz - hh],
        [cx - hw, cy - hl, cz + hh],
        [cx + hw, cy - hl, cz + hh],
        [cx + hw, cy + hl, cz + hh],
        [cx - hw, cy + hl, cz + hh],
    ]


def test_infer_category_manipulation():
    assert _infer_category_from_label("coffee_mug") == "manipulation"
    assert _infer_category_from_label("Chair") == "manipulation"
    assert _infer_category_from_label("book") == "manipulation"
    # Should not false-match "door" inside "outdoor".
    assert _infer_category_from_label("outdoor_table") == "manipulation"


def test_infer_category_articulation():
    assert _infer_category_from_label("cabinet_door") == "articulation"
    assert _infer_category_from_label("kitchen_drawer") == "articulation"
    assert _infer_category_from_label("refrigerator") == "articulation"
    assert _infer_category_from_label("window_frame") == "articulation"


def test_infer_category_navigation():
    assert _infer_category_from_label("hallway") == "navigation"
    assert _infer_category_from_label("room_area") == "navigation"
    assert _infer_category_from_label("main_corridor") == "navigation"


def test_obb_from_corners_computes_center_and_extents():
    corners = _make_box_corners([1.0, 2.0, 0.5], [0.4, 0.3, 0.6])
    result = _obb_from_corners(corners)
    assert result is not None
    np.testing.assert_allclose(result["center"], [1.0, 2.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(result["extents"], [0.4, 0.3, 0.6], atol=1e-6)
    assert np.allclose(result["axes"], np.eye(3).tolist())


def test_obb_from_corners_accepts_dict_points():
    """Actual InteriorGS labels.json uses {x,y,z} dicts, not [x,y,z] lists."""
    cx, cy, cz, hw, hl, hh = 1.0, 2.0, 0.5, 0.2, 0.15, 0.3
    corners = [
        {"x": cx - hw, "y": cy - hl, "z": cz - hh},
        {"x": cx + hw, "y": cy - hl, "z": cz - hh},
        {"x": cx + hw, "y": cy + hl, "z": cz - hh},
        {"x": cx - hw, "y": cy + hl, "z": cz - hh},
        {"x": cx - hw, "y": cy - hl, "z": cz + hh},
        {"x": cx + hw, "y": cy - hl, "z": cz + hh},
        {"x": cx + hw, "y": cy + hl, "z": cz + hh},
        {"x": cx - hw, "y": cy + hl, "z": cz + hh},
    ]
    result = _obb_from_corners(corners)
    assert result is not None
    np.testing.assert_allclose(result["center"], [cx, cy, cz], atol=1e-6)
    np.testing.assert_allclose(result["extents"], [hw * 2, hl * 2, hh * 2], atol=1e-6)


def test_obb_from_corners_returns_none_for_malformed():
    assert _obb_from_corners([[1, 2]]) is None  # not 3D
    assert _obb_from_corners([[1, 2, 3]]) is None  # only 1 point
    assert _obb_from_corners("bad") is None  # wrong type


def test_ingest_interiorgs_actual_file_format(tmp_path):
    """labels.json as a list (actual InteriorGS on-disk format, not normalised dict)."""
    fac = FacilityConfig(name="Test", ply_path=tmp_path / "scene.ply")

    def _dict_corners(center, size):
        cx, cy, cz = center
        hw, hl, hh = size[0] / 2, size[1] / 2, size[2] / 2
        return [
            {"x": cx - hw, "y": cy - hl, "z": cz - hh},
            {"x": cx + hw, "y": cy - hl, "z": cz - hh},
            {"x": cx + hw, "y": cy + hl, "z": cz - hh},
            {"x": cx - hw, "y": cy + hl, "z": cz - hh},
            {"x": cx - hw, "y": cy - hl, "z": cz + hh},
            {"x": cx + hw, "y": cy - hl, "z": cz + hh},
            {"x": cx + hw, "y": cy + hl, "z": cz + hh},
            {"x": cx - hw, "y": cy + hl, "z": cz + hh},
        ]

    # A list (not dict) with ins_id / label keys
    labels = [
        {
            "ins_id": "42",
            "label": "coffee_mug",
            "bounding_box": _dict_corners([1.0, 2.0, 0.5], [0.12, 0.12, 0.15]),
        },
        {
            "ins_id": "43",
            "label": "window",
            "bounding_box": _dict_corners([3.0, 0.0, 1.2], [1.2, 0.1, 0.8]),
        },
    ]
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))

    payload = ingest_interiorgs(labels_path, None, fac)
    assert payload["source"] == "interiorgs"

    manip = payload["manipulation_candidates"]
    artic = payload["articulation_hints"]
    assert len(manip) == 1
    assert manip[0]["label"] == "coffee_mug"
    assert manip[0]["instance_id"] == "42"
    assert len(artic) == 1
    assert artic[0]["label"] == "window"
    assert artic[0]["instance_id"] == "43"
    np.testing.assert_allclose(manip[0]["boundingBox"]["center"], [1.0, 2.0, 0.5], atol=1e-5)


def test_obb_from_position_size():
    result = _obb_from_position_size([3.0, 1.0, 0.5], [0.6, 0.4, 1.0])
    assert result is not None
    np.testing.assert_allclose(result["center"], [3.0, 1.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(result["extents"], [0.6, 0.4, 1.0], atol=1e-6)


def test_ingest_interiorgs_labels_only(tmp_path):
    """Basic ingestion: labels.json with mixed objects, no structure.json."""
    fac = FacilityConfig(name="Test", ply_path=tmp_path / "scene.ply")

    labels = {
        "objects": [
            {
                "instance_id": "obj_001",
                "semantic_label": "coffee_mug",
                "bounding_box": _make_box_corners([1.0, 2.0, 0.5], [0.12, 0.12, 0.15]),
            },
            {
                "instance_id": "obj_002",
                "semantic_label": "cabinet_door",
                "bounding_box": _make_box_corners([2.0, 3.0, 1.0], [0.6, 0.04, 0.8]),
            },
            {
                "instance_id": "obj_003",
                "semantic_label": "sofa",
                "bounding_box": _make_box_corners([0.0, 0.0, 0.25], [1.8, 0.9, 0.5]),
            },
        ]
    }
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))

    payload = ingest_interiorgs(labels_path, None, fac)

    assert payload["source"] == "interiorgs"
    assert payload["resolved_up_axis"] == "z"
    assert payload["scene_type"] == "indoor"  # no structure.json

    manip = payload["manipulation_candidates"]
    artic = payload["articulation_hints"]
    assert len(manip) == 2  # coffee_mug + sofa
    assert len(artic) == 1  # cabinet_door

    labels_seen = {e["label"] for e in manip}
    assert "coffee_mug" in labels_seen
    assert "sofa" in labels_seen
    assert artic[0]["label"] == "cabinet_door"

    # OBB center should round-trip correctly
    mug = next(e for e in manip if e["label"] == "coffee_mug")
    np.testing.assert_allclose(mug["boundingBox"]["center"], [1.0, 2.0, 0.5], atol=1e-5)
    np.testing.assert_allclose(mug["boundingBox"]["extents"], [0.12, 0.12, 0.15], atol=1e-5)

    task_ids = {t["task_id"] for t in payload["tasks"]}
    assert "pick_place_manipulation" in task_ids
    assert "open_close_access_points" in task_ids
    assert any(t["task_id"].startswith("Pick up ") for t in payload["tasks"])
    assert any(t["task_id"].startswith("Open and close ") for t in payload["tasks"])


def test_ingest_interiorgs_scene_type_from_structure(tmp_path):
    """structure.json room_type drives scene_type."""
    fac = FacilityConfig(name="Test", ply_path=tmp_path / "scene.ply")

    labels = {
        "objects": [
            {
                "instance_id": "obj_001",
                "semantic_label": "coffee_mug",
                "bounding_box": _make_box_corners([0.0, 0.0, 0.5], [0.1, 0.1, 0.2]),
            }
        ]
    }
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))

    structure = {
        "rooms": [
            {"room_type": "Kitchen", "profile": []},
            {"room_type": "Kitchen", "profile": []},
            {"room_type": "Living Room", "profile": []},
        ],
        "holes": [],
        "ins": [],
    }
    structure_path = tmp_path / "structure.json"
    structure_path.write_text(json.dumps(structure))

    payload = ingest_interiorgs(labels_path, structure_path, fac)
    assert payload["scene_type"] == "kitchen"  # most common


def test_ingest_interiorgs_adds_holes_from_structure(tmp_path):
    """structure.json DOOR/WINDOW holes become articulation hints."""
    fac = FacilityConfig(name="Test", ply_path=tmp_path / "scene.ply")

    labels = {"objects": []}
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))

    door_profile = _make_box_corners([5.0, 0.0, 1.0], [0.9, 0.1, 2.1])
    window_profile = _make_box_corners([0.0, 3.0, 1.2], [1.2, 0.1, 0.8])
    structure = {
        "rooms": [],
        "holes": [
            {"type": "DOOR", "profile": door_profile, "thickness": 0.1},
            {"type": "WINDOW", "profile": window_profile, "thickness": 0.1},
        ],
        "ins": [],
    }
    structure_path = tmp_path / "structure.json"
    structure_path.write_text(json.dumps(structure))

    payload = ingest_interiorgs(labels_path, structure_path, fac)
    artic = payload["articulation_hints"]
    assert len(artic) == 2
    artic_labels = {e["label"] for e in artic}
    assert "door" in artic_labels
    assert "window" in artic_labels
    assert all(e["source"] == "interiorgs_structure" for e in artic)

    # Hole IDs should be deterministic across runs.
    payload2 = ingest_interiorgs(labels_path, structure_path, fac)
    ids1 = sorted(e["instance_id"] for e in payload["articulation_hints"])
    ids2 = sorted(e["instance_id"] for e in payload2["articulation_hints"])
    assert ids1 == ids2


def test_ingest_interiorgs_ins_not_duplicated_when_label_covered(tmp_path):
    """structure.json ins entries are skipped if labels.json already has that class."""
    fac = FacilityConfig(name="Test", ply_path=tmp_path / "scene.ply")

    labels = {
        "objects": [
            {
                "instance_id": "chair_001",
                "semantic_label": "Chair",
                "bounding_box": _make_box_corners([1.0, 1.0, 0.45], [0.5, 0.5, 0.9]),
            }
        ]
    }
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))

    structure = {
        "rooms": [],
        "holes": [],
        "ins": [
            # "office_chair" semantic class should be considered covered by "Chair"
            {"label": "office_chair", "position": [2.0, 2.0, 0.45], "size": [0.5, 0.5, 0.9]},
            # "Table" not yet covered — should be added
            {"label": "Table", "position": [3.0, 3.0, 0.4], "size": [1.2, 0.8, 0.75]},
        ],
    }
    structure_path = tmp_path / "structure.json"
    structure_path.write_text(json.dumps(structure))

    payload = ingest_interiorgs(labels_path, structure_path, fac)
    manip_labels = [e["label"] for e in payload["manipulation_candidates"]]
    # Chair class should appear exactly once (from labels.json)
    assert manip_labels.count("Chair") == 1
    assert "office_chair" not in manip_labels
    # Table should appear (from structure ins)
    assert "Table" in manip_labels


def test_bootstrap_stage_fails_when_interiorgs_has_no_usable_hints(
    sample_ply, tmp_path, monkeypatch
):
    """labels.json present but empty/malformed should fail with manual guidance."""
    (sample_ply.parent / "labels.json").write_text(json.dumps({"objects": []}))

    # VLM should still not be called in this path.
    def _fail_if_called(**kwargs):
        raise AssertionError("VLM was called despite labels.json being present")

    monkeypatch.setattr(
        "blueprint_validation.stages.s0_task_hints_bootstrap.detect_and_generate_specs",
        _fail_if_called,
    )

    config = ValidationConfig()
    fac = FacilityConfig(name="A", ply_path=sample_ply)
    stage = TaskHintsBootstrapStage()
    result = stage.execute(config, fac, tmp_path / "outputs" / "a", {})
    assert result.status == "failed"
    assert "manual analysis is required" in (result.detail or "").lower()


def test_bootstrap_stage_uses_interiorgs_when_labels_json_present(
    sample_ply, tmp_path, monkeypatch
):
    """Stage should use InteriorGS ingestion and never call VLM when labels.json exists."""
    # Place labels.json in same directory as the PLY
    labels = {
        "objects": [
            {
                "instance_id": "mug_001",
                "semantic_label": "coffee_mug",
                "bounding_box": _make_box_corners([1.0, 1.0, 0.5], [0.1, 0.1, 0.15]),
            },
            {
                "instance_id": "door_001",
                "semantic_label": "cabinet_door",
                "bounding_box": _make_box_corners([3.0, 0.0, 1.0], [0.6, 0.05, 0.8]),
            },
        ]
    }
    (sample_ply.parent / "labels.json").write_text(json.dumps(labels))

    # VLM must NOT be called — if it is, the test fails
    def _fail_if_called(**kwargs):
        raise AssertionError("VLM was called despite labels.json being present")

    monkeypatch.setattr(
        "blueprint_validation.stages.s0_task_hints_bootstrap.detect_and_generate_specs",
        _fail_if_called,
    )

    config = ValidationConfig()
    fac = FacilityConfig(name="A", ply_path=sample_ply)
    stage = TaskHintsBootstrapStage()
    work_dir = tmp_path / "outputs" / "a"

    result = stage.execute(config, fac, work_dir, {})
    assert result.status == "success"
    assert result.metrics["source"] == "interiorgs"
    assert result.metrics["num_candidates"] == 1  # coffee_mug
    assert result.metrics["num_articulation"] == 1  # cabinet_door
    assert result.metrics["resolved_up_axis"] == "z"

    payload = read_json(Path(result.outputs["task_hints_path"]))
    assert payload["source"] == "interiorgs"
    assert payload["manipulation_candidates"][0]["label"] == "coffee_mug"
    assert payload["articulation_hints"][0]["label"] == "cabinet_door"


def test_bootstrap_stage_falls_through_to_vlm_without_labels_json(
    sample_ply, tmp_path, monkeypatch
):
    """Without labels.json the stage should proceed to VLM as before."""
    # Ensure no labels.json is present
    assert not (sample_ply.parent / "labels.json").exists()

    vlm_result = SceneDetectionResult(
        specs=[
            CameraPathSpec(type="manipulation", approach_point=[1.0, 1.0, 0.5], arc_radius_m=0.6)
        ],
        detections=[
            DetectedRegion(
                label="box", center_3d=np.array([1.0, 1.0, 0.5]), category="manipulation"
            )
        ],
        scene_type="warehouse",
        suggested_tasks=[],
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s0_task_hints_bootstrap.detect_and_generate_specs",
        lambda **kwargs: vlm_result,
    )

    config = ValidationConfig()
    fac = FacilityConfig(name="A", ply_path=sample_ply)
    stage = TaskHintsBootstrapStage()

    result = stage.execute(config, fac, tmp_path / "outputs" / "a", {})
    assert result.status == "success"
    assert result.metrics["source"] == "vlm"
